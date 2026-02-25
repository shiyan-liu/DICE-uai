#!/usr/bin/env python3
"""
Local pairwise judge based on DeepSeek-R1.
Supports deep thinking mode and constrained A/B/T output via logits.
"""

import torch
import logging
import gc
import re
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


class LocalPairwiseJudge:
    """Pairwise judge using a local DeepSeek-R1 model."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DICE.LocalPairwise")

        self.model_path = "/root/autodl-tmp/deepseek-deployment/deepseek-r1-8b"
        self.enable_deep_thinking = getattr(config, 'enable_deep_thinking', True)

        self.model = None
        self.tokenizer = None
        self.choice_tokens = {}

        self._initialize_model()

    def _initialize_model(self):
        """Load model with memory-optimized settings."""
        self.logger.info(f"Loading DeepSeek-R1 model: {self.model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, use_fast=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = False
            self.model.eval()

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.choice_tokens = {
                "A": self.tokenizer.convert_tokens_to_ids("A"),
                "B": self.tokenizer.convert_tokens_to_ids("B"),
                "T": self.tokenizer.convert_tokens_to_ids("T")
            }

            self.logger.info("Model loaded successfully")
            self.logger.info(f"Choice token IDs: {self.choice_tokens}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def judge_pair(
        self, question: str, qa_a: Dict[str, Any], qa_b: Dict[str, Any],
        granularity: str, atoms: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute pairwise judgment."""
        self.logger.info(f"Judging at {granularity} granularity")

        try:
            logits_result = self._get_logits_judgment(question, qa_a, qa_b)

            if logits_result is None:
                return self._create_default_judgment("Judgment failed: no logits result")

            parsed = self._parse_logits_result(logits_result, granularity)
            if parsed is None:
                return self._create_default_judgment("Judgment failed: cannot parse logits")

            return parsed

        except Exception as e:
            self.logger.error(f"Judgment failed: {e}")
            return self._create_default_judgment(f"Judgment failed: {str(e)}")

    def _get_logits_judgment(self, question: str, qa_a: Dict[str, Any], qa_b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Route to deep-thinking or direct mode."""
        try:
            if self.enable_deep_thinking:
                self.logger.info("Using deep thinking mode")
                return self._get_logits_with_deep_thinking(question, qa_a, qa_b)
            else:
                self.logger.info("Using direct output mode")
                return self._get_logits_direct_mode(question, qa_a, qa_b)
        except Exception as e:
            self.logger.error(f"Logits judgment failed: {e}")
            return None

    def _get_logits_with_deep_thinking(self, question: str, qa_a: Dict[str, Any], qa_b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Two-phase: deep thinking + constrained A/B/T output."""
        thinking_result = self._generate_deep_thinking(question, qa_a, qa_b)
        if thinking_result is None:
            return None

        torch.cuda.empty_cache()
        gc.collect()

        choice_result = self._generate_final_choice(thinking_result["full_context"])
        if choice_result is None:
            return None

        torch.cuda.empty_cache()
        gc.collect()

        reasoning_choice = self._extract_choice_from_reasoning(thinking_result["reasoning"])
        consistent = (reasoning_choice == choice_result["choice"]) if reasoning_choice else True

        self.logger.info(
            f"Deep thinking result: {choice_result['choice']} "
            f"(P: A={choice_result['prob_a']:.3f}, B={choice_result['prob_b']:.3f}, T={choice_result['prob_t']:.3f})"
        )

        return {
            "reasoning": thinking_result["reasoning"],
            "choice": choice_result["choice"],
            "logit_a": choice_result["logit_a"],
            "logit_b": choice_result["logit_b"],
            "logit_t": choice_result["logit_t"],
            "prob_a": choice_result["prob_a"],
            "prob_b": choice_result["prob_b"],
            "prob_t": choice_result["prob_t"],
            "raw_response": choice_result.get("final_answer", ""),
            "generated_token": choice_result["generated_token"],
            "verification_consistent": consistent,
            "reasoning_choice": reasoning_choice
        }

    def _get_logits_direct_mode(self, question: str, qa_a: Dict[str, Any], qa_b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Direct output mode: generate full judgment then extract logits from last token."""
        prompt = self._build_direct_judgment_prompt(question, qa_a, qa_b)

        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=4096, padding=False
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=1500, do_sample=True,
                    temperature=0.2, top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                    return_dict_in_generate=True, output_scores=True
                )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs.sequences[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            self.logger.debug(f"Direct judgment output: {generated_text[:200]}...")

            if outputs.scores and len(outputs.scores) > 0 and len(generated_tokens) > 0:
                last_step_logits = outputs.scores[-1][0]
                last_token_id = generated_tokens[-1].item()
                last_token_text = self.tokenizer.decode([last_token_id], skip_special_tokens=True)

                logits_dict = self._compute_logits_directly(last_step_logits)

                if last_token_text.strip().upper() in ["A", "B", "T"]:
                    choice = last_token_text.strip().upper()
                else:
                    choice = self._extract_choice_from_text(generated_text)
                    if not choice:
                        if logits_dict["prob_a"] > logits_dict["prob_b"] and logits_dict["prob_a"] > logits_dict["prob_t"]:
                            choice = "A"
                        elif logits_dict["prob_b"] > logits_dict["prob_a"] and logits_dict["prob_b"] > logits_dict["prob_t"]:
                            choice = "B"
                        else:
                            choice = "T"
            else:
                choice = self._extract_choice_from_text(generated_text) or "T"
                logits_dict = self._create_fallback_logits(choice)

            self.logger.info(
                f"Direct judgment: {choice} "
                f"(P: A={logits_dict['prob_a']:.3f}, B={logits_dict['prob_b']:.3f}, T={logits_dict['prob_t']:.3f})"
            )

            del inputs, outputs

            return {
                "reasoning": generated_text,
                "choice": choice,
                "logit_a": logits_dict["logit_a"],
                "logit_b": logits_dict["logit_b"],
                "logit_t": logits_dict["logit_t"],
                "prob_a": logits_dict["prob_a"],
                "prob_b": logits_dict["prob_b"],
                "prob_t": logits_dict["prob_t"],
                "raw_response": generated_text,
                "generated_token": choice,
                "verification_consistent": True,
                "reasoning_choice": choice
            }

        except Exception as e:
            self.logger.error(f"Direct judgment failed: {e}")
            return None

    def _generate_deep_thinking(self, question: str, qa_a: Dict[str, Any], qa_b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Phase 1: generate deep thinking reasoning."""
        prompt = self._build_analysis_prompt(question, qa_a, qa_b)

        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=3072, padding=False
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                reasoning_outputs = self.model.generate(
                    **inputs, max_new_tokens=1400,
                    do_sample=True, temperature=0.1, top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            input_length = inputs["input_ids"].shape[1]
            reasoning_generated = reasoning_outputs[0][input_length:]
            reasoning_text = self.tokenizer.decode(reasoning_generated, skip_special_tokens=True)

            is_complete = self._check_reasoning_completeness(reasoning_text)
            if not is_complete:
                self.logger.warning("Reasoning may be truncated, proceeding anyway")

            self.logger.info(
                f"Reasoning: {len(reasoning_text)} chars, {len(reasoning_generated)} tokens, "
                f"complete={is_complete}"
            )

            del inputs, reasoning_outputs, reasoning_generated

            full_context = prompt + reasoning_text + "\n\n基于以上深度分析，我的最终判决是："

            return {"reasoning": reasoning_text, "full_context": full_context}

        except Exception as e:
            self.logger.error(f"Deep thinking generation failed: {e}")
            return None

    def _generate_final_choice(self, full_context: str) -> Optional[Dict[str, Any]]:
        """Phase 2: constrained generation of a single A/B/T token."""
        choice_prompt = (
            full_context
            + "\n\n现在请给出你的最终判决，只输出一个字母：\n\n"
            "如果系统A更好，输出：A\n如果系统B更好，输出：B\n如果两者相当，输出：T\n\n我的选择是："
        )

        try:
            inputs = self.tokenizer(
                choice_prompt, return_tensors="pt", truncation=True,
                max_length=4096, padding=False
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                original_logits = logits.clone()

                # Constrain to A/B/T tokens only
                constrained_logits = torch.full_like(logits, -float('inf'))
                for key in ["A", "B", "T"]:
                    constrained_logits[self.choice_tokens[key]] = logits[self.choice_tokens[key]]

                next_token_id = torch.multinomial(
                    torch.softmax(constrained_logits / 0.3, dim=-1), num_samples=1
                ).item()

                choice_found = None
                for key, tid in self.choice_tokens.items():
                    if next_token_id == tid:
                        choice_found = key
                        break

            del inputs

            if choice_found and original_logits is not None:
                logits_dict = self._compute_logits_directly(original_logits)
                self.logger.info(
                    f"Final choice: {choice_found} "
                    f"(P: A={logits_dict['prob_a']:.3f}, B={logits_dict['prob_b']:.3f}, T={logits_dict['prob_t']:.3f})"
                )
                return {
                    "choice": choice_found,
                    "generated_token": choice_found,
                    "final_answer": choice_found,
                    **logits_dict
                }
            else:
                generated_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
                self.logger.warning(f"No valid A/B/T choice found, generated: '{generated_text}'")
                fallback_choice = self._extract_choice_from_text(generated_text)
                if fallback_choice:
                    self.logger.info(f"Extracted fallback choice: {fallback_choice}")
                    return self._create_fallback_logits(fallback_choice)

            return None

        except Exception as e:
            self.logger.error(f"Final choice generation failed: {e}")
            return None

    def _check_reasoning_completeness(self, reasoning_text: str) -> bool:
        """Check whether reasoning appears complete or truncated."""
        if not reasoning_text:
            return False

        completion_indicators = [
            "最终判决", "结论", "因此", "总结", "综上所述",
            "系统A", "系统B", "更好", "获胜", "优于",
            "判断", "选择", "决定"
        ]

        reasoning_lower = reasoning_text.lower()
        has_conclusion = any(ind in reasoning_lower for ind in completion_indicators)

        text_stripped = reasoning_text.strip()
        if not text_stripped:
            return False

        ends_properly = text_stripped.endswith(('.', '。', '!', '！', '?', '？'))

        has_comparison = (
            ("系统a" in reasoning_lower and "系统b" in reasoning_lower) or
            ("system a" in reasoning_lower and "system b" in reasoning_lower)
        )

        return has_conclusion and (ends_properly or len(reasoning_text) > 1000) and has_comparison

    def _compute_logits_directly(self, logits_tensor: torch.Tensor) -> Dict[str, float]:
        """Compute A/B/T probabilities from raw logits."""
        logit_a = float(logits_tensor[self.choice_tokens["A"]].cpu())
        logit_b = float(logits_tensor[self.choice_tokens["B"]].cpu())
        logit_t = float(logits_tensor[self.choice_tokens["T"]].cpu())

        probs = torch.softmax(torch.tensor([logit_a, logit_b, logit_t]), dim=0)

        return {
            "logit_a": logit_a, "logit_b": logit_b, "logit_t": logit_t,
            "prob_a": float(probs[0]), "prob_b": float(probs[1]), "prob_t": float(probs[2])
        }

    def _extract_choice_from_reasoning(self, reasoning: str) -> Optional[str]:
        """Extract A/B/T choice from reasoning text."""
        if not reasoning:
            return None

        choice_patterns = [
            "选择A", "选择B", "选择T",
            "判决A", "判决B", "判决T",
            "答案A", "答案B", "答案T",
            "系统A更好", "系统B更好", "两系统相当"
        ]

        reasoning_lower = reasoning.lower()
        for pattern in choice_patterns:
            if pattern.lower() in reasoning_lower:
                if "a" in pattern.lower():
                    return "A"
                elif "b" in pattern.lower():
                    return "B"
                elif "t" in pattern.lower() or "相当" in pattern:
                    return "T"

        return None

    def _extract_choice_from_text(self, text: str) -> Optional[str]:
        """Extract A/B/T choice from generated text."""
        if not text:
            return None

        text_upper = text.strip().upper()

        # Check last lines for standalone A/B/T
        lines = text_upper.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line in ['A', 'B', 'T']:
                return line

        # Check last few lines for choice patterns
        for line in reversed(lines[-3:]):
            line = line.strip()
            if '：A' in line or ':A' in line:
                return 'A'
            elif '：B' in line or ':B' in line:
                return 'B'
            elif '：T' in line or ':T' in line:
                return 'T'

        # Regex for standalone A/B/T
        matches = re.findall(r'(?:^|\s|：|:)([ABT])(?:\s|$|。|，|！)', text_upper)
        if matches:
            return matches[-1]

        # Last resort: find isolated A/B/T
        for choice in ['A', 'B', 'T']:
            if choice in text_upper:
                pos = text_upper.rfind(choice)
                if ((pos == 0 or not text_upper[pos-1].isalnum()) and
                        (pos == len(text_upper)-1 or not text_upper[pos+1].isalnum())):
                    return choice

        return None

    def _create_fallback_logits(self, choice: str) -> Dict[str, Any]:
        """Create synthetic logits when real logits are unavailable."""
        if choice == "A":
            logit_a, logit_b, logit_t = 2.0, -1.0, -1.0
        elif choice == "B":
            logit_a, logit_b, logit_t = -1.0, 2.0, -1.0
        else:
            logit_a, logit_b, logit_t = -1.0, -1.0, 2.0

        probs = torch.softmax(torch.tensor([logit_a, logit_b, logit_t]), dim=0)
        return {
            "choice": choice, "generated_token": choice,
            "logit_a": logit_a, "logit_b": logit_b, "logit_t": logit_t,
            "prob_a": float(probs[0]), "prob_b": float(probs[1]), "prob_t": float(probs[2])
        }

    def _build_analysis_prompt(self, question: str, qa_a: Dict[str, Any], qa_b: Dict[str, Any]) -> str:
        """Build the deep-analysis prompt with full evidence."""
        answer_a = qa_a.get("rag_answer", qa_a.get("answer", "无回答"))
        answer_b = qa_b.get("rag_answer", qa_b.get("answer", "无回答"))
        expected_answer = qa_a.get("expected_answer", "")
        groundtruth = qa_a.get("groundtruth", qa_b.get("groundtruth", ""))

        evidence_a = qa_a.get("retrieved_docs", qa_a.get("context", []))
        evidence_b = qa_b.get("retrieved_docs", qa_b.get("context", []))
        evidence_text_a = self._format_evidence(evidence_a)
        evidence_text_b = self._format_evidence(evidence_b)

        prompt = f"""你是一个专业的RAG系统回答质量评估专家。请对比分析两个RAG系统对同一问题的回答质量。

问题：{question}

标准答案：{expected_answer}
标准答案对应的知识库里的证据：{groundtruth}

系统A的回答：
{answer_a}

系统A的检索证据：
{evidence_text_a}

系统B的回答：
{answer_b}

系统B的检索证据：
{evidence_text_b}

特别注意：请务必在1000token以内给出答案！！！

评估标准：
1.先比系统AB的答案相较于<标准答案>的准确性，在覆盖了标答基础上增加的额外信息不能算作加分项，一切以标准答案为准，都覆盖了标准答案的关键意思的答案必须判平局，答案质量类似直接判平局即可，不用管后面两条规则
2.如果答案准确性相差无几，你只需比较<标准答案对应的知识库里的证据>是否完整/部分包含在系统检索出的证据中，不用纠结于证据质量
3.如果上面两点都相差无几，那么就判平局

注意：
1.由于token有限，请你在800token以内完成深度思考的全过程，并给出答案，<一定>不要超出1000token限制，所以为了节省token，要求你不能重复思考相同的内容
2.在评估的最后，明确说明是A获胜/B获胜/平局

特殊判决规则：
- 如果一方给出答案，另一方回答"信息不足"，要判断给出答案的一方是否正确，是否胡编（指的是完全错误，与标准答案完全不一致，而非部分错误），若胡编则判另一方（诚实的一方）赢。若一方部分正确，另一方信息不足，则判部分正确的一方获胜
- 完全答对 > 部分答对 > 信息不足 > 完全错误

请进行深度分析："""

        self.logger.info(f"Prompt constructed, length: {len(prompt)}")
        return prompt

    def _build_direct_judgment_prompt(self, question: str, qa_a: Dict[str, Any], qa_b: Dict[str, Any]) -> str:
        """Build prompt for direct (non-deep-thinking) judgment."""
        answer_a = qa_a.get("rag_answer", qa_a.get("answer", "无回答"))
        answer_b = qa_b.get("rag_answer", qa_b.get("answer", "无回答"))
        expected_answer = qa_a.get("expected_answer", "")
        groundtruth = qa_a.get("groundtruth", qa_b.get("groundtruth", ""))

        evidence_a = qa_a.get("retrieved_docs", qa_a.get("context", []))
        evidence_b = qa_b.get("retrieved_docs", qa_b.get("context", []))
        evidence_text_a = self._format_evidence(evidence_a)
        evidence_text_b = self._format_evidence(evidence_b)

        prompt = f"""你是一个专业的RAG系统回答质量评估专家。请对比两个RAG系统的回答质量，给出最终判决。

问题：{question}

标准答案：{expected_answer}

系统A的回答：{answer_a}
系统A的检索证据：{evidence_text_a}

系统B的回答：{answer_b}
系统B的检索证据：{evidence_text_b}

评估标准：
1. 准确性：回答是否正确，是否包含标准答案的关键信息
2. 完整性：回答是否全面，是否遗漏重要信息
3. 相关性：回答是否针对问题，是否包含无关信息

特殊规则：
- 如果一方给出答案，另一方回答"信息不足"，要判断给出答案的一方是否正确
- 完全答对 > 部分答对 > 部分答错 > 信息不足 > 完全错误

请详细分析比较两个系统的回答质量，然后在分析的最后一行明确给出你的最终判决。

最终判决格式要求：
- 如果系统A更好，最后一行输出：A
- 如果系统B更好，最后一行输出：B
- 如果两者相当，最后一行输出：T

开始分析："""

        self.logger.info(f"Direct judgment prompt constructed, length: {len(prompt)}")
        return prompt

    def _format_evidence(self, evidence_list: List) -> str:
        """Format retrieved evidence for display."""
        if not evidence_list:
            return "无检索证据"

        formatted = []
        for i, doc in enumerate(evidence_list[:3]):
            if isinstance(doc, dict):
                content = doc.get("content", doc.get("text", str(doc)))
            else:
                content = str(doc)

            if not content or content.strip() == "":
                content = "(empty)"

            formatted.append(f"[Evidence {i+1}] {content[:200]}...")

        return "\n".join(formatted) if formatted else "无有效检索证据"

    def _parse_logits_result(self, logits_result: Dict[str, Any], granularity: str) -> Optional[Dict[str, Any]]:
        """Parse logits result into standard judgment format."""
        try:
            choice = logits_result.get("choice", "Unknown")
            prob_a = logits_result.get("prob_a", 0.333)
            prob_b = logits_result.get("prob_b", 0.333)
            prob_t = logits_result.get("prob_t", 0.333)

            if choice == "A":
                winner, confidence = "A wins", prob_a
            elif choice == "B":
                winner, confidence = "B wins", prob_b
            elif choice == "T":
                winner, confidence = "Tie", prob_t
            else:
                winner, confidence = "Unknown", max(prob_a, prob_b, prob_t)

            sorted_probs = sorted([prob_a, prob_b, prob_t])
            margin = sorted_probs[-1] - sorted_probs[-2]

            return {
                "label": winner,
                "reason": logits_result.get("reasoning", "Based on DeepSeek-R1 analysis"),
                "granularity": granularity,
                "confidence": confidence,
                "margin_score": margin,
                "prob_a": prob_a, "prob_b": prob_b, "prob_t": prob_t,
                "logit_a": logits_result.get("logit_a", 0.0),
                "logit_b": logits_result.get("logit_b", 0.0),
                "logit_t": logits_result.get("logit_t", 0.0),
                "raw_response": logits_result.get("raw_response", ""),
                "generated_token": logits_result.get("generated_token", choice),
                "verification_consistent": logits_result.get("verification_consistent", True),
                "reasoning_choice": logits_result.get("reasoning_choice", choice)
            }

        except Exception as e:
            self.logger.error(f"Failed to parse logits result: {e}")
            return None

    def _create_default_judgment(self, reason: str) -> Dict[str, Any]:
        """Create a default tie judgment."""
        return {
            "label": "Soft tie", "reason": reason,
            "granularity": "passage", "confidence": 0.333,
            "margin_score": 0.0,
            "prob_a": 0.333, "prob_b": 0.333, "prob_t": 0.333,
            "logit_a": 0.0, "logit_b": 0.0, "logit_t": 0.0,
            "raw_response": "", "generated_token": "T",
            "verification_consistent": False, "reasoning_choice": "Unknown"
        }
