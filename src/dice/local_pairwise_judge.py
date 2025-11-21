#!/usr/bin/env python3
"""
基于本地DeepSeek-R1模型的Pairwise判决器 - 显存优化版
实现深度思考(800 tokens) + 强制字母输出 + 全面证据分析
"""

import torch
import logging
import gc
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


class LocalPairwiseJudge:
    """基于本地DeepSeek-R1的Pairwise判决器 - 优化版"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DICE.LocalPairwise")
        
        # 本地模型路径
        self.model_path = "/root/autodl-tmp/deepseek-deployment/deepseek-r1-8b"
        
        # 控制是否启用深度思考模式
        self.enable_deep_thinking = getattr(config, 'enable_deep_thinking', True)  # 默认开启深度思考
        
        # 模型和tokenizer
        self.model = None
        self.tokenizer = None
        self.choice_tokens = {}
        
        # 初始化模型
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化本地模型 - 显存优化版"""
        self.logger.info(f"🚀 加载本地DeepSeek-R1模型: {self.model_path}")
        
        try:
            # 显存优化：使用float16和high_cpu_mem_usage=False
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # 显存优化：使用float16，device_map auto，低CPU内存使用
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,  # 减少一半显存
                trust_remote_code=True,
                low_cpu_mem_usage=True      # 减少CPU内存使用
            )
            
            # 显存优化：关闭缓存
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = False
            
            self.model.eval()  # 设置为评估模式
            
            # 设置pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 预计算选择token的ID (A/B/T 三选项)
            self.choice_tokens = {
                "A": self.tokenizer.convert_tokens_to_ids("A"),
                "B": self.tokenizer.convert_tokens_to_ids("B"), 
                "T": self.tokenizer.convert_tokens_to_ids("T")
            }
            
            self.logger.info("✅ 模型加载完成")
            self.logger.info(f"🎯 选择token IDs: {self.choice_tokens}")
            
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            raise

    def judge_pair(
        self, 
        question: str, 
        qa_a: Dict[str, Any], 
        qa_b: Dict[str, Any], 
        granularity: str,
        atoms: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """执行pairwise判决"""
        
        self.logger.info(f"执行{granularity}粒度判决")
        
        try:
            # 调用logits判决 
            logits_result = self._get_logits_judgment(question, qa_a, qa_b)
            
            if logits_result is None:
                return self._create_default_judgment("判决失败: 无法获取logits结果")
            
            # 解析判决结果
            parsed_judgment = self._parse_logits_result(logits_result, granularity)
            if parsed_judgment is None:
                return self._create_default_judgment("判决失败: 无法解析logits结果")
            
            return parsed_judgment
            
        except Exception as e:
            self.logger.error(f"❌ 判决失败: {e}")
            return self._create_default_judgment(f"判决失败: {str(e)}")
    
    def _get_logits_judgment(self, question: str, qa_a: Dict[str, Any], qa_b: Dict[str, Any]) -> Dict[str, Any]:
        """DeepSeek-R1判决模式 - 可选深度思考或直接输出"""
        
        try:
            if self.enable_deep_thinking:
                self.logger.info("🧠 使用深度思考模式")
                return self._get_logits_with_deep_thinking(question, qa_a, qa_b)
            else:
                self.logger.info("⚡ 使用直接输出模式")
                return self._get_logits_direct_mode(question, qa_a, qa_b)
                
        except Exception as e:
            self.logger.error(f"❌ Logits判决失败: {e}")
            return None
    
    def _get_logits_with_deep_thinking(self, question: str, qa_a: Dict[str, Any], qa_b: Dict[str, Any]) -> Dict[str, Any]:
        """深度思考模式"""
        # 第一阶段：深度思考
        thinking_result = self._generate_deep_thinking(question, qa_a, qa_b)
        if thinking_result is None:
            return None
        
        # 显存清理
        torch.cuda.empty_cache()
        gc.collect()
        
        # 第二阶段：强制字母输出 (A/B/T)
        choice_result = self._generate_final_choice(thinking_result["full_context"])
        if choice_result is None:
            return None
        
        # 显存清理
        torch.cuda.empty_cache()
        gc.collect()
        
        # 验证一致性
        reasoning_choice = self._extract_choice_from_reasoning(thinking_result["reasoning"])
        consistent = (reasoning_choice == choice_result["choice"]) if reasoning_choice else True
        
        # 简化日志输出
        self.logger.info(f"✅ 深度思考判决完成: {choice_result['choice']} (概率: A={choice_result['prob_a']:.3f}, B={choice_result['prob_b']:.3f}, T={choice_result['prob_t']:.3f})")
        
        # 构建最终结果
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
    
    def _get_logits_direct_mode(self, question: str, qa_a: Dict[str, Any], qa_b: Dict[str, Any]) -> Dict[str, Any]:
        """直接输出模式 - 完整输出判决后提取最后token的logits"""
        
        # 构建直接判决prompt
        prompt = self._build_direct_judgment_prompt(question, qa_a, qa_b)
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=4096,
                padding=False
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 让模型完整输出判决过程，确保有足够空间完成分析
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1500,  # 增加到1500，确保分析完整
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                    return_dict_in_generate=True,
                    output_scores=True  # 获取每个step的logits
                )
            
            # 提取完整生成的判决
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs.sequences[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"\n⚡ DeepSeek-R1 直接判决:")
            print(f"   完整输出: {generated_text}")
            
            # 关键：从最后一个token获取logits（这是模型完整思考后的决策）
            if outputs.scores and len(outputs.scores) > 0 and len(generated_tokens) > 0:
                # 获取最后一个生成步骤的logits
                last_step_logits = outputs.scores[-1][0]  # 最后一步的logits
                last_token_id = generated_tokens[-1].item()  # 最后一个token ID
                last_token_text = self.tokenizer.decode([last_token_id], skip_special_tokens=True)
                
                print(f"   最后一个token: '{last_token_text}' (ID: {last_token_id})")
                
                # 基于最后一步的logits计算A/B/T概率
                logits_dict = self._compute_logits_directly(last_step_logits)
                
                # 确定选择：优先从最后token判断，其次从文本分析
                if last_token_text.strip().upper() in ["A", "B", "T"]:
                    choice = last_token_text.strip().upper()
                    print(f"   ✅ 最后token直接是选择: {choice}")
                else:
                    # 从完整文本中提取选择
                    choice = self._extract_choice_from_text(generated_text)
                    if not choice:
                        # 如果文本分析也失败，基于logits概率选择
                        if logits_dict["prob_a"] > logits_dict["prob_b"] and logits_dict["prob_a"] > logits_dict["prob_t"]:
                            choice = "A"
                        elif logits_dict["prob_b"] > logits_dict["prob_a"] and logits_dict["prob_b"] > logits_dict["prob_t"]:
                            choice = "B"
                        else:
                            choice = "T"
                    print(f"   🔍 从文本/logits推断选择: {choice}")
                
            else:
                # 无法获取logits，使用文本分析
                choice = self._extract_choice_from_text(generated_text)
                if not choice:
                    choice = "T"
                logits_dict = self._create_fallback_logits(choice)
                print(f"   ⚠️ 无法获取logits，使用文本分析: {choice}")
            
            print(f"   原始Logits: A={logits_dict['logit_a']:.3f}, B={logits_dict['logit_b']:.3f}, T={logits_dict['logit_t']:.3f}")
            print(f"   概率分布: A={logits_dict['prob_a']:.3f}, B={logits_dict['prob_b']:.3f}, T={logits_dict['prob_t']:.3f}")
            print(f"   最终选择: {choice}\n")
            
            # 清理
            del inputs, outputs
            
            self.logger.info(f"✅ 直接判决完成: {choice} (概率: A={logits_dict['prob_a']:.3f}, B={logits_dict['prob_b']:.3f}, T={logits_dict['prob_t']:.3f})")
            
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
            self.logger.error(f"❌ 直接判决失败: {e}")
            return None
    
    def _generate_deep_thinking(self, question: str, qa_a: Dict[str, Any], qa_b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """第一阶段：DeepSeek-R1深度思考生成（800 tokens）"""
        
        # 构建完整的分析prompt
        prompt = self._build_analysis_prompt(question, qa_a, qa_b)
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=3072,  # 减少输入长度，为思考留出更多空间（3072 + 2048 = 5120 < 8192模型上下文）
                padding=False
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # DeepSeek-R1深度思考生成：确保不被截断
            with torch.no_grad():
                reasoning_outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1400,
                    do_sample=True,    
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
            # 提取生成的推理文本
            input_length = inputs["input_ids"].shape[1]
            reasoning_generated = reasoning_outputs[0][input_length:]
            reasoning_text = self.tokenizer.decode(reasoning_generated, skip_special_tokens=True)
            
            # 🔍 检查深度思考是否完整
            is_complete = self._check_reasoning_completeness(reasoning_text)
            if not is_complete:
                self.logger.warning("⚠️ 深度思考可能未完成，但继续进行判决")
            
            # 🖨️ 调试：深度思考完成
            print(f"\n{'='*60}")
            print(f"🧠 DeepSeek-R1 深度思考内容 ({'完整' if is_complete else '可能截断'}):")
            print(f"{'='*60}")
            print(reasoning_text)
            print(f"{'='*60}")
            print(f"📊 思考长度: {len(reasoning_text)} 字符, {len(reasoning_generated)} tokens")
            print(f"✅ 思考完整性: {'完整' if is_complete else '可能截断'}\n")
            
            # 清理
            del inputs, reasoning_outputs, reasoning_generated
            
            # 构建第二阶段的上下文       
            full_context = prompt + reasoning_text + "\n\n基于以上深度分析，我的最终判决是："
            
            return {
                "reasoning": reasoning_text,
                "full_context": full_context
            }
            
        except Exception as e:
            self.logger.error(f"❌ 深度思考生成失败: {e}")
            return None
    
    def _generate_final_choice(self, full_context: str) -> Optional[Dict[str, Any]]:
        """第二阶段：强制生成单字母选择A/B/T"""
        
        # 构建非常明确的单字母选择prompt
        choice_prompt = full_context + "\n\n现在请给出你的最终判决，只输出一个字母：\n\n如果系统A更好，输出：A\n如果系统B更好，输出：B\n如果两者相当，输出：T\n\n我的选择是："
        
        try:
            inputs = self.tokenizer(
                choice_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                padding=False
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 强制约束生成：只允许A/B/T
            choice_found = None
            original_logits = None
            generated_tokens = []
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]  # 获取最后位置的logits
                original_logits = logits.clone()  # 保存原始logits
                
                # 创建约束logits：只允许A、B、T token
                constrained_logits = torch.full_like(logits, -float('inf'))
                constrained_logits[self.choice_tokens["A"]] = logits[self.choice_tokens["A"]]
                constrained_logits[self.choice_tokens["B"]] = logits[self.choice_tokens["B"]]
                constrained_logits[self.choice_tokens["T"]] = logits[self.choice_tokens["T"]]
                
                # 使用约束后的logits进行采样
                next_token_id = torch.multinomial(
                    torch.softmax(constrained_logits / 0.3, dim=-1), 
                    num_samples=1
                ).item()
                
                # 确定选择
                if next_token_id == self.choice_tokens["A"]:
                    choice_found = "A"
                elif next_token_id == self.choice_tokens["B"]:
                    choice_found = "B"
                elif next_token_id == self.choice_tokens["T"]:
                    choice_found = "T"
                else:
                    # 这种情况不应该发生，但作为安全措施
                    choice_found = None
                
                generated_tokens = [next_token_id]
            
            # 清理inputs
            del inputs
            
            if choice_found and original_logits is not None:
                # 生成的最终回答（仅显示简短内容）
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # 🖨️ 调试：打印最终回答
                print(f"\n🎯 DeepSeek-R1 最终回答:")
                print(f"   生成内容: '{generated_text}'")
                print(f"   最后一个token: '{choice_found}'")
                
                # 计算A/B/T的logits和概率（使用原始未约束的logits）
                logits_dict = self._compute_logits_directly(original_logits)
                
                print(f"   Logits: A={logits_dict['logit_a']:.3f}, B={logits_dict['logit_b']:.3f}, T={logits_dict['logit_t']:.3f}")
                print(f"   概率: A={logits_dict['prob_a']:.3f}, B={logits_dict['prob_b']:.3f}, T={logits_dict['prob_t']:.3f}")
                print(f"   确定选择: {choice_found}\n")
                
                return {
                    "choice": choice_found,
                    "generated_token": choice_found,
                    "final_answer": generated_text,
                    **logits_dict
                }
            else:
                # 如果没有找到有效选择，分析生成的内容
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # 🖨️ 调试：打印未找到选择的情况
                print(f"\n⚠️ 未找到有效A/B/T选择!")
                print(f"   生成的内容: '{generated_text}'")
                print(f"   生成的Token IDs: {generated_tokens}")
                
                self.logger.warning(f"⚠️ 未找到有效选择A/B/T，生成内容: '{generated_text}'")
                
                # 尝试从生成文本中提取A/B/T
                fallback_choice = self._extract_choice_from_text(generated_text)
                if fallback_choice:
                    print(f"   🔄 从文本提取到选择: {fallback_choice}\n")
                    self.logger.info(f"🔄 从文本中提取到选择: {fallback_choice}")
                    # 返回模拟的logits
                    return self._create_fallback_logits(fallback_choice)
                else:
                    print(f"   ❌ 无法提取任何有效选择\n")
                    self.logger.warning(f"⚠️ 无法从推理或logits中提取明确选择")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 最终回答生成失败: {e}")
            return None
    
    def _check_reasoning_completeness(self, reasoning_text: str) -> bool:
        """检查推理是否完整 - 判断是否被截断"""
        if not reasoning_text:
            return False
        
        # 检查推理是否有明确的结论标志
        completion_indicators = [
            "最终判决", "结论", "因此", "总结", "综上所述", 
            "系统A", "系统B", "更好", "获胜", "优于",
            "判断", "选择", "决定"
        ]
        
        reasoning_lower = reasoning_text.lower()
        has_conclusion = any(indicator in reasoning_lower for indicator in completion_indicators)
        
        # 检查文本是否突然截断（以不完整的句子结尾）
        text_stripped = reasoning_text.strip()
        if not text_stripped:
            return False
        
        # 检查最后的字符是否表明完整性
        last_chars = text_stripped[-50:].lower()  # 检查最后50个字符
        
        # 如果以句号、感叹号、问号结尾，且有结论性内容，认为是完整的
        ends_properly = text_stripped.endswith(('.', '。', '!', '！', '?', '？'))
        
        # 检查是否包含系统对比的内容
        has_comparison = ("系统a" in reasoning_lower and "系统b" in reasoning_lower) or \
                        ("system a" in reasoning_lower and "system b" in reasoning_lower)
        
        # 综合判断
        is_complete = has_conclusion and (ends_properly or len(reasoning_text) > 1000) and has_comparison
        
        return is_complete
    
    def _compute_logits_directly(self, logits_tensor: torch.Tensor) -> Dict[str, float]:
        """直接计算logits - 获取A/B/T的真实概率"""
        
        # 获取A/B/T的logits
        logit_a = float(logits_tensor[self.choice_tokens["A"]].cpu())
        logit_b = float(logits_tensor[self.choice_tokens["B"]].cpu())
        logit_t = float(logits_tensor[self.choice_tokens["T"]].cpu())
        
        # 计算三选项概率
        logits_abc = torch.tensor([logit_a, logit_b, logit_t])
        probs = torch.softmax(logits_abc, dim=0)
        
        return {
                "logit_a": logit_a,
                "logit_b": logit_b,
                "logit_t": logit_t,
            "prob_a": float(probs[0]),
            "prob_b": float(probs[1]),
            "prob_t": float(probs[2])
        }
    
    def _extract_choice_from_reasoning(self, reasoning: str) -> Optional[str]:
        """从推理文本中提取选择"""
        if not reasoning:
            return None
        
        # 寻找明确的选择表达
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
        """从文本中提取A/B/T选择"""
        if not text:
            return None
        
        text = text.strip().upper()
        if "A" in text:
            return "A"
        elif "B" in text:
            return "B"
        elif "T" in text:
            return "T"
        
        return None
    
    def _create_fallback_logits(self, choice: str) -> Dict[str, Any]:
        """创建回退logits（基于文本分析的选择）"""
        # 模拟logits：给选中的选项高分，其他低分
        if choice == "A":
            logit_a, logit_b, logit_t = 2.0, -1.0, -1.0
        elif choice == "B":
            logit_a, logit_b, logit_t = -1.0, 2.0, -1.0
        else:  # T
            logit_a, logit_b, logit_t = -1.0, -1.0, 2.0
        
        # 计算概率
        logits_tensor = torch.tensor([logit_a, logit_b, logit_t])
        probs = torch.softmax(logits_tensor, dim=0)
        
        return {
            "choice": choice,
            "generated_token": choice,
            "logit_a": logit_a,
            "logit_b": logit_b,
            "logit_t": logit_t,
            "prob_a": float(probs[0]),
            "prob_b": float(probs[1]),
            "prob_t": float(probs[2])
        }
    
    def _build_analysis_prompt(self, question: str, qa_a: Dict[str, Any], qa_b: Dict[str, Any]) -> str:
        """构建深度分析prompt - 包含全面的证据和答案信息"""
        
        # 安全获取数据，处理空字段
        answer_a = qa_a.get("rag_answer", qa_a.get("answer", "无回答"))
        answer_b = qa_b.get("rag_answer", qa_b.get("answer", "无回答"))
        expected_answer = qa_a.get("expected_answer", "")
        groundtruth = qa_a.get("groundtruth", qa_b.get("groundtruth", ""))
        
        # 获取检索证据
        evidence_a = qa_a.get("retrieved_docs", qa_a.get("context", []))
        evidence_b = qa_b.get("retrieved_docs", qa_b.get("context", []))
        
        # 格式化证据
        evidence_text_a = self._format_evidence(evidence_a)
        evidence_text_b = self._format_evidence(evidence_b)
        
        # # 调试输出
        # self.logger.info(f"🔍 answer_a: {answer_a[:100]}...")
        # self.logger.info(f"🔍 answer_b: {answer_b[:100]}...")
        # self.logger.info(f"🔍 expected_answer: {expected_answer[:100]}...")
        # self.logger.info(f"🔍 groundtruth: {groundtruth[:100]}...")
        
        # 构建完整的评估prompt
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

        self.logger.info(f"📝 构造prompt完成，长度: {len(prompt)}")
        return prompt
    
    def _build_direct_judgment_prompt(self, question: str, qa_a: Dict[str, Any], qa_b: Dict[str, Any]) -> str:
        """构建直接判决prompt - 无深度思考模式"""
        
        # 安全获取数据，处理空字段
        answer_a = qa_a.get("rag_answer", qa_a.get("answer", "无回答"))
        answer_b = qa_b.get("rag_answer", qa_b.get("answer", "无回答"))
        expected_answer = qa_a.get("expected_answer", "")
        groundtruth = qa_a.get("groundtruth", qa_b.get("groundtruth", ""))
        
        # 获取检索证据
        evidence_a = qa_a.get("retrieved_docs", qa_a.get("context", []))
        evidence_b = qa_b.get("retrieved_docs", qa_b.get("context", []))
        
        # 格式化证据
        evidence_text_a = self._format_evidence(evidence_a)
        evidence_text_b = self._format_evidence(evidence_b)
        
        # 构建直接判决prompt
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

        self.logger.info(f"📝 构造直接判决prompt完成，长度: {len(prompt)}")
        return prompt
    
    def _format_evidence(self, evidence_list: List) -> str:
        """格式化检索证据，处理空值"""
        if not evidence_list:
            return "无检索证据"
        
        formatted = []
        for i, doc in enumerate(evidence_list[:3]):  # 只显示前3条
            if isinstance(doc, dict):
                content = doc.get("content", doc.get("text", str(doc)))
            else:
                content = str(doc)
            
            # 处理空内容
            if not content or content.strip() == "":
                content = "空内容"
            
            formatted.append(f"[证据{i+1}] {content[:200]}...")
        
        return "\n".join(formatted) if formatted else "无有效检索证据"

    def _extract_choice_from_text(self, text: str) -> Optional[str]:
        """从生成的文本中提取A/B/T选择"""
        if not text:
            return None
        
        # 去除多余空格，转为大写
        text = text.strip().upper()
        
        # 方法1：查找文本最后一行单独的A/B/T
        lines = text.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line in ['A', 'B', 'T']:
                return line
        
        # 方法2：查找最后几行中包含选择模式的内容
        last_lines = lines[-3:]  # 检查最后3行
        for line in reversed(last_lines):
            line = line.strip()
            # 模式如："最终选择：A" 或 "我的判决：B" 等
            if '：A' in line or ':A' in line:
                return 'A'
            elif '：B' in line or ':B' in line:
                return 'B'
            elif '：T' in line or ':T' in line:
                return 'T'
        
        # 方法3：查找整个文本中最后一个单独的A/B/T
        import re
        # 查找独立的A/B/T字母（前后有空格、换行或标点）
        matches = re.findall(r'(?:^|\s|：|:)([ABT])(?:\s|$|。|，|！)', text)
        if matches:
            return matches[-1]  # 返回最后一个匹配
        
        # 方法4：简单地查找最后出现的A/B/T
        for choice in ['A', 'B', 'T']:
            if choice in text:
                last_pos = text.rfind(choice)
                # 确保不是在其他单词中间
                if (last_pos == 0 or not text[last_pos-1].isalnum()) and \
                   (last_pos == len(text)-1 or not text[last_pos+1].isalnum()):
                    return choice
        
        return None

    def _parse_logits_result(self, logits_result: Dict[str, Any], granularity: str) -> Dict[str, Any]:
        """解析logits结果为标准判决格式"""
        
        try:
            choice = logits_result.get("choice", "Unknown")
            prob_a = logits_result.get("prob_a", 0.333)
            prob_b = logits_result.get("prob_b", 0.333)
            prob_t = logits_result.get("prob_t", 0.333)
            
            # 确定获胜者和标签
            if choice == "A":
                winner = "A wins"
                confidence = prob_a
            elif choice == "B":
                winner = "B wins"
                confidence = prob_b
            elif choice == "T":
                winner = "Tie"
                confidence = prob_t
            else:
                winner = "Unknown"
                confidence = max(prob_a, prob_b, prob_t)
            
            # 计算margin
            max_prob = max(prob_a, prob_b, prob_t)
            second_prob = sorted([prob_a, prob_b, prob_t])[-2]
            margin = max_prob - second_prob
            
            self.logger.info(f"✅ 解析判决完成: {choice == 'A' or choice == 'B' or choice == 'T'}")
            
            # 返回完整结果
            result = {
                "label": winner,
                "reason": logits_result.get("reasoning", "基于DeepSeek-R1深度分析"),
                "granularity": granularity,
                "confidence": confidence,
                "margin_score": margin,
                
                # 直接字段（dice_simplified.py需要）
                "prob_a": prob_a,
                "prob_b": prob_b,
                "prob_t": prob_t,
                "logit_a": logits_result.get("logit_a", 0.0),
                "logit_b": logits_result.get("logit_b", 0.0),
                "logit_t": logits_result.get("logit_t", 0.0),
                
                # 其他字段
                "raw_response": logits_result.get("raw_response", ""),
                "generated_token": logits_result.get("generated_token", choice),
                "verification_consistent": logits_result.get("verification_consistent", True),
                "reasoning_choice": logits_result.get("reasoning_choice", choice)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 解析logits结果失败: {e}")
            return None
    
    def _create_default_judgment(self, reason: str) -> Dict[str, Any]:
        """创建默认判决结果"""
        return {
            "label": "Soft tie",
            "reason": reason,
            "granularity": "passage",
            "confidence": 0.333,
            "margin_score": 0.0,
            "prob_a": 0.333,
            "prob_b": 0.333,
            "prob_t": 0.333,
            "logit_a": 0.0,
            "logit_b": 0.0,
            "logit_t": 0.0,
            "raw_response": "",
            "generated_token": "T",
            "verification_consistent": False,
            "reasoning_choice": "Unknown"
        }