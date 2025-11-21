#!/usr/bin/env python3
"""
DICE Pairwise判决模块
实现检索-证据双通道pairwise判决和Margin-Aware Tie分解
"""

import json
import logging
import math
import os
from typing import Dict, Any, Tuple, List
import numpy as np
from openai import OpenAI

class PairwiseJudge:
    """Pairwise判决器"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DICE.Pairwise")
        
        # 初始化OpenAI兼容的API客户端
        api_key = config.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请设置DEEPSEEK_API_KEY环境变量或在config中提供api_key")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=config.base_url
        )
    
    def judge_pair(
        self, 
        question: str, 
        qa_a: Dict[str, Any], 
        qa_b: Dict[str, Any], 
        granularity: str,
        atoms: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行pairwise判决
        
        Args:
            question: 问题文本
            qa_a: 系统A的QA数据
            qa_b: 系统B的QA数据
            granularity: 当前粒度
            atoms: 该粒度的原子单元
            
        Returns:
            判决结果
        """
        self.logger.info(f"执行{granularity}粒度判决")
        
        # 构造prompt
        prompt = self._build_prompt(question, qa_a, qa_b, granularity, atoms)
        
        # 调用LLM进行判决
        raw_judgment = self._call_llm(prompt)
        
        # 解析判决结果
        parsed_judgment = self._parse_judgment(raw_judgment, granularity)
        
        # Margin-Aware Tie处理
        if parsed_judgment["label"] == "Tie":
            margin_score = self._compute_margin_aware_tie(
                question, qa_a, qa_b, granularity, atoms
            )
            parsed_judgment["margin_score"] = margin_score
            parsed_judgment["score"] = 0.5 + margin_score  # 调整到[0,1]区间
        else:
            parsed_judgment["margin_score"] = 0.0
            parsed_judgment["score"] = 1.0 if parsed_judgment["label"] == "A wins" else 0.0
        
        return parsed_judgment
    
    def _build_prompt(
        self, 
        question: str, 
        qa_a: Dict[str, Any], 
        qa_b: Dict[str, Any], 
        granularity: str,
        atoms: Dict[str, Any]
    ) -> str:
        """构建pairwise判决prompt"""
        
        # 获取标准答案
        groundtruth = qa_a.get("groundtruth", qa_a.get("expected_answer", ""))
        
        base_prompt = f"""你是一个专业的RAG系统评估专家。请对两个系统在{granularity}粒度上的表现进行比较。

问题: {question}

标准答案: {groundtruth}

系统A:
证据: {self._format_evidence(qa_a.get('context', []))}
回答: {qa_a.get('rag_answer', '')}

系统B:
证据: {self._format_evidence(qa_b.get('context', []))}
回答: {qa_b.get('rag_answer', '')}

"""
        
        # 根据粒度添加特定指导
        granularity_guide = self._get_granularity_guide(granularity, atoms)
        
        full_prompt = base_prompt + granularity_guide + """

请严格按照以下格式回答:
判决: [A wins/B wins/Tie]
理由: [一句话说明判决原因，不超过30字]

注意:
1. 必须考虑检索证据的质量和答案的准确性
2. 如果两者差异很小或各有优劣，选择Tie
3. 优先考虑事实准确性，其次考虑完整性
"""
        
        return full_prompt
    
    def _format_evidence(self, contexts: List[str]) -> str:
        """格式化证据文本"""
        if not contexts:
            return "[无检索证据]"
        
        formatted = []
        for i, ctx in enumerate(contexts[:3]):  # 最多显示3个证据
            # 截断过长的证据
            truncated = ctx[:200] + "..." if len(ctx) > 200 else ctx
            formatted.append(f"[证据{i+1}] {truncated}")
        
        return "\n".join(formatted)
    
    def _get_granularity_guide(self, granularity: str, atoms: Dict[str, Any]) -> str:
        """获取粒度特定的评估指导"""
        
        if granularity == "token":
            return f"""
Token粒度评估指导:
- 重点关注关键词汇、数字、专有名词的准确性
- 检查重要事实性token是否准确提取
- Token级别差异: {self._summarize_token_atoms(atoms)}

评估标准: 哪个系统在关键token的准确性和完整性上更优？
"""
        
        elif granularity == "sentence":
            return f"""
Sentence粒度评估指导:
- 重点关注句子的语义完整性和逻辑性
- 检查句子是否有证据支撑
- 句子级别差异: {self._summarize_sentence_atoms(atoms)}

评估标准: 哪个系统的句子表达更准确、更有证据支撑？
"""
        
        elif granularity == "passage":
            return f"""
Passage粒度评估指导:
- 重点关注检索证据的覆盖度和相关性
- 检查证据与答案的一致性
- 段落级别差异: {self._summarize_passage_atoms(atoms)}

评估标准: 哪个系统的检索证据更全面、更相关？
"""
        
        elif granularity == "kg":
            return f"""
KG粒度评估指导:
- 重点关注知识三元组的准确性和完整性
- 检查实体关系是否正确
- 知识图谱差异: {self._summarize_kg_atoms(atoms)}

评估标准: 哪个系统的知识结构更准确、更完整？
"""
        
        return ""
    
    def _summarize_token_atoms(self, atoms: Dict[str, Any]) -> str:
        """汇总token原子信息"""
        comparisons = atoms.get("comparison_units", [])
        unique_a = sum(1 for c in comparisons if c["in_a"] and not c["in_b"])
        unique_b = sum(1 for c in comparisons if c["in_b"] and not c["in_a"])
        common = sum(1 for c in comparisons if c["in_a"] and c["in_b"])
        
        return f"A独有{unique_a}个token, B独有{unique_b}个token, 共同{common}个token"
    
    def _summarize_sentence_atoms(self, atoms: Dict[str, Any]) -> str:
        """汇总sentence原子信息"""
        sentences_a = atoms.get("sentences_a", [])
        sentences_b = atoms.get("sentences_b", [])
        
        return f"A有{len(sentences_a)}个句子, B有{len(sentences_b)}个句子"
    
    def _summarize_passage_atoms(self, atoms: Dict[str, Any]) -> str:
        """汇总passage原子信息"""
        comparison = atoms.get("comparison_units", {})
        
        return f"A有{comparison.get('passage_count_a', 0)}个段落, B有{comparison.get('passage_count_b', 0)}个段落, 实体重叠度{comparison.get('entity_overlap', 0):.2f}"
    
    def _summarize_kg_atoms(self, atoms: Dict[str, Any]) -> str:
        """汇总KG原子信息"""
        comparison = atoms.get("comparison_units", {})
        
        return f"A有{comparison.get('triple_count_a', 0)}个三元组, B有{comparison.get('triple_count_b', 0)}个三元组, 重叠度{comparison.get('overlap_ratio', 0):.2f}"
    
    def _call_llm(self, prompt: str) -> str:
        """调用LLM进行判决"""
        try:
            self.logger.info("🔄 正在调用LLM...")
            self.logger.debug(f"📝 发送的prompt: {prompt[:200]}...")
            
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的RAG系统评估专家，请客观公正地进行评估。"
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=self.config.judge_temperature,
                max_tokens=self.config.max_tokens,
                top_p=0.9
            )
            
            content = response.choices[0].message.content
            self.logger.info(f"✅ LLM响应成功")
            self.logger.info(f"📄 原始响应内容: {repr(content)}")
            self.logger.info(f"📄 格式化响应内容:\n{content}")
            
            return content
            
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            # 返回默认判决
            return "判决: Tie\n理由: LLM调用失败，无法进行判决"
    
    def _parse_llm_response(self, response: str) -> Tuple[str, str]:
        """从LLM的原始响应中解析判决和理由"""
        self.logger.info("🔍 开始解析LLM响应...")
        self.logger.debug(f"待解析的响应: {repr(response)}")
        
        lines = response.strip().split('\n')
        judgment = "Tie"  # 默认值
        reasoning = "未能从LLM响应中解析出理由。"
        
        self.logger.debug(f"分割后的行数: {len(lines)}")
        for i, line in enumerate(lines):
            self.logger.debug(f"第{i+1}行: {repr(line)}")
            
            # 🔧 改进：处理多种判决格式，包括带###前缀的
            if ("判决:" in line.lower() or "judgement:" in line.lower() or 
                "判决：" in line.lower() or "judgment:" in line.lower()):
                # 🔧 处理中英文冒号
                if ":" in line:
                    decision_part = line.split(":", 1)[-1].strip()
                elif "：" in line:
                    decision_part = line.split("：", 1)[-1].strip()
                else:
                    decision_part = line.strip()
                self.logger.info(f"🎯 找到判决行: {repr(line)}")
                self.logger.info(f"🎯 提取的判决部分: {repr(decision_part)}")
                
                if "a wins" in decision_part.lower():
                    judgment = "A wins"
                elif "b wins" in decision_part.lower():
                    judgment = "B wins"
                else:
                    judgment = "Tie"
                self.logger.info(f"🎯 最终判决: {judgment}")
                
            elif line.lower().startswith("理由:") or line.lower().startswith("reason:"):
                reasoning = line.split(":", 1)[-1].strip()
                self.logger.info(f"💭 找到理由行: {repr(line)}")
                self.logger.info(f"💭 提取的理由: {repr(reasoning)}")
        
        # 如果没有找到明确的理由，使用整个响应作为理由
        if reasoning == "未能从LLM响应中解析出理由。":
            reasoning = response
            self.logger.warning("⚠️ 未找到明确理由，使用整个响应")
        
        self.logger.info(f"✅ 解析完成 - 判决: {judgment}, 理由: {reasoning[:50]}...")
        return judgment, reasoning

    def _call_llm_with_logits(self, prompt: str) -> Dict[str, Any]:
        """调用LLM并返回logits信息"""
        # 如果是在线API模式，它可能不支持logprobs，进行优雅降级
        if "https://api.deepseek.com" in self.config.base_url:
            self.logger.warning("在线API模式不支持logprobs，降级为常规调用。")
            raw_response = self._call_llm(prompt)
            judgment, reasoning = self._parse_llm_response(raw_response)
            
            # 🔧 修复：根据实际判决结果设置logits，而不是固定为0.0
            # 这样可以避免所有判决都被强制为Tie
            if judgment == "A wins":
                logit_a, logit_b = 2.0, -2.0  # A明显胜出
            elif judgment == "B wins":
                logit_a, logit_b = -2.0, 2.0  # B明显胜出
            else:  # Tie
                logit_a, logit_b = 0.0, 0.0   # 平局
            
            result = {
                "content": raw_response,
                "choice": judgment,
                "logit_a": logit_a,
                "logit_b": logit_b,
                "raw_response": {"message": {"content": raw_response}} # 模拟结构
            }
            
            self.logger.info("📊 常规调用结果:")
            self.logger.info(f"   📄 内容: {repr(result['content'])}")
            self.logger.info(f"   🎯 选择: {result['choice']}")
            self.logger.info(f"   📈 logit_a: {result['logit_a']} (基于判决调整)")
            self.logger.info(f"   📈 logit_b: {result['logit_b']} (基于判决调整)")
            
            return result

        try:
            # 创建选择题格式来获取logits
            choice_prompt = f"""{prompt}

请从以下选项中选择：
A. A wins
B. B wins  
C. Tie

你的选择："""

            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的RAG系统评估专家，请客观公正地进行评估。"
                    },
                    {
                        "role": "user", 
                        "content": choice_prompt
                    }
                ],
                temperature=self.config.judge_temperature,
                max_tokens=self.config.max_tokens,
                top_p=0.9,
                logprobs=True,  # 请求logprobs
                top_logprobs=3  # 返回前3个token的logprobs
            )
            
            # 提取logits信息
            content = response.choices[0].message.content
            
            # 模拟logits计算（如果API不直接支持logprobs）
            # 基于LLM的温度和输出概率估算logits
            choice_map = {"A": "A wins", "B": "B wins", "C": "Tie"}
            
            # 解析选择
            choice = None
            for key, value in choice_map.items():
                if key in content.upper() or value in content:
                    choice = value
                    break
            
            if not choice:
                choice = "Tie"
            
            # 估算logits（基于温度和选择确定性）
            # 这里使用一个简化的方法，实际项目中应该使用真实的logprobs
            base_logit = 0.0
            if "明显" in content or "显然" in content or "clearly" in content.lower():
                confidence_logit = 2.0  # 高确定性
            elif "可能" in content or "或许" in content or "maybe" in content.lower():
                confidence_logit = -1.0  # 低确定性
            else:
                confidence_logit = 0.5  # 中等确定性
            
            # 根据选择分配logits
            if choice == "A wins":
                logit_a, logit_b = base_logit + confidence_logit, base_logit - confidence_logit
            elif choice == "B wins":
                logit_a, logit_b = base_logit - confidence_logit, base_logit + confidence_logit
            else:  # Tie
                logit_a, logit_b = base_logit, base_logit
            
            return {
                "content": content,
                "choice": choice,
                "logit_a": logit_a,
                "logit_b": logit_b,
                "raw_response": response
            }
            
        except Exception as e:
            self.logger.error(f"LLM logits调用失败: {e}")
            # 返回默认值
            return {
                "content": "判决: Tie\n理由: LLM调用失败，无法进行判决",
                "choice": "Tie",
                "logit_a": 0.0,
                "logit_b": 0.0,
                "raw_response": None
            }
    
    def _parse_judgment(self, raw_judgment: str, granularity: str) -> Dict[str, Any]:
        """解析LLM的判决结果"""
        lines = raw_judgment.strip().split('\n')
        
        label = "Tie"  # 默认值
        reason = "解析失败"
        
        for line in lines:
            line = line.strip()
            if line.startswith("判决:") or line.startswith("判决："):
                # 提取判决
                decision_part = line.split(":", 1)[-1].split("：", 1)[-1].strip()
                if "A wins" in decision_part or "A胜" in decision_part or "A更好" in decision_part:
                    label = "A wins"
                elif "B wins" in decision_part or "B胜" in decision_part or "B更好" in decision_part:
                    label = "B wins"
                else:
                    label = "Tie"
            
            elif line.startswith("理由:") or line.startswith("理由："):
                reason = line.split(":", 1)[-1].split("：", 1)[-1].strip()
        
        return {
            "label": label,
            "reason": reason,
            "granularity": granularity,
            "raw_response": raw_judgment
        }
    
    def _compute_margin_aware_tie(
        self, 
        question: str, 
        qa_a: Dict[str, Any], 
        qa_b: Dict[str, Any], 
        granularity: str,
        atoms: Dict[str, Any]
    ) -> float:
        """
        计算Margin-Aware Tie的软得分
        
        Returns:
            软得分 [-0.05, 0.05]，正值表示偏向A，负值表示偏向B
        """
        try:
            # 构造比较prompt以获取置信度
            confidence_prompt = f"""请对以下两个系统在{granularity}粒度上的表现进行精细比较，给出你的置信度评估。

问题: {question}

系统A: {qa_a.get('rag_answer', '')}
系统B: {qa_b.get('rag_answer', '')}

请回答: 如果必须选择一个更好的系统，你会选择哪个？请用1-10的数字表示你的置信度(1=非常不确定，10=非常确定)。

格式: 选择: [A/B], 置信度: [1-10]"""
            
            response = self._call_llm(confidence_prompt)
            
            # 解析置信度
            choice, confidence = self._parse_confidence(response)
            
            # 转换为margin score
            if choice == "A":
                margin_score = (confidence - 5) / 100  # 映射到[-0.05, 0.05]
            elif choice == "B":
                margin_score = -(confidence - 5) / 100
            else:
                margin_score = 0.0
            
            # 限制范围
            margin_score = max(-0.05, min(0.05, margin_score))
            
            return margin_score
            
        except Exception as e:
            self.logger.warning(f"Margin-Aware Tie计算失败: {e}")
            return 0.0
    
    def _parse_confidence(self, response: str) -> Tuple[str, int]:
        """解析置信度响应"""
        lines = response.strip().split('\n')
        choice = "A"  # 默认
        confidence = 5  # 默认中等置信度
        
        for line in lines:
            line = line.strip()
            if "选择:" in line or "选择：" in line:
                if "B" in line:
                    choice = "B"
                else:
                    choice = "A"
            
            if "置信度:" in line or "置信度：" in line:
                # 提取数字
                import re
                numbers = re.findall(r'\d+', line)
                if numbers:
                    confidence = int(numbers[0])
                    confidence = max(1, min(10, confidence))  # 限制范围
        
        return choice, confidence
    
    def get_judgment_statistics(self, judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取判决统计信息"""
        if not judgments:
            return {}
        
        a_wins = sum(1 for j in judgments if j["label"] == "A wins")
        b_wins = sum(1 for j in judgments if j["label"] == "B wins") 
        ties = sum(1 for j in judgments if j["label"] == "Tie")
        
        total = len(judgments)
        
        return {
            "total_judgments": total,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "a_win_rate": a_wins / total if total > 0 else 0,
            "b_win_rate": b_wins / total if total > 0 else 0,
            "tie_rate": ties / total if total > 0 else 0,
            "avg_margin_score": np.mean([j.get("margin_score", 0) for j in judgments])
        } 