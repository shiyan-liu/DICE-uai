#!/usr/bin/env python3
"""
DICE四粒度原子构建模块
实现token、sentence、passage、KG四个粒度的原子单元构建
"""

import re
import json
import logging
from typing import List, Dict, Any, Tuple
import jieba
import spacy
from collections import defaultdict

class GranularityBuilder:
    """四粒度原子构建器"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DICE.Granularity")
        
        # 初始化NLP工具
        self._init_nlp_tools()
    
    def _init_nlp_tools(self):
        """初始化NLP工具"""
        try:
            # 尝试加载spacy中文模型
            self.nlp = spacy.load("zh_core_web_sm")
        except OSError:
            self.logger.warning("spacy中文模型未找到，使用简化版本")
            self.nlp = None
        
        # 初始化jieba
        jieba.setLogLevel(logging.WARNING)
    
    def build_granularities(
        self, 
        question: str, 
        qa_a: Dict[str, Any], 
        qa_b: Dict[str, Any],
        groundtruth: str = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        构建四个粒度的原子单元
        
        Args:
            question: 问题文本
            qa_a: 系统A的QA数据 
            qa_b: 系统B的QA数据
            groundtruth: 标准答案（可选）
            
        Returns:
            四个粒度的原子单元字典
        """
        granularities = {}
        
        # Token粒度
        if self.config.enable_token:
            granularities["token"] = self._build_token_granularity(question, qa_a, qa_b, groundtruth)
        
        # Sentence粒度
        if self.config.enable_sentence:
            granularities["sentence"] = self._build_sentence_granularity(question, qa_a, qa_b, groundtruth)
        
        # Passage粒度
        if self.config.enable_passage:
            granularities["passage"] = self._build_passage_granularity(question, qa_a, qa_b, groundtruth)
        
        # KG粒度
        if self.config.enable_kg:
            granularities["kg"] = self._build_kg_granularity(question, qa_a, qa_b, groundtruth)
        
        return granularities
    
    def _build_token_granularity(self, question: str, qa_a: Dict, qa_b: Dict, groundtruth: str = None) -> Dict[str, Any]:
        """构建Token粒度原子单元"""
        # 提取关键信息
        answer_a = qa_a.get("rag_answer", "")
        answer_b = qa_b.get("rag_answer", "")
        context_a = qa_a.get("context", [])
        context_b = qa_b.get("context", [])
        
        # Token级别的事实单元提取
        tokens_a = self._extract_fact_tokens(answer_a, context_a)
        tokens_b = self._extract_fact_tokens(answer_b, context_b)
        tokens_gt = self._extract_fact_tokens(groundtruth or "", []) if groundtruth else []
        
        return {
            "question": question,
            "tokens_a": tokens_a,
            "tokens_b": tokens_b,
            "tokens_groundtruth": tokens_gt,
            "comparison_units": self._compare_token_facts(tokens_a, tokens_b, tokens_gt)
        }
    
    def _extract_fact_tokens(self, answer: str, context: List[str]) -> List[Dict[str, Any]]:
        """提取事实性token"""
        tokens = []
        
        # 确保answer是字符串类型
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        elif not isinstance(answer, str):
            answer = str(answer) if answer else ""
        
        # 使用jieba分词
        answer_tokens = list(jieba.cut(answer))
        
        # 提取数字、日期、专有名词等关键token
        for token in answer_tokens:
            token_info = {
                "text": token,
                "type": self._classify_token(token),
                "evidence": self._find_token_evidence(token, context)
            }
            
            # 只保留关键token
            if token_info["type"] != "common":
                tokens.append(token_info)
        
        return tokens
    
    def _classify_token(self, token: str) -> str:
        """分类token类型"""
        # 数字
        if re.match(r'^\d+(\.\d+)?%?$', token):
            return "number"
        
        # 日期
        if re.match(r'\d{4}年|\d{1,2}月|\d{1,2}日', token):
            return "date"
        
        # 专有名词（简化版本）
        if len(token) > 1 and not re.match(r'^[的了在是和与等]', token):
            return "entity"
        
        return "common"
    
    def _find_token_evidence(self, token: str, context: List[str]) -> List[str]:
        """在上下文中找到token的证据"""
        evidence = []
        for i, ctx in enumerate(context):
            if token in ctx:
                # 提取包含token的句子片段
                sentences = re.split(r'[。！？；]', ctx)
                for sent in sentences:
                    if token in sent:
                        evidence.append(f"Context{i+1}: {sent.strip()}")
        return evidence
    
    def _compare_token_facts(self, tokens_a: List[Dict], tokens_b: List[Dict], tokens_gt: List[Dict] = None) -> List[Dict]:
        """比较两个系统的token级事实"""
        comparisons = []
        
        # 构建token映射
        tokens_a_map = {t["text"]: t for t in tokens_a}
        tokens_b_map = {t["text"]: t for t in tokens_b}
        
        all_tokens = set(tokens_a_map.keys()) | set(tokens_b_map.keys())
        
        for token in all_tokens:
            comparison = {
                "token": token,
                "in_a": token in tokens_a_map,
                "in_b": token in tokens_b_map,
                "evidence_a": tokens_a_map.get(token, {}).get("evidence", []),
                "evidence_b": tokens_b_map.get(token, {}).get("evidence", [])
            }
            comparisons.append(comparison)
        
        return comparisons
    
    def _build_sentence_granularity(self, question: str, qa_a: Dict, qa_b: Dict, groundtruth: str = None) -> Dict[str, Any]:
        """构建Sentence粒度原子单元"""
        answer_a = qa_a.get("rag_answer", "")
        answer_b = qa_b.get("rag_answer", "")
        context_a = qa_a.get("context", [])
        context_b = qa_b.get("context", [])
        
        # 句子级别的切片
        sentences_a = self._split_sentences(answer_a)
        sentences_b = self._split_sentences(answer_b)
        
        # 为每个句子找证据
        sentences_a_with_evidence = [
            {
                "text": sent,
                "evidence": self._find_sentence_evidence(sent, context_a),
                "semantic_type": self._classify_sentence_type(sent)
            }
            for sent in sentences_a
        ]
        
        sentences_b_with_evidence = [
            {
                "text": sent,
                "evidence": self._find_sentence_evidence(sent, context_b),
                "semantic_type": self._classify_sentence_type(sent)
            }
            for sent in sentences_b
        ]
        
        return {
            "question": question,
            "sentences_a": sentences_a_with_evidence,
            "sentences_b": sentences_b_with_evidence,
            "comparison_units": self._compare_sentences(sentences_a_with_evidence, sentences_b_with_evidence)
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """中文分句"""
        sentences = re.split(r'[。！？；]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _classify_sentence_type(self, sentence: str) -> str:
        """分类句子语义类型"""
        if re.search(r'\d+(\.\d+)?%?', sentence):
            return "factual"
        elif re.search(r'因为|由于|所以|因此', sentence):
            return "causal"
        elif re.search(r'如果|假如|倘若', sentence):
            return "conditional"
        else:
            return "descriptive"
    
    def _find_sentence_evidence(self, sentence: str, context: List[str]) -> List[str]:
        """为句子找证据"""
        evidence = []
        
        # 提取句子中的关键词
        keywords = [word for word in jieba.cut(sentence) if len(word) > 1]
        
        for i, ctx in enumerate(context):
            # 计算重叠程度
            overlap = sum(1 for keyword in keywords if keyword in ctx)
            if overlap >= max(1, len(keywords) * 0.3):  # 至少30%重叠
                evidence.append(f"Context{i+1}: {ctx[:200]}...")
        
        return evidence
    
    def _compare_sentences(self, sentences_a: List[Dict], sentences_b: List[Dict]) -> List[Dict]:
        """比较句子级别的内容"""
        comparisons = []
        
        # 语义相似度比较（简化版本）
        for i, sent_a in enumerate(sentences_a):
            best_match = None
            best_similarity = 0
            
            for j, sent_b in enumerate(sentences_b):
                similarity = self._compute_sentence_similarity(sent_a["text"], sent_b["text"])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (j, sent_b)
            
            comparison = {
                "sentence_a": sent_a,
                "sentence_b": best_match[1] if best_match else None,
                "similarity": best_similarity,
                "semantic_match": best_similarity > 0.5
            }
            comparisons.append(comparison)
        
        return comparisons
    
    def _compute_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """计算句子相似度（简化版本，基于词汇重叠）"""
        words1 = set(jieba.cut(sent1))
        words2 = set(jieba.cut(sent2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _build_passage_granularity(self, question: str, qa_a: Dict, qa_b: Dict, groundtruth: str = None) -> Dict[str, Any]:
        """构建Passage粒度原子单元"""
        context_a = qa_a.get("context", [])
        context_b = qa_b.get("context", [])
        answer_a = qa_a.get("rag_answer", "")
        answer_b = qa_b.get("rag_answer", "")
        
        # 段落级别的块
        passages_a = self._create_passage_chunks(context_a, answer_a)
        passages_b = self._create_passage_chunks(context_b, answer_b)
        
        return {
            "question": question,
            "passages_a": passages_a,
            "passages_b": passages_b,
            "comparison_units": self._compare_passages(passages_a, passages_b)
        }
    
    def _create_passage_chunks(self, context: List[str], answer: str) -> List[Dict[str, Any]]:
        """创建段落块"""
        chunks = []
        
        # 将每个context作为一个段落
        for i, ctx in enumerate(context):
            chunk = {
                "id": f"passage_{i+1}",
                "text": ctx,
                "answer_support": self._measure_answer_support(ctx, answer),
                "key_entities": self._extract_passage_entities(ctx),
                "length": len(ctx)
            }
            chunks.append(chunk)
        
        return chunks
    
    def _measure_answer_support(self, passage: str, answer: str) -> float:
        """测量段落对答案的支持度"""
        answer_words = set(jieba.cut(answer))
        passage_words = set(jieba.cut(passage))
        
        if not answer_words:
            return 0.0
        
        support = len(answer_words & passage_words) / len(answer_words)
        return support
    
    def _extract_passage_entities(self, passage: str) -> List[str]:
        """提取段落中的实体"""
        # 简化版实体提取
        entities = []
        
        # 提取数字
        numbers = re.findall(r'\d+(\.\d+)?%?', passage)
        entities.extend(numbers)
        
        # 提取可能的专有名词（长度>2的词）
        words = jieba.cut(passage)
        for word in words:
            if len(word) > 2 and not re.match(r'^[的了在是和与等]', word):
                entities.append(word)
        
        return list(set(entities))[:10]  # 限制数量
    
    def _compare_passages(self, passages_a: List[Dict], passages_b: List[Dict]) -> List[Dict]:
        """比较段落级别的内容"""
        comparison = {
            "passage_count_a": len(passages_a),
            "passage_count_b": len(passages_b),
            "total_length_a": sum(p["length"] for p in passages_a),
            "total_length_b": sum(p["length"] for p in passages_b),
            "avg_support_a": np.mean([p["answer_support"] for p in passages_a]) if passages_a else 0,
            "avg_support_b": np.mean([p["answer_support"] for p in passages_b]) if passages_b else 0,
            "entity_overlap": self._compute_entity_overlap(passages_a, passages_b)
        }
        
        return comparison
    
    def _compute_entity_overlap(self, passages_a: List[Dict], passages_b: List[Dict]) -> float:
        """计算实体重叠度"""
        entities_a = set()
        entities_b = set()
        
        for p in passages_a:
            entities_a.update(p["key_entities"])
        
        for p in passages_b:
            entities_b.update(p["key_entities"])
        
        if not entities_a or not entities_b:
            return 0.0
        
        intersection = entities_a & entities_b
        union = entities_a | entities_b
        
        return len(intersection) / len(union) if union else 0.0
    
    def _build_kg_granularity(self, question: str, qa_a: Dict, qa_b: Dict, groundtruth: str = None) -> Dict[str, Any]:
        """构建KG粒度原子单元（知识图谱三元组）"""
        answer_a = qa_a.get("rag_answer", "")
        answer_b = qa_b.get("rag_answer", "")
        context_a = qa_a.get("context", [])
        context_b = qa_b.get("context", [])
        
        # 提取三元组
        triples_a = self._extract_triples(answer_a, context_a)
        triples_b = self._extract_triples(answer_b, context_b)
        
        return {
            "question": question,
            "triples_a": triples_a,
            "triples_b": triples_b,
            "comparison_units": self._compare_triples(triples_a, triples_b)
        }
    
    def _extract_triples(self, answer: str, context: List[str]) -> List[Dict[str, Any]]:
        """提取知识三元组（简化版本）"""
        triples = []
        
        # 合并答案和上下文
        full_text = answer + " " + " ".join(context)
        
        # 使用规则提取简单三元组
        triples.extend(self._extract_numeric_triples(full_text))
        triples.extend(self._extract_action_triples(full_text))
        triples.extend(self._extract_property_triples(full_text))
        
        return triples
    
    def _extract_numeric_triples(self, text: str) -> List[Dict[str, Any]]:
        """提取数值型三元组"""
        triples = []
        
        # 模式：主语 + 数值关系
        patterns = [
            r'(\w+).*?(\d+(\.\d+)?%?)',
            r'(\w+).*?为.*?(\d+(\.\d+)?%?)',
            r'(\w+).*?是.*?(\d+(\.\d+)?%?)',
            r'(\w+).*?达到.*?(\d+(\.\d+)?%?)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                subject = match.group(1)
                value = match.group(2)
                triple = {
                    "subject": subject,
                    "predicate": "has_value",
                    "object": value,
                    "type": "numeric",
                    "evidence": match.group(0)
                }
                triples.append(triple)
        
        return triples
    
    def _extract_action_triples(self, text: str) -> List[Dict[str, Any]]:
        """提取动作型三元组"""
        triples = []
        
        # 简单的动作模式
        action_words = ['提高', '降低', '增加', '减少', '宣布', '发布', '实施']
        
        for action in action_words:
            pattern = f'(\\w+).*?{action}.*?(\\w+)'
            matches = re.finditer(pattern, text)
            for match in matches:
                subject = match.group(1)
                obj = match.group(2)
                triple = {
                    "subject": subject,
                    "predicate": action,
                    "object": obj,
                    "type": "action",
                    "evidence": match.group(0)
                }
                triples.append(triple)
        
        return triples
    
    def _extract_property_triples(self, text: str) -> List[Dict[str, Any]]:
        """提取属性型三元组"""
        triples = []
        
        # 属性关系模式
        patterns = [
            r'(\w+).*?的.*?(\w+)',
            r'(\w+).*?属于.*?(\w+)',
            r'(\w+).*?来自.*?(\w+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                subject = match.group(1)
                obj = match.group(2)
                triple = {
                    "subject": subject,
                    "predicate": "has_property",
                    "object": obj,
                    "type": "property", 
                    "evidence": match.group(0)
                }
                triples.append(triple)
        
        return triples[:10]  # 限制数量
    
    def _compare_triples(self, triples_a: List[Dict], triples_b: List[Dict]) -> Dict[str, Any]:
        """比较三元组"""
        # 构建三元组字符串表示用于比较
        triples_a_str = set()
        triples_b_str = set()
        
        for triple in triples_a:
            triple_str = f"{triple['subject']}-{triple['predicate']}-{triple['object']}"
            triples_a_str.add(triple_str)
        
        for triple in triples_b:
            triple_str = f"{triple['subject']}-{triple['predicate']}-{triple['object']}"
            triples_b_str.add(triple_str)
        
        # 计算overlap
        intersection = triples_a_str & triples_b_str
        union = triples_a_str | triples_b_str
        
        return {
            "triple_count_a": len(triples_a),
            "triple_count_b": len(triples_b),
            "unique_to_a": len(triples_a_str - triples_b_str),
            "unique_to_b": len(triples_b_str - triples_a_str),
            "common_triples": len(intersection),
            "overlap_ratio": len(intersection) / len(union) if union else 0.0
        }

# 添加numpy导入
import numpy as np 