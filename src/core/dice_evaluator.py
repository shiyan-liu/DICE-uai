"""
DICE评估器核心实现
实现检索-证据双通道判决和四粒度熵权融合
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy.stats import beta
from sklearn.preprocessing import normalize
import math

@dataclass
class DICEResult:
    """DICE评估结果"""
    question: str
    granularity_results: Dict[str, Dict[str, Any]]
    combined_delta: float
    ci_95: Tuple[float, float]
    ece: float
    consistency_scores: Dict[str, float]

@dataclass
class RAGPair:
    """RAG系统对比对"""
    question: str
    answer_a: str
    answer_b: str
    evidence_a: List[str]
    evidence_b: List[str]
    system_a_id: str = "System_A"
    system_b_id: str = "System_B"

class DICEEvaluator:
    """
    DICE评估器主类
    实现检索-证据双通道pairwise判决和多粒度融合
    """
    
    def __init__(self, 
                 llm_client=None,
                 tie_boundary: float = 0.1,
                 k_elo: int = 32,
                 bootstrap_samples: int = 1000,
                 random_seed: int = 42):
        """
        初始化DICE评估器
        
        Args:
            llm_client: LLM客户端（OpenAI或Anthropic）
            tie_boundary: Tie判决的边界阈值
            k_elo: Elo评分的K因子
            bootstrap_samples: Bootstrap采样次数
            random_seed: 随机种子
        """
        self.llm_client = llm_client
        self.tie_boundary = tie_boundary
        self.k_elo = k_elo
        self.bootstrap_samples = bootstrap_samples
        self.random_seed = random_seed
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 初始化日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 粒度权重
        self.granularity_weights = {}
        
    def evaluate_pair(self, rag_pair: RAGPair) -> DICEResult:
        """
        评估单个RAG系统对比对
        
        Args:
            rag_pair: RAG系统对比对
            
        Returns:
            DICEResult: 评估结果
        """
        self.logger.info(f"开始评估问题: {rag_pair.question}")
        
        # Step 1: 四粒度原子构建
        granularity_atoms = self._build_granularity_atoms(rag_pair)
        
        # Step 2: Pairwise判决
        granularity_results = {}
        for granularity in ['token', 'sentence', 'passage', 'kg']:
            result = self._pairwise_judgment(rag_pair, granularity, granularity_atoms[granularity])
            granularity_results[granularity] = result
            
        # Step 3: Margin-Aware Tie分解
        for granularity, result in granularity_results.items():
            if result['label'] == 'Tie':
                result['score'] = self._margin_aware_tie(result['raw_score'])
            else:
                result['score'] = 1.0 if result['label'] == 'A wins' else 0.0
                
        # Step 4: 检索-证据一致性校准
        consistency_scores = self._calculate_consistency_scores(rag_pair)
        
        # Step 5: 四粒度熵权融合
        combined_delta = self._entropy_weighted_fusion(granularity_results)
        
        # Step 6: 统计输出（Bootstrap置信区间）
        ci_95 = self._bootstrap_confidence_interval(granularity_results)
        ece = self._calculate_ece(granularity_results)
        
        return DICEResult(
            question=rag_pair.question,
            granularity_results=granularity_results,
            combined_delta=combined_delta,
            ci_95=ci_95,
            ece=ece,
            consistency_scores=consistency_scores
        )
    
    def _build_granularity_atoms(self, rag_pair: RAGPair) -> Dict[str, Dict]:
        """构建四粒度原子单元"""
        atoms = {
            'token': self._build_token_atoms(rag_pair),
            'sentence': self._build_sentence_atoms(rag_pair),
            'passage': self._build_passage_atoms(rag_pair),
            'kg': self._build_kg_atoms(rag_pair)
        }
        return atoms
    
    def _build_token_atoms(self, rag_pair: RAGPair) -> Dict:
        """构建Token级原子"""
        tokens_a = rag_pair.answer_a.split()
        tokens_b = rag_pair.answer_b.split()
        
        return {
            'tokens_a': tokens_a,
            'tokens_b': tokens_b,
            'unique_a': set(tokens_a) - set(tokens_b),
            'unique_b': set(tokens_b) - set(tokens_a),
            'common': set(tokens_a) & set(tokens_b)
        }
    
    def _build_sentence_atoms(self, rag_pair: RAGPair) -> Dict:
        """构建句子级原子"""
        import re
        
        sentences_a = re.split(r'[.!?。！？]', rag_pair.answer_a)
        sentences_b = re.split(r'[.!?。！？]', rag_pair.answer_b)
        
        # 过滤空句子
        sentences_a = [s.strip() for s in sentences_a if s.strip()]
        sentences_b = [s.strip() for s in sentences_b if s.strip()]
        
        return {
            'sentences_a': sentences_a,
            'sentences_b': sentences_b,
            'count_a': len(sentences_a),
            'count_b': len(sentences_b)
        }
    
    def _build_passage_atoms(self, rag_pair: RAGPair) -> Dict:
        """构建段落级原子"""
        return {
            'passage_a': rag_pair.answer_a,
            'passage_b': rag_pair.answer_b,
            'evidence_a': rag_pair.evidence_a,
            'evidence_b': rag_pair.evidence_b,
            'evidence_count_a': len(rag_pair.evidence_a),
            'evidence_count_b': len(rag_pair.evidence_b)
        }
    
    def _build_kg_atoms(self, rag_pair: RAGPair) -> Dict:
        """构建知识图谱三元组原子（简化版本）"""
        # 这里使用简化的实体识别方法
        # 在实际应用中可以使用NER工具
        import re
        
        def extract_entities(text):
            # 简单的实体抽取：数字、专有名词等
            entities = re.findall(r'\b[A-Z][a-zA-Z]+\b|\b\d+(?:\.\d+)?\b', text)
            return entities
        
        entities_a = extract_entities(rag_pair.answer_a)
        entities_b = extract_entities(rag_pair.answer_b)
        
        return {
            'entities_a': entities_a,
            'entities_b': entities_b,
            'unique_entities_a': set(entities_a) - set(entities_b),
            'unique_entities_b': set(entities_b) - set(entities_a),
            'common_entities': set(entities_a) & set(entities_b)
        }
    
    def _pairwise_judgment(self, rag_pair: RAGPair, granularity: str, atoms: Dict) -> Dict:
        """执行pairwise判决"""
        if self.llm_client is None:
            # 如果没有LLM客户端，使用规则基础的判决
            return self._rule_based_judgment(rag_pair, granularity, atoms)
        
        # 构建prompt
        prompt = self._build_judgment_prompt(rag_pair, granularity, atoms)
        
        try:
            # 调用LLM进行判决
            response = self.llm_client.generate(prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            self.logger.warning(f"LLM调用失败: {e}, 使用规则基础判决")
            return self._rule_based_judgment(rag_pair, granularity, atoms)
    
    def _rule_based_judgment(self, rag_pair: RAGPair, granularity: str, atoms: Dict) -> Dict:
        """规则基础的判决方法（当LLM不可用时）"""
        if granularity == 'token':
            len_a = len(atoms['tokens_a'])
            len_b = len(atoms['tokens_b'])
            unique_a = len(atoms['unique_a'])
            unique_b = len(atoms['unique_b'])
            
            # 基于长度和独特token数量判决
            score_a = len_a + unique_a * 0.5
            score_b = len_b + unique_b * 0.5
            
        elif granularity == 'sentence':
            count_a = atoms['count_a']
            count_b = atoms['count_b']
            
            # 基于句子数量和平均长度
            avg_len_a = len(rag_pair.answer_a) / max(count_a, 1)
            avg_len_b = len(rag_pair.answer_b) / max(count_b, 1)
            
            score_a = count_a + avg_len_a * 0.01
            score_b = count_b + avg_len_b * 0.01
            
        elif granularity == 'passage':
            len_a = len(rag_pair.answer_a)
            len_b = len(rag_pair.answer_b)
            evidence_a = atoms['evidence_count_a']
            evidence_b = atoms['evidence_count_b']
            
            score_a = len_a + evidence_a * 10
            score_b = len_b + evidence_b * 10
            
        else:  # kg
            entities_a = len(atoms['entities_a'])
            entities_b = len(atoms['entities_b'])
            
            score_a = entities_a
            score_b = entities_b
        
        # 判决逻辑
        diff = abs(score_a - score_b)
        max_score = max(score_a, score_b)
        
        if max_score == 0 or diff / max_score < self.tie_boundary:
            label = 'Tie'
            reason = f"{granularity}级别差异微小"
        elif score_a > score_b:
            label = 'A wins'
            reason = f"{granularity}级别A系统表现更好"
        else:
            label = 'B wins'
            reason = f"{granularity}级别B系统表现更好"
            
        return {
            'label': label,
            'reason': reason,
            'raw_score': score_a - score_b,
            'score_a': score_a,
            'score_b': score_b
        }
    
    def _build_judgment_prompt(self, rag_pair: RAGPair, granularity: str, atoms: Dict) -> str:
        """构建判决prompt"""
        prompt = f"""请从{granularity}粒度比较两个RAG系统的回答质量。

问题: {rag_pair.question}

系统A:
证据: {'; '.join(rag_pair.evidence_a)}
回答: {rag_pair.answer_a}

系统B:
证据: {'; '.join(rag_pair.evidence_b)}
回答: {rag_pair.answer_b}

评估粒度: {granularity}

请从准确性、完整性、相关性三个维度进行{granularity}级别的比较，只能选择以下三个选项之一：
- A wins: 系统A明显更好
- B wins: 系统B明显更好  
- Tie: 两系统表现相当

回答格式：
判决: [A wins/B wins/Tie]
理由: [一句话解释，不超过20字]"""

        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict:
        """解析LLM响应"""
        lines = response.strip().split('\n')
        
        label = 'Tie'
        reason = '无法解析'
        
        for line in lines:
            if '判决:' in line or 'Label:' in line:
                if 'A wins' in line:
                    label = 'A wins'
                elif 'B wins' in line:
                    label = 'B wins'
                else:
                    label = 'Tie'
            elif '理由:' in line or 'Reason:' in line:
                reason = line.split(':', 1)[1].strip()
                
        return {
            'label': label,
            'reason': reason,
            'raw_score': 0.5 if label == 'Tie' else (1.0 if label == 'A wins' else 0.0)
        }
    
    def _margin_aware_tie(self, raw_score: float) -> float:
        """Margin-Aware Tie分解"""
        # 使用logistic函数将raw_score映射到[0,1]区间
        # 温度参数用于控制软化程度
        temperature = 2.0
        sigmoid_score = 1 / (1 + np.exp(-raw_score / temperature))
        
        # 确保Tie的软得分在0.4-0.6之间
        if 0.4 <= sigmoid_score <= 0.6:
            return sigmoid_score
        else:
            return 0.5  # 默认Tie得分
    
    def _calculate_consistency_scores(self, rag_pair: RAGPair) -> Dict[str, float]:
        """计算检索-证据一致性得分"""
        def calculate_entailment_score(answer: str, evidence: List[str]) -> float:
            """计算回答与证据的蕴含得分"""
            if not evidence:
                return 0.0
                
            # 简化的一致性计算：基于词汇重叠
            answer_words = set(answer.lower().split())
            total_overlap = 0
            
            for doc in evidence:
                doc_words = set(doc.lower().split())
                overlap = len(answer_words & doc_words)
                total_overlap += overlap / max(len(answer_words), 1)
                
            return total_overlap / len(evidence)
        
        consistency_a = calculate_entailment_score(rag_pair.answer_a, rag_pair.evidence_a)
        consistency_b = calculate_entailment_score(rag_pair.answer_b, rag_pair.evidence_b)
        
        return {
            'consistency_a': consistency_a,
            'consistency_b': consistency_b,
            'consistency_diff': consistency_a - consistency_b
        }
    
    def _entropy_weighted_fusion(self, granularity_results: Dict) -> float:
        """四粒度熵权融合"""
        # 计算每个粒度的分布熵
        entropies = {}
        scores = {}
        
        for granularity, result in granularity_results.items():
            score = result['score']
            scores[granularity] = score
            
            # 计算分布熵（基于得分的确定性）
            if score == 0.5:  # 完全不确定
                entropy = 1.0
            else:
                # 基于距离0.5的程度计算熵
                certainty = abs(score - 0.5) * 2  # [0,1]
                entropy = 1 - certainty
                
            entropies[granularity] = entropy
        
        # 计算熵权重（熵越低权重越高）
        entropy_weights = {}
        total_inv_entropy = sum(1 / (e + 1e-8) for e in entropies.values())
        
        for granularity, entropy in entropies.items():
            weight = (1 / (entropy + 1e-8)) / total_inv_entropy
            entropy_weights[granularity] = weight
        
        # 加权融合
        weighted_score = sum(entropy_weights[g] * scores[g] for g in scores.keys())
        
        # 转换为Elo差值
        if weighted_score == 0.5:
            elo_delta = 0
        else:
            # 基于胜率计算Elo差值
            win_rate = weighted_score
            if win_rate >= 1.0:
                elo_delta = 400  # 最大差值
            elif win_rate <= 0.0:
                elo_delta = -400
            else:
                elo_delta = 400 * math.log10(win_rate / (1 - win_rate))
                
        return elo_delta
    
    def _bootstrap_confidence_interval(self, granularity_results: Dict) -> Tuple[float, float]:
        """Bootstrap置信区间"""
        scores = [result['score'] for result in granularity_results.values()]
        
        bootstrap_deltas = []
        for _ in range(self.bootstrap_samples):
            # 重采样
            bootstrap_scores = np.random.choice(scores, size=len(scores), replace=True)
            
            # 计算加权得分
            weighted_score = np.mean(bootstrap_scores)
            
            # 转换为Elo差值
            if weighted_score == 0.5:
                elo_delta = 0
            else:
                win_rate = weighted_score
                if win_rate >= 1.0:
                    elo_delta = 400
                elif win_rate <= 0.0:
                    elo_delta = -400
                else:
                    elo_delta = 400 * math.log10(win_rate / (1 - win_rate))
                    
            bootstrap_deltas.append(elo_delta)
        
        # 计算95%置信区间
        ci_lower = np.percentile(bootstrap_deltas, 2.5)
        ci_upper = np.percentile(bootstrap_deltas, 97.5)
        
        return (ci_lower, ci_upper)
    
    def _calculate_ece(self, granularity_results: Dict) -> float:
        """计算Expected Calibration Error"""
        scores = [result['score'] for result in granularity_results.values()]
        
        # 简化的ECE计算
        confidences = [abs(score - 0.5) * 2 for score in scores]
        accuracies = [1.0 if score != 0.5 else 0.5 for score in scores]
        
        ece = np.mean([abs(conf - acc) for conf, acc in zip(confidences, accuracies)])
        return ece
    
    def evaluate_multiple_pairs(self, rag_pairs: List[RAGPair]) -> List[DICEResult]:
        """评估多个RAG系统对比对"""
        results = []
        
        for i, pair in enumerate(rag_pairs):
            self.logger.info(f"评估进度: {i+1}/{len(rag_pairs)}")
            result = self.evaluate_pair(pair)
            results.append(result)
            
        return results
    
    def save_results(self, results: List[DICEResult], output_path: str):
        """保存评估结果到文件"""
        output_data = []
        
        for result in results:
            data = {
                'question': result.question,
                'granularity_results': result.granularity_results,
                'combined_delta': result.combined_delta,
                'ci_95': result.ci_95,
                'ece': result.ece,
                'consistency_scores': result.consistency_scores
            }
            output_data.append(data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for data in output_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                
        self.logger.info(f"结果已保存到: {output_path}") 