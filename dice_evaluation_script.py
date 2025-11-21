#!/usr/bin/env python3
"""
DICE准确率评估脚本
用于验证DICE系统的可信度，通过与人工标注的"金标准"进行对比
"""

import json
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from datetime import datetime
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import kendalltau
import pandas as pd

from src.dice.dice_simplified import SimplifiedDICEEvaluator, SimplifiedDICEConfig
from ragas_evaluator import RagasEvaluator, RagasConfig, RagasValidationEvaluator


class DICEValidationEvaluator:
    """DICE验证评估器 - 用于评估DICE本身的准确性"""
    
    def __init__(self, config: SimplifiedDICEConfig, tournament_result_file: str = None):
        self.config = config
        self.logger = logging.getLogger("DICEValidation")
        self.dice_evaluator = SimplifiedDICEEvaluator(config)
        self.tournament_result_file = tournament_result_file
        self.tournament_results = None
        
        # 设置日志
        self._setup_logger()
        
        # 如果提供了tournament结果文件，则加载它
        if self.tournament_result_file and Path(self.tournament_result_file).exists():
            self._load_tournament_results()
    
    def _setup_logger(self):
        """设置日志"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _load_tournament_results(self):
        """加载tournament结果文件"""
        try:
            self.logger.info(f"开始加载tournament结果文件: {self.tournament_result_file}")
            with open(self.tournament_result_file, 'r', encoding='utf-8') as f:
                self.tournament_results = json.load(f)
            self.logger.info(f"成功加载tournament结果文件，包含 {len(self.tournament_results.get('swiss_results', {}).get('match_records', []))} 个对决记录")
        except Exception as e:
            self.logger.error(f"加载tournament结果文件失败: {e}")
            self.tournament_results = None
    
    def _find_tournament_match(self, system_a: str, system_b: str, question: str) -> Dict[str, Any]:
        """在tournament结果中查找匹配的对决"""
        if not self.tournament_results:
            return None
        
        # 查找匹配的系统对
        match_records = self.tournament_results.get('swiss_results', {}).get('match_records', [])
        
        for match in match_records:
            match_system_a = match.get('system_a', '')
            match_system_b = match.get('system_b', '')
            
            # 检查系统对是否匹配（考虑顺序）
            if ((match_system_a == system_a and match_system_b == system_b) or 
                (match_system_a == system_b and match_system_b == system_a)):
                
                # 在comparison结果中查找匹配的问题
                comparison = match.get('comparison', {})
                question_results = comparison.get('question_results', [])
                
                for q_result in question_results:
                    if q_result.get('question', '') == question:
                        return q_result
        
        return None
    
    def sample_evaluation_pairs(self, qacg_files: List[str], num_samples: int = 200, 
                               random_seed: int = 42) -> List[Dict[str, Any]]:
        """采样评估对"""
        import random
        random.seed(random_seed)
        
        all_pairs = []
        for file_path in qacg_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_pairs.extend(data)
        
        if len(all_pairs) < num_samples:
            self.logger.warning(f"可用数据对数量({len(all_pairs)})少于请求的采样数量({num_samples})")
            return all_pairs
        
        return random.sample(all_pairs, num_samples)
    
    def run_dice_evaluation(self, evaluation_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """运行DICE评估"""
        results = []
        
        for i, pair in enumerate(evaluation_pairs):
            try:
                # 从QACG格式中提取问答对
                qa_a = pair.get('qa_a', {})
                qa_b = pair.get('qa_b', {})
                
                question = qa_a.get('question', '')
                system_a = pair.get('system_a', '')
                system_b = pair.get('system_b', '')
                
                # 首先尝试从tournament结果中查找匹配
                tournament_match = self._find_tournament_match(system_a, system_b, question)
                
                if tournament_match:
                    # 使用tournament中的已有结果
                    self.logger.info(f"使用tournament结果: {system_a} vs {system_b} - {question[:50]}...")
                    
                    passage_judgment = tournament_match.get('passage_judgment', {})
                    score_a = passage_judgment.get('prob_a', 0.0)
                    score_b = passage_judgment.get('prob_b', 0.0)
                    dice_score = score_a - score_b
                    
                    result = {
                        'index': i,
                        'question': question,
                        'system_a': system_a,
                        'system_b': system_b,
                        'answer_a': qa_a.get('rag_answer', ''),
                        'answer_b': qa_b.get('rag_answer', ''),
                        'context_a': qa_a.get('context', []),
                        'context_b': qa_b.get('context', []),
                        'dice_score': dice_score,
                        'dice_explanation': passage_judgment.get('reason', ''),
                        'human_annotation': pair.get('human_annotation', ''),
                        'prob_a': score_a,
                        'prob_b': score_b,
                        'win_type': passage_judgment.get('win_type', 'Unknown'),
                        'source': 'tournament'  # 标记来源
                    }
                else:
                    # 没有找到tournament结果，进行新的推理
                    self.logger.info(f"未找到tournament结果，进行新推理: {system_a} vs {system_b} - {question[:50]}...")
                    
                    # 构建问答对格式
                    target_qa_a = {
                        'answer': qa_a.get('rag_answer', ''),
                        'context': qa_a.get('context', [])
                    }
                    
                    target_qa_b = {
                        'answer': qa_b.get('rag_answer', ''),
                        'context': qa_b.get('context', [])
                    }
                    
                    # 使用DICE的pairwise judge进行评估
                    judgment = self.dice_evaluator.pairwise_judge.judge_pair(
                        question=question,
                        qa_a=target_qa_a,
                        qa_b=target_qa_b,
                        granularity="passage"  # 使用passage粒度进行评估
                    )
                    
                    # 从判决结果中提取分数
                    passage_judgment = judgment.get('passage_judgment', {})
                    score_a = passage_judgment.get('prob_a', 0.0)
                    score_b = passage_judgment.get('prob_b', 0.0)
                    
                    # 计算相对分数（系统A相对于系统B的优势）
                    dice_score = score_a - score_b
                    
                    result = {
                        'index': i,
                        'question': question,
                        'system_a': system_a,
                        'system_b': system_b,
                        'answer_a': qa_a.get('rag_answer', ''),
                        'answer_b': qa_b.get('rag_answer', ''),
                        'context_a': qa_a.get('context', []),
                        'context_b': qa_b.get('context', []),
                        'dice_score': dice_score,
                        'dice_explanation': passage_judgment.get('reason', ''),
                        'human_annotation': pair.get('human_annotation', ''),
                        'prob_a': score_a,
                        'prob_b': score_b,
                        'win_type': passage_judgment.get('win_type', 'Unknown'),
                        'source': 'new_inference'  # 标记来源
                    }
                
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"已完成 {i + 1}/{len(evaluation_pairs)} 个评估")
                    
            except Exception as e:
                self.logger.error(f"评估第{i}个样本时出错: {e}")
                # 添加一个默认结果
                result = {
                    'index': i,
                    'question': pair.get('qa_a', {}).get('question', ''),
                    'system_a': pair.get('system_a', ''),
                    'system_b': pair.get('system_b', ''),
                    'answer_a': pair.get('qa_a', {}).get('rag_answer', ''),
                    'answer_b': pair.get('qa_b', {}).get('rag_answer', ''),
                    'context_a': pair.get('qa_a', {}).get('context', []),
                    'context_b': pair.get('qa_b', {}).get('context', []),
                    'dice_score': 0.0,
                    'dice_explanation': f'评估出错: {str(e)}',
                    'human_annotation': pair.get('human_annotation', ''),
                    'prob_a': 0.0,
                    'prob_b': 0.0,
                    'win_type': 'Error',
                    'source': 'error'
                }
                results.append(result)
                continue
        
        return results
    
    def load_human_annotations(self, annotation_file: str) -> Dict[int, str]:
        """加载人工标注"""
        annotations = {}
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if 'index' in item and 'human_annotation' in item:
                        annotations[item['index']] = item['human_annotation']
        except Exception as e:
            self.logger.error(f"加载人工标注文件失败: {e}")
        return annotations
    
    def calculate_agreement(self, results: List[Dict[str, Any]], gold_labels: Dict[int, str]) -> Dict[str, float]:
        """计算一致性指标"""
        dice_scores = []
        human_scores = []
        
        for result in results:
            if result['index'] in gold_labels:
                dice_scores.append(result['dice_score'])
                # 将人工标注转换为数值分数
                human_annotation = gold_labels[result['index']]
                if human_annotation.lower() in ['a', 'system_a', 'good', 'correct', 'accurate']:
                    human_scores.append(1.0)  # 系统A更好
                elif human_annotation.lower() in ['b', 'system_b', 'bad', 'incorrect', 'inaccurate']:
                    human_scores.append(-1.0)  # 系统B更好
                else:
                    human_scores.append(0.0)  # 平局或中性
        
        if len(dice_scores) == 0:
            return {'correlation': 0.0, 'kappa': 0.0}
        
        # 计算皮尔逊相关系数
        correlation = np.corrcoef(dice_scores, human_scores)[0, 1] if len(dice_scores) > 1 else 0.0
        
        # 计算Cohen's Kappa (将分数转换为二分类)
        dice_binary = [1 if score > 0 else 0 for score in dice_scores]  # 正数表示A更好
        human_binary = [1 if score > 0 else 0 for score in human_scores]  # 正数表示A更好
        kappa = cohen_kappa_score(dice_binary, human_binary) if len(dice_scores) > 1 else 0.0
        
        return {
            'correlation': correlation,
            'kappa': kappa,
            'sample_size': len(dice_scores)
        }
    
    def calculate_elo_correlation(self, results: List[Dict[str, Any]], gold_labels: Dict[int, str]) -> Dict[str, float]:
        """计算ELO相关性"""
        # 这里可以实现ELO评分系统的相关性计算
        # 暂时返回基本的相关性指标
        return self.calculate_agreement(results, gold_labels)
    
    def analyze_disagreement_cases(self, results: List[Dict[str, Any]], gold_labels: Dict[int, str]) -> List[Dict[str, Any]]:
        """分析不一致案例"""
        disagreement_cases = []
        
        for result in results:
            if result['index'] in gold_labels:
                dice_score = result['dice_score']
                human_annotation = gold_labels[result['index']]
                
                # 判断是否不一致
                dice_a_better = dice_score > 0  # DICE认为系统A更好
                human_a_better = human_annotation.lower() in ['a', 'system_a', 'good', 'correct', 'accurate']
                
                if dice_a_better != human_a_better:
                    disagreement_cases.append({
                        'index': result['index'],
                        'question': result['question'],
                        'system_a': result.get('system_a', ''),
                        'system_b': result.get('system_b', ''),
                        'answer_a': result.get('answer_a', ''),
                        'answer_b': result.get('answer_b', ''),
                        'dice_score': dice_score,
                        'human_annotation': human_annotation,
                        'disagreement_type': 'dice_a_better_human_b_better' if dice_a_better else 'dice_b_better_human_a_better'
                    })
        
        return disagreement_cases
    
    def print_disagreement_analysis(self, disagreement_cases: List[Dict[str, Any]]) -> None:
        """打印不一致分析"""
        if not disagreement_cases:
            self.logger.info("没有发现不一致案例")
            return
        
        self.logger.info(f"发现 {len(disagreement_cases)} 个不一致案例:")
        
        for case in disagreement_cases[:5]:  # 只显示前5个
            self.logger.info(f"案例 {case['index']}: DICE分数={case['dice_score']:.3f}, 人工标注={case['human_annotation']}")
            self.logger.info(f"问题: {case['question'][:100]}...")
    
    def generate_validation_report(self, results: List[Dict[str, Any]], gold_labels: Dict[int, str]) -> Dict[str, Any]:
        """生成验证报告"""
        agreement_metrics = self.calculate_agreement(results, gold_labels)
        disagreement_cases = self.analyze_disagreement_cases(results, gold_labels)
        
        report = {
            'total_samples': len(results),
            'annotated_samples': len([r for r in results if r['index'] in gold_labels]),
            'agreement_metrics': agreement_metrics,
            'disagreement_count': len(disagreement_cases),
            'disagreement_rate': len(disagreement_cases) / len(results) if results else 0.0,
            'dice_scores_summary': {
                'mean': np.mean([r['dice_score'] for r in results]) if results else 0.0,
                'std': np.std([r['dice_score'] for r in results]) if results else 0.0,
                'min': min([r['dice_score'] for r in results]) if results else 0.0,
                'max': max([r['dice_score'] for r in results]) if results else 0.0
            }
        }
        
        return report


class UnifiedValidationEvaluator:
    """统一验证评估器 - 支持DICE和RAGAS两种评估方法"""
    
    def __init__(self, evaluation_method: str = "dice", dice_config: SimplifiedDICEConfig = None, 
                 ragas_config: RagasConfig = None, tournament_result_file: str = None):
        self.evaluation_method = evaluation_method.lower()
        self.logger = logging.getLogger("UnifiedValidation")
        
        # 根据评估方法初始化相应的评估器
        if self.evaluation_method == "dice":
            if dice_config is None:
                raise ValueError("使用DICE方法时必须提供dice_config")
            self.evaluator = DICEValidationEvaluator(dice_config, tournament_result_file)
        elif self.evaluation_method == "ragas":
            if ragas_config is None:
                raise ValueError("使用RAGAS方法时必须提供ragas_config")
            self.evaluator = RagasValidationEvaluator(ragas_config)
        else:
            raise ValueError(f"不支持的评估方法: {evaluation_method}")
        
        # 设置日志
        self._setup_logger()
        
        self.logger.info(f"初始化统一验证评估器，使用方法: {self.evaluation_method.upper()}")
    
    def _setup_logger(self):
        """设置日志"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _derive_dice_label(self, result: Dict[str, Any]) -> str:
        """统一推断DICE标签的逻辑，避免分值尺度误判导致统计错误。"""
        explicit_label = result.get("dice_judgment")
        if explicit_label in {"A wins", "B wins", "Tie"}:
            return explicit_label
        
        score = result.get("dice_score")
        if isinstance(score, (int, float)):
            # 若是[0,1]分尺度，则以0.5为中性阈值，加入轻微缓冲
            if 0.0 <= score <= 1.0:
                if score > 0.55:
                    return "A wins"
                if score < 0.45:
                    return "B wins"
                return "Tie"
            # 否则视为对称分制（如[-1,1]），以0为中性阈值，加入轻微缓冲
            if score > 0.1:
                return "A wins"
            if score < -0.1:
                return "B wins"
            return "Tie"
        
        # 回退：若有prob_a/prob_b可比较
        prob_a = result.get("prob_a")
        prob_b = result.get("prob_b")
        if isinstance(prob_a, (int, float)) and isinstance(prob_b, (int, float)):
            delta = prob_a - prob_b
            if delta > 0.05:
                return "A wins"
            if delta < -0.05:
                return "B wins"
            return "Tie"
        
        return "Tie"
        
    def sample_evaluation_pairs(self, qacg_files: List[str], num_samples: int = 200, 
                               random_seed: int = 42) -> List[Dict[str, Any]]:
        """
        从70题中随机抽取200对(q, cA, aA, cB, aB)用于人工标注
        
        Args:
            qacg_files: QACG文件路径列表
            num_samples: 采样数量
            random_seed: 随机种子
            
        Returns:
            采样的评估对列表
        """
        self.logger.info(f"开始采样 {num_samples} 对评估样本")
        random.seed(random_seed)
        
        # 加载所有系统数据
        all_systems_data = {}
        for file_path in qacg_files:
            system_name = Path(file_path).stem.replace("qacg_", "")
            with open(file_path, 'r', encoding='utf-8') as f:
                all_systems_data[system_name] = json.load(f)
        
        systems = list(all_systems_data.keys())
        if len(systems) < 2:
            raise ValueError(f"需要至少2个系统，实际获得{len(systems)}个")
        
        self.logger.info(f"加载了 {len(systems)} 个系统: {systems}")
        
        # 确定数据长度（使用最短的系统数据长度）
        min_length = min(len(data) for data in all_systems_data.values())
        self.logger.info(f"每个系统有 {min_length} 题数据")
        
        # 生成所有可能的系统对和题目组合
        all_combinations = []
        for i, system_a in enumerate(systems):
            for j, system_b in enumerate(systems):
                if i < j:  # 避免重复对比
                    for q_idx in range(min_length):
                        qa_a = all_systems_data[system_a][q_idx]
                        qa_b = all_systems_data[system_b][q_idx]
                        
                        # 确保两个系统回答的是同一个问题
                        if qa_a["question"] == qa_b["question"]:
                            combination = {
                                "question_idx": q_idx,
                                "system_a": system_a,
                                "system_b": system_b,
                                "qa_a": qa_a,
                                "qa_b": qa_b,
                                "question": qa_a["question"],
                                "answer_a": qa_a.get("rag_answer", ""),
                                "answer_b": qa_b.get("rag_answer", ""),
                                "expected_answer": qa_a.get("expected_answer", ""),
                                "context_a": qa_a.get("context", []),
                                "context_b": qa_b.get("context", []),
                                "groundtruth": qa_a.get("groundtruth", qa_a.get("expected_answer", ""))
                            }
                            all_combinations.append(combination)
        
        self.logger.info(f"总共有 {len(all_combinations)} 个可能的组合")
        
        # 随机采样
        if len(all_combinations) < num_samples:
            self.logger.warning(f"可用组合数 ({len(all_combinations)}) 少于需求样本数 ({num_samples})")
            sampled_pairs = all_combinations
        else:
            sampled_pairs = random.sample(all_combinations, num_samples)
        
        self.logger.info(f"成功采样 {len(sampled_pairs)} 对评估样本")
        return sampled_pairs
    
    def run_evaluation(self, evaluation_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """运行相应的评估方法"""
        if self.evaluation_method == "dice":
            return self.run_dice_evaluation(evaluation_pairs)
        elif self.evaluation_method == "ragas":
            return self.evaluator.run_ragas_evaluation(evaluation_pairs)
    
    def load_human_annotations(self, annotation_file: str) -> Dict[int, str]:
        """代理到具体评估器的标注加载方法"""
        return self.evaluator.load_human_annotations(annotation_file)
    
    def calculate_agreement(self, results: List[Dict[str, Any]], 
                          gold_labels: Dict[int, str]) -> Dict[str, float]:
        """代理到具体评估器的一致性计算方法"""
        return self.evaluator.calculate_agreement(results, gold_labels)
    
    def calculate_elo_correlation(self, results: List[Dict[str, Any]], 
                                gold_labels: Dict[int, str]) -> Dict[str, float]:
        """代理到具体评估器的Elo相关性计算方法"""
        return self.evaluator.calculate_elo_correlation(results, gold_labels)
    
    def analyze_disagreement_cases(self, results: List[Dict[str, Any]], 
                                  gold_labels: Dict[int, str]) -> List[Dict[str, Any]]:
        """代理到具体评估器的分歧分析方法"""
        return self.evaluator.analyze_disagreement_cases(results, gold_labels)
    
    def print_disagreement_analysis(self, disagreement_cases: List[Dict[str, Any]]):
        """代理到具体评估器的分歧打印方法"""
        return self.evaluator.print_disagreement_analysis(disagreement_cases)
    
    def generate_validation_report(self, agreement_metrics: Dict[str, Any], 
                                 correlation_metrics: Dict[str, Any],
                                 results: List[Dict[str, Any]],
                                 gold_labels: Dict[int, str],
                                 output_file: str):
        """代理到具体评估器的报告生成方法"""
        return self.evaluator.generate_validation_report(
            agreement_metrics, correlation_metrics, results, gold_labels, output_file
        )
    
    def run_dice_evaluation(self, evaluation_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        使用DICE评估所有采样的对比对
        
        Args:
            evaluation_pairs: 评估对列表
            
        Returns:
            DICE评估结果列表
        """
        self.logger.info(f"开始DICE评估 {len(evaluation_pairs)} 对样本")
        
        dice_results = []
        for i, pair in enumerate(evaluation_pairs):
            self.logger.info(f"评估第 {i+1}/{len(evaluation_pairs)} 对")
            
            # 使用DICE进行评估
            qa_a = pair["qa_a"]
            qa_b = pair["qa_b"]
            
            # 使用DICE评估器的_pairwise_comparison方法
            result = self.evaluator.dice_evaluator._pairwise_comparison(
                [qa_a], [qa_b], 
                pair["system_a"], pair["system_b"],
                max_questions=1
            )
            
            # 提取关键信息
            if result["question_results"]:
                question_result = result["question_results"][0]
                passage_judgment = question_result.get("passage_judgment", {})
                
                dice_result = {
                    "pair_id": i,  # 使用索引作为pair_id，与标注模板保持一致
                    "question": pair["question"],
                    "system_a": pair["system_a"],
                    "system_b": pair["system_b"],
                    "dice_judgment": passage_judgment.get("label", "Tie"),
                    "dice_score": passage_judgment.get("score", 0.5),
                    "dice_reason": passage_judgment.get("reason", ""),
                    "dice_margin_score": passage_judgment.get("margin_score", 0.0),
                    "combined_delta": question_result.get("elo_delta", 0.0),
                    "original_pair": pair
                }
            else:
                # 备用结果
                dice_result = {
                    "pair_id": i,  # 使用索引作为pair_id，与标注模板保持一致
                    "question": pair["question"],
                    "system_a": pair["system_a"],
                    "system_b": pair["system_b"],
                    "dice_judgment": "Tie",
                    "dice_score": 0.5,
                    "dice_reason": "评估失败",
                    "dice_margin_score": 0.0,
                    "combined_delta": 0.0,
                    "original_pair": pair
                }
            
            dice_results.append(dice_result)
        
        return dice_results
    
    def load_human_annotations(self, annotation_file: str) -> Dict[int, str]:
        """
        加载人工标注结果
        
        Args:
            annotation_file: 人工标注文件路径
            
        Returns:
            Dict[pair_id, gold_label]: 金标准标注
        """
        self.logger.info(f"加载人工标注: {annotation_file}")
        
        if not Path(annotation_file).exists():
            self.logger.error(f"标注文件不存在: {annotation_file}")
            # 创建示例标注文件
            self._create_annotation_template(annotation_file)
            raise FileNotFoundError(f"请完成人工标注后重新运行: {annotation_file}")
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)
        
        # 获取实际的标注数组
        if isinstance(annotation_data, dict) and "annotations" in annotation_data:
            annotations = annotation_data["annotations"]
        else:
            annotations = annotation_data
        
        # 转换为简单的dict格式
        gold_labels = {}
        for item in annotations:
            pair_id = item["pair_id"]
            # 使用多数票决定金标准
            votes = item["expert_votes"]
            
            # 检查是否所有投票都为空
            valid_votes = [vote for vote in votes if vote and vote.strip()]
            if not valid_votes:
                self.logger.warning(f"pair_id {pair_id} 的expert_votes为空，跳过此项")
                continue
            
            # 检查投票是否有效
            valid_labels = {"A wins", "B wins", "Tie"}
            filtered_votes = [vote for vote in valid_votes if vote in valid_labels]
            if not filtered_votes:
                self.logger.warning(f"pair_id {pair_id} 没有有效的投票标签，跳过此项")
                continue
            
            vote_counts = defaultdict(int)
            for vote in filtered_votes:
                vote_counts[vote] += 1
            gold_label = max(vote_counts.items(), key=lambda x: x[1])[0]
            gold_labels[pair_id] = gold_label
        
        self.logger.info(f"加载了 {len(gold_labels)} 个金标准标注")
        
        if len(gold_labels) == 0:
            raise ValueError("没有找到任何有效的标注数据。请确保：\n"
                           "1. expert_votes不为空\n"
                           "2. 投票值为 'A wins'、'B wins' 或 'Tie'\n"
                           "3. 至少有一位专家完成了标注")
        
        return gold_labels
    
    def _create_annotation_template(self, annotation_file: str):
        """创建人工标注模板文件"""
        self.logger.info(f"创建标注模板: {annotation_file}")
        
        template = {
            "instructions": "请3位专家独立完成标注，每个pair_id对应一个评估对，请为每位专家在expert_votes中填入 'A wins'、'B wins' 或 'Tie'",
            "annotation_guide": {
                "A wins": "系统A明显优于系统B",
                "B wins": "系统B明显优于系统A", 
                "Tie": "两个系统表现相当，难以区分优劣"
            },
            "annotations": [
                {
                    "pair_id": 0,
                    "question": "示例问题",
                    "system_a": "system_a_name",
                    "answer_a": "系统A的回答",
                    "system_b": "system_b_name", 
                    "answer_b": "系统B的回答",
                    "expert_votes": ["A wins", "B wins", "A wins"]  # 3位专家的投票
                }
            ]
        }
        
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
    
    def calculate_agreement(self, dice_results: List[Dict[str, Any]], 
                          gold_labels: Dict[int, str]) -> Dict[str, float]:
        """
        计算DICE与金标准的一致性
        
        Args:
            dice_results: DICE评估结果
            gold_labels: 金标准标注
            
        Returns:
            一致性指标字典
        """
        self.logger.info("计算一致性指标")
        
        # 准备数据
        dice_labels = []
        human_labels = []
        
        for result in dice_results:
            # 使用pair_id字段
            result_index = result.get("pair_id", result.get("index", -1))
            if result_index in gold_labels:
                # 统一从结果中推断标签（兼容不同分数尺度与结构）
                dice_label = self._derive_dice_label(result)
                dice_labels.append(dice_label)
                human_labels.append(gold_labels[result_index])
        
        if len(dice_labels) == 0:
            raise ValueError("没有找到匹配的标注数据")
        
        self.logger.info(f"匹配到 {len(dice_labels)} 个标注对")
        
        # 计算κ值
        kappa = cohen_kappa_score(human_labels, dice_labels)
        
        # 计算准确率
        accuracy = sum(1 for d, h in zip(dice_labels, human_labels) if d == h) / len(dice_labels)
        
        # 分类统计
        label_stats = {}
        for label in ["A wins", "B wins", "Tie"]:
            human_count = human_labels.count(label)
            dice_count = dice_labels.count(label)
            label_stats[label] = {
                "human_count": human_count,
                "dice_count": dice_count,
                "agreement": sum(1 for d, h in zip(dice_labels, human_labels) 
                              if d == h == label) if human_count > 0 else 0
            }
        
        return {
            "kappa": kappa,
            "accuracy": accuracy,
            "total_pairs": len(dice_labels),
            "label_statistics": label_stats
        }
    
    def calculate_elo_correlation(self, dice_results: List[Dict[str, Any]], 
                                gold_labels: Dict[int, str]) -> Dict[str, float]:
        """
        计算DICE-Elo与人工-Elo排序的相关性
        
        Args:
            dice_results: DICE评估结果
            gold_labels: 金标准标注
            
        Returns:
            相关性指标字典
        """
        self.logger.info("计算Elo排序相关性")
        
        # 收集所有系统
        all_systems = set()
        for result in dice_results:
            all_systems.add(result["system_a"])
            all_systems.add(result["system_b"])
        all_systems = list(all_systems)
        
        # 计算DICE-Elo分数
        dice_elo = {system: 1500.0 for system in all_systems}  # 初始Elo
        human_elo = {system: 1500.0 for system in all_systems}
        
        k_factor = 32
        
        for result in dice_results:
            # 兼容两种数据结构：使用pair_id或index字段
            pair_id = result.get("pair_id", result.get("index", -1))
            if pair_id not in gold_labels:
                continue
                
            system_a = result["system_a"]
            system_b = result["system_b"]
            
            # DICE结果 - 统一推断方式
            dice_judgment = self._derive_dice_label(result)
            
            if dice_judgment == "A wins":
                dice_score_a, dice_score_b = 1.0, 0.0
            elif dice_judgment == "B wins":
                dice_score_a, dice_score_b = 0.0, 1.0
            else:
                dice_score_a, dice_score_b = 0.5, 0.5
            
            # 人工标注结果
            human_label = gold_labels[pair_id]
            if human_label == "A wins":
                human_score_a, human_score_b = 1.0, 0.0
            elif human_label == "B wins":
                human_score_a, human_score_b = 0.0, 1.0
            else:
                human_score_a, human_score_b = 0.5, 0.5
            
            # 更新DICE-Elo
            expected_a = 1 / (1 + 10**((dice_elo[system_b] - dice_elo[system_a]) / 400))
            dice_elo[system_a] += k_factor * (dice_score_a - expected_a)
            dice_elo[system_b] += k_factor * (dice_score_b - (1 - expected_a))
            
            # 更新Human-Elo
            expected_a = 1 / (1 + 10**((human_elo[system_b] - human_elo[system_a]) / 400))
            human_elo[system_a] += k_factor * (human_score_a - expected_a)
            human_elo[system_b] += k_factor * (human_score_b - (1 - expected_a))
        
        # 计算排序
        dice_ranking = sorted(all_systems, key=lambda x: dice_elo[x], reverse=True)
        human_ranking = sorted(all_systems, key=lambda x: human_elo[x], reverse=True)
        
        # 计算Kendall-τ相关性
        dice_ranks = [dice_ranking.index(system) for system in all_systems]
        human_ranks = [human_ranking.index(system) for system in all_systems]
        
        tau, p_value = kendalltau(dice_ranks, human_ranks)
        
        return {
            "kendall_tau": tau,
            "p_value": p_value,
            "dice_elo_scores": dice_elo,
            "human_elo_scores": human_elo,
            "dice_ranking": dice_ranking,
            "human_ranking": human_ranking
        }
    
    def analyze_disagreement_cases(self, dice_results: List[Dict[str, Any]], 
                                  gold_labels: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        分析DICE判断与人工标注不一致的case
        
        Args:
            dice_results: DICE评估结果
            gold_labels: 金标准标注
            
        Returns:
            不一致的case列表
        """
        self.logger.info("分析不一致的case")
        
        disagreement_cases = []
        
        for result in dice_results:
            # 兼容两种数据结构：使用pair_id或index字段
            pair_id = result.get("pair_id", result.get("index", -1))
            if pair_id in gold_labels:
                # 统一推断DICE判决
                dice_judgment = self._derive_dice_label(result)
                
                human_judgment = gold_labels[pair_id]
                
                if dice_judgment != human_judgment:
                    # 兼容两种数据结构：使用original_pair或直接从result获取
                    pair_data = result.get("original_pair", result)
                    case = {
                        "pair_id": pair_id,
                        "question": pair_data.get("question", result.get("question", "")),
                        "system_a": result["system_a"],
                        "system_b": result["system_b"],
                        "answer_a": pair_data.get("answer_a", result.get("answer_a", "")),
                        "answer_b": pair_data.get("answer_b", result.get("answer_b", "")),
                        "context_a": pair_data.get("context_a", result.get("context_a", []))[:2],  # 只显示前2个context
                        "context_b": pair_data.get("context_b", result.get("context_b", []))[:2],
                        "groundtruth": pair_data.get("groundtruth", result.get("groundtruth", "")),
                        "dice_judgment": dice_judgment,
                        "dice_score": result.get("dice_score", 0.0),
                        "dice_reason": result.get("dice_reason", result.get("dice_explanation", "")),
                        "human_judgment": human_judgment,
                        "disagreement_type": f"DICE: {dice_judgment} vs Human: {human_judgment}"
                    }
                    disagreement_cases.append(case)
        
        self.logger.info(f"发现 {len(disagreement_cases)} 个不一致的case")
        return disagreement_cases
    
    def print_disagreement_analysis(self, disagreement_cases: List[Dict[str, Any]]):
        """
        打印不一致case的详细分析
        """
        if not disagreement_cases:
            print("\n✅ 所有case都一致，没有发现分歧")
            return
        
        print(f"\n🔍 发现 {len(disagreement_cases)} 个不一致的case:")
        print("="*80)
        
        for i, case in enumerate(disagreement_cases[:10]):  # 只显示前10个
            print(f"\n📋 Case {case['pair_id']+1} (pair_id: {case['pair_id']})")
            print(f"🔥 分歧类型: {case['disagreement_type']}")
            print(f"❓ 问题: {case['question']}")
            print()
            
            print(f"🤖 系统A ({case['system_a']}):")
            print(f"   回答: {case['answer_a'][:300]}{'...' if len(case['answer_a']) > 200 else ''}")
            print()
            
            print(f"🤖 系统B ({case['system_b']}):")
            print(f"   回答: {case['answer_b'][:300]}{'...' if len(case['answer_b']) > 200 else ''}")
            print()
            
            print(f"📝 标准答案: {case['groundtruth'][:200]}{'...' if len(case['groundtruth']) > 200 else ''}")
            print()
            
            print(f"🎯 DICE判断: {case['dice_judgment']} (置信度: {case['dice_score']:.3f})")
            print(f"   理由: {case['dice_reason'][:500]}{'...' if len(case['dice_reason']) > 500 else ''}")
            print()
            
            print(f"👥 人工判断: {case['human_judgment']}")
            print("-" * 80)
        
        if len(disagreement_cases) > 10:
            print(f"\n... 还有 {len(disagreement_cases) - 10} 个不一致的case未显示")
        
        # 统计不同类型的分歧
        disagreement_stats = {}
        for case in disagreement_cases:
            disagreement_type = case['disagreement_type']
            disagreement_stats[disagreement_type] = disagreement_stats.get(disagreement_type, 0) + 1
        
        print(f"\n📊 分歧类型统计:")
        for disagreement_type, count in sorted(disagreement_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(disagreement_cases)) * 100
            print(f"   {disagreement_type}: {count} 个 ({percentage:.1f}%)")

    def generate_validation_report(self, agreement_metrics: Dict[str, Any], 
                                 correlation_metrics: Dict[str, Any],
                                 dice_results: List[Dict[str, Any]],
                                 gold_labels: Dict[int, str],
                                 output_file: str):
        """
        生成验证报告
        
        Args:
            agreement_metrics: 一致性指标
            correlation_metrics: 相关性指标  
            dice_results: DICE评估结果
            gold_labels: 金标准标注
            output_file: 输出文件路径
        """
        self.logger.info(f"生成验证报告: {output_file}")
        
        # 分析不一致的case
        disagreement_cases = self.analyze_disagreement_cases(dice_results, gold_labels)
        
        report = {
            "validation_summary": {
                "kappa_score": agreement_metrics["kappa"],
                "accuracy": agreement_metrics["accuracy"],
                "kendall_tau": correlation_metrics["kendall_tau"],
                "validation_passed": (
                    agreement_metrics["kappa"] >= 0.85 and 
                    correlation_metrics["kendall_tau"] >= 0.9
                )
            },
            "detailed_metrics": {
                "agreement_analysis": agreement_metrics,
                "correlation_analysis": correlation_metrics
            },
            "disagreement_analysis": {
                "total_disagreements": len(disagreement_cases),
                "disagreement_rate": len(disagreement_cases) / len(dice_results) if dice_results else 0,
                "sample_cases": disagreement_cases
            },
            "conclusion": self._generate_conclusion(agreement_metrics, correlation_metrics)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印摘要
        self._print_validation_summary(report)
        
        # 打印不一致case分析
        self.print_disagreement_analysis(disagreement_cases)
    
    def _generate_conclusion(self, agreement_metrics: Dict[str, Any], 
                           correlation_metrics: Dict[str, Any]) -> str:
        """生成结论"""
        kappa = agreement_metrics["kappa"]
        tau = correlation_metrics["kendall_tau"]
        
        # 检查是否为2系统的特殊情况
        num_systems = len(correlation_metrics.get("dice_ranking", []))
        if num_systems == 2:
            if tau == -1.0:
                conclusion = "📊 2系统验证：DICE与人工排序完全相反（τ=-1.0）。"
                if kappa >= 0.6:
                    conclusion += f"但κ值({kappa:.3f})表明总体一致性尚可，可能存在系统偏好差异。"
                else:
                    conclusion += f"且κ值({kappa:.3f})较低，建议检查判决逻辑或增加更多系统进行验证。"
                return conclusion
            elif tau == 1.0:
                return f"✅ 2系统验证：DICE与人工排序完全一致（τ=1.0），κ值={kappa:.3f}。"
        
        # 标准的多系统评估
        if kappa >= 0.85 and tau >= 0.9:
            return "✅ DICE系统验证通过！κ值和Kendall-τ均达标，系统可信度高，可用于后续评估。"
        elif kappa >= 0.85:
            return "⚠️ DICE系统部分通过。κ值达标但排序相关性不足，建议检查Elo计算逻辑。"
        elif tau >= 0.9:
            return "⚠️ DICE系统部分通过。排序相关性达标但一致性不足，建议检查判决逻辑。"
        else:
            return "❌ DICE系统验证失败。κ值和Kendall-τ均未达标，需要重新调整评估策略。"
    
    def _print_validation_summary(self, report: Dict[str, Any]):
        """打印验证摘要"""
        summary = report["validation_summary"]
        
        print("\n" + "="*60)
        print("🔬 DICE系统验证结果")
        print("="*60)
        print(f"κ 值 (目标≥0.85): {summary['kappa_score']:.3f}")
        print(f"准确率: {summary['accuracy']:.3f}")
        print(f"Kendall-τ (目标≥0.9): {summary['kendall_tau']:.3f}")
        print(f"验证状态: {'✅ 通过' if summary['validation_passed'] else '❌ 未通过'}")
        print("\n" + report["conclusion"])
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="多RAG系统准确率验证评估")
    parser.add_argument("--qacg_files", nargs="+", required=True,
                       help="QACG文件路径列表")
    parser.add_argument("--num_samples", type=int, default=200,
                       help="采样评估对数量")
    parser.add_argument("--annotation_file", type=str, 
                       default="dice_human_annotations.json",
                       help="人工标注文件路径")
    parser.add_argument("--output_dir", type=str, default="dice_validation_output",
                       help="输出目录")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--llm_model", type=str, default="deepseek-chat",
                       help="LLM模型")
    parser.add_argument("--tournament_result_file", type=str, 
                       default="dice_simplified_output/tournament_result.json",
                       help="tournament结果文件路径，用于复用已有判断")
    parser.add_argument("--ragas", action="store_true",
                       help="使用RAGAS方法进行评估（默认使用DICE方法）")
    parser.add_argument("--ragas_metrics", nargs="+", 
                       default=["answer_relevancy", "context_precision", "context_recall", "faithfulness", "answer_correctness"],
                       help="RAGAS评估指标列表")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 根据评估方法创建配置和评估器
    import os
    
    if args.ragas:
        # RAGAS配置 - 使用DeepSeek
        ragas_config = RagasConfig(
            llm_model=args.llm_model,
            metrics=args.ragas_metrics,
            api_key=os.environ.get("DEEPSEEK_API_KEY", "xxxxxxx"),  # 使用DeepSeek API
            base_url="https://api.deepseek.com"
        )
        evaluator = UnifiedValidationEvaluator(
            evaluation_method="ragas",
            ragas_config=ragas_config
        )
        evaluation_method = "RAGAS"
    else:
        # DICE配置
        dice_config = SimplifiedDICEConfig(
            llm_model=args.llm_model,
            output_dir=str(output_dir),
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com"
        )
        evaluator = UnifiedValidationEvaluator(
            evaluation_method="dice",
            dice_config=dice_config,
            tournament_result_file=args.tournament_result_file
        )
        evaluation_method = "DICE"
    
    print(f"🔬 {evaluation_method}系统验证评估")
    print(f"📁 QACG文件数量: {len(args.qacg_files)}")
    print(f"📊 采样数量: {args.num_samples}")
    print(f"🔧 评估方法: {evaluation_method}")
    
    try:
        # 步骤1: 采样评估对
        print("\n📋 步骤1: 采样评估对...")
        evaluation_pairs = evaluator.sample_evaluation_pairs(
            args.qacg_files, args.num_samples, args.random_seed
        )
        
        # 保存采样结果
        pairs_file = output_dir / "evaluation_pairs.json"
        with open(pairs_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_pairs, f, ensure_ascii=False, indent=2)
        print(f"✅ 采样完成，保存至: {pairs_file}")
        
        # 步骤1.5: 检查或创建人工标注文件
        print(f"\n📝 步骤1.5: 检查人工标注文件: {args.annotation_file}")
        annotation_file_path = Path(args.annotation_file)
        
        if not annotation_file_path.exists():
            print("⚠️  人工标注文件不存在，创建标注模板...")
            
            # 创建标注模板
            annotation_data = []
            for i, pair in enumerate(evaluation_pairs):
                annotation_item = {
                    "pair_id": i,
                    "question": pair["question"],
                    "system_a": pair["system_a"],
                    "answer_a": pair["answer_a"],
                    "context_a": pair["context_a"][:3],  # 只显示前3个context
                    "system_b": pair["system_b"],
                    "answer_b": pair["answer_b"],
                    "context_b": pair["context_b"][:3],
                    "groundtruth": pair["groundtruth"],
                    "expert_votes": ["", "", ""]  # 3位专家填入：A wins/B wins/Tie
                }
                annotation_data.append(annotation_item)
            
            template = {
                "instructions": "请3位专家独立完成标注。对于每个pair_id，请为每位专家在expert_votes中填入 'A wins'、'B wins' 或 'Tie'",
                "annotation_guide": {
                    "A wins": "系统A的检索质量和回答质量明显优于系统B",
                    "B wins": "系统B的检索质量和回答质量明显优于系统A", 
                    "Tie": "两个系统表现相当，难以区分优劣"
                },
                "evaluation_criteria": [
                    "1. 检索证据的相关性和完整性",
                    "2. 回答的准确性和逻辑性", 
                    "3. 证据与回答的一致性",
                    "4. 与标准答案的符合程度"
                ],
                "annotations": annotation_data
            }
            
            with open(args.annotation_file, 'w', encoding='utf-8') as f:
                json.dump(template, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 已创建标注模板: {args.annotation_file}")
            print("💡 标注说明:")
            print("   - 每个pair_id需要3位专家独立投票")
            print("   - 投票选项: 'A wins'、'B wins'、'Tie'")
            print("   - 请根据检索质量和回答质量进行判断")
            print("⚠️  如需生成验证报告，请先完成标注后再运行")
            print("✅ 程序将继续执行DICE评估...\n")
        else:
            print(f"✅ 人工标注文件已存在: {args.annotation_file}")
        
        # 步骤2: 检查或运行DICE评估
        results_file = output_dir / f"{evaluation_method.lower()}_results.json"
        evaluation_results = None
        
        print(f"\n🤖 步骤2: 检查{evaluation_method}评估结果文件...")
        if results_file.exists():
            print(f"✅ 发现已有评估结果文件: {results_file}")
            print("📂 加载已有评估结果，跳过重新评估...")
            
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    evaluation_results = json.load(f)
                print(f"✅ 成功加载 {len(evaluation_results)} 个评估结果")
                
                # 验证评估结果是否与当前采样对匹配
                if len(evaluation_results) != len(evaluation_pairs):
                    print(f"⚠️  评估结果数量({len(evaluation_results)})与采样对数量({len(evaluation_pairs)})不匹配")
                    print("🔄 将重新运行评估...")
                    evaluation_results = None
                else:
                    print("✅ 评估结果数量匹配，将使用已有结果")
            except Exception as e:
                print(f"❌ 加载评估结果失败: {e}")
                print("🔄 将重新运行评估...")
                evaluation_results = None
        
        # 如果没有加载到有效的评估结果，则运行评估
        if evaluation_results is None:
            print(f"\n🤖 运行{evaluation_method}系统评估...")
            evaluation_results = evaluator.run_evaluation(evaluation_pairs)
            
            # 保存评估结果
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            print(f"✅ {evaluation_method}评估完成，保存至: {results_file}")
        
        # 步骤3: 尝试加载人工标注并生成报告
        print(f"\n📊 步骤3: 检查人工标注完成情况...")
        
        try:
            # 尝试加载人工标注
            gold_labels = evaluator.load_human_annotations(args.annotation_file)
            
            if len(gold_labels) == 0:
                print("⚠️  人工标注文件存在但没有有效标注")
                print("💡 请完成标注后重新运行以生成验证报告")
                print(f"✅ DICE评估结果已保存至: {results_file}")
                return
            
            print(f"✅ 成功加载 {len(gold_labels)} 个人工标注")
            
            # 计算一致性指标
            print("\n📊 步骤4: 计算一致性指标...")
            agreement_metrics = evaluator.calculate_agreement(evaluation_results, gold_labels)
            
            # 计算Elo相关性
            print("📊 步骤5: 计算Elo排序相关性...")
            correlation_metrics = evaluator.calculate_elo_correlation(evaluation_results, gold_labels)
            
            # 生成报告
            print("📝 步骤6: 生成验证报告...")
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            report_file = output_dir / f"validation_report_{timestamp}.json"
            evaluator.generate_validation_report(
                agreement_metrics, correlation_metrics, evaluation_results, gold_labels, str(report_file)
            )
            
            print(f"\n✅ 验证报告已保存至: {report_file}")
            
        except Exception as e:
            print(f"⚠️  无法生成验证报告: {e}")
            print("💡 这可能是因为:")
            print("   1. 人工标注文件尚未完成标注")
            print("   2. 标注格式不正确")
            print("   3. expert_votes字段为空")
            print(f"\n✅ DICE评估已完成，结果已保存至: {results_file}")
            print("📝 完成人工标注后，可重新运行脚本生成验证报告")
        
    except Exception as e:
        print(f"❌ 评估过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()