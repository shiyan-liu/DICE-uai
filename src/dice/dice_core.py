#!/usr/bin/env python3
"""
DICE核心模块
实现检索-证据双通道离散化评估框架
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
from .granularity import GranularityBuilder
from .local_pairwise_judge import LocalPairwiseJudge as PairwiseJudge
from .elo_fusion import EloFusion

class NumpyJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy和其他特殊类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

@dataclass
class DICEConfig:
    """DICE配置类"""
    # 基础配置
    llm_model: str = "qwen2.5:7b"  # 用于判决的LLM模型
    judge_temperature: float = 0.1  # 判决温度
    
    # 四粒度配置
    enable_token: bool = True     # 启用token粒度
    enable_sentence: bool = True  # 启用sentence粒度  
    enable_passage: bool = True   # 启用passage粒度
    enable_kg: bool = True        # 启用KG粒度
    
    # Tie处理配置
    tie_threshold: float = 0.05   # Tie边界阈值
    margin_temperature: float = 0.1  # Margin-Aware Tie温度
    
    # Elo配置
    initial_elo: float = 1000.0   # 初始Elo分数
    k_factor: int = 32           # Elo K因子
    dynamic_k: bool = True       # 是否启用动态K
    
    # 统计配置
    bootstrap_samples: int = 1000  # Bootstrap重采样次数
    confidence_level: float = 0.95 # 置信水平
    
    # 输出配置
    output_dir: str = "dice_output"
    save_intermediate: bool = True  # 保存中间结果
    detailed_output: bool = False  # 详细输出选项

class DICEEvaluator:
    """DICE评估器核心类"""
    
    def __init__(self, config: DICEConfig):
        self.config = config
        self.logger = self._setup_logger()
        
        # 初始化各个模块
        self.granularity_builder = GranularityBuilder(config)
        self.pairwise_judge = PairwiseJudge(config)
        self.elo_fusion = EloFusion(config)
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("DICE评估器初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger("DICE")
        logger.setLevel(logging.INFO)
        
        # 不添加额外的handler，使用全局配置
        # 避免重复日志输出
        return logger
    
    def load_qacg_data(self, file_path: str) -> List[Dict[str, Any]]:
        """加载QACG生成的数据"""
        self.logger.info(f"加载数据文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"加载了 {len(data)} 条数据")
        return data
    
    def evaluate_pair(self, qa_a: Dict[str, Any], qa_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估一对QA系统的表现
        
        Args:
            qa_a: 系统A的QA数据
            qa_b: 系统B的QA数据
            
        Returns:
            评估结果字典
        """
        question = qa_a["question"]
        self.logger.info(f"评估问题: {question[:50]}...")
        
        # 步骤1: 四粒度原子构建
        groundtruth = qa_a.get("groundtruth", qa_a.get("expected_answer", ""))  # 获取标准答案
        
        # 确保groundtruth是字符串类型，如果是列表则取第一个元素或转换为字符串
        if isinstance(groundtruth, list):
            groundtruth = groundtruth[0] if groundtruth else ""
        elif not isinstance(groundtruth, str):
            groundtruth = str(groundtruth) if groundtruth else ""
            
        granularities = self.granularity_builder.build_granularities(
            question, qa_a, qa_b, groundtruth
        )
        
        # 步骤2-3: Pairwise判决 + Margin-Aware Tie
        judgments = {}
        for granularity, atoms in granularities.items():
            if self._is_granularity_enabled(granularity):
                judgment = self.pairwise_judge.judge_pair(
                    question, qa_a, qa_b, granularity, atoms
                )
                judgments[granularity] = judgment
        
        # 步骤4-5: 四粒度熵权融合
        fusion_result = self.elo_fusion.fuse_judgments(judgments)
        
        # 步骤6: 统计输出
        stats = self.elo_fusion.compute_statistics(fusion_result)
        
        result = {
            "question": question,
            "system_a": qa_a.get("metadata", {}).get("system_name", "system_a"),
            "system_b": qa_b.get("metadata", {}).get("system_name", "system_b"),
            "granularity_results": judgments,
            "fusion_result": fusion_result,
            "statistics": stats,
            "combined_delta": fusion_result.get("elo_delta", 0),
            "ci_95": stats.get("ci_95", [0, 0]),
            "ece": stats.get("ece", 0.0)
        }
        
        # 如果启用详细输出，添加QACG四元组和四个维度的A、B值
        if self.config.detailed_output:
            result["detailed_info"] = {
                "qacg_quadruple_a": {
                    "question": question,
                    "answer": qa_a.get("rag_answer", ""),
                    "context": qa_a.get("context", []),
                    "groundtruth": qa_a.get("groundtruth", qa_a.get("expected_answer", ""))
                },
                "qacg_quadruple_b": {
                    "question": question,
                    "answer": qa_b.get("rag_answer", ""),
                    "context": qa_b.get("context", []),
                    "groundtruth": qa_b.get("groundtruth", qa_b.get("expected_answer", ""))
                },
                "granularity_atoms": granularities,
                "four_dimensions": {
                    "token_dimension": {
                        "a_values": granularities.get("token", {}).get("tokens_a", []),
                        "b_values": granularities.get("token", {}).get("tokens_b", [])
                    },
                    "sentence_dimension": {
                        "a_values": granularities.get("sentence", {}).get("sentences_a", []),
                        "b_values": granularities.get("sentence", {}).get("sentences_b", [])
                    },
                    "passage_dimension": {
                        "a_values": granularities.get("passage", {}).get("passages_a", []),
                        "b_values": granularities.get("passage", {}).get("passages_b", [])
                    },
                    "kg_dimension": {
                        "a_values": granularities.get("kg", {}).get("triples_a", []),
                        "b_values": granularities.get("kg", {}).get("triples_b", [])
                    }
                }
            }
        
        return result
    
    def evaluate_all_pairs(self, data_files: List[str]) -> Dict[str, Any]:
        """
        评估所有系统对的表现
        
        Args:
            data_files: 数据文件路径列表
            
        Returns:
            全局评估结果
        """
        self.logger.info(f"开始评估 {len(data_files)} 个系统")
        
        # 加载所有数据
        all_data = {}
        for file_path in data_files:
            system_name = Path(file_path).stem  # 从文件名提取系统名
            all_data[system_name] = self.load_qacg_data(file_path)
        
        systems = list(all_data.keys())
        self.logger.info(f"系统列表: {systems}")
        
        # 执行成对比较
        pair_results = []
        for i, system_a in enumerate(systems):
            for j, system_b in enumerate(systems):
                if i < j:  # 避免重复比较
                    self.logger.info(f"比较系统: {system_a} vs {system_b}")
                    
                    # 对每个问题进行成对比较
                    question_results = []
                    data_a = all_data[system_a]
                    data_b = all_data[system_b]
                    
                    min_len = min(len(data_a), len(data_b))
                    for k in range(min_len):
                        if data_a[k]["question"] == data_b[k]["question"]:
                            result = self.evaluate_pair(data_a[k], data_b[k])
                            question_results.append(result)
                    
                    pair_result = {
                        "system_a": system_a,
                        "system_b": system_b,
                        "question_results": question_results,
                        "summary": self._summarize_pair_results(question_results)
                    }
                    pair_results.append(pair_result)
        
        # 全局统计和Elo矩阵
        global_stats = self._compute_global_statistics(pair_results)
        elo_matrix = self._build_elo_matrix(systems, pair_results)
        
        final_result = {
            "systems": systems,
            "pair_results": pair_results,
            "global_statistics": global_stats,
            "elo_matrix": elo_matrix,
            "config": asdict(self.config)  # 使用asdict转换dataclass
        }
        
        # 保存结果
        self._save_results(final_result)
        
        return final_result
    
    def _is_granularity_enabled(self, granularity: str) -> bool:
        """检查粒度是否启用"""
        granularity_map = {
            "token": self.config.enable_token,
            "sentence": self.config.enable_sentence,
            "passage": self.config.enable_passage,
            "kg": self.config.enable_kg
        }
        return granularity_map.get(granularity, True)
    
    def _summarize_pair_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总成对比较结果"""
        if not results:
            return {}
        
        total_questions = len(results)
        a_wins = sum(1 for r in results if r["fusion_result"].get("winner") == "A")
        b_wins = sum(1 for r in results if r["fusion_result"].get("winner") == "B")
        ties = total_questions - a_wins - b_wins
        
        avg_delta = np.mean([r["combined_delta"] for r in results])
        
        return {
            "total_questions": total_questions,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "win_rate_a": a_wins / total_questions if total_questions > 0 else 0,
            "win_rate_b": b_wins / total_questions if total_questions > 0 else 0,
            "tie_rate": ties / total_questions if total_questions > 0 else 0,
            "avg_elo_delta": avg_delta
        }
    
    def _compute_global_statistics(self, pair_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算全局统计数据"""
        total_comparisons = sum(len(pr["question_results"]) for pr in pair_results)
        
        # 收集所有Elo delta
        all_deltas = []
        for pr in pair_results:
            for qr in pr["question_results"]:
                all_deltas.append(qr["combined_delta"])
        
        stats = {
            "total_comparisons": total_comparisons,
            "total_pairs": len(pair_results),
            "avg_elo_delta": np.mean(all_deltas) if all_deltas else 0,
            "std_elo_delta": np.std(all_deltas) if all_deltas else 0,
            "elo_delta_range": [min(all_deltas), max(all_deltas)] if all_deltas else [0, 0]
        }
        
        return stats
    
    def _build_elo_matrix(self, systems: List[str], pair_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建Elo评分矩阵"""
        n_systems = len(systems)
        elo_scores = {system: self.config.initial_elo for system in systems}
        
        # 计算每个系统的相对Elo分数
        for pr in pair_results:
            system_a = pr["system_a"]
            system_b = pr["system_b"]
            summary = pr["summary"]
            
            # 基于胜率调整Elo
            win_rate_a = summary.get("win_rate_a", 0.5)
            win_rate_b = summary.get("win_rate_b", 0.5)
            expected_a = 1 / (1 + 10 ** ((elo_scores[system_b] - elo_scores[system_a]) / 400))
            expected_b = 1 - expected_a
            
            k = self.config.k_factor
            elo_scores[system_a] += k * (win_rate_a - expected_a)
            elo_scores[system_b] += k * (win_rate_b - expected_b)
        
        # 排序
        sorted_systems = sorted(systems, key=lambda x: elo_scores[x], reverse=True)
        
        return {
            "elo_scores": elo_scores,
            "ranking": sorted_systems,
            "score_differences": {
                f"{sys1}_vs_{sys2}": elo_scores[sys1] - elo_scores[sys2]
                for sys1 in systems for sys2 in systems if sys1 != sys2
            }
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """保存评估结果"""
        output_path = Path(self.config.output_dir) / "dice_results.json"
        
        # 使用自定义编码器处理numpy和其他特殊类型
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
        
        self.logger.info(f"结果已保存到: {output_path}")
        
        # 保存汇总报告
        self._save_summary_report(results)
    
    def _save_summary_report(self, results: Dict[str, Any]):
        """保存汇总报告"""
        report_path = Path(self.config.output_dir) / "dice_summary.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# DICE评估报告\n\n")
            
            # 系统排名
            f.write("## 系统排名\n\n")
            elo_matrix = results["elo_matrix"]
            for i, system in enumerate(elo_matrix["ranking"], 1):
                elo_score = elo_matrix["elo_scores"][system]
                f.write(f"{i}. {system}: {elo_score:.1f}\n")
            
            # 全局统计
            f.write("\n## 全局统计\n\n")
            stats = results["global_statistics"]
            f.write(f"- 总比较次数: {stats['total_comparisons']}\n")
            f.write(f"- 系统对数: {stats['total_pairs']}\n")
            f.write(f"- 平均Elo差: {stats['avg_elo_delta']:.2f}\n")
            f.write(f"- Elo差标准差: {stats['std_elo_delta']:.2f}\n")
            
            # 成对比较结果
            f.write("\n## 成对比较结果\n\n")
            for pr in results["pair_results"]:
                f.write(f"### {pr['system_a']} vs {pr['system_b']}\n\n")
                summary = pr["summary"]
                f.write(f"- A胜率: {summary['win_rate_a']:.1%}\n")
                f.write(f"- B胜率: {summary['win_rate_b']:.1%}\n")
                f.write(f"- 平局率: {summary['tie_rate']:.1%}\n")
                f.write(f"- 平均Elo差: {summary['avg_elo_delta']:.2f}\n\n")
        
        self.logger.info(f"汇总报告已保存到: {report_path}")

# 便捷函数
def create_dice_evaluator(
    llm_model: str = "qwen2.5",
    output_dir: str = "dice_output"
) -> DICEEvaluator:
    """创建DICE评估器的便捷函数"""
    config = DICEConfig(
        llm_model=llm_model,
        output_dir=output_dir
    )
    return DICEEvaluator(config) 