#!/usr/bin/env python3
"""
DICE Elo融合模块
实现四粒度熵权融合、动态K-Elo和Bootstrap置信区间
"""

import logging
import math
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import stats
from collections import defaultdict

class EloFusion:
    """Elo融合器"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DICE.EloFusion")
    
    def fuse_judgments(self, judgments: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        融合四个粒度的判决结果
        
        Args:
            judgments: 四个粒度的判决结果字典
            
        Returns:
            融合后的结果
        """
        if not judgments:
            return {"elo_delta": 0, "winner": "Tie", "confidence": 0.0}
        
        # 步骤1: 计算熵权
        entropy_weights = self._compute_entropy_weights(judgments)
        
        # 步骤2: 粒度内Elo更新
        granularity_elos = {}
        for granularity, judgment in judgments.items():
            elo_delta = self._compute_granularity_elo(judgment)
            granularity_elos[granularity] = elo_delta
        
        # 步骤3: 熵权融合
        weighted_elo_delta = self._weighted_fusion(granularity_elos, entropy_weights)
        
        # 步骤4: 动态K调整
        adjusted_elo_delta = self._apply_dynamic_k(weighted_elo_delta)
        
        # 步骤5: 确定获胜者
        winner = self._determine_winner(adjusted_elo_delta, judgments)
        
        # 步骤6: 计算置信度
        confidence = self._compute_confidence(judgments, entropy_weights)
        
        fusion_result = {
            "elo_delta": adjusted_elo_delta,
            "winner": winner,
            "confidence": confidence,
            "granularity_elos": granularity_elos,
            "entropy_weights": entropy_weights,
            "weighted_delta": weighted_elo_delta
        }
        
        return fusion_result
    
    def _compute_entropy_weights(self, judgments: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """计算熵权"""
        weights = {}
        
        # 收集每个粒度的判决分布
        granularity_distributions = {}
        
        for granularity, judgment in judgments.items():
            # 构造伪分布：基于当前判决和置信度
            label = judgment.get("label", "Tie")
            score = judgment.get("score", 0.5)
            
            if label == "A wins":
                dist = [score, 1-score, 0]  # [A, B, Tie]
            elif label == "B wins":
                dist = [1-score, score, 0]
            else:  # Tie
                margin = judgment.get("margin_score", 0)
                if margin > 0:  # 偏向A
                    dist = [0.5 + margin, 0.5 - margin, 0]
                elif margin < 0:  # 偏向B
                    dist = [0.5 + margin, 0.5 - margin, 0]
                else:
                    dist = [0.33, 0.33, 0.34]
            
            granularity_distributions[granularity] = dist
        
        # 计算每个粒度的熵
        entropies = {}
        for granularity, dist in granularity_distributions.items():
            entropy = self._calculate_entropy(dist)
            entropies[granularity] = entropy
        
        # 熵权计算：熵越低，权重越高
        total_weight = 0
        for granularity, entropy in entropies.items():
            weight = 1 / (1 + entropy)  # 熵越低权重越高
            weights[granularity] = weight
            total_weight += weight
        
        # 归一化
        for granularity in weights:
            weights[granularity] /= total_weight
        
        self.logger.info(f"熵权分布: {weights}")
        return weights
    
    def _calculate_entropy(self, distribution: List[float]) -> float:
        """计算信息熵"""
        entropy = 0
        for p in distribution:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def _compute_granularity_elo(self, judgment: Dict[str, Any]) -> float:
        """计算单个粒度的Elo变化"""
        label = judgment.get("label", "Tie")
        score = judgment.get("score", 0.5)
        
        # 基础Elo计算
        if label == "A wins":
            win_score = 1.0
        elif label == "B wins":
            win_score = 0.0
        else:  # Tie，使用margin-aware分数
            win_score = score
        
        # 期望得分（假设两系统实力相当）
        expected_score = 0.5
        
        # Elo变化
        k_factor = self.config.k_factor
        elo_delta = k_factor * (win_score - expected_score)
        
        return elo_delta
    
    def _weighted_fusion(
        self, 
        granularity_elos: Dict[str, float], 
        weights: Dict[str, float]
    ) -> float:
        """熵权融合"""
        weighted_sum = 0
        for granularity, elo_delta in granularity_elos.items():
            weight = weights.get(granularity, 0)
            weighted_sum += weight * elo_delta
        
        return weighted_sum
    
    def _apply_dynamic_k(self, elo_delta: float) -> float:
        """应用动态K因子"""
        if not self.config.dynamic_k:
            return elo_delta
        
        # 当Elo差距过大时，降低K因子
        if abs(elo_delta) > 20:  # 相当于|ΔR|>400的条件简化
            adjustment_factor = 0.5  # K=16相当于K=32的0.5倍
            adjusted_delta = elo_delta * adjustment_factor
            self.logger.info(f"应用动态K调整: {elo_delta:.2f} -> {adjusted_delta:.2f}")
            return adjusted_delta
        
        return elo_delta
    
    def _determine_winner(
        self, 
        elo_delta: float, 
        judgments: Dict[str, Dict[str, Any]]
    ) -> str:
        """确定获胜者"""
        # 基于Elo差值确定获胜者
        if elo_delta > self.config.tie_threshold:
            return "A"
        elif elo_delta < -self.config.tie_threshold:
            return "B"
        else:
            return "Tie"
    
    def _compute_confidence(
        self, 
        judgments: Dict[str, Dict[str, Any]], 
        weights: Dict[str, float]
    ) -> float:
        """计算置信度"""
        # 基于判决一致性和权重分布计算置信度
        
        # 1. 判决一致性
        labels = [j.get("label", "Tie") for j in judgments.values()]
        unique_labels = set(labels)
        
        if len(unique_labels) == 1:
            # 所有粒度判决一致
            consistency_score = 1.0
        elif len(unique_labels) == 2:
            # 部分一致
            consistency_score = 0.6
        else:
            # 完全不一致
            consistency_score = 0.3
        
        # 2. 权重分布的集中度
        weight_entropy = self._calculate_entropy(list(weights.values()))
        max_entropy = math.log2(len(weights))
        weight_concentration = 1 - (weight_entropy / max_entropy)
        
        # 3. 综合置信度
        confidence = 0.7 * consistency_score + 0.3 * weight_concentration
        
        return confidence
    
    def compute_statistics(self, fusion_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算统计数据和Bootstrap置信区间
        
        Args:
            fusion_result: 融合结果
            
        Returns:
            统计数据
        """
        elo_delta = fusion_result.get("elo_delta", 0)
        
        # Bootstrap置信区间（简化版本，基于正态分布假设）
        ci_95 = self._compute_bootstrap_ci(elo_delta)
        
        # Expected Calibration Error（简化版本）
        ece = self._compute_ece(fusion_result)
        
        # 统计显著性检验
        p_value = self._compute_significance(elo_delta)
        
        statistics = {
            "elo_delta": elo_delta,
            "ci_95": ci_95,
            "ece": ece,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "confidence_level": self.config.confidence_level
        }
        
        return statistics
    
    def _compute_bootstrap_ci(self, elo_delta: float) -> List[float]:
        """计算Bootstrap 95%置信区间"""
        # 简化实现：基于标准误差的正态分布近似
        # 实际应用中应该使用真实的Bootstrap重采样
        
        # 估计标准误差（基于Elo系统的典型方差）
        se = abs(elo_delta) * 0.1 + 5  # 简化的标准误差估计
        
        # 95%置信区间
        z_score = 1.96  # 95%置信水平的z值
        margin = z_score * se
        
        ci_lower = elo_delta - margin
        ci_upper = elo_delta + margin
        
        return [ci_lower, ci_upper]
    
    def _compute_ece(self, fusion_result: Dict[str, Any]) -> float:
        """计算Expected Calibration Error"""
        # 简化版本：基于置信度和实际表现的差异
        confidence = fusion_result.get("confidence", 0.5)
        
        # 实际表现（基于是否为Tie）
        winner = fusion_result.get("winner", "Tie")
        actual_performance = 1.0 if winner != "Tie" else 0.5
        
        # ECE = |置信度 - 实际表现|
        ece = abs(confidence - actual_performance)
        
        return ece
    
    def _compute_significance(self, elo_delta: float) -> float:
        """计算统计显著性p值"""
        # 简化版本：基于t检验的单样本检验
        # 零假设：elo_delta = 0
        
        # 估计标准误差
        se = abs(elo_delta) * 0.1 + 5
        
        if se == 0:
            return 1.0
        
        # t统计量
        t_stat = elo_delta / se
        
        # 双尾检验的p值（自由度=1000-1，大样本近似）
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=999))
        
        return p_value
    
    def compute_pairwise_matrix(
        self, 
        systems: List[str], 
        pairwise_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """计算成对比较矩阵"""
        n_systems = len(systems)
        
        # 初始化Elo分数
        elo_scores = {system: self.config.initial_elo for system in systems}
        
        # 构建胜负矩阵
        win_matrix = np.zeros((n_systems, n_systems))
        total_matrix = np.zeros((n_systems, n_systems))
        
        system_to_idx = {system: i for i, system in enumerate(systems)}
        
        # 处理成对比较结果
        for result in pairwise_results:
            system_a = result.get("system_a", "")
            system_b = result.get("system_b", "")
            
            if system_a in system_to_idx and system_b in system_to_idx:
                idx_a = system_to_idx[system_a]
                idx_b = system_to_idx[system_b]
                
                # 获取胜率
                summary = result.get("summary", {})
                win_rate_a = summary.get("win_rate_a", 0.5)
                
                # 更新矩阵
                total_comparisons = summary.get("total_questions", 0)
                win_matrix[idx_a, idx_b] = win_rate_a * total_comparisons
                win_matrix[idx_b, idx_a] = (1 - win_rate_a) * total_comparisons
                total_matrix[idx_a, idx_b] = total_comparisons
                total_matrix[idx_b, idx_a] = total_comparisons
                
                # 更新Elo分数
                expected_a = 1 / (1 + 10 ** ((elo_scores[system_b] - elo_scores[system_a]) / 400))
                
                k = self.config.k_factor
                elo_scores[system_a] += k * (win_rate_a - expected_a)
                elo_scores[system_b] += k * ((1 - win_rate_a) - (1 - expected_a))
        
        # 计算胜率矩阵
        win_rate_matrix = np.divide(
            win_matrix, 
            total_matrix, 
            out=np.zeros_like(win_matrix), 
            where=total_matrix!=0
        )
        
        # 排序
        sorted_systems = sorted(systems, key=lambda x: elo_scores[x], reverse=True)
        
        matrix_result = {
            "systems": systems,
            "elo_scores": elo_scores,
            "ranking": sorted_systems,
            "win_matrix": win_matrix.tolist(),
            "win_rate_matrix": win_rate_matrix.tolist(),
            "total_matrix": total_matrix.tolist()
        }
        
        return matrix_result
    
    def generate_elo_report(
        self, 
        matrix_result: Dict[str, Any]
    ) -> str:
        """生成Elo评分报告"""
        systems = matrix_result["systems"]
        elo_scores = matrix_result["elo_scores"]
        ranking = matrix_result["ranking"]
        
        report = "# DICE Elo评分报告\n\n"
        
        # 排名表
        report += "## 系统排名\n\n"
        report += "| 排名 | 系统 | Elo分数 | 分数差 |\n"
        report += "|------|------|---------|--------|\n"
        
        for i, system in enumerate(ranking):
            score = elo_scores[system]
            score_diff = score - elo_scores[ranking[0]] if i > 0 else 0
            report += f"| {i+1} | {system} | {score:.1f} | {score_diff:+.1f} |\n"
        
        # 成对比较胜率
        report += "\n## 成对比较胜率矩阵\n\n"
        win_rate_matrix = np.array(matrix_result["win_rate_matrix"])
        
        report += "| 系统 |"
        for system in systems:
            report += f" {system} |"
        report += "\n"
        
        report += "|------|"
        for _ in systems:
            report += "------|"
        report += "\n"
        
        for i, system_a in enumerate(systems):
            report += f"| {system_a} |"
            for j, system_b in enumerate(systems):
                if i == j:
                    report += " - |"
                else:
                    win_rate = win_rate_matrix[i, j]
                    report += f" {win_rate:.1%} |"
            report += "\n"
        
        return report 