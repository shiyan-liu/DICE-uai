#!/usr/bin/env python3
"""
DICE 精简版 - 锦标赛和基线对比场景
专注于 passage 粒度 + 检索-证据双通道判决
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import itertools
from collections import defaultdict
import math
from dataclasses import dataclass
import concurrent.futures
import threading

# 导入本地判决器
from .local_pairwise_judge import LocalPairwiseJudge

# 添加tqdm进度条支持
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # 定义一个简单的替代品
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0
        
        def __iter__(self):
            if self.iterable:
                for item in self.iterable:
                    yield item
                    self.update(1)
            return self
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def update(self, n=1):
            self.n += n
        
        def set_description(self, desc):
            self.desc = desc
        
        def close(self):
            pass

# 添加sklearn导入和异常处理
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .local_pairwise_judge import LocalPairwiseJudge as PairwiseJudge


@dataclass
class SimplifiedDICEConfig:
    """精简版DICE配置"""
    # LLM配置 - 在线API
    llm_model: str = "deepseek-chat"
    api_key: str = ""  # 从环境变量获取: DEEPSEEK_API_KEY
    base_url: str = "https://api.deepseek.com"
    judge_temperature: float = 0.1
    max_tokens: int = 2048
    
    # DeepSeek-R1本地模型配置
    enable_deep_thinking: bool = True  # 是否启用深度思考模式，默认开启
    
    # 评估配置
    max_questions: int = 70
    early_stop_elo_diff: float = 400.0
    early_stop_ci_threshold: float = 30.0
    
    # Elo配置
    initial_elo: float = 1000.0
    k_factor: int = 32
    
    # 并发配置 - 双GPU优化
    max_workers: int = 4  # 最大并发worker数量（双GPU优化：2卡×2worker）
    batch_size: int = 8   # 每批处理的问题数量（双GPU显存总量~48GB）
    
    # 输出配置
    output_dir: str = "dice_simplified_output"
    save_detailed: bool = True


class SimplifiedDICEEvaluator:
    """DICE精简版评估器"""
    
    def __init__(self, config: SimplifiedDICEConfig = None):
        self.config = config or SimplifiedDICEConfig()
        self.logger = logging.getLogger("DICE.Simplified")
        self._setup_logger()
        
        # 初始化判决器（仅使用passage粒度）
        self.pairwise_judge = LocalPairwiseJudge(self.config)
        
        # 并发相关
        self._lock = threading.Lock()  # 用于同步日志输出
        
        # 虚拟基线生成指令
        self.baseline_prompts = {
            "Good": {
                "instruction": "作为一个高质量的RAG系统，请基于给定问题和标准答案生成详细准确的回答。要求：1)提供完整的关键信息，2)逻辑清晰条理分明，3)基于权威可靠的资料，4)准确性高且表述专业。",
                "context_instruction": "请生成3条高质量、高相关性的检索证据，内容应该详细、准确，能够充分支撑回答。",
                "quality_level": "high"
            },
            "Medium": {
                "instruction": "作为一个中等水平的RAG系统，请基于给定问题生成基本正确的回答。要求：1)包含主要信息但可能缺少细节，2)表述基本准确但不够深入，3)信息完整性中等。",
                "context_instruction": "请生成3条中等质量的检索证据，内容基本相关但可能缺少一些关键细节。",
                "quality_level": "medium"
            },
            "Bad": {
                "instruction": "作为一个低质量的RAG系统，请基于给定问题生成质量较差的回答。要求：1)信息不够准确或有遗漏，2)表述可能含糊不清，3)可能包含错误或无关信息。",
                "context_instruction": "请生成3条低质量的检索证据，内容相关性较低，可能包含错误或不够准确的信息。",
                "quality_level": "low"
            }
        }
        
    def _log_question_result(self, result: Dict[str, Any], completed_count: int, total_questions: int):
        """线程安全的问题结果日志输出 - 显示soft win信息"""
        passage_judgment = result["passage_judgment"]
        question = result["question"]
        score_a = result["score_a"]
        score_b = result["score_b"]
        
        self.logger.info(f"    问题 {completed_count}/{total_questions}: {question[:60]}...")
        self.logger.info(f"    🏆 判决: {passage_judgment.get('win_type', 'Unknown')}")
        self.logger.info(f"    📈 Logits: A={passage_judgment.get('logit_a', 0):.2f}, B={passage_judgment.get('logit_b', 0):.2f}, T={passage_judgment.get('logit_t', 0):.2f}")
        self.logger.info(f"    📊 概率: A={passage_judgment.get('prob_a', 0):.3f}, B={passage_judgment.get('prob_b', 0):.3f}, T={passage_judgment.get('prob_t', 0):.3f}")
        self.logger.info(f"    🔥 概率差距: {passage_judgment.get('prob_diff', 0):.3f} ({'Hard' if passage_judgment.get('prob_diff', 0) >= 0.1 else 'Soft'} win)")
        self.logger.info(f"    🎯 得分: A={score_a:.3f}, B={score_b:.3f}")
        # 简化理由输出
        # self.logger.info(f"    💭 理由: {passage_judgment.get('reason', '')}...")
        self.logger.info("")
    
    def _setup_logger(self):
        """设置日志"""
        self.logger.setLevel(logging.INFO)
        # 设置propagate=False以避免重复输出到根logger
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def scenario_a_tournament(self, qacg_files: List[str]) -> Dict[str, Any]:
        """
        场景A: 八系统锦标赛
        
        Args:
            qacg_files: QACG文件路径列表（8个系统）
            
        Returns:
            锦标赛结果
        """
        self.logger.info("🏆 开始场景A: 八系统锦标赛（动态Elo配对系统）")
        
        # 1. 加载系统数据
        systems = self._load_systems(qacg_files)
        system_names = list(systems.keys())
        
        if len(system_names) != 8:
            raise ValueError(f"需要8个系统，实际获得{len(system_names)}个")
        
        # 2. 瑞士轮锦标赛（4轮，每轮4场，共16场比赛）
        swiss_results = self._swiss_tournament(system_names, systems, num_rounds=4)
        
        # 3. 最终排名（基于瑞士轮Elo分数）
        final_ranking = self._calculate_dynamic_ranking(swiss_results["final_elo_scores"])
        
        # 4. 95% CI分析
        all_pairwise_results = swiss_results["all_pairwise_results"]
        ci_analysis = self._bootstrap_ci_analysis(all_pairwise_results, system_names)
        
        # 5. 失败模式动态聚类分析
        failure_clusters = self._cluster_failure_modes(all_pairwise_results)
        
        # 汇总结果
        tournament_result = {
            "config": self._config_to_dict(),
            "tournament_type": "swiss_tournament",
            "swiss_results": swiss_results,
            "final_ranking": final_ranking,
            "final_elo_scores": swiss_results["final_elo_scores"],
            "total_llm_calls": swiss_results["total_llm_calls"],
            "ci_analysis": ci_analysis,
            "failure_analysis": failure_clusters
        }
        
        # 保存结果
        self._save_tournament_result(tournament_result)
        return tournament_result
    
    def scenario_c_full_round_robin(self, qacg_files: List[str]) -> Dict[str, Any]:
        """
        场景C: 全对全两两配对（完整循环赛）
        - 记录所有配对比赛；每个系统之间只对战一次
        """
        self.logger.info("🏆 开始场景C: 全对全两两配对（完整循环赛）")
        
        # 1. 加载系统数据
        systems = self._load_systems(qacg_files)
        system_names = list(systems.keys())
        
        if len(system_names) < 2:
            raise ValueError(f"需要至少2个系统，实际获得{len(system_names)}个")
        
        # 2. 初始化Elo
        elo_scores = {system: 1500.0 for system in system_names}
        all_pairwise_results = []
        match_records = []
        total_llm_calls = 0
        
        # 3. 遍历所有唯一配对（组合）
        pair_idx = 0
        total_pairs = len(system_names) * (len(system_names) - 1) // 2
        for sys_a, sys_b in itertools.combinations(system_names, 2):
            pair_idx += 1
            self.logger.info(f"  📊 第{pair_idx}/{total_pairs}场: {sys_a} (ELO: {elo_scores[sys_a]:.1f}) vs {sys_b} (ELO: {elo_scores[sys_b]:.1f})")
            
            # 执行对比
            comparison = self._pairwise_comparison(
                systems[sys_a], systems[sys_b], sys_a, sys_b, 
                max_questions=self.config.max_questions
            )
            all_pairwise_results.append(comparison)
            total_llm_calls += len(comparison["question_results"])
            
            # 更新Elo
            old_elo_a, old_elo_b = elo_scores[sys_a], elo_scores[sys_b]
            self._update_elo_scores_dynamic(elo_scores, comparison, sys_a, sys_b)
            
            # 记录比赛
            match_records.append({
                "match_num": pair_idx,
                "system_a": sys_a,
                "system_b": sys_b,
                "old_elo_a": old_elo_a,
                "old_elo_b": old_elo_b,
                "new_elo_a": elo_scores[sys_a],
                "new_elo_b": elo_scores[sys_b],
                "winner": self._determine_winner(comparison),
                "comparison": comparison
            })
        
        # 4. 最终排名与分析
        final_ranking = self._calculate_dynamic_ranking(elo_scores)
        ci_analysis = self._bootstrap_ci_analysis(all_pairwise_results, system_names)
        failure_clusters = self._cluster_failure_modes(all_pairwise_results)
        
        result = {
            "config": self._config_to_dict(),
            "tournament_type": "full_round_robin",
            "round_robin_results": {
                "match_records": match_records,
                "all_pairwise_results": all_pairwise_results,
                "final_elo_scores": elo_scores,
                "total_llm_calls": total_llm_calls,
                "total_matches": len(match_records)
            },
            "final_ranking": final_ranking,
            "final_elo_scores": elo_scores,
            "total_llm_calls": total_llm_calls,
            "ci_analysis": ci_analysis,
            "failure_analysis": failure_clusters
        }
        
        # 5. 保存
        self._save_tournament_result(result)
        return result
    
    def _swiss_tournament(self, system_names: List[str], all_systems: Dict[str, List[Dict]], 
                         num_rounds: int) -> Dict[str, Any]:
        """瑞士轮锦标赛实现"""
        self.logger.info(f"🔄 开始瑞士轮锦标赛，共{num_rounds}轮")
        
        # 初始化选手状态
        standings = {}
        for system in system_names:
            standings[system] = {
                "elo": self.config.initial_elo,
                "swiss_points": 0.0,  # 瑞士轮积分
                "wins": 0,
                "draws": 0, 
                "losses": 0,
                "sb_score": 0.0,  # SB分（对手分数总和）
                "opponents": []  # 对战过的对手
            }
        
        rounds = []
        total_llm_calls = 0
        
        for round_num in range(1, num_rounds + 1):
            self.logger.info(f"🏁 第{round_num}轮开始")
            
            # 配对
            pairings = self._swiss_pairing(standings, round_num)
            
            # 进行比赛
            round_results = []
            round_pairwise_results = []
            
            for sys_a, sys_b in pairings:
                self.logger.info(f"  📊 {sys_a} vs {sys_b}")
                
                # 执行对比
                comparison = self._pairwise_comparison(
                    all_systems[sys_a], all_systems[sys_b], sys_a, sys_b,
                    max_questions=max(3, self.config.max_questions // num_rounds)  # 每轮使用部分题目
                )
                round_pairwise_results.append(comparison)
                total_llm_calls += len(comparison["question_results"])
                
                # 计算比赛结果
                result = self._calculate_match_result(comparison)
                round_results.append({
                    "system_a": sys_a,
                    "system_b": sys_b,
                    "result": result,
                    "comparison": comparison
                })
                
                # 更新ELO分数
                self._update_elo_scores_swiss(standings, comparison, sys_a, sys_b)
                
                # 更新瑞士轮积分和记录
                self._update_swiss_standings(standings, sys_a, sys_b, result)
            
            # 保存本轮结果
            rounds.append({
                "round": round_num,
                "pairings": pairings,
                "results": round_results,
                "pairwise_results": round_pairwise_results,
                "standings_after_round": self._get_current_standings_snapshot(standings)
            })
            
            # 计算SB分（需要在每轮后更新）
            self._update_sb_scores(standings)
            
            self.logger.info(f"第{round_num}轮结束，当前排名:")
            current_ranking = self._get_current_ranking(standings)
            for i, (system, stats) in enumerate(current_ranking[:3], 1):
                self.logger.info(f"  {i}. {system}: {stats['swiss_points']:.1f}分 (ELO: {stats['elo']:.1f})")
        
        return {
            "rounds": rounds,
            "final_standings": standings,
            "total_llm_calls": total_llm_calls
        }
    
    def _swiss_pairing(self, standings: Dict[str, Dict], round_num: int) -> List[Tuple[str, str]]:
        """瑞士轮配对算法"""
        if round_num == 1:
            # 第一轮：大小模型交叉配对，测试真实差距
            systems = list(standings.keys())
            large_systems = [s for s in systems if "large" in s]
            small_systems = [s for s in systems if "small" in s]
            
            pairings = []
            # 确保每个大模型都有小模型对手
            for i in range(min(len(large_systems), len(small_systems))):
                pairings.append((large_systems[i], small_systems[i]))
            
            # 如果有剩余系统，配对剩下的
            remaining_large = large_systems[len(small_systems):]
            remaining_small = small_systems[len(large_systems):]
            
            for i in range(0, len(remaining_large), 2):
                if i + 1 < len(remaining_large):
                    pairings.append((remaining_large[i], remaining_large[i + 1]))
                    
            for i in range(0, len(remaining_small), 2):
                if i + 1 < len(remaining_small):
                    pairings.append((remaining_small[i], remaining_small[i + 1]))
            
            return pairings
        else:
            # 根据积分和ELO分数配对
            systems_by_score = sorted(
                standings.keys(),
                key=lambda x: (standings[x]["swiss_points"], standings[x]["elo"]),
                reverse=True
            )
            
            pairings = []
            paired = set()
            
            for i, system_a in enumerate(systems_by_score):
                if system_a in paired:
                    continue
                
                # 寻找最佳对手（积分相近且未对战过）
                best_opponent = None
                for j in range(i + 1, len(systems_by_score)):
                    system_b = systems_by_score[j]
                    if (system_b not in paired and 
                        system_b not in standings[system_a]["opponents"]):
                        best_opponent = system_b
                        break
                
                # 如果找不到未对战的对手，选择最近的对手
                if not best_opponent:
                    for j in range(i + 1, len(systems_by_score)):
                        system_b = systems_by_score[j]
                        if system_b not in paired:
                            best_opponent = system_b
                            break
                
                if best_opponent:
                    pairings.append((system_a, best_opponent))
                    paired.add(system_a)
                    paired.add(best_opponent)
            
            return pairings
    
    def _calculate_match_result(self, comparison: Dict[str, Any]) -> str:
        """计算比赛结果（胜/平/负）"""
        summary = comparison["summary"]
        win_rate_a = summary["win_rate_a"]
        
        if win_rate_a > 0.6:
            return "A_wins"
        elif win_rate_a < 0.4:
            return "B_wins"
        else:
            return "draw"
    
    def _update_elo_scores_swiss(self, standings: Dict[str, Dict], 
                               comparison: Dict[str, Any], sys_a: str, sys_b: str):
        """更新ELO分数（瑞士轮版本，使用加权算法）"""
        summary = comparison["summary"]
        win_rate_a = summary["win_rate_a"]
        win_rate_b = summary["win_rate_b"]
        
        elo_a = standings[sys_a]["elo"]
        elo_b = standings[sys_b]["elo"]
        
        # 计算期望胜率
        expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        expected_b = 1 - expected_a
        
        # 计算分差和加权系数
        rating_diff = abs(elo_a - elo_b)
        base_k = self.config.k_factor
        
        # 加权系数：基于分差的非线性函数
        weight_factor = 0.5 + 1.5 * (1 - math.exp(-rating_diff / 200))
        
        # 爆冷奖励
        upset_bonus_a = 1.0
        upset_bonus_b = 1.0
        
        if elo_a < elo_b and win_rate_a > 0.5:  # A爆冷击败B
            upset_bonus_a = 1.0 + (rating_diff / 400)
            upset_bonus_b = 1.0 + (rating_diff / 600)
        elif elo_b < elo_a and win_rate_b > 0.5:  # B爆冷击败A
            upset_bonus_b = 1.0 + (rating_diff / 400)
            upset_bonus_a = 1.0 + (rating_diff / 600)
        
        # 计算最终K因子
        k_a = base_k * weight_factor * upset_bonus_a
        k_b = base_k * weight_factor * upset_bonus_b
        
        # 更新ELO
        standings[sys_a]["elo"] += k_a * (win_rate_a - expected_a)
        standings[sys_b]["elo"] += k_b * (win_rate_b - expected_b)
    
    def _update_swiss_standings(self, standings: Dict[str, Dict], 
                              sys_a: str, sys_b: str, result: str):
        """更新瑞士轮积分和战绩"""
        # 记录对手
        standings[sys_a]["opponents"].append(sys_b)
        standings[sys_b]["opponents"].append(sys_a)
        
        # 更新积分和战绩
        if result == "A_wins":
            standings[sys_a]["swiss_points"] += 1.0
            standings[sys_a]["wins"] += 1
            standings[sys_b]["losses"] += 1
        elif result == "B_wins":
            standings[sys_b]["swiss_points"] += 1.0
            standings[sys_b]["wins"] += 1
            standings[sys_a]["losses"] += 1
        else:  # draw
            standings[sys_a]["swiss_points"] += 0.5
            standings[sys_b]["swiss_points"] += 0.5
            standings[sys_a]["draws"] += 1
            standings[sys_b]["draws"] += 1
    
    def _update_sb_scores(self, standings: Dict[str, Dict]):
        """更新SB分（对手分数总和）"""
        for system in standings:
            sb_score = 0.0
            for opponent in standings[system]["opponents"]:
                sb_score += standings[opponent]["swiss_points"]
            standings[system]["sb_score"] = sb_score
    
    def _get_current_standings_snapshot(self, standings: Dict[str, Dict]) -> Dict[str, Dict]:
        """获取当前积分榜快照"""
        return {system: stats.copy() for system, stats in standings.items()}
    
    def _get_current_ranking(self, standings: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
        """获取当前排名"""
        return sorted(
            standings.items(),
            key=lambda x: (x[1]["swiss_points"], x[1]["elo"], x[1]["sb_score"]),
            reverse=True
        )
    
    def _calculate_swiss_ranking(self, final_standings: Dict[str, Dict]) -> List[str]:
        """计算瑞士轮最终排名"""
        # 排名规则：
        # 1. 瑞士轮积分（胜1分，平0.5分，负0分）
        # 2. ELO分数
        # 3. SB分（对手分数总和）
        # 4. 胜场数
        # 5. 系统名称（字典序）
        
        ranked_systems = sorted(
            final_standings.items(),
            key=lambda x: (
                x[1]["swiss_points"],    # 主要：瑞士轮积分
                x[1]["elo"],            # 次要：ELO分数
                x[1]["sb_score"],       # 第三：SB分
                x[1]["wins"],           # 第四：胜场数
                x[0]                    # 第五：系统名称
            ),
            reverse=True
        )
        
        return [system for system, _ in ranked_systems]
    
    def _bootstrap_ci_analysis(self, pairwise_results: List[Dict], system_names: List[str]) -> Dict[str, Any]:
        """执行bootstrap CI分析"""
        all_score_diffs = []
        for result in pairwise_results:
            for qr in result["question_results"]:
                # 使用得分差值代替elo_delta
                score_diff = qr["score_a"] - qr["score_b"]
                all_score_diffs.append(score_diff)

        if not all_score_diffs:
            return {
                "mean_score_diff": 0.0,
                "ci_95": "0.00 - 0.00",
                "significance": "无数据"
            }

        # 计算平均得分差
        mean_score_diff = np.mean(all_score_diffs)

        # 执行bootstrap CI
        try:
            from scipy.stats import bootstrap
            boot_results = bootstrap((all_score_diffs,), np.mean, confidence_level=0.95, n_resamples=1000)
            ci_95 = boot_results.confidence_interval
            ci_95_str = f"{ci_95.low:.2f} - {ci_95.high:.2f}"
            
            # 显著性判断 (基于CI)
            significance = "显著" if not (ci_95.low <= 0 <= ci_95.high) else "不显著"
        except Exception as e:
            self.logger.warning(f"Bootstrap CI计算失败: {e}")
            ci_95_str = "计算失败"
            significance = "未知"

        return {
            "mean_score_diff": mean_score_diff,
            "ci_95": ci_95_str,
            "significance": significance
        }
    
    def _cluster_failure_modes(self, pairwise_results: List[Dict]) -> Dict[str, Any]:
        """动态语义聚类分析失败模式 - 基于LLM回答的语义相似度"""
        # 收集所有失败原因文本
        failure_reasons = []
        reason_to_systems = {}
        
        for result in pairwise_results:
            for qr in result["question_results"]:
                passage_judgment = qr.get("passage_judgment", {})
                reason = passage_judgment.get("reason", "")
                if reason and len(reason.strip()) > 10:  # 过滤太短的原因
                    failure_reasons.append(reason.strip())
                    if reason not in reason_to_systems:
                        reason_to_systems[reason] = set()
                    reason_to_systems[reason].add(result["system_a"])
                    reason_to_systems[reason].add(result["system_b"])

        if len(failure_reasons) < 5:
            # 数据不足，返回简单统计
            return {
                "cluster_0": {
                    "label": "失败原因分析",
                    "systems": list(set().union(*reason_to_systems.values())) if reason_to_systems else [],
                    "reasons": failure_reasons,
                    "top_keywords": self._extract_top_keywords(failure_reasons),
                    "size": len(failure_reasons)
                }
            }

        try:
            if not SKLEARN_AVAILABLE:
                self.logger.warning("sklearn不可用，跳过动态语义聚类，返回简单统计")
                return {
                    "cluster_0": {
                        "label": "失败原因分析(简化模式)",
                        "systems": list(set().union(*reason_to_systems.values())) if reason_to_systems else [],
                        "reasons": failure_reasons,
                        "top_keywords": self._extract_top_keywords(failure_reasons),
                        "size": len(failure_reasons)
                    }
                }
            
            # 动态TF-IDF向量化（中文分词友好）
            vectorizer = TfidfVectorizer(
                max_features=200, 
                stop_words=None, 
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.8,
                token_pattern=r'[\u4e00-\u9fff]+|[a-zA-Z]+\d*'  # 中文字符或英文单词
            )
            tfidf_matrix = vectorizer.fit_transform(failure_reasons)
            
            # 动态确定聚类数量（基于数据规模和语义相似度）
            n_clusters = self._determine_optimal_clusters(tfidf_matrix, failure_reasons)
            
            if n_clusters <= 1:
                # 聚类效果不佳，返回统一分析
                return {
                    "cluster_0": {
                        "label": "通用失败模式",
                        "systems": list(set().union(*reason_to_systems.values())) if reason_to_systems else [],
                        "reasons": failure_reasons,
                        "top_keywords": self._extract_top_keywords(failure_reasons),
                        "size": len(failure_reasons)
                    }
                }
            
            # K-means聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # 动态生成聚类标签
            feature_names = vectorizer.get_feature_names_out()
            clusters = {}
            
            for cluster_id in range(n_clusters):
                cluster_reasons = [failure_reasons[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                cluster_systems = set()
                
                # 收集该聚类对应的系统
                for reason in cluster_reasons:
                    if reason in reason_to_systems:
                        cluster_systems.update(reason_to_systems[reason])
                
                # 动态生成聚类标签（基于TF-IDF权重最高的词）
                cluster_label = self._generate_cluster_label(cluster_reasons, feature_names, kmeans.cluster_centers_[cluster_id])
                
                # 提取该聚类的关键词
                top_keywords = self._extract_cluster_keywords(cluster_reasons, feature_names)
                
                clusters[f"cluster_{cluster_id}"] = {
                    "label": cluster_label,
                    "systems": list(cluster_systems),
                    "reasons": cluster_reasons,
                    "top_keywords": top_keywords,
                    "size": len(cluster_reasons)
                }
            
            # 按聚类大小排序
            sorted_clusters = dict(sorted(clusters.items(), key=lambda x: x[1]["size"], reverse=True))
            
            return sorted_clusters
            
        except Exception as e:
            self.logger.warning(f"动态语义聚类失败: {e}")
            # 返回简单的关键词统计
            return {
                "cluster_0": {
                    "label": "失败模式分析",
                    "systems": list(set().union(*reason_to_systems.values())) if reason_to_systems else [],
                    "reasons": failure_reasons,
                    "top_keywords": self._extract_top_keywords(failure_reasons),
                    "size": len(failure_reasons)
                }
            }
    
    def _determine_optimal_clusters(self, tfidf_matrix, failure_reasons: List[str]) -> int:
        """动态确定最优聚类数量"""
        n_samples = len(failure_reasons)
        
        # 基于数据规模确定聚类数量范围
        if n_samples < 5:
            return 1
        elif n_samples < 15:
            max_clusters = 2
        elif n_samples < 30:
            max_clusters = 3
        else:
            max_clusters = min(5, n_samples // 8)
        
        # 使用轮廓系数选择最优聚类数
        try:
            from sklearn.metrics import silhouette_score
            best_n_clusters = 1
            best_score = -1
            
            for n in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                labels = kmeans.fit_predict(tfidf_matrix)
                score = silhouette_score(tfidf_matrix, labels)
                
                if score > best_score and score > 0.3:  # 要求一定的聚类质量
                    best_score = score
                    best_n_clusters = n
            
            return best_n_clusters
            
        except Exception:
            # 轮廓分析失败，使用启发式规则
            return min(3, max(1, n_samples // 10))
    
    def _generate_cluster_label(self, cluster_reasons: List[str], 
                              feature_names: list, cluster_center: list) -> str:
        """基于TF-IDF权重动态生成聚类标签"""
        try:
            # 获取权重最高的前3个特征
            top_indices = sorted(range(len(cluster_center)), 
                               key=lambda i: cluster_center[i], reverse=True)[:3]
            top_features = [feature_names[i] for i in top_indices if cluster_center[i] > 0]
            
            if not top_features:
                return "未分类失败模式"
            
            # 基于关键特征生成有意义的标签
            label_mapping = {
                ('检索', '缺失', '段落'): "检索缺关键段",
                ('数字', '错误', '计算'): "数值计算错误", 
                ('逻辑', '跳跃', '推理'): "逻辑推理问题",
                ('证据', '不足', '支撑'): "证据支撑不足",
                ('回答', '不完整', '缺失'): "回答不完整",
                ('理解', '错误', '理解'): "理解偏差",
                ('格式', '错误', '结构'): "格式结构问题"
            }
            
            # 尝试匹配预定义模式
            top_features_str = ' '.join(top_features)
            for pattern, label in label_mapping.items():
                if any(keyword in top_features_str for keyword in pattern):
                    return label
            
            # 如果没有匹配，基于最重要的特征生成标签
            main_feature = top_features[0]
            if '检索' in main_feature or '查找' in main_feature:
                return "检索相关问题"
            elif '回答' in main_feature or '答案' in main_feature:
                return "回答质量问题"
            elif '逻辑' in main_feature or '推理' in main_feature:
                return "逻辑推理问题"
            elif '数字' in main_feature or '计算' in main_feature:
                return "数值处理问题"
            else:
                return f"{main_feature}相关问题"
                
        except Exception:
            return "失败模式"
    
    def _extract_cluster_keywords(self, cluster_reasons: List[str], feature_names: list) -> List[Tuple[str, int]]:
        """提取聚类的关键词及频次"""
        try:
            if not SKLEARN_AVAILABLE:
                # 简化的关键词提取
                return self._extract_top_keywords(cluster_reasons)
            
            # 重新对该聚类的文本进行TF-IDF分析
            vectorizer = TfidfVectorizer(
                max_features=50,
                ngram_range=(1, 2),
                token_pattern=r'[\u4e00-\u9fff]+|[a-zA-Z]+\d*'
            )
            tfidf_matrix = vectorizer.fit_transform(cluster_reasons)
            feature_names = vectorizer.get_feature_names_out()
            
            # 计算TF-IDF总分
            tfidf_scores = tfidf_matrix.sum(axis=0).A1
            
            # 获取前5个关键词
            top_indices = sorted(range(len(tfidf_scores)), 
                               key=lambda i: tfidf_scores[i], reverse=True)[:5]
            
            top_keywords = []
            for idx in top_indices:
                if tfidf_scores[idx] > 0:
                    keyword = feature_names[idx]
                    # 计算该词在文本中的出现次数
                    count = sum(1 for reason in cluster_reasons if keyword in reason)
                    top_keywords.append((keyword, count))
            
            return top_keywords
            
        except Exception:
            return self._extract_top_keywords(cluster_reasons)
    
    def _extract_top_keywords(self, reasons: List[str]) -> List[Tuple[str, int]]:
        """简单的关键词提取（备用方法）"""
        common_keywords = [
            "检索", "缺失", "不足", "错误", "不准确", "不完整", "不相关", 
            "逻辑", "推理", "证据", "支撑", "回答", "数字", "计算", "理解",
            "段落", "文档", "信息", "关键", "重要", "遗漏", "偏差"
        ]
        
        keyword_counts = defaultdict(int)
        all_text = ' '.join(reasons)
        
        for keyword in common_keywords:
            count = all_text.count(keyword)
            if count > 0:
                keyword_counts[keyword] = count
        
        # 返回前5个关键词
        return sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _swiss_tournament(self, system_names: List[str], all_systems: Dict[str, List[Dict]], 
                         num_rounds: int = 4) -> Dict[str, Any]:
        """瑞士轮锦标赛 - 4轮比赛，每轮4场，每队每轮只比一场"""
        self.logger.info(f"🔄 开始瑞士轮锦标赛，{num_rounds}轮比赛")
        
        # 初始化所有队伍Elo=1500（无先验信息）
        elo_scores = {system: 1500.0 for system in system_names}
        match_history = set()  # 记录已对战的队伍对
        all_pairwise_results = []
        match_records = []
        total_llm_calls = 0
        
        # 瑞士轮进度条
        tournament_progress = tqdm(range(1, num_rounds + 1), 
                                 desc="🏆 瑞士轮进度", 
                                 unit="轮",
                                 ncols=100,
                                 colour='green')
        
        for round_num in tournament_progress:
            self.logger.info(f"🏁 第{round_num}轮比赛")
            
            # 为当前轮次选择配对
            round_pairs = self._select_swiss_round_pairs(elo_scores, match_history, system_names)
            
            if not round_pairs:
                self.logger.info("无法找到更多有效配对，提前结束")
                tournament_progress.close()
                break
            
            # 执行当前轮次的所有比赛
            for match_idx, (sys_a, sys_b) in enumerate(round_pairs, 1):
                match_num = (round_num - 1) * 4 + match_idx
                self.logger.info(f"  📊 第{match_num}场: {sys_a} (ELO: {elo_scores[sys_a]:.1f}) vs {sys_b} (ELO: {elo_scores[sys_b]:.1f})")
                
                # 记录这场对战
                match_history.add((sys_a, sys_b))
                match_history.add((sys_b, sys_a))  # 双向记录
                
                # 执行对比
                comparison = self._pairwise_comparison(
                    all_systems[sys_a], all_systems[sys_b], sys_a, sys_b,
                    max_questions=self.config.max_questions
                )
                all_pairwise_results.append(comparison)
                total_llm_calls += len(comparison["question_results"])
                
                # 更新Elo分数
                old_elo_a, old_elo_b = elo_scores[sys_a], elo_scores[sys_b]
                self._update_elo_scores_dynamic(elo_scores, comparison, sys_a, sys_b)
                
                # 记录详细比赛信息
                match_records.append({
                    "round": round_num,
                    "match_num": match_num,
                    "system_a": sys_a,
                    "system_b": sys_b,
                    "old_elo_a": old_elo_a,
                    "old_elo_b": old_elo_b,
                    "new_elo_a": elo_scores[sys_a],
                    "new_elo_b": elo_scores[sys_b],
                    "winner": self._determine_winner(comparison),
                    "comparison": comparison
                })
            
            # 输出当前轮次后的排名
            current_ranking = sorted(system_names, key=lambda x: elo_scores[x], reverse=True)
            self.logger.info(f"  第{round_num}轮后排名: {current_ranking[0]}({elo_scores[current_ranking[0]]:.1f}) > {current_ranking[1]}({elo_scores[current_ranking[1]]:.1f}) > {current_ranking[2]}({elo_scores[current_ranking[2]]:.1f})")
            
            # 更新进度条描述
            tournament_progress.set_description(f"🏆 第{round_num}轮完成 - 领先: {current_ranking[0]}")
        
        # 关闭进度条
        tournament_progress.close()
        
        return {
            "match_records": match_records,
            "all_pairwise_results": all_pairwise_results,
            "final_elo_scores": elo_scores,
            "total_llm_calls": total_llm_calls,
            "total_matches": len(match_records),
            "total_rounds": num_rounds
        }
    
    def _select_swiss_round_pairs(self, elo_scores: Dict[str, float], match_history: set, 
                                 system_names: List[str]) -> List[Tuple[str, str]]:
        """为瑞士轮选择当前轮次的配对 - 改进版本"""
        # 生成所有可能的对战组合
        all_possible_pairs = []
        for i, sys_a in enumerate(system_names):
            for sys_b in system_names[i+1:]:
                if (sys_a, sys_b) not in match_history:
                    elo_diff = abs(elo_scores[sys_a] - elo_scores[sys_b])
                    all_possible_pairs.append((sys_a, sys_b, elo_diff))
        
        # 按Elo差距排序（优先选择Elo接近的对战）
        all_possible_pairs.sort(key=lambda x: x[2])
        
        # 使用回溯算法找到最优的4场对战组合
        best_combination = self._find_best_round_combination(all_possible_pairs, len(system_names) // 2)
        
        if best_combination:
            return [(pair[0], pair[1]) for pair in best_combination]
        else:
            self.logger.warning("无法找到有效的瑞士轮配对组合")
            return []
    
    def _find_best_round_combination(self, all_pairs: List[Tuple[str, str, float]], 
                                   target_pairs: int) -> List[Tuple[str, str, float]]:
        """使用回溯算法找到最优的轮次对战组合"""
        def backtrack(used_systems: set, current_pairs: List[Tuple[str, str, float]], 
                     pair_index: int) -> List[Tuple[str, str, float]]:
            # 如果已经找到足够的配对，返回结果
            if len(current_pairs) == target_pairs:
                return current_pairs.copy()
            
            # 如果已经检查完所有可能的配对，返回None
            if pair_index >= len(all_pairs):
                return None
            
            # 尝试包含当前配对
            sys_a, sys_b, elo_diff = all_pairs[pair_index]
            if sys_a not in used_systems and sys_b not in used_systems:
                used_systems.add(sys_a)
                used_systems.add(sys_b)
                current_pairs.append(all_pairs[pair_index])
                
                result = backtrack(used_systems, current_pairs, pair_index + 1)
                if result:
                    return result
                
                # 回溯
                current_pairs.pop()
                used_systems.remove(sys_a)
                used_systems.remove(sys_b)
            
            # 尝试跳过当前配对
            return backtrack(used_systems, current_pairs, pair_index + 1)
        
        # 开始回溯搜索
        result = backtrack(set(), [], 0)
        return result if result else []
    
    def _dynamic_elo_tournament(self, system_names: List[str], all_systems: Dict[str, List[Dict]], 
                               max_matches: int) -> Dict[str, Any]:
        """动态Elo配对锦标赛 - 根据updated_recommandation.md"""
        self.logger.info(f"🔄 开始动态Elo配对锦标赛，最大{max_matches}场比赛")
        
        # 初始化所有队伍Elo=1500（无先验信息）
        elo_scores = {system: 1500.0 for system in system_names}
        match_history = set()  # 记录已对战的队伍对
        all_pairwise_results = []
        match_records = []
        total_llm_calls = 0
        
        # 动态配对直到达到最大场次 - 添加总体进度条
        tournament_progress = tqdm(range(1, max_matches + 1), 
                                 desc="🏆 锦标赛进度", 
                                 unit="场比赛",
                                 ncols=100,
                                 colour='green')
        
        for match_num in tournament_progress:
            self.logger.info(f"🏁 第{match_num}场比赛")
            
            # 选择当前Elo最接近的未对战过的两队
            best_pair = self._find_best_elo_pair(elo_scores, match_history)
            
            if not best_pair:
                self.logger.info("所有可能的对战已完成，提前结束")
                tournament_progress.close()
                break
                
            sys_a, sys_b = best_pair
            self.logger.info(f"  📊 {sys_a} (ELO: {elo_scores[sys_a]:.1f}) vs {sys_b} (ELO: {elo_scores[sys_b]:.1f})")
            
            # 记录这场对战
            match_history.add((sys_a, sys_b))
            match_history.add((sys_b, sys_a))  # 双向记录
            
            # 执行对比
            comparison = self._pairwise_comparison(
                all_systems[sys_a], all_systems[sys_b], sys_a, sys_b,
                max_questions=self.config.max_questions  # 使用用户设置的完整题目数
            )
            all_pairwise_results.append(comparison)
            total_llm_calls += len(comparison["question_results"])
            
            # 更新Elo分数（使用加权算法）
            old_elo_a, old_elo_b = elo_scores[sys_a], elo_scores[sys_b]
            self._update_elo_scores_dynamic(elo_scores, comparison, sys_a, sys_b)
            
            # 记录详细比赛信息
            match_records.append({
                "match_num": match_num,
                "system_a": sys_a,
                "system_b": sys_b,
                "old_elo_a": old_elo_a,
                "old_elo_b": old_elo_b,
                "new_elo_a": elo_scores[sys_a],
                "new_elo_b": elo_scores[sys_b],
                "winner": self._determine_winner(comparison),
                "comparison": comparison
            })
            
            # 输出当前排名（前3名）
            current_ranking = sorted(system_names, key=lambda x: elo_scores[x], reverse=True)
            self.logger.info(f"  当前排名: {current_ranking[0]}({elo_scores[current_ranking[0]]:.1f}) > {current_ranking[1]}({elo_scores[current_ranking[1]]:.1f}) > {current_ranking[2]}({elo_scores[current_ranking[2]]:.1f})")
            
            # 更新进度条描述
            tournament_progress.set_description(f"🏆 第{match_num}场完成 - 领先: {current_ranking[0]}")
            
            # 收敛机制已移除 - 运行完整场次以获得更准确排名
        
        # 关闭进度条
        tournament_progress.close()
        
        return {
            "match_records": match_records,
            "all_pairwise_results": all_pairwise_results,
            "final_elo_scores": elo_scores,
            "total_llm_calls": total_llm_calls,
            "total_matches": len(match_records)
        }
    
    def _find_best_elo_pair(self, elo_scores: Dict[str, float], match_history: set) -> Tuple[str, str]:
        """寻找Elo最接近的未对战过的两队"""
        systems = list(elo_scores.keys())
        best_pair = None
        min_elo_diff = float('inf')
        
        for i, sys_a in enumerate(systems):
            for j, sys_b in enumerate(systems[i+1:], i+1):
                # 检查是否已对战过
                if (sys_a, sys_b) in match_history:
                    continue
                
                # 计算Elo差距
                elo_diff = abs(elo_scores[sys_a] - elo_scores[sys_b])
                
                if elo_diff < min_elo_diff:
                    min_elo_diff = elo_diff
                    best_pair = (sys_a, sys_b)
        
        return best_pair
    
    def _update_elo_scores_dynamic(self, elo_scores: Dict[str, float], 
                                 comparison: Dict[str, Any], sys_a: str, sys_b: str):
        """动态Elo更新（新的soft win评分机制）- 简化版本"""
        summary = comparison["summary"]
        
        # 新的评分机制已经计算好了elo_delta
        elo_delta = summary["elo_delta"]
        
        old_elo_a = elo_scores[sys_a]
        old_elo_b = elo_scores[sys_b]
        
        # 直接应用elo_delta（A获得的分数变化）
        elo_scores[sys_a] += elo_delta
        elo_scores[sys_b] -= elo_delta  # B的变化与A相反
        
        # 记录详细变化信息（便于调试）
        self.logger.debug(f"Elo更新: {sys_a}({old_elo_a:.1f}→{elo_scores[sys_a]:.1f}, +{elo_delta:.1f}) vs {sys_b}({old_elo_b:.1f}→{elo_scores[sys_b]:.1f}, {-elo_delta:.1f})")
    
    def _determine_winner(self, comparison: Dict[str, Any]) -> str:
        """确定比赛胜者"""
        summary = comparison["summary"]
        win_rate_a = summary["win_rate_a"]
        
        if win_rate_a > 0.6:
            return "A"
        elif win_rate_a < 0.4:
            return "B"
        else:
            return "Tie"
    
    # 收敛检查方法已移除 - 使用完整场次评估以获得更准确的排名
    
    def _calculate_dynamic_ranking(self, final_elo_scores: Dict[str, float]) -> List[str]:
        """基于最终Elo分数计算排名"""
        return sorted(final_elo_scores.keys(), key=lambda x: final_elo_scores[x], reverse=True)
    
    def _parse_tournament_rankings(self, tournament_report_path: str = None) -> Dict[str, Dict]:
        """解析锦标赛排名，提取1、5、8名的系统信息"""
        if not tournament_report_path:
            # 使用默认路径
            tournament_report_path = "dice_simplified_output/tournament_report.md"
        
        try:
            with open(tournament_report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析排名
            rankings = {}
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                if '**bge-' in line and '**:' in line:
                    # 提取排名（从当前行直接提取）
                    rank = None
                    
                    # 从行首提取排名数字
                    line_stripped = line.strip()
                    if line_stripped.startswith('1.'):
                        rank = 1
                    elif line_stripped.startswith('2.'):
                        rank = 2
                    elif line_stripped.startswith('3.'):
                        rank = 3
                    elif line_stripped.startswith('4.'):
                        rank = 4
                    elif line_stripped.startswith('5.'):
                        rank = 5
                    elif line_stripped.startswith('6.'):
                        rank = 6
                    elif line_stripped.startswith('7.'):
                        rank = 7
                    elif line_stripped.startswith('8.'):
                        rank = 8
                    
                    if rank is None:
                        continue
                    
                    parts = line.split('**:')
                    if len(parts) == 2:
                        system_name = parts[0].replace('**', '').strip()
                        # 移除排名前缀（如 "1. ", "2. " 等）
                        if '. ' in system_name:
                            system_name = system_name.split('. ', 1)[1]
                        elo_score = float(parts[1].strip().split()[0])
                        
                        rankings[rank] = {
                            'system_name': system_name,
                            'elo_score': elo_score,
                            'rank': rank
                        }
            
            # 确保有1、5、8名
            required_ranks = [1, 5, 8]
            result = {}
            
            for rank in required_ranks:
                if rank in rankings:
                    rank_name = {1: "1st_Place", 5: "5th_Place", 8: "8th_Place"}[rank]
                    result[rank_name] = rankings[rank]
                else:
                    self.logger.warning(f"未找到第{rank}名的系统信息")
            
            self.logger.info(f"解析到锦标赛排名: {list(result.keys())}")
            return result
            
        except Exception as e:
            self.logger.error(f"解析锦标赛排名失败: {e}")
            # 返回默认的虚拟基线
            return {
                "1st_Place": {"system_name": "tournament_1st", "elo_score": 1520.0, "rank": 1},
                "5th_Place": {"system_name": "tournament_5th", "elo_score": 1495.0, "rank": 5},
                "8th_Place": {"system_name": "tournament_8th", "elo_score": 1480.0, "rank": 8}
            }
    
    def _create_tournament_baseline_data(self, target_data: List[Dict], baseline_info: Dict) -> Tuple[List[Dict], int]:
        """基于锦标赛排名创建基线数据"""
        baseline_name = baseline_info['system_name']
        elo_score = baseline_info['elo_score']
        rank = baseline_info['rank']
        
        # 根据排名调整生成质量
        if rank == 1:
            quality_level = "high"
            instruction = f"作为锦标赛第1名的系统({baseline_name}, Elo: {elo_score:.1f})，请生成高质量回答。要求：1)提供完整准确的信息，2)逻辑清晰条理分明，3)基于权威资料，4)表述专业准确。"
        elif rank == 5:
            quality_level = "medium"
            instruction = f"作为锦标赛第5名的系统({baseline_name}, Elo: {elo_score:.1f})，请生成中等质量回答。要求：1)包含主要信息但可能缺少细节，2)表述基本准确但不够深入，3)信息完整性中等。"
        else:  # rank == 8
            quality_level = "low"
            instruction = f"作为锦标赛第8名的系统({baseline_name}, Elo: {elo_score:.1f})，请生成较低质量回答。要求：1)信息不够准确或有遗漏，2)表述可能含糊不清，3)可能包含错误或无关信息。"
        
        # 生成基线数据
        baseline_data = []
        generation_calls = 0
        
        for item in target_data:
            question = item['question']
            groundtruth = item['groundtruth']
            
            # 生成基线回答
            baseline_answer = self._generate_baseline_answer(question, groundtruth, instruction, quality_level)
            generation_calls += 1
            
            # 生成基线上下文
            baseline_contexts = self._generate_baseline_contexts(question, groundtruth, quality_level)
            generation_calls += 3  # 3个上下文
            
            baseline_data.append({
                'question': question,
                'groundtruth': groundtruth,
                'answer': baseline_answer,
                'context': baseline_contexts
            })
        
        return baseline_data, generation_calls
    
    def _summarize_tournament_baseline_comparison(self, baseline_results: Dict, target_system: str) -> Dict:
        """总结锦标赛基线对比结果"""
        summary = {
            "target_system": target_system,
            "comparisons": {}
        }
        
        for rank_name, result in baseline_results.items():
            baseline_info = result["baseline_info"]
            comparison = result["comparison"]
            
            # 计算胜率
            total_questions = len(comparison["question_results"])
            wins = sum(1 for qr in comparison["question_results"] 
                      if qr["passage_judgment"]["win_type"] == "A wins")
            ties = sum(1 for qr in comparison["question_results"] 
                      if qr["passage_judgment"]["win_type"] == "Tie")
            
            win_rate = wins / total_questions if total_questions > 0 else 0
            tie_rate = ties / total_questions if total_questions > 0 else 0
            
            # 判断结论
            if win_rate > 0.6:
                conclusion = f"显著优于{rank_name}"
            elif win_rate > 0.4:
                conclusion = f"略优于{rank_name}"
            elif win_rate > 0.2:
                conclusion = f"与{rank_name}相当"
            else:
                conclusion = f"不如{rank_name}"
            
            summary["comparisons"][rank_name] = {
                "baseline_system": baseline_info["system_name"],
                "baseline_elo": baseline_info["elo_score"],
                "baseline_rank": baseline_info["rank"],
                "win_rate": win_rate,
                "tie_rate": tie_rate,
                "total_questions": total_questions,
                "conclusion": conclusion
            }
        
        return summary
    
    def scenario_b_baseline_comparison(self, qacg_file: str, target_system: str = None, 
                                     tournament_report_path: str = None) -> Dict[str, Any]:
        """
        场景B: 单系统vs锦标赛排名基线
        
        Args:
            qacg_file: 目标系统的QACG文件
            target_system: 系统名称（可选）
            tournament_report_path: 锦标赛报告文件路径（可选）
            
        Returns:
            基线对比结果
        """
        self.logger.info("🎯 开始场景B: 单系统vs锦标赛排名基线")
        
        # 1. 加载目标系统
        target_data = self._load_qacg_file(qacg_file)
        if not target_system:
            target_system = Path(qacg_file).stem.replace("qacg_", "")
        
        # 2. 解析锦标赛排名
        tournament_rankings = self._parse_tournament_rankings(tournament_report_path)
        
        # 3. 与锦标赛排名基线对比
        baseline_results = {}
        total_calls = 0
        
        for rank_name, baseline_info in tournament_rankings.items():
            self.logger.info(f"🔄 {target_system} vs {rank_name} ({baseline_info['system_name']}) 对比")
            
            # 构造基线数据
            baseline_data, baseline_generation_calls = self._create_tournament_baseline_data(
                target_data, baseline_info
            )
            
            # 执行对比
            comparison_result = self._pairwise_comparison(
                target_data, baseline_data, 
                f"{target_system}", f"{rank_name}_{baseline_info['system_name']}",
                max_questions=self.config.max_questions
            )
            
            # 保存基线数据以供详细对比使用
            comparison_result["baseline_data"] = baseline_data
            comparison_result["baseline_generation_calls"] = baseline_generation_calls
            comparison_result["baseline_info"] = baseline_info
            baseline_results[rank_name] = comparison_result
            total_calls += len(comparison_result["question_results"]) + baseline_generation_calls
        
        # 4. 统计分析
        comparison_summary = self._summarize_tournament_baseline_comparison(baseline_results, target_system)
        
        # 5. 生成详细QACG对比数据
        detailed_qacg_comparisons = self._generate_detailed_qacg_comparisons(target_data, target_system, baseline_results)
        
        result = {
            "config": self._config_to_dict(),
            "target_system": target_system,
            "tournament_rankings": tournament_rankings,
            "baseline_comparisons": baseline_results,
            "summary": comparison_summary,
            "detailed_qacg_comparisons": detailed_qacg_comparisons,
            "total_llm_calls": total_calls
        }
        
        # 保存结果
        self._save_baseline_result(result)
        return result
    
    def _load_systems(self, qacg_files: List[str]) -> Dict[str, List[Dict]]:
        """加载所有系统数据"""
        systems = {}
        for file_path in qacg_files:
            system_name = Path(file_path).stem.replace("qacg_", "")
            systems[system_name] = self._load_qacg_file(file_path)
        return systems
    
    def _load_qacg_file(self, file_path: str) -> List[Dict]:
        """加载QACG文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data[:self.config.max_questions]  # 限制题目数量
    
    def _create_groups(self, system_names: List[str]) -> List[List[str]]:
        """创建分组 (根据系统数量自动分组)"""
        # 简单按顺序分组，实际可以根据预期实力分组
        mid = len(system_names) // 2
        return [system_names[:mid], system_names[mid:]]
    
    def _group_stage(self, group_systems: List[str], all_systems: Dict[str, List[Dict]], 
                    stage_name: str = "小组赛") -> Dict[str, Any]:
        """组内对战"""
        self.logger.info(f"🔄 {stage_name}: {group_systems}")
        
        # 初始化Elo分数
        elo_scores = {system: self.config.initial_elo for system in group_systems}
        pairwise_results = []
        total_calls = 0
        
        # 所有两两对战
        for sys_a, sys_b in itertools.combinations(group_systems, 2):
            self.logger.info(f"  📊 {sys_a} vs {sys_b}")
            
            # 执行对比
            comparison = self._pairwise_comparison(
                all_systems[sys_a], all_systems[sys_b], sys_a, sys_b
            )
            pairwise_results.append(comparison)
            total_calls += len(comparison["question_results"])
            
            # 更新Elo分数
            self._update_elo_scores(elo_scores, comparison, sys_a, sys_b)
        
        # 排名
        ranking = sorted(group_systems, key=lambda x: elo_scores[x], reverse=True)
        
        return {
            "stage": stage_name,
            "systems": group_systems,
            "pairwise_results": pairwise_results,
            "elo_scores": elo_scores,
            "ranking": ranking,
            "total_llm_calls": total_calls
        }
    
    def _judge_single_question(self, question_data: Tuple[int, Dict, Dict, str]) -> Tuple[int, Dict[str, Any]]:
        """
        判决单个问题（用于并发处理）- 使用新的soft win机制
        
        Args:
            question_data: (index, qa_a, qa_b, groundtruth)
            
        Returns:
            (index, question_result): 索引和判决结果
        """
        i, qa_a, qa_b, groundtruth = question_data
        
        try:
            # 只进行passage粒度判决
            question = qa_a["question"]
            expected_answer = qa_a.get("expected_answer", "")
            
            # # 打印当前问题的标准答案和正确证据
            # print(f"\n📋 问题 {i+1}: {question}")
            # print(f"📝 标准答案: {expected_answer}")
            # print(f"📄 正确证据: {groundtruth}")
            # print("-" * 80)
            
            # 构建passage粒度对比
            passage_judgment = self._judge_passage_only(question, qa_a, qa_b, groundtruth)
            
            # 计算soft win得分
            score_a, score_b = self._calculate_soft_win_score(passage_judgment)
            
            question_result = {
                "question": question,
                "passage_judgment": passage_judgment,
                "score_a": score_a,
                "score_b": score_b,
                "winner": passage_judgment.get("win_type", "Tie"),
                "index": i  # 保持原始顺序
            }
            
            return i, question_result
            
        except Exception as e:
            # 处理异常情况
            self.logger.error(f"问题 {i+1} 判决失败: {e}")
            error_result = {
                "question": qa_a.get("question", ""),
                "passage_judgment": {
                    "label": "Tie",
                    "reason": f"判决失败: {str(e)}",
                    "score": 0.5,
                    "margin_score": 0.0,
                    "granularity": "passage",
                    "logit_a": 0.0,
                    "logit_b": 0.0,
                    "logit_t": 0.0,
                    "prob_a": 0.33,
                    "prob_b": 0.33,
                    "prob_t": 0.33,
                    "win_type": "Error tie",
                    "score_a": 0.5,
                    "score_b": 0.5,
                    "prob_diff": 0.0
                },
                "score_a": 0.5,
                "score_b": 0.5,
                "winner": "Error tie",
                "index": i
            }
            return i, error_result

    def _pairwise_comparison(self, data_a: List[Dict], data_b: List[Dict], 
                           name_a: str, name_b: str, max_questions: int = None) -> Dict[str, Any]:
        """执行成对比较（支持并发处理）"""
        if max_questions is None:
            max_questions = self.config.max_questions
        
        total_questions = min(len(data_a), len(data_b), max_questions)
        
        self.logger.info(f"🚀 开始并发处理 {total_questions} 个问题...")
        self.logger.info(f"⚙️ 并发配置: {self.config.max_workers} workers, 批大小: {self.config.batch_size}")
        
        # 准备所有问题数据
        questions_data = []
        for i in range(total_questions):
            qa_a = data_a[i]
            qa_b = data_b[i]
            groundtruth = qa_a.get("groundtruth", qa_a.get("expected_answer", ""))
            questions_data.append((i, qa_a, qa_b, groundtruth))
        
        # 并发处理 - 添加问题处理进度条
        question_results = []
        completed_count = 0
        
        # 创建问题级进度条
        question_progress = tqdm(total=total_questions, 
                               desc=f"📝 {name_a} vs {name_b}", 
                               unit="题",
                               ncols=100,
                               colour='blue',
                               leave=False)
        
        # 分批处理
        for batch_start in range(0, len(questions_data), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(questions_data))
            batch_data = questions_data[batch_start:batch_end]
            
            self.logger.info(f"🔄 处理批次 {batch_start//self.config.batch_size + 1}: 问题 {batch_start+1}-{batch_end}")
            
            # 使用ThreadPoolExecutor进行并发处理
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(batch_data))) as executor:
                # 提交任务
                future_to_index = {executor.submit(self._judge_single_question, question_data): question_data[0] 
                                 for question_data in batch_data}
                
                # 收集结果
                batch_results = []
                for future in concurrent.futures.as_completed(future_to_index):
                    try:
                        i, result = future.result()
                        batch_results.append((i, result))
                        completed_count += 1
                        
                        # 更新进度条
                        question_progress.update(1)
                        winner = result.get("winner", "Unknown")
                        question_progress.set_description(f"📝 {name_a} vs {name_b} - 最新: {winner}")
                        
                        # 输出详细的判决结果（重要！包含理由等信息）
                        with self._lock:
                            self._log_question_result(result, completed_count, total_questions)
                            
                    except Exception as e:
                        i = future_to_index[future]
                        self.logger.error(f"问题 {i+1} 处理异常: {e}")
                
                # 按原始顺序排序
                batch_results.sort(key=lambda x: x[0])
                question_results.extend([result for _, result in batch_results])
            
            # 早停机制已移除 - 按用户要求去除所有收敛/早停机制
        
        # 关闭问题进度条
        question_progress.close()
        
        self.logger.info(f"✅ 并发处理完成，共处理 {len(question_results)} 个问题")
        
        # 汇总结果 - 使用新的累计评分机制
        summary = self._summarize_pairwise_result_with_soft_win(question_results, name_a, name_b)
        
        return {
            "system_a": name_a,
            "system_b": name_b,
            "question_results": question_results,
            "summary": summary
        }
    
    def _judge_passage_only(self, question: str, qa_a: Dict, qa_b: Dict, groundtruth: str) -> Dict[str, Any]:
        """仅进行passage粒度判决（检索-证据双通道）"""
        # 构建检索-证据双通道prompt
        context_a = qa_a.get("context", [])
        context_b = qa_b.get("context", [])
        answer_a = qa_a.get("rag_answer", "")
        answer_b = qa_b.get("rag_answer", "")
        expected_answer = qa_a.get("expected_answer", "")
        
        # 简化的passage级判决prompt
        prompt = f"""作为RAG系统评估专家，请对比两个系统的检索-回答质量。

问题: {question}
标准答案: {groundtruth}

系统A:
检索证据: {' '.join(context_a[:3])}  
回答: {answer_a}

系统B:
检索证据: {' '.join(context_b[:3])}
回答: {answer_b}

请从以下角度对比:
1. 检索证据的相关性和完整性
2. 回答的准确性和逻辑性
3. 证据与回答的一致性
4. 在一方给出答案，另一方回答"信息不足"的情况下，要是给出答案的那一方答案完全错误（与标准答案完全不一致），算信息不足的一方赢
5. 对于答案质量请遵守如下法则：完全答对>部分答对>部分答错>信息不足>完全错误


判决格式：
判决: [A wins/B wins/Tie]
理由: [基于上述原则的具体分析]"""

        try:
            # 使用judge_pair获得深度思考结果
            judge_result = self.pairwise_judge.judge_pair(
                question=question,
                qa_a={
                    "rag_answer": answer_a, 
                    "retrieved_docs": context_a,
                    "expected_answer": expected_answer,
                    "groundtruth": groundtruth
                },
                qa_b={
                    "rag_answer": answer_b, 
                    "retrieved_docs": context_b,
                    "expected_answer": expected_answer,  # 两个系统的标准答案相同
                    "groundtruth": groundtruth  # 两个系统的标准证据相同
                },
                granularity="passage",
                atoms={}
            )
            
            # 从深度判决结果中提取信息
            label = judge_result.get("label", "Tie")
            response = judge_result.get("reason", "")
            logit_a = judge_result.get("logit_a", 0.0)
            logit_b = judge_result.get("logit_b", 0.0)
            logit_t = judge_result.get("logit_t", 0.0)
            prob_a = judge_result.get("prob_a", 0.33)
            prob_b = judge_result.get("prob_b", 0.33)
            prob_t = judge_result.get("prob_t", 0.33)
            
            # 简化日志：只输出关键信息
            # self.logger.info(f"🔍 从judge_result获取的logits: A={logit_a}, B={logit_b}, T={logit_t}")
            # self.logger.info(f"🔍 从judge_result获取的概率: A={prob_a:.3f}, B={prob_b:.3f}, T={prob_t:.3f}")
            # self.logger.info(f"🔍 judge_result所有键: {list(judge_result.keys())}")
            
            # 🔧 改进理由解析逻辑
            reason = "基于LLM判决结果"  # 默认描述
            lines = response.strip().split('\n')
            
            # 多种方式尝试提取理由
            found_reason = False
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # 匹配各种理由格式
                if (line.lower().startswith("理由:") or line.lower().startswith("理由：") or 
                    line.lower().startswith("reason:") or line.lower().startswith("原因:")):
                    extracted_reason = line.split(":", 1)[-1].split("：", 1)[-1].strip()
                    if extracted_reason:  # 确保提取到的理由不为空
                        reason = extracted_reason
                        found_reason = True
                        break
                elif "因为" in line or "由于" in line or "所以" in line:
                    reason = line.strip()
                    found_reason = True
                    break
                elif line.startswith("-") or line.startswith("*"):
                    # 可能是列表格式的理由
                    reason = line[1:].strip()
                    found_reason = True
                    break
            
            # 🔧 关键修复：如果没有找到标准格式的理由，使用整个响应的摘要
            if not found_reason and len(response.strip()) > 0:
                # 从完整响应中提取有意义的内容作为理由
                response_clean = response.strip()
                
                # 清理废话：去除无用的选择提示
                unwanted_patterns = [
                    "A\n", "B\n", "T\n", "请根据以上信息，给出判决",
                    "判决: [A wins/B wins/Tie]", "理由: [简要说明原因]",
                    "你的选择是（只输出一个字母）：", "请选择：",
                    "A - 系统A更好", "B - 系统B更好", "T - 两系统相当"
                ]
                
                for pattern in unwanted_patterns:
                    response_clean = response_clean.replace(pattern, "")
                
                # 清理多余的换行和空格
                response_clean = " ".join(response_clean.split())
                
                # 如果响应很长，提取关键部分，但不截断
                if len(response_clean) > 300:
                    # 寻找判决相关的关键句子
                    key_sentences = []
                    for line in lines:
                        line = line.strip()
                        # 跳过废话行
                        if line in ["A", "B", "T", ""] or any(unwanted in line for unwanted in unwanted_patterns):
                            continue
                        if any(keyword in line for keyword in ["系统A", "系统B", "更优", "更好", "胜出", "准确", "完整", "相关", "一致"]):
                            key_sentences.append(line)
                    
                    if key_sentences:
                        reason = " ".join(key_sentences)  # 取所有关键句子，不截断
                    else:
                        reason = response_clean  # 保留完整响应，不截断
                else:
                    reason = response_clean
            
            # 如果仍然没有找到合适的理由，尝试用判决的逻辑
            if reason == "基于LLM判决结果" and label != "Tie":
                if label == "A wins":
                    reason = "系统A在评估指标上表现更优"
                elif label == "B wins":
                    reason = "系统B在评估指标上表现更优"
            
            # 计算margin_score（Margin-Aware Tie）
            # 修复逻辑：只有当T的概率不是最高时，才考虑A/B的margin
            if label == "Tie":
                # 当判决为Tie时，检查是否真的是明显的平局
                # 如果T的概率确实最高，就保持Tie；否则考虑A/B的细微差别
                max_prob = max(prob_a, prob_b, prob_t)
                if max_prob == prob_t:
                    # T概率最高，确实应该是Tie
                    margin_score = 0.0
                    score = 0.5
                else:
                    # A或B概率最高但被误判为Tie，使用margin_score微调
                    margin_score = self._calculate_margin_score(logit_a, logit_b)
                    if abs(margin_score) > 0.05:  # margin_threshold
                        score = 0.5 + margin_score
                        if margin_score > 0:
                            label = "A soft wins"
                        else:
                            label = "B soft wins"
                    else:
                        score = 0.5
            else:
                # 非Tie判决，计算margin_score用于记录
                margin_score = self._calculate_margin_score(logit_a, logit_b)
                score = 1.0 if label == "A wins" else (0.0 if label == "B wins" else 0.5)
            
            return {
                "label": label,
                "reason": reason,
                "score": score,
                "margin_score": margin_score,
                "raw_response": response,
                "granularity": "passage",
                "logit_a": logit_a,
                "logit_b": logit_b,
                "logit_t": logit_t,
                "prob_a": prob_a,
                "prob_b": prob_b,
                "prob_t": prob_t
            }
            
        except Exception as e:
            self.logger.error(f"Passage判决失败: {e}")
            return {
                "label": "Tie",
                "reason": f"判决失败: {str(e)}",
                "score": 0.5,
                "margin_score": 0.0,
                "raw_response": "",
                "granularity": "passage",
                "logit_a": 0.0,
                "logit_b": 0.0,
                "logit_t": 0.0,
                "prob_a": 0.33,
                "prob_b": 0.33,
                "prob_t": 0.33
            }
    
    def _calculate_margin_score(self, logit_a: float, logit_b: float) -> float:
        """计算Margin-Aware Tie的margin_score - 直接使用logits"""
        # 计算 logit_A - logit_B 的差值
        logit_diff = logit_a - logit_b
        
        # 经温度 0.1 的 softmax 映射到 (0,1)
        temperature = 0.1
        margin_raw = 1.0 / (1.0 + math.exp(-logit_diff / temperature))
        
        # 映射到(-0.5, 0.5)范围，用于调整score
        margin_score = (margin_raw - 0.5)
        
        return margin_score
    
    def _calculate_soft_win_score(self, passage_judgment: Dict[str, Any]) -> Tuple[float, float]:
        """
        计算soft win得分机制
        
        Args:
            passage_judgment: 包含prob_a, prob_b, prob_t的判决结果
            
        Returns:
            (score_a, score_b): A和B系统的得分
        """
        prob_a = passage_judgment.get("prob_a", 0.33)
        prob_b = passage_judgment.get("prob_b", 0.33)
        prob_t = passage_judgment.get("prob_t", 0.33)
        
        # 找出最高概率和次高概率
        probs_sorted = sorted([prob_a, prob_b, prob_t], reverse=True)
        max_prob = probs_sorted[0]
        second_prob = probs_sorted[1]
        
        # 计算概率差距
        prob_diff = max_prob - second_prob
        
        # 阈值0.1判断是hard win还是soft win
        if prob_diff >= 0.1:
            # Hard win: 胜者得1分，败者得0分
            if max_prob == prob_a:
                score_a, score_b = 1.0, 0.0
                win_type = "A hard wins"
            elif max_prob == prob_b:
                score_a, score_b = 0.0, 1.0
                win_type = "B hard wins"
            else:  # prob_t是最高
                score_a, score_b = 0.5, 0.5
                win_type = "Hard tie"
        else:
            # Soft win: 使用概率作为得分，但只在A和B之间分配
            # 将T的概率按比例分配给A和B
            if prob_a + prob_b > 0:
                total_ab = prob_a + prob_b
                # 将T概率按A和B的相对比例分配
                score_a = prob_a + prob_t * (prob_a / total_ab)
                score_b = prob_b + prob_t * (prob_b / total_ab)
            else:
                score_a, score_b = 0.5, 0.5
            
            # 确保分数在[0,1]范围内
            score_a = max(0.0, min(1.0, score_a))
            score_b = max(0.0, min(1.0, score_b))
            
            if score_a > score_b:
                win_type = "A soft wins"
            elif score_b > score_a:
                win_type = "B soft wins"
            else:
                win_type = "Soft tie"
        
        # 记录到judgment中用于日志显示
        passage_judgment["win_type"] = win_type
        passage_judgment["score_a"] = score_a
        passage_judgment["score_b"] = score_b
        passage_judgment["prob_diff"] = prob_diff
        
        return score_a, score_b
    
    def _summarize_pairwise_result_with_soft_win(self, question_results: List[Dict], name_a: str, name_b: str) -> Dict[str, Any]:
        """
        汇总成对比较结果 - 使用新的soft win累计评分机制
        
        Args:
            question_results: 问题判决结果列表
            name_a: 系统A名称
            name_b: 系统B名称
            
        Returns:
            汇总结果，包含累计得分和Elo更新
        """
        if not question_results:
            return {
                "total_score_a": 0.0,
                "total_score_b": 0.0,
                "elo_delta": 0.0,
                "winner": "Tie",
                "confidence": 0.0,
                "question_details": []
            }
        
        # 累计所有问题的得分
        total_score_a = sum(result["score_a"] for result in question_results)
        total_score_b = sum(result["score_b"] for result in question_results)
        total_questions = len(question_results)
        
        # 计算平均得分率
        avg_score_a = total_score_a / total_questions
        avg_score_b = total_score_b / total_questions
        
        # 基于累计得分差距计算Elo更新
        score_diff = total_score_a - total_score_b
        
        # 将得分差距转换为胜率用于Elo计算
        # 得分范围：[-total_questions, +total_questions]
        # 转换为胜率范围：[0, 1]
        max_diff = total_questions
        normalized_diff = score_diff / max_diff  # [-1, 1]
        
        # 使用sigmoid函数将差距转换为胜率
        # 这样可以平滑处理各种得分差距
        import math
        win_rate_a = 1 / (1 + math.exp(-5 * normalized_diff))  # 5是调节参数，控制转换的陡峭程度
        
        # 计算Elo更新 - 使用标准Elo公式
        k_factor = self.config.k_factor
        elo_delta = k_factor * (win_rate_a - 0.5)
        
        # 确定胜者
        if abs(score_diff) < 0.1:  # 非常接近
            winner = "Tie"
            confidence = 0.5 + abs(score_diff) / (2 * max_diff)
        elif score_diff > 0:
            winner = f"{name_a} wins"
            confidence = win_rate_a
        else:
            winner = f"{name_b} wins"
            confidence = 1 - win_rate_a
        
        # 统计不同类型的判决
        hard_wins_a = sum(1 for r in question_results if r["passage_judgment"].get("win_type", "").startswith("A hard"))
        hard_wins_b = sum(1 for r in question_results if r["passage_judgment"].get("win_type", "").startswith("B hard"))
        soft_wins_a = sum(1 for r in question_results if r["passage_judgment"].get("win_type", "").startswith("A soft"))
        soft_wins_b = sum(1 for r in question_results if r["passage_judgment"].get("win_type", "").startswith("B soft"))
        ties = sum(1 for r in question_results if "tie" in r["passage_judgment"].get("win_type", "").lower())
        
        self.logger.info(f"🏆 累计评分结果:")
        self.logger.info(f"  📊 总分: {name_a}={total_score_a:.2f}, {name_b}={total_score_b:.2f} (共{total_questions}题)")
        self.logger.info(f"  📈 平均得分率: {name_a}={avg_score_a:.3f}, {name_b}={avg_score_b:.3f}")
        self.logger.info(f"  🎯 判决统计: A硬胜{hard_wins_a}, A软胜{soft_wins_a}, B硬胜{hard_wins_b}, B软胜{soft_wins_b}, 平局{ties}")
        self.logger.info(f"  ⚖️ Elo更新: {elo_delta:.1f} ({winner}, 置信度{confidence:.3f})")
        
        return {
            "total_score_a": total_score_a,
            "total_score_b": total_score_b,
            "avg_score_a": avg_score_a,
            "avg_score_b": avg_score_b,
            "score_diff": score_diff,
            "win_rate_a": win_rate_a,
            "elo_delta": elo_delta,
            "winner": winner,
            "confidence": confidence,
            "total_questions": total_questions,
            "hard_wins_a": hard_wins_a,
            "hard_wins_b": hard_wins_b,
            "soft_wins_a": soft_wins_a,
            "soft_wins_b": soft_wins_b,
            "ties": ties,
            "question_details": question_results
        }
    
    # 早停方法已移除 - 按用户要求去除所有收敛/早停机制
    
    def _update_elo_scores(self, elo_scores: Dict[str, float], 
                         comparison: Dict[str, Any], sys_a: str, sys_b: str):
        """更新Elo分数"""
        summary = comparison["summary"]
        win_rate_a = summary["win_rate_a"]
        win_rate_b = summary["win_rate_b"]
        
        # 计算期望胜率
        expected_a = 1 / (1 + 10 ** ((elo_scores[sys_b] - elo_scores[sys_a]) / 400))
        expected_b = 1 - expected_a
        
        # 更新Elo
        k = self.config.k_factor
        elo_scores[sys_a] += k * (win_rate_a - expected_a)
        elo_scores[sys_b] += k * (win_rate_b - expected_b)
    
    def _summarize_pairwise_result(self, question_results: List[Dict], 
                                 name_a: str, name_b: str) -> Dict[str, Any]:
        """汇总成对比较结果"""
        total_questions = len(question_results)
        a_wins = sum(1 for r in question_results if r["winner"] == "A wins")
        b_wins = sum(1 for r in question_results if r["winner"] == "B wins")
        ties = total_questions - a_wins - b_wins
        
        return {
            "total_questions": total_questions,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "win_rate_a": a_wins / total_questions if total_questions > 0 else 0,
            "win_rate_b": b_wins / total_questions if total_questions > 0 else 0,
            "tie_rate": ties / total_questions if total_questions > 0 else 0,
            "avg_elo_delta": np.mean([r["elo_delta"] for r in question_results]) if question_results else 0
        }
    
    def _create_baseline_data(self, target_data: List[Dict], baseline_name: str) -> Tuple[List[Dict], int]:
        """创建基线对比数据 - 使用LLM生成真实的QACG对"""
        self.logger.info(f"生成 {baseline_name} 基线的真实QACG数据...")
        baseline_data = []
        baseline_prompt = self.baseline_prompts[baseline_name]
        llm_calls = 0
        
        for i, qa in enumerate(target_data):
            question = qa["question"]
            groundtruth = qa.get("groundtruth", qa.get("expected_answer", ""))
            
            self.logger.info(f"  生成第 {i+1}/{len(target_data)} 个{baseline_name}基线回答")
            
            # 生成基线回答
            generated_answer = self._generate_baseline_answer(question, groundtruth, baseline_prompt)
            llm_calls += 1
            
            # 生成基线检索证据
            generated_context = self._generate_baseline_context(question, groundtruth, baseline_prompt)
            llm_calls += 1
            
            baseline_qa = {
                "question": question,
                "rag_answer": generated_answer,
                "context": generated_context,
                "groundtruth": groundtruth,
                "metadata": {
                    "system_type": "baseline",
                    "baseline_quality": baseline_name.lower(),
                    "generated_by": "llm_baseline_generator"
                }
            }
            baseline_data.append(baseline_qa)
        
        return baseline_data, llm_calls
    
    def _generate_baseline_answer(self, question: str, groundtruth: str, baseline_prompt: Dict) -> str:
        """使用LLM生成基线回答"""
        prompt = f"""
{baseline_prompt["instruction"]}

问题: {question}
参考标准答案: {groundtruth}

请基于上述要求生成一个{baseline_prompt["quality_level"]}质量的回答:
"""
        
        try:
            response = self.pairwise_judge._call_llm(prompt)
            return response.strip()
        except Exception as e:
            self.logger.error(f"生成基线回答失败: {e}")
            # 降级到默认回答
            fallback_answers = {
                "high": f"基于相关资料，{groundtruth}",
                "medium": f"根据信息显示，{groundtruth[:len(groundtruth)//2]}...",
                "low": "信息不够明确，可能需要更多资料。"
            }
            return fallback_answers.get(baseline_prompt["quality_level"], "无法生成回答")
    
    def _generate_baseline_context(self, question: str, groundtruth: str, baseline_prompt: Dict) -> List[str]:
        """使用LLM生成基线检索证据"""
        prompt = f"""
{baseline_prompt["context_instruction"]}

问题: {question}
参考信息: {groundtruth}

请生成3条符合{baseline_prompt["quality_level"]}质量要求的检索证据，每条证据应该独立成段：

证据1：
证据2：
证据3：
"""
        
        try:
            response = self.pairwise_judge._call_llm(prompt)
            # 解析响应，提取3条证据
            lines = response.strip().split('\n')
            contexts = []
            current_context = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("证据") and "：" in line:
                    if current_context:
                        contexts.append(current_context.strip())
                    current_context = line.split("：", 1)[1]
                elif line and not line.startswith("证据"):
                    current_context += " " + line
            
            if current_context:
                contexts.append(current_context.strip())
            
            # 确保有3条证据
            while len(contexts) < 3:
                fallback_contexts = {
                    "high": f"这是基于权威资料的高质量证据，详细说明了{question}的相关信息。",
                    "medium": f"这是关于{question}的基本信息，提供了部分相关内容。",
                    "low": f"这是与{question}相关的一般性信息，可能不够准确。"
                }
                contexts.append(fallback_contexts.get(baseline_prompt["quality_level"], "相关信息不足"))
            
            return contexts[:3]
            
        except Exception as e:
            self.logger.error(f"生成基线证据失败: {e}")
            # 降级到默认证据
            fallback_contexts = {
                "high": [
                    f"权威资料显示，{groundtruth[:50]}...",
                    f"详细分析表明，{question}涉及多个方面的考量。",
                    "基于可靠来源的信息，以上内容具有较高准确性。"
                ],
                "medium": [
                    f"相关信息表明，{groundtruth[:30]}...",
                    f"关于{question}的基本信息如上所述。",
                    "这些信息基本准确但可能不够完整。"
                ],
                "low": [
                    f"据了解，{groundtruth[:20]}...",
                    f"关于{question}的信息可能不够准确。",
                    "需要进一步验证相关内容的准确性。"
                ]
            }
            return fallback_contexts.get(baseline_prompt["quality_level"], ["信息不足"])
    
    def _summarize_baseline_comparison(self, baseline_results: Dict[str, Any], 
                                     target_system: str) -> Dict[str, Any]:
        """汇总基线对比结果"""
        summary = {
            "target_system": target_system,
            "comparisons": {}
        }
        
        for baseline_name, result in baseline_results.items():
            win_rate = result["summary"]["win_rate_a"]  # target系统的胜率
            total_questions = result["summary"]["total_questions"]
            
            # 统计显著性简化判断
            if win_rate > 0.6:
                conclusion = f"显著优于{baseline_name}基线"
            elif win_rate < 0.4:
                conclusion = f"显著劣于{baseline_name}基线"
            else:
                conclusion = f"与{baseline_name}基线相当"
            
            summary["comparisons"][baseline_name] = {
                "win_rate": win_rate,
                "total_questions": total_questions,
                "conclusion": conclusion
            }
        
        return summary
    
    def _generate_detailed_qacg_comparisons(self, target_data: List[Dict], target_system: str, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成详细的QACG对比数据 - 重用已生成的基线数据"""
        self.logger.info("整理详细QACG对比数据...")
        
        detailed_comparisons = {
            "target_system": target_system,
            "total_questions": len(target_data),
            "qacg_pairs": []
        }
        
        # 限制输出数量以避免文件过大
        sample_size = min(len(target_data), self.config.max_questions)
        
        # 从baseline_results中提取已生成的基线数据
        baseline_data_by_name = {}
        for baseline_name, result in baseline_results.items():
            baseline_data_by_name[baseline_name] = result.get("baseline_data", [])
        
        for i, target_qa in enumerate(target_data[:sample_size]):
            question = target_qa["question"]
            
            # 构建对比对
            qacg_pair = {
                "question_id": i + 1,
                "question": question,
                "groundtruth": target_qa.get("groundtruth", target_qa.get("expected_answer", "")),
                
                # 目标系统的QACG
                "target_system": {
                    "name": target_system,
                    "answer": target_qa.get("rag_answer", ""),
                    "context": target_qa.get("context", []),
                    "metadata": target_qa.get("metadata", {})
                },
                
                # 各个基线的QACG
                "baselines": {}
            }
            
            # 使用已生成的基线数据
            for baseline_name in self.baseline_prompts.keys():
                if baseline_name in baseline_data_by_name and i < len(baseline_data_by_name[baseline_name]):
                    baseline_qa = baseline_data_by_name[baseline_name][i]
                    baseline_qacg = {
                        "name": f"Baseline_{baseline_name}",
                        "answer": baseline_qa.get("rag_answer", ""),
                        "context": baseline_qa.get("context", []),
                        "quality_level": baseline_name.lower(),
                        "description": self._get_baseline_description(baseline_name),
                        "generation_instruction": self.baseline_prompts[baseline_name]["instruction"],
                        "metadata": baseline_qa.get("metadata", {})
                    }
                else:
                    # 备用基线数据（如果出现数据不匹配）
                    baseline_qacg = {
                        "name": f"Baseline_{baseline_name}",
                        "answer": f"未能生成{baseline_name}质量的基线回答",
                        "context": [f"未能生成{baseline_name}质量的基线证据"],
                        "quality_level": baseline_name.lower(),
                        "description": self._get_baseline_description(baseline_name),
                        "generation_instruction": self.baseline_prompts[baseline_name]["instruction"],
                        "metadata": {"error": "baseline_generation_failed"}
                    }
                
                qacg_pair["baselines"][baseline_name] = baseline_qacg
            
            detailed_comparisons["qacg_pairs"].append(qacg_pair)
        
        return detailed_comparisons
    
    def _get_baseline_description(self, baseline_name: str) -> str:
        """获取基线描述"""
        descriptions = {
            "Good": "高质量基线：提供详细准确的回答，包含完整关键信息，逻辑清晰",
            "Medium": "中等质量基线：提供基本正确但不够详细的回答，存在信息缺失", 
            "Bad": "低质量基线：回答不够准确，存在明显错误或遗漏"
        }
        return descriptions.get(baseline_name, "未知基线")
    
    def _analyze_failures(self, pairwise_results: List[Dict]) -> Dict[str, Any]:
        """分析失败原因（词云数据）"""
        failure_reasons = []
        
        for result in pairwise_results:
            for qr in result["question_results"]:
                passage_judgment = qr.get("passage_judgment", {})
                reason = passage_judgment.get("reason", "")
                if reason:
                    failure_reasons.append(reason)
        
        # 简化的词频统计
        reason_counts = defaultdict(int)
        for reason in failure_reasons:
            # 简单的关键词提取
            keywords = ["准确", "完整", "相关", "证据", "逻辑", "错误", "缺失", "模糊"]
            for keyword in keywords:
                if keyword in reason:
                    reason_counts[keyword] += 1
        
        return {
            "total_reasons": len(failure_reasons),
            "keyword_counts": dict(reason_counts),
            "top_reasons": sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """配置转字典"""
        return {
            "llm_model": self.config.llm_model,
            "max_questions": self.config.max_questions,
            "early_stop_elo_diff": self.config.early_stop_elo_diff,
            "early_stop_ci_threshold": self.config.early_stop_ci_threshold,
            "initial_elo": self.config.initial_elo,
            "k_factor": self.config.k_factor
        }
    
    def _save_tournament_result(self, result: Dict[str, Any]):
        """保存锦标赛结果"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 保存详细结果
        with open(output_dir / "tournament_result.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存简要报告
        self._save_tournament_report(result, output_dir)
        
        self.logger.info(f"🏆 锦标赛结果已保存到: {output_dir}")
    
    def _save_baseline_result(self, result: Dict[str, Any]):
        """保存基线对比结果"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 保存详细结果
        with open(output_dir / "baseline_comparison.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存详细的QACG对比数据到单独文件
        if "detailed_qacg_comparisons" in result:
            with open(output_dir / "qacg_detailed_comparisons.json", 'w', encoding='utf-8') as f:
                json.dump(result["detailed_qacg_comparisons"], f, ensure_ascii=False, indent=2, default=str)
            self.logger.info(f"📋 详细QACG对比数据已保存到: {output_dir / 'qacg_detailed_comparisons.json'}")
        
        # 保存简要报告
        self._save_baseline_report(result, output_dir)
        
        self.logger.info(f"🎯 基线对比结果已保存到: {output_dir}")
    
    def _save_tournament_report(self, result: Dict[str, Any], output_dir: Path):
        """保存锦标赛报告（支持瑞士轮和动态Elo配对）"""
        tournament_type = result.get("tournament_type", "swiss_tournament")
        
        with open(output_dir / "tournament_report.md", 'w', encoding='utf-8') as f:
            if tournament_type == "swiss_tournament":
                f.write("# DICE精简版锦标赛报告 (瑞士轮系统)\n\n")
            elif tournament_type == "full_round_robin":
                f.write("# DICE精简版锦标赛报告 (完整循环赛)\n\n")
            else:
                f.write("# DICE精简版锦标赛报告 (动态Elo配对系统)\n\n")
            
            # 最终排名
            f.write("## 🏆 最终排名 (动态Elo)\n\n")
            final_ranking = result["final_ranking"]
            final_elo_scores = result["final_elo_scores"]
            
            for i, system in enumerate(final_ranking, 1):
                elo_score = final_elo_scores[system]
                # 前3名标记
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
                f.write(f"{i}. **{system}**: {elo_score:.1f} {medal}\n")
            
            # 比赛过程
            if tournament_type == "swiss_tournament":
                f.write("\n## 📊 瑞士轮比赛过程\n\n")
                swiss_results = result["swiss_results"]
                match_records = swiss_results["match_records"]
                total_rounds = swiss_results.get("total_rounds", 4)
                
                f.write(f"总比赛场次: {len(match_records)}场 ({total_rounds}轮，每轮4场)\n\n")
                
                # 按轮次显示比赛
                f.write("### 轮次比赛回顾\n")
                current_round = 1
                for i, match in enumerate(match_records):
                    if match.get('round', 1) != current_round:
                        current_round = match.get('round', 1)
                        f.write(f"\n#### 第{current_round}轮\n")
                    
                    f.write(f"**第{match['match_num']}场**: {match['system_a']} (ELO: {match['old_elo_a']:.1f}) vs {match['system_b']} (ELO: {match['old_elo_b']:.1f})\n")
                    f.write(f"- 胜者: {match['winner']}\n")
                    f.write(f"- Elo变化: {match['system_a']} ({match['old_elo_a']:.1f}→{match['new_elo_a']:.1f}), {match['system_b']} ({match['old_elo_b']:.1f}→{match['new_elo_b']:.1f})\n\n")
                
                # 瑞士轮系统说明
                f.write("## 🎯 瑞士轮系统说明\n\n")
                f.write("- **轮次配对**: 4轮比赛，每轮4场，每队每轮只比一场\n")
                f.write("- **智能配对**: 每轮选择Elo最接近的未对战过的两队\n")
                f.write("- **动态调整**: 实时更新Elo分数，反映真实实力变化\n")
                f.write("- **无种子队**: 初始Elo=1500，完全基于比赛结果学习\n")
                f.write("- **公平性**: 确保每对系统只对战一次\n\n")
            elif tournament_type == "full_round_robin":
                f.write("\n## 📊 完整循环赛比赛过程\n\n")
                rr = result["round_robin_results"]
                match_records = rr.get("match_records", [])
                f.write(f"总比赛场次: {len(match_records)}场（全对全，每对系统仅一次对战）\n\n")
                
                # 按顺序显示比赛
                f.write("### 比赛回顾\n")
                for match in match_records:
                    f.write(f"**第{match['match_num']}场**: {match['system_a']} (ELO: {match['old_elo_a']:.1f}) vs {match['system_b']} (ELO: {match['old_elo_b']:.1f})\n")
                    f.write(f"- 胜者: {match['winner']}\n")
                    f.write(f"- Elo变化: {match['system_a']} ({match['old_elo_a']:.1f}→{match['new_elo_a']:.1f}), {match['system_b']} ({match['old_elo_b']:.1f}→{match['new_elo_b']:.1f})\n\n")
                
                # 循环赛说明
                f.write("## 🎯 完整循环赛说明\n\n")
                f.write("- **配对方式**: 所有系统两两对战一次（共N(N-1)/2场）\n")
                f.write("- **评分方式**: 使用soft win累计评分与动态Elo更新\n")
                f.write("- **可比性**: 覆盖全部配对，避免抽样不完整的偏差\n\n")
            else:
                f.write("\n## 📊 动态配对过程\n\n")
                # 安全获取dynamic_results
                dynamic_results = result.get("dynamic_results")
                if dynamic_results:
                    match_records = dynamic_results.get("match_records", [])
            f.write(f"总比赛场次: {len(match_records)}场\n\n")
            
            # 显示关键比赛
            f.write("### 关键比赛回顾\n")
            for i, match in enumerate(match_records):  # 显示前10场关键比赛
                f.write(f"**第{match['match_num']}场**: {match['system_a']} (ELO: {match['old_elo_a']:.1f}) vs {match['system_b']} (ELO: {match['old_elo_b']:.1f})\n")
                f.write(f"- 胜者: {match['winner']}\n")
                f.write(f"- Elo变化: {match['system_a']} ({match['old_elo_a']:.1f}→{match['new_elo_a']:.1f}), {match['system_b']} ({match['old_elo_b']:.1f}→{match['new_elo_b']:.1f})\n\n")
            else:
                f.write("总比赛场次: 未知\n\n")
                f.write("### 关键比赛回顾\n")
                f.write("比赛记录不可用\n\n")
            
            # 动态Elo系统说明
            f.write("## 🎯 动态Elo配对系统说明\n\n")
            f.write("- **智能配对**: 每轮选择Elo最接近的未对战过的两队\n")
            f.write("- **动态调整**: 实时更新Elo分数，反映真实实力变化\n")
            f.write("- **高效性**: 最大化信息增益，减少冗余比赛\n")
            f.write("- **无种子队**: 初始Elo=1500，完全基于比赛结果学习\n")
            f.write("- **收敛判断**: 当排名稳定或达到最大场次时结束\n\n")
            
            # 失败分析 - 使用动态聚类结果
            f.write("## 📊 动态失败模式聚类分析\n\n")
            failure_clusters = result["failure_analysis"]
            for cluster_id, cluster_data in failure_clusters.items():
                f.write(f"### {cluster_data['label']}\n")
                f.write(f"- 相关系统: {', '.join(cluster_data['systems'][:5])}{'...' if len(cluster_data['systems']) > 5 else ''}\n")
                f.write(f"- 失败案例数: {cluster_data['size']}\n")
                
                # 显示动态提取的关键词
                top_keywords = cluster_data.get('top_keywords', [])
                if top_keywords:
                    keyword_str = ', '.join([f'{k}({v}次)' for k, v in top_keywords[:3]])
                    f.write(f"- 关键词: {keyword_str}\n")
                f.write("\n")
            
            # 调用量统计
            total_calls = result["total_llm_calls"]
            total_matches = len(match_records)
            f.write(f"## 📈 性能统计\n\n")
            f.write(f"- 总比赛场次: {total_matches}场 (vs 传统联赛28场，减少{(28-total_matches)/28*100:.1f}%)\n")
            f.write(f"- 总LLM调用次数: {total_calls}\n")
            f.write(f"- 估计用时: ~{total_calls/40:.1f}分钟 (8×A100)\n")
            f.write(f"- 每队平均对战: {total_matches*2/8:.1f}场\n")

            # CI分析
            ci_analysis = result.get("ci_analysis", {})
            if ci_analysis:
                f.write(f"\n## 📊 95% 置信区间分析\n\n")
                f.write(f"- 平均得分差值: {ci_analysis.get('mean_score_diff', 0):.2f}\n")
                f.write(f"- 95% CI: {ci_analysis.get('ci_95', 'N/A')}\n")
                f.write(f"- 统计显著性: {ci_analysis.get('significance', 'N/A')}\n")
    
    def _save_baseline_report(self, result: Dict[str, Any], output_dir: Path):
        """保存基线对比报告"""
        with open(output_dir / "baseline_report.md", 'w', encoding='utf-8') as f:
            f.write("# DICE精简版基线对比报告\n\n")
            
            target_system = result["target_system"]
            f.write(f"## 🎯 目标系统: {target_system}\n\n")
            
            # 对比结果
            f.write("## 📊 基线对比结果\n\n")
            summary = result["summary"]
            
            for baseline_name, comparison in summary["comparisons"].items():
                win_rate = comparison["win_rate"]
                conclusion = comparison["conclusion"]
                f.write(f"### vs {baseline_name} 基线\n")
                f.write(f"- 胜率: {win_rate:.1%}\n")
                f.write(f"- 结论: {conclusion}\n\n")
            
            # 性能统计
            total_calls = result["total_llm_calls"]
            f.write(f"## 📈 性能统计\n\n")
            f.write(f"- 总LLM调用次数: {total_calls}\n")
            f.write(f"- 估计用时: ~{total_calls/40:.1f}分钟\n")


def create_simplified_evaluator(config: SimplifiedDICEConfig = None) -> SimplifiedDICEEvaluator:
    """创建精简版DICE评估器"""
    return SimplifiedDICEEvaluator(config) 