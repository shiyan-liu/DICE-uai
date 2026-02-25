#!/usr/bin/env python3
"""
DICE simplified engine - tournament and baseline comparison scenarios.
Passage-granularity pairwise judgment with retrieval-evidence dual-channel.
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

from .llm_judge import LocalPairwiseJudge
from .llm_judge import LocalPairwiseJudge as PairwiseJudge

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback when tqdm is not installed
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

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class SimplifiedDICEConfig:
    """Simplified DICE configuration."""
    # LLM config - online API
    llm_model: str = "deepseek-chat"
    api_key: str = ""  # from env: DEEPSEEK_API_KEY
    base_url: str = "https://api.deepseek.com"
    judge_temperature: float = 0.1
    max_tokens: int = 2048

    # DeepSeek-R1 local model config
    enable_deep_thinking: bool = True

    # Evaluation config
    max_questions: int = 70
    early_stop_elo_diff: float = 400.0
    early_stop_ci_threshold: float = 30.0

    # Elo config
    initial_elo: float = 1000.0
    k_factor: int = 32

    # Concurrency config (dual-GPU)
    max_workers: int = 4   # max concurrent workers (2 GPUs x 2 workers)
    batch_size: int = 8    # questions per batch (~48GB total VRAM)

    # Output config
    output_dir: str = "dice_simplified_output"
    save_detailed: bool = True


class SimplifiedDICEEvaluator:
    """Simplified DICE evaluator."""

    def __init__(self, config: SimplifiedDICEConfig = None):
        self.config = config or SimplifiedDICEConfig()
        self.logger = logging.getLogger("DICE.Simplified")
        self._setup_logger()

        # Passage-level pairwise judge
        self.pairwise_judge = LocalPairwiseJudge(self.config)

        self._lock = threading.Lock()

        # Virtual baseline generation instructions (Chinese LLM prompts - keep as-is)
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
        """Thread-safe per-question result logging with soft win info."""
        passage_judgment = result["passage_judgment"]
        question = result["question"]
        score_a = result["score_a"]
        score_b = result["score_b"]

        self.logger.info(f"    Q {completed_count}/{total_questions}: {question[:60]}...")
        self.logger.info(f"    Verdict: {passage_judgment.get('win_type', 'Unknown')}")
        self.logger.info(f"    Logits: A={passage_judgment.get('logit_a', 0):.2f}, B={passage_judgment.get('logit_b', 0):.2f}, T={passage_judgment.get('logit_t', 0):.2f}")
        self.logger.info(f"    Probs: A={passage_judgment.get('prob_a', 0):.3f}, B={passage_judgment.get('prob_b', 0):.3f}, T={passage_judgment.get('prob_t', 0):.3f}")
        self.logger.info(f"    Prob gap: {passage_judgment.get('prob_diff', 0):.3f} ({'Hard' if passage_judgment.get('prob_diff', 0) >= 0.1 else 'Soft'} win)")
        self.logger.info(f"    Scores: A={score_a:.3f}, B={score_b:.3f}")
        self.logger.info("")
    
    def _setup_logger(self):
        """Configure logger."""
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def scenario_a_tournament(self, qacg_files: List[str]) -> Dict[str, Any]:
        """Scenario A: 8-system tournament (dynamic Elo pairing)."""
        self.logger.info("Starting Scenario A: 8-system tournament (dynamic Elo pairing)")

        systems = self._load_systems(qacg_files)
        system_names = list(systems.keys())

        if len(system_names) != 8:
            raise ValueError(f"Expected 8 systems, got {len(system_names)}")

        # Swiss tournament: 4 rounds, 4 matches per round
        swiss_results = self._swiss_tournament(system_names, systems, num_rounds=4)

        final_ranking = self._calculate_dynamic_ranking(swiss_results["final_elo_scores"])

        all_pairwise_results = swiss_results["all_pairwise_results"]
        ci_analysis = self._bootstrap_ci_analysis(all_pairwise_results, system_names)

        failure_clusters = self._cluster_failure_modes(all_pairwise_results)

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
        
        self._save_tournament_result(tournament_result)
        return tournament_result

    def scenario_c_full_round_robin(self, qacg_files: List[str]) -> Dict[str, Any]:
        """Scenario C: full round-robin pairwise comparison (each pair plays once)."""
        self.logger.info("Starting Scenario C: full round-robin")

        systems = self._load_systems(qacg_files)
        system_names = list(systems.keys())

        if len(system_names) < 2:
            raise ValueError(f"Need at least 2 systems, got {len(system_names)}")

        elo_scores = {system: 1500.0 for system in system_names}
        all_pairwise_results = []
        match_records = []
        total_llm_calls = 0

        pair_idx = 0
        total_pairs = len(system_names) * (len(system_names) - 1) // 2
        for sys_a, sys_b in itertools.combinations(system_names, 2):
            pair_idx += 1
            self.logger.info(f"  Match {pair_idx}/{total_pairs}: {sys_a} (ELO: {elo_scores[sys_a]:.1f}) vs {sys_b} (ELO: {elo_scores[sys_b]:.1f})")

            comparison = self._pairwise_comparison(
                systems[sys_a], systems[sys_b], sys_a, sys_b, 
                max_questions=self.config.max_questions
            )
            all_pairwise_results.append(comparison)
            total_llm_calls += len(comparison["question_results"])
            
            old_elo_a, old_elo_b = elo_scores[sys_a], elo_scores[sys_b]
            self._update_elo_scores_dynamic(elo_scores, comparison, sys_a, sys_b)

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
        
        self._save_tournament_result(result)
        return result
    
    def _swiss_tournament(self, system_names: List[str], all_systems: Dict[str, List[Dict]],
                         num_rounds: int) -> Dict[str, Any]:
        """Swiss-system tournament implementation."""
        self.logger.info(f"Starting Swiss tournament, {num_rounds} rounds")

        standings = {}
        for system in system_names:
            standings[system] = {
                "elo": self.config.initial_elo,
                "swiss_points": 0.0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "sb_score": 0.0,  # Solkoff / Buchholz score
                "opponents": []
            }
        
        rounds = []
        total_llm_calls = 0
        
        for round_num in range(1, num_rounds + 1):
            self.logger.info(f"Round {round_num} started")

            pairings = self._swiss_pairing(standings, round_num)

            round_results = []
            round_pairwise_results = []
            
            for sys_a, sys_b in pairings:
                self.logger.info(f"  {sys_a} vs {sys_b}")

                comparison = self._pairwise_comparison(
                    all_systems[sys_a], all_systems[sys_b], sys_a, sys_b,
                    max_questions=max(3, self.config.max_questions // num_rounds)
                )
                round_pairwise_results.append(comparison)
                total_llm_calls += len(comparison["question_results"])
                
                result = self._calculate_match_result(comparison)
                round_results.append({
                    "system_a": sys_a,
                    "system_b": sys_b,
                    "result": result,
                    "comparison": comparison
                })
                
                self._update_elo_scores_swiss(standings, comparison, sys_a, sys_b)
                self._update_swiss_standings(standings, sys_a, sys_b, result)
            
            rounds.append({
                "round": round_num,
                "pairings": pairings,
                "results": round_results,
                "pairwise_results": round_pairwise_results,
                "standings_after_round": self._get_current_standings_snapshot(standings)
            })
            
            self._update_sb_scores(standings)

            self.logger.info(f"Round {round_num} finished, current ranking:")
            current_ranking = self._get_current_ranking(standings)
            for i, (system, stats) in enumerate(current_ranking[:3], 1):
                self.logger.info(f"  {i}. {system}: {stats['swiss_points']:.1f}pts (ELO: {stats['elo']:.1f})")
        
        return {
            "rounds": rounds,
            "final_standings": standings,
            "total_llm_calls": total_llm_calls
        }
    
    def _swiss_pairing(self, standings: Dict[str, Dict], round_num: int) -> List[Tuple[str, str]]:
        """Swiss-system pairing algorithm."""
        if round_num == 1:
            # Round 1: cross-pair large vs small models
            systems = list(standings.keys())
            large_systems = [s for s in systems if "large" in s]
            small_systems = [s for s in systems if "small" in s]

            pairings = []
            for i in range(min(len(large_systems), len(small_systems))):
                pairings.append((large_systems[i], small_systems[i]))

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
            # Pair by score and Elo
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
                
                # Find best opponent (close score, not yet faced)
                best_opponent = None
                for j in range(i + 1, len(systems_by_score)):
                    system_b = systems_by_score[j]
                    if (system_b not in paired and 
                        system_b not in standings[system_a]["opponents"]):
                        best_opponent = system_b
                        break
                
                # Fallback: pick nearest unpaired opponent
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
        """Determine match result (win/draw/loss)."""
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
        """Update Elo scores (Swiss variant with weighted algorithm)."""
        summary = comparison["summary"]
        win_rate_a = summary["win_rate_a"]
        win_rate_b = summary["win_rate_b"]
        
        elo_a = standings[sys_a]["elo"]
        elo_b = standings[sys_b]["elo"]
        
        expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        expected_b = 1 - expected_a

        rating_diff = abs(elo_a - elo_b)
        base_k = self.config.k_factor

        # Non-linear weight based on rating gap
        weight_factor = 0.5 + 1.5 * (1 - math.exp(-rating_diff / 200))

        # Upset bonus
        upset_bonus_a = 1.0
        upset_bonus_b = 1.0

        if elo_a < elo_b and win_rate_a > 0.5:
            upset_bonus_a = 1.0 + (rating_diff / 400)
            upset_bonus_b = 1.0 + (rating_diff / 600)
        elif elo_b < elo_a and win_rate_b > 0.5:
            upset_bonus_b = 1.0 + (rating_diff / 400)
            upset_bonus_a = 1.0 + (rating_diff / 600)

        k_a = base_k * weight_factor * upset_bonus_a
        k_b = base_k * weight_factor * upset_bonus_b

        standings[sys_a]["elo"] += k_a * (win_rate_a - expected_a)
        standings[sys_b]["elo"] += k_b * (win_rate_b - expected_b)
    
    def _update_swiss_standings(self, standings: Dict[str, Dict],
                              sys_a: str, sys_b: str, result: str):
        """Update Swiss standings and records."""
        standings[sys_a]["opponents"].append(sys_b)
        standings[sys_b]["opponents"].append(sys_a)
        
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
        """Update Solkoff/Buchholz scores (sum of opponents' points)."""
        for system in standings:
            sb_score = 0.0
            for opponent in standings[system]["opponents"]:
                sb_score += standings[opponent]["swiss_points"]
            standings[system]["sb_score"] = sb_score
    
    def _get_current_standings_snapshot(self, standings: Dict[str, Dict]) -> Dict[str, Dict]:
        """Get current standings snapshot."""
        return {system: stats.copy() for system, stats in standings.items()}

    def _get_current_ranking(self, standings: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
        """Get current ranking."""
        return sorted(
            standings.items(),
            key=lambda x: (x[1]["swiss_points"], x[1]["elo"], x[1]["sb_score"]),
            reverse=True
        )

    def _calculate_swiss_ranking(self, final_standings: Dict[str, Dict]) -> List[str]:
        """Calculate final Swiss tournament ranking."""
        # Ranking rules:
        # 1. Swiss points (win=1, draw=0.5, loss=0)
        # 2. Elo score
        # 3. SB score (opponent points sum)
        # 4. Win count
        # 5. System name (lexicographic order)
        
        ranked_systems = sorted(
            final_standings.items(),
            key=lambda x: (
                x[1]["swiss_points"],
                x[1]["elo"],
                x[1]["sb_score"],
                x[1]["wins"],
                x[0]
            ),
            reverse=True
        )

        return [system for system, _ in ranked_systems]
    
    def _bootstrap_ci_analysis(self, pairwise_results: List[Dict], system_names: List[str]) -> Dict[str, Any]:
        """Execute bootstrap 95% confidence interval analysis."""
        all_score_diffs = []
        for result in pairwise_results:
            for qr in result["question_results"]:
                # Use score difference instead of elo_delta
                score_diff = qr["score_a"] - qr["score_b"]
                all_score_diffs.append(score_diff)

        if not all_score_diffs:
            return {
                "mean_score_diff": 0.0,
                "ci_95": "0.00 - 0.00",
                "significance": "no_data"
            }

        # Calculate mean score difference
        mean_score_diff = np.mean(all_score_diffs)

        # Execute bootstrap CI
        try:
            from scipy.stats import bootstrap
            boot_results = bootstrap((all_score_diffs,), np.mean, confidence_level=0.95, n_resamples=1000)
            ci_95 = boot_results.confidence_interval
            ci_95_str = f"{ci_95.low:.2f} - {ci_95.high:.2f}"

            # Significance judgment (based on CI)
            significance = "significant" if not (ci_95.low <= 0 <= ci_95.high) else "not_significant"
        except Exception as e:
            self.logger.warning(f"Bootstrap CI calculation failed: {e}")
            ci_95_str = "calculation_failed"
            significance = "unknown"

        return {
            "mean_score_diff": mean_score_diff,
            "ci_95": ci_95_str,
            "significance": significance
        }
    
    def _cluster_failure_modes(self, pairwise_results: List[Dict]) -> Dict[str, Any]:
        """Analyze failure modes using dynamic semantic clustering based on LLM response similarity."""
        # Collect all failure reason texts
        failure_reasons = []
        reason_to_systems = {}

        for result in pairwise_results:
            for qr in result["question_results"]:
                passage_judgment = qr.get("passage_judgment", {})
                reason = passage_judgment.get("reason", "")
                if reason and len(reason.strip()) > 10:  # Filter overly short reasons
                    failure_reasons.append(reason.strip())
                    if reason not in reason_to_systems:
                        reason_to_systems[reason] = set()
                    reason_to_systems[reason].add(result["system_a"])
                    reason_to_systems[reason].add(result["system_b"])

        if len(failure_reasons) < 5:
            # Insufficient data, return simple statistics
            return {
                "cluster_0": {
                    "label": "Failure reason analysis",
                    "systems": list(set().union(*reason_to_systems.values())) if reason_to_systems else [],
                    "reasons": failure_reasons,
                    "top_keywords": self._extract_top_keywords(failure_reasons),
                    "size": len(failure_reasons)
                }
            }

        try:
            if not SKLEARN_AVAILABLE:
                self.logger.warning("sklearn unavailable, skipping dynamic semantic clustering, returning simple statistics")
                return {
                    "cluster_0": {
                        "label": "Failure reason analysis (simplified mode)",
                        "systems": list(set().union(*reason_to_systems.values())) if reason_to_systems else [],
                        "reasons": failure_reasons,
                        "top_keywords": self._extract_top_keywords(failure_reasons),
                        "size": len(failure_reasons)
                    }
                }

            # Dynamic TF-IDF vectorization (Chinese tokenization friendly)
            vectorizer = TfidfVectorizer(
                max_features=200,
                stop_words=None,
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.8,
                token_pattern=r'[\u4e00-\u9fff]+|[a-zA-Z]+\d*'  # Chinese characters or English words
            )
            tfidf_matrix = vectorizer.fit_transform(failure_reasons)

            # Dynamically determine optimal number of clusters (based on data scale and semantic similarity)
            n_clusters = self._determine_optimal_clusters(tfidf_matrix, failure_reasons)

            if n_clusters <= 1:
                # Poor clustering, return unified analysis
                return {
                    "cluster_0": {
                        "label": "General failure pattern",
                        "systems": list(set().union(*reason_to_systems.values())) if reason_to_systems else [],
                        "reasons": failure_reasons,
                        "top_keywords": self._extract_top_keywords(failure_reasons),
                        "size": len(failure_reasons)
                    }
                }

            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)

            # Dynamic cluster label generation
            feature_names = vectorizer.get_feature_names_out()
            clusters = {}

            for cluster_id in range(n_clusters):
                cluster_reasons = [failure_reasons[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                cluster_systems = set()

                # Collect systems for this cluster
                for reason in cluster_reasons:
                    if reason in reason_to_systems:
                        cluster_systems.update(reason_to_systems[reason])

                # Dynamically generate cluster label (based on highest TF-IDF weighted terms)
                cluster_label = self._generate_cluster_label(cluster_reasons, feature_names, kmeans.cluster_centers_[cluster_id])

                # Extract keywords for this cluster
                top_keywords = self._extract_cluster_keywords(cluster_reasons, feature_names)

                clusters[f"cluster_{cluster_id}"] = {
                    "label": cluster_label,
                    "systems": list(cluster_systems),
                    "reasons": cluster_reasons,
                    "top_keywords": top_keywords,
                    "size": len(cluster_reasons)
                }

            # Sort clusters by size
            sorted_clusters = dict(sorted(clusters.items(), key=lambda x: x[1]["size"], reverse=True))

            return sorted_clusters

        except Exception as e:
            self.logger.warning(f"Dynamic semantic clustering failed: {e}")
            # Return simple keyword statistics
            return {
                "cluster_0": {
                    "label": "Failure pattern analysis",
                    "systems": list(set().union(*reason_to_systems.values())) if reason_to_systems else [],
                    "reasons": failure_reasons,
                    "top_keywords": self._extract_top_keywords(failure_reasons),
                    "size": len(failure_reasons)
                }
            }
    
    def _determine_optimal_clusters(self, tfidf_matrix, failure_reasons: List[str]) -> int:
        """Dynamically determine optimal number of clusters."""
        n_samples = len(failure_reasons)

        # Determine cluster count range based on data scale
        if n_samples < 5:
            return 1
        elif n_samples < 15:
            max_clusters = 2
        elif n_samples < 30:
            max_clusters = 3
        else:
            max_clusters = min(5, n_samples // 8)

        # Use silhouette coefficient to select optimal cluster count
        try:
            from sklearn.metrics import silhouette_score
            best_n_clusters = 1
            best_score = -1

            for n in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                labels = kmeans.fit_predict(tfidf_matrix)
                score = silhouette_score(tfidf_matrix, labels)

                if score > best_score and score > 0.3:  # Require certain clustering quality
                    best_score = score
                    best_n_clusters = n

            return best_n_clusters

        except Exception:
            # Silhouette analysis failed, use heuristic rules
            return min(3, max(1, n_samples // 10))
    
    def _generate_cluster_label(self, cluster_reasons: List[str],
                              feature_names: list, cluster_center: list) -> str:
        """Dynamically generate cluster labels based on TF-IDF weights."""
        try:
            # Get top 3 features by weight
            top_indices = sorted(range(len(cluster_center)),
                               key=lambda i: cluster_center[i], reverse=True)[:3]
            top_features = [feature_names[i] for i in top_indices if cluster_center[i] > 0]

            if not top_features:
                return "Uncategorized failure pattern"

            # Generate meaningful labels based on key features
            label_mapping = {
                ('retrieval', 'missing', 'passage'): "Retrieval missing key passages",
                ('numeric', 'error', 'calculation'): "Numeric calculation error",
                ('logic', 'jump', 'reasoning'): "Logic reasoning issue",
                ('evidence', 'insufficient', 'support'): "Insufficient evidence support",
                ('answer', 'incomplete', 'missing'): "Answer incomplete",
                ('understanding', 'error', 'understanding'): "Understanding deviation",
                ('format', 'error', 'structure'): "Format structure issue"
            }

            # Try matching predefined patterns
            top_features_str = ' '.join(top_features)
            for pattern, label in label_mapping.items():
                if any(keyword in top_features_str for keyword in pattern):
                    return label

            # If no match, generate label based on most important feature
            main_feature = top_features[0]
            if 'retrieval' in main_feature or 'search' in main_feature:
                return "Retrieval-related issue"
            elif 'answer' in main_feature or 'response' in main_feature:
                return "Answer quality issue"
            elif 'logic' in main_feature or 'reasoning' in main_feature:
                return "Logic reasoning issue"
            elif 'numeric' in main_feature or 'calculation' in main_feature:
                return "Numeric processing issue"
            else:
                return f"{main_feature}-related issue"

        except Exception:
            return "Failure pattern"
    
    def _extract_cluster_keywords(self, cluster_reasons: List[str], feature_names: list) -> List[Tuple[str, int]]:
        """Extract keywords and their frequency from cluster."""
        try:
            if not SKLEARN_AVAILABLE:
                # Simplified keyword extraction
                return self._extract_top_keywords(cluster_reasons)

            # Re-analyze TF-IDF for this cluster's text
            vectorizer = TfidfVectorizer(
                max_features=50,
                ngram_range=(1, 2),
                token_pattern=r'[\u4e00-\u9fff]+|[a-zA-Z]+\d*'
            )
            tfidf_matrix = vectorizer.fit_transform(cluster_reasons)
            feature_names = vectorizer.get_feature_names_out()

            # Calculate TF-IDF total scores
            tfidf_scores = tfidf_matrix.sum(axis=0).A1

            # Get top 5 keywords
            top_indices = sorted(range(len(tfidf_scores)),
                               key=lambda i: tfidf_scores[i], reverse=True)[:5]

            top_keywords = []
            for idx in top_indices:
                if tfidf_scores[idx] > 0:
                    keyword = feature_names[idx]
                    # Count how many times this keyword appears in texts
                    count = sum(1 for reason in cluster_reasons if keyword in reason)
                    top_keywords.append((keyword, count))

            return top_keywords

        except Exception:
            return self._extract_top_keywords(cluster_reasons)

    def _extract_top_keywords(self, reasons: List[str]) -> List[Tuple[str, int]]:
        """Simple keyword extraction (fallback method)."""
        common_keywords = [
            "retrieval", "missing", "insufficient", "error", "inaccurate", "incomplete", "irrelevant",
            "logic", "reasoning", "evidence", "support", "answer", "numeric", "calculation", "understanding",
            "passage", "document", "information", "key", "important", "omission", "deviation"
        ]

        keyword_counts = defaultdict(int)
        all_text = ' '.join(reasons)

        for keyword in common_keywords:
            count = all_text.count(keyword)
            if count > 0:
                keyword_counts[keyword] = count

        # Return top 5 keywords
        return sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _swiss_tournament(self, system_names: List[str], all_systems: Dict[str, List[Dict]],
                         num_rounds: int = 4) -> Dict[str, Any]:
        """Swiss tournament - 4 rounds, 4 matches per round, each team plays once per round."""
        self.logger.info(f"Starting Swiss tournament with {num_rounds} rounds")

        # Initialize all teams with Elo=1500 (no prior information)
        elo_scores = {system: 1500.0 for system in system_names}
        match_history = set()  # Record pairs that have played
        all_pairwise_results = []
        match_records = []
        total_llm_calls = 0

        # Swiss tournament progress bar
        tournament_progress = tqdm(range(1, num_rounds + 1),
                                 desc="Swiss tournament progress",
                                 unit="round",
                                 ncols=100,
                                 colour='green')

        for round_num in tournament_progress:
            self.logger.info(f"Round {round_num} started")

            # Select pairings for this round
            round_pairs = self._select_swiss_round_pairs(elo_scores, match_history, system_names)

            if not round_pairs:
                self.logger.info("No more valid pairings available, ending early")
                tournament_progress.close()
                break

            # Execute all matches for this round
            for match_idx, (sys_a, sys_b) in enumerate(round_pairs, 1):
                match_num = (round_num - 1) * 4 + match_idx
                self.logger.info(f"  Match {match_num}: {sys_a} (ELO: {elo_scores[sys_a]:.1f}) vs {sys_b} (ELO: {elo_scores[sys_b]:.1f})")

                # Record this match
                match_history.add((sys_a, sys_b))
                match_history.add((sys_b, sys_a))  # Bidirectional record

                # Execute comparison
                comparison = self._pairwise_comparison(
                    all_systems[sys_a], all_systems[sys_b], sys_a, sys_b,
                    max_questions=self.config.max_questions
                )
                all_pairwise_results.append(comparison)
                total_llm_calls += len(comparison["question_results"])

                # Update Elo scores
                old_elo_a, old_elo_b = elo_scores[sys_a], elo_scores[sys_b]
                self._update_elo_scores_dynamic(elo_scores, comparison, sys_a, sys_b)

                # Record detailed match info
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

            # Output current ranking after this round
            current_ranking = sorted(system_names, key=lambda x: elo_scores[x], reverse=True)
            self.logger.info(f"  Round {round_num} complete, ranking: {current_ranking[0]}({elo_scores[current_ranking[0]]:.1f}) > {current_ranking[1]}({elo_scores[current_ranking[1]]:.1f}) > {current_ranking[2]}({elo_scores[current_ranking[2]]:.1f})")

            # Update progress bar description
            tournament_progress.set_description(f"Round {round_num} complete - Leader: {current_ranking[0]}")

        # Close progress bar
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
        """Select pairings for current round of Swiss tournament - improved version."""
        # Generate all possible match combinations
        all_possible_pairs = []
        for i, sys_a in enumerate(system_names):
            for sys_b in system_names[i+1:]:
                if (sys_a, sys_b) not in match_history:
                    elo_diff = abs(elo_scores[sys_a] - elo_scores[sys_b])
                    all_possible_pairs.append((sys_a, sys_b, elo_diff))

        # Sort by Elo difference (prioritize pairings with close Elos)
        all_possible_pairs.sort(key=lambda x: x[2])

        # Use backtracking to find optimal 4-match combination
        best_combination = self._find_best_round_combination(all_possible_pairs, len(system_names) // 2)

        if best_combination:
            return [(pair[0], pair[1]) for pair in best_combination]
        else:
            self.logger.warning("Unable to find valid Swiss tournament pairing combination")
            return []

    def _find_best_round_combination(self, all_pairs: List[Tuple[str, str, float]],
                                   target_pairs: int) -> List[Tuple[str, str, float]]:
        """Use backtracking to find optimal round match combination."""
        def backtrack(used_systems: set, current_pairs: List[Tuple[str, str, float]],
                     pair_index: int) -> List[Tuple[str, str, float]]:
            # If found enough pairings, return result
            if len(current_pairs) == target_pairs:
                return current_pairs.copy()

            # If checked all possible pairings, return None
            if pair_index >= len(all_pairs):
                return None

            # Try including current pairing
            sys_a, sys_b, elo_diff = all_pairs[pair_index]
            if sys_a not in used_systems and sys_b not in used_systems:
                used_systems.add(sys_a)
                used_systems.add(sys_b)
                current_pairs.append(all_pairs[pair_index])

                result = backtrack(used_systems, current_pairs, pair_index + 1)
                if result:
                    return result

                # Backtrack
                current_pairs.pop()
                used_systems.remove(sys_a)
                used_systems.remove(sys_b)

            # Try skipping current pairing
            return backtrack(used_systems, current_pairs, pair_index + 1)

        # Start backtracking search
        result = backtrack(set(), [], 0)
        return result if result else []
    
    def _dynamic_elo_tournament(self, system_names: List[str], all_systems: Dict[str, List[Dict]],
                               max_matches: int) -> Dict[str, Any]:
        """Dynamic Elo pairing tournament (based on recommendations)."""
        self.logger.info(f"Starting dynamic Elo pairing tournament with max {max_matches} matches")

        # Initialize all teams with Elo=1500 (no prior information)
        elo_scores = {system: 1500.0 for system in system_names}
        match_history = set()  # Record pairs that have played
        all_pairwise_results = []
        match_records = []
        total_llm_calls = 0

        # Dynamic pairing until max matches - add overall progress bar
        tournament_progress = tqdm(range(1, max_matches + 1),
                                 desc="Tournament progress",
                                 unit="match",
                                 ncols=100,
                                 colour='green')

        for match_num in tournament_progress:
            self.logger.info(f"Match {match_num} started")

            # Select closest Elo pair that hasn't played yet
            best_pair = self._find_best_elo_pair(elo_scores, match_history)

            if not best_pair:
                self.logger.info("All possible matches complete, ending early")
                tournament_progress.close()
                break

            sys_a, sys_b = best_pair
            self.logger.info(f"  {sys_a} (ELO: {elo_scores[sys_a]:.1f}) vs {sys_b} (ELO: {elo_scores[sys_b]:.1f})")

            # Record this match
            match_history.add((sys_a, sys_b))
            match_history.add((sys_b, sys_a))  # Bidirectional record

            # Execute comparison
            comparison = self._pairwise_comparison(
                all_systems[sys_a], all_systems[sys_b], sys_a, sys_b,
                max_questions=self.config.max_questions  # Use full full question count from config
            )
            all_pairwise_results.append(comparison)
            total_llm_calls += len(comparison["question_results"])

            # Update Elo scores (using weighted algorithm)
            old_elo_a, old_elo_b = elo_scores[sys_a], elo_scores[sys_b]
            self._update_elo_scores_dynamic(elo_scores, comparison, sys_a, sys_b)

            # Record detailed match info
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

            # Output current ranking (top 3)
            current_ranking = sorted(system_names, key=lambda x: elo_scores[x], reverse=True)
            self.logger.info(f"  Current ranking: {current_ranking[0]}({elo_scores[current_ranking[0]]:.1f}) > {current_ranking[1]}({elo_scores[current_ranking[1]]:.1f}) > {current_ranking[2]}({elo_scores[current_ranking[2]]:.1f})")

            # Update progress bar description
            tournament_progress.set_description(f"Match {match_num} complete - Leader: {current_ranking[0]}")

            # Convergence mechanism removed - run full matches for more accurate ranking

        # Close progress bar
        tournament_progress.close()

        return {
            "match_records": match_records,
            "all_pairwise_results": all_pairwise_results,
            "final_elo_scores": elo_scores,
            "total_llm_calls": total_llm_calls,
            "total_matches": len(match_records)
        }
    
    def _find_best_elo_pair(self, elo_scores: Dict[str, float], match_history: set) -> Tuple[str, str]:
        """Find two teams with closest Elo that haven't played yet."""
        systems = list(elo_scores.keys())
        best_pair = None
        min_elo_diff = float('inf')

        for i, sys_a in enumerate(systems):
            for j, sys_b in enumerate(systems[i+1:], i+1):
                # Check if they've played before
                if (sys_a, sys_b) in match_history:
                    continue

                # Calculate Elo difference
                elo_diff = abs(elo_scores[sys_a] - elo_scores[sys_b])

                if elo_diff < min_elo_diff:
                    min_elo_diff = elo_diff
                    best_pair = (sys_a, sys_b)

        return best_pair

    def _update_elo_scores_dynamic(self, elo_scores: Dict[str, float],
                                 comparison: Dict[str, Any], sys_a: str, sys_b: str):
        """Update Elo scores (new soft win scoring mechanism) - simplified version."""
        summary = comparison["summary"]

        # New scoring mechanism already calculated elo_delta
        elo_delta = summary["elo_delta"]

        old_elo_a = elo_scores[sys_a]
        old_elo_b = elo_scores[sys_b]

        # Apply elo_delta directly (change in A's score)
        elo_scores[sys_a] += elo_delta
        elo_scores[sys_b] -= elo_delta  # B's change is opposite

        # Record detailed change info (for debugging)
        self.logger.debug(f"Elo update: {sys_a}({old_elo_a:.1f}->{elo_scores[sys_a]:.1f}, +{elo_delta:.1f}) vs {sys_b}({old_elo_b:.1f}->{elo_scores[sys_b]:.1f}, {-elo_delta:.1f})")
    
    def _determine_winner(self, comparison: Dict[str, Any]) -> str:
        """Determine match winner."""
        summary = comparison["summary"]
        win_rate_a = summary["win_rate_a"]

        if win_rate_a > 0.6:
            return "A"
        elif win_rate_a < 0.4:
            return "B"
        else:
            return "Tie"

    # Early stopping methods removed per user request - use full match count for accurate ranking
    
    def _calculate_dynamic_ranking(self, final_elo_scores: Dict[str, float]) -> List[str]:
        """Calculate ranking based on final Elo scores."""
        return sorted(final_elo_scores.keys(), key=lambda x: final_elo_scores[x], reverse=True)

    def _parse_tournament_rankings(self, tournament_report_path: str = None) -> Dict[str, Dict]:
        """Parse tournament rankings, extracting 1st, 5th, and 8th place system info."""
        if not tournament_report_path:
            # Use default path
            tournament_report_path = "dice_simplified_output/tournament_report.md"

        try:
            with open(tournament_report_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse rankings
            rankings = {}
            lines = content.split('\n')

            for i, line in enumerate(lines):
                if '**bge-' in line and '**:' in line:
                    # Extract ranking (directly from current line)
                    rank = None

                    # Extract rank number from line start
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
                        # Remove rank prefix like "1. ", "2. ", etc.
                        if '. ' in system_name:
                            system_name = system_name.split('. ', 1)[1]
                        elo_score = float(parts[1].strip().split()[0])

                        rankings[rank] = {
                            'system_name': system_name,
                            'elo_score': elo_score,
                            'rank': rank
                        }

            # Ensure we have 1st, 5th, 8th place
            required_ranks = [1, 5, 8]
            result = {}

            for rank in required_ranks:
                if rank in rankings:
                    rank_name = {1: "1st_Place", 5: "5th_Place", 8: "8th_Place"}[rank]
                    result[rank_name] = rankings[rank]
                else:
                    self.logger.warning(f"System at rank {rank} not found")

            self.logger.info(f"Parsed tournament rankings: {list(result.keys())}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to parse tournament rankings: {e}")
            # Return default virtual baselines
            return {
                "1st_Place": {"system_name": "tournament_1st", "elo_score": 1520.0, "rank": 1},
                "5th_Place": {"system_name": "tournament_5th", "elo_score": 1495.0, "rank": 5},
                "8th_Place": {"system_name": "tournament_8th", "elo_score": 1480.0, "rank": 8}
            }
    
    def _create_tournament_baseline_data(self, target_data: List[Dict], baseline_info: Dict) -> Tuple[List[Dict], int]:
        """Create baseline data based on tournament ranking."""
        baseline_name = baseline_info['system_name']
        elo_score = baseline_info['elo_score']
        rank = baseline_info['rank']

        # Adjust generation quality based on rank
        if rank == 1:
            quality_level = "high"
            instruction = f"As system ranked 1st in tournament ({baseline_name}, Elo: {elo_score:.1f}), generate high-quality answer. Requirements: 1)Provide complete accurate information, 2)Clear logical structure, 3)Based on authoritative materials, 4)Professional expression."
        elif rank == 5:
            quality_level = "medium"
            instruction = f"As system ranked 5th in tournament ({baseline_name}, Elo: {elo_score:.1f}), generate medium-quality answer. Requirements: 1)Main information with possible details missing, 2)Basically accurate but not deep enough, 3)Medium information completeness."
        else:  # rank == 8
            quality_level = "low"
            instruction = f"As system ranked 8th in tournament ({baseline_name}, Elo: {elo_score:.1f}), generate lower-quality answer. Requirements: 1)Information may be inaccurate or incomplete, 2)Expression may be unclear, 3)Possible errors or irrelevant information."

        # Generate baseline data
        baseline_data = []
        generation_calls = 0

        for item in target_data:
            question = item['question']
            groundtruth = item['groundtruth']

            # Generate baseline answer
            baseline_answer = self._generate_baseline_answer(question, groundtruth, instruction, quality_level)
            generation_calls += 1

            # Generate baseline contexts
            baseline_contexts = self._generate_baseline_contexts(question, groundtruth, quality_level)
            generation_calls += 3  # 3 contexts

            baseline_data.append({
                'question': question,
                'groundtruth': groundtruth,
                'answer': baseline_answer,
                'context': baseline_contexts
            })

        return baseline_data, generation_calls
    
    def _summarize_tournament_baseline_comparison(self, baseline_results: Dict, target_system: str) -> Dict:
        """Summarize tournament baseline comparison results."""
        summary = {
            "target_system": target_system,
            "comparisons": {}
        }

        for rank_name, result in baseline_results.items():
            baseline_info = result["baseline_info"]
            comparison = result

            # Calculate win rate
            total_questions = len(comparison["question_results"])
            wins = sum(1 for qr in comparison["question_results"]
                      if qr["passage_judgment"]["win_type"] == "A wins")
            ties = sum(1 for qr in comparison["question_results"]
                      if qr["passage_judgment"]["win_type"] == "Tie")

            win_rate = wins / total_questions if total_questions > 0 else 0
            tie_rate = ties / total_questions if total_questions > 0 else 0

            # Determine conclusion
            if win_rate > 0.6:
                conclusion = f"significantly better than {rank_name}"
            elif win_rate > 0.4:
                conclusion = f"slightly better than {rank_name}"
            elif win_rate > 0.2:
                conclusion = f"comparable to {rank_name}"
            else:
                conclusion = f"inferior to {rank_name}"

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
        Scenario B: Single system vs tournament ranking baselines.

        Args:
            qacg_file: Target system's QACG file
            target_system: System name (optional)
            tournament_report_path: Tournament report file path (optional)

        Returns:
            Baseline comparison result
        """
        self.logger.info("Starting Scenario B: Single system vs tournament ranking baselines")

        # 1. Load target system
        target_data = self._load_qacg_file(qacg_file)
        if not target_system:
            target_system = Path(qacg_file).stem.replace("qacg_", "")

        # 2. Parse tournament rankings
        tournament_rankings = self._parse_tournament_rankings(tournament_report_path)

        # 3. Compare against tournament ranking baselines
        baseline_results = {}
        total_calls = 0

        for rank_name, baseline_info in tournament_rankings.items():
            self.logger.info(f"Comparing {target_system} vs {rank_name} ({baseline_info['system_name']})")

            # Construct baseline data
            baseline_data, baseline_generation_calls = self._create_tournament_baseline_data(
                target_data, baseline_info
            )

            # Execute comparison
            comparison_result = self._pairwise_comparison(
                target_data, baseline_data,
                f"{target_system}", f"{rank_name}_{baseline_info['system_name']}",
                max_questions=self.config.max_questions
            )

            # Save baseline data for detailed comparison
            comparison_result["baseline_data"] = baseline_data
            comparison_result["baseline_generation_calls"] = baseline_generation_calls
            comparison_result["baseline_info"] = baseline_info
            baseline_results[rank_name] = comparison_result
            total_calls += len(comparison_result["question_results"]) + baseline_generation_calls

        # 4. Statistical analysis
        comparison_summary = self._summarize_tournament_baseline_comparison(baseline_results, target_system)

        # 5. Generate detailed QACG comparison data
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

        # Save results
        self._save_baseline_result(result)
        return result
    
    def _load_systems(self, qacg_files: List[str]) -> Dict[str, List[Dict]]:
        """Load all systems data."""
        systems = {}
        for file_path in qacg_files:
            system_name = Path(file_path).stem.replace("qacg_", "")
            systems[system_name] = self._load_qacg_file(file_path)
        return systems

    def _load_qacg_file(self, file_path: str) -> List[Dict]:
        """Load QACG file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data[:self.config.max_questions]  # Limit to configured number of questions
    
    def _create_groups(self, system_names: List[str]) -> List[List[str]]:
        """Create system groups by splitting names."""
        mid = len(system_names) // 2
        return [system_names[:mid], system_names[mid:]]

    def _group_stage(self, group_systems: List[str], all_systems: Dict[str, List[Dict]],
                    stage_name: str = "Group Stage") -> Dict[str, Any]:
        """Execute within-group pairwise comparisons."""
        self.logger.info(f"Starting {stage_name} for systems: {group_systems}")

        elo_scores = {system: self.config.initial_elo for system in group_systems}
        pairwise_results = []
        total_calls = 0

        for sys_a, sys_b in itertools.combinations(group_systems, 2):
            self.logger.info(f"Comparing {sys_a} vs {sys_b}")

            comparison = self._pairwise_comparison(
                all_systems[sys_a], all_systems[sys_b], sys_a, sys_b
            )
            pairwise_results.append(comparison)
            total_calls += len(comparison["question_results"])

            self._update_elo_scores(elo_scores, comparison, sys_a, sys_b)

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
        """Judge a single question (for concurrent processing) using soft-win mechanism.

        Args:
            question_data: (index, qa_a, qa_b, groundtruth)

        Returns:
            (index, question_result): Index and judgment result
        """
        i, qa_a, qa_b, groundtruth = question_data

        try:
            question = qa_a["question"]
            expected_answer = qa_a.get("expected_answer", "")

            passage_judgment = self._judge_passage_only(question, qa_a, qa_b, groundtruth)

            score_a, score_b = self._calculate_soft_win_score(passage_judgment)

            question_result = {
                "question": question,
                "passage_judgment": passage_judgment,
                "score_a": score_a,
                "score_b": score_b,
                "winner": passage_judgment.get("win_type", "Tie"),
                "index": i
            }

            return i, question_result

        except Exception as e:
            self.logger.error(f"Question {i+1} judgment failed: {e}")
            error_result = {
                "question": qa_a.get("question", ""),
                "passage_judgment": {
                    "label": "Tie",
                    "reason": f"Judgment failed: {str(e)}",
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
        """Execute pairwise comparison (with concurrent processing support)."""
        if max_questions is None:
            max_questions = self.config.max_questions

        total_questions = min(len(data_a), len(data_b), max_questions)

        self.logger.info(f"Starting concurrent processing of {total_questions} questions...")
        self.logger.info(f"Concurrency config: {self.config.max_workers} workers, batch size: {self.config.batch_size}")

        # Prepare all question data
        questions_data = []
        for i in range(total_questions):
            qa_a = data_a[i]
            qa_b = data_b[i]
            groundtruth = qa_a.get("groundtruth", qa_a.get("expected_answer", ""))
            questions_data.append((i, qa_a, qa_b, groundtruth))

        # Concurrent processing - add question level progress bar
        question_results = []
        completed_count = 0

        # Create question-level progress bar
        question_progress = tqdm(total=total_questions,
                               desc=f"{name_a} vs {name_b}",
                               unit="question",
                               ncols=100,
                               colour='blue',
                               leave=False)

        # Process in batches
        for batch_start in range(0, len(questions_data), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(questions_data))
            batch_data = questions_data[batch_start:batch_end]

            self.logger.info(f"Processing batch {batch_start//self.config.batch_size + 1}: questions {batch_start+1}-{batch_end}")

            # Use ThreadPoolExecutor for concurrent processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(batch_data))) as executor:
                # Submit tasks
                future_to_index = {executor.submit(self._judge_single_question, question_data): question_data[0]
                                 for question_data in batch_data}

                # Collect results
                batch_results = []
                for future in concurrent.futures.as_completed(future_to_index):
                    try:
                        i, result = future.result()
                        batch_results.append((i, result))
                        completed_count += 1

                        # Update progress bar
                        question_progress.update(1)
                        winner = result.get("winner", "Unknown")
                        question_progress.set_description(f"{name_a} vs {name_b} - Latest: {winner}")

                        # Output detailed judgment result (important! includes reasoning)
                        with self._lock:
                            self._log_question_result(result, completed_count, total_questions)
                            
                    except Exception as e:
                        i = future_to_index[future]
                        self.logger.error(f"Question {i+1} processing error: {e}")

                # Sort by original order
                batch_results.sort(key=lambda x: x[0])
                question_results.extend([result for _, result in batch_results])

            # Early stopping mechanism removed per user request

        # Close question progress bar
        question_progress.close()

        self.logger.info(f"Concurrent processing complete, {len(question_results)} questions processed")

        # Summarize results - using new cumulative scoring mechanism
        summary = self._summarize_pairwise_result_with_soft_win(question_results, name_a, name_b)

        return {
            "system_a": name_a,
            "system_b": name_b,
            "question_results": question_results,
            "summary": summary
        }

    def _judge_passage_only(self, question: str, qa_a: Dict, qa_b: Dict, groundtruth: str) -> Dict[str, Any]:
        """Perform passage-granularity judgment only (retrieval-evidence dual-channel)."""
        # Build retrieval-evidence dual-channel prompt
        context_a = qa_a.get("context", [])
        context_b = qa_b.get("context", [])
        answer_a = qa_a.get("rag_answer", "")
        answer_b = qa_b.get("rag_answer", "")
        expected_answer = qa_a.get("expected_answer", "")

        # Simplified passage-level judgment prompt (in Chinese for LLM)
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
            # self.logger.info(f" 从judge_result获取的logits: A={logit_a}, B={logit_b}, T={logit_t}")
            # self.logger.info(f" 从judge_result获取的概率: A={prob_a:.3f}, B={prob_b:.3f}, T={prob_t:.3f}")
            # self.logger.info(f" judge_result所有键: {list(judge_result.keys())}")
            
            # Improve reason parsing logic
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
        
        self.logger.info(f" 累计评分结果:")
        self.logger.info(f"   总分: {name_a}={total_score_a:.2f}, {name_b}={total_score_b:.2f} (共{total_questions}题)")
        self.logger.info(f"  📈 平均得分率: {name_a}={avg_score_a:.3f}, {name_b}={avg_score_b:.3f}")
        self.logger.info(f"   判决统计: A硬胜{hard_wins_a}, A软胜{soft_wins_a}, B硬胜{hard_wins_b}, B软胜{soft_wins_b}, 平局{ties}")
        self.logger.info(f"   Elo更新: {elo_delta:.1f} ({winner}, 置信度{confidence:.3f})")
        
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
        """Create baseline comparison data using LLM-generated QACG pairs."""
        self.logger.info(f"Generating baseline data for quality level: {baseline_name}")
        baseline_data = []
        baseline_prompt = self.baseline_prompts[baseline_name]
        llm_calls = 0

        for i, qa in enumerate(target_data):
            question = qa["question"]
            groundtruth = qa.get("groundtruth", qa.get("expected_answer", ""))

            self.logger.info(f"Generating baseline answer {i+1}/{len(target_data)} for quality: {baseline_name}")

            generated_answer = self._generate_baseline_answer(question, groundtruth, baseline_prompt)
            llm_calls += 1

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
        """Generate baseline answer using LLM."""
        prompt = f"""
{baseline_prompt["instruction"]}

Question: {question}
Reference answer: {groundtruth}

Generate a {baseline_prompt["quality_level"]} quality response based on the above requirements:
"""

        try:
            response = self.pairwise_judge._call_llm(prompt)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Failed to generate baseline answer: {e}")
            fallback_answers = {
                "high": f"Based on available information, {groundtruth}",
                "medium": f"According to the information, {groundtruth[:len(groundtruth)//2]}...",
                "low": "Insufficient information to provide a clear answer."
            }
            return fallback_answers.get(baseline_prompt["quality_level"], "Unable to generate answer")

    def _generate_baseline_context(self, question: str, groundtruth: str, baseline_prompt: Dict) -> List[str]:
        """Generate baseline retrieval evidence using LLM."""
        prompt = f"""
{baseline_prompt["context_instruction"]}

Question: {question}
Reference information: {groundtruth}

Generate 3 pieces of evidence that meet the {baseline_prompt["quality_level"]} quality requirements:

Evidence 1:
Evidence 2:
Evidence 3:
"""

        try:
            response = self.pairwise_judge._call_llm(prompt)
            lines = response.strip().split('\n')
            contexts = []
            current_context = ""

            for line in lines:
                line = line.strip()
                if line.startswith("Evidence") and ":" in line:
                    if current_context:
                        contexts.append(current_context.strip())
                    current_context = line.split(":", 1)[1]
                elif line and not line.startswith("Evidence"):
                    current_context += " " + line

            if current_context:
                contexts.append(current_context.strip())

            while len(contexts) < 3:
                fallback_contexts = {
                    "high": f"High-quality evidence based on authoritative sources regarding {question}.",
                    "medium": f"Basic information about {question} with partial relevant content.",
                    "low": f"General information related to {question}, may not be entirely accurate."
                }
                contexts.append(fallback_contexts.get(baseline_prompt["quality_level"], "Related information unavailable"))

            return contexts[:3]

        except Exception as e:
            self.logger.error(f"Failed to generate baseline evidence: {e}")
            fallback_contexts = {
                "high": [
                    f"Authoritative sources indicate, {groundtruth[:50]}...",
                    f"Detailed analysis shows {question} involves multiple considerations.",
                    "Based on reliable sources, the above information has high accuracy."
                ],
                "medium": [
                    f"Related information indicates, {groundtruth[:30]}...",
                    f"Regarding {question}, basic information is as described above.",
                    "This information is generally accurate but may be incomplete."
                ],
                "low": [
                    f"As reported, {groundtruth[:20]}...",
                    f"Information about {question} may not be entirely accurate.",
                    "Further verification of relevant content accuracy is recommended."
                ]
            }
            return fallback_contexts.get(baseline_prompt["quality_level"], ["Insufficient information"])
    
    def _summarize_baseline_comparison(self, baseline_results: Dict[str, Any],
                                     target_system: str) -> Dict[str, Any]:
        """Summarize baseline comparison results."""
        summary = {
            "target_system": target_system,
            "comparisons": {}
        }

        for baseline_name, result in baseline_results.items():
            win_rate = result["summary"]["win_rate_a"]
            total_questions = result["summary"]["total_questions"]

            if win_rate > 0.6:
                conclusion = f"Significantly better than {baseline_name} baseline"
            elif win_rate < 0.4:
                conclusion = f"Significantly worse than {baseline_name} baseline"
            else:
                conclusion = f"Comparable to {baseline_name} baseline"

            summary["comparisons"][baseline_name] = {
                "win_rate": win_rate,
                "total_questions": total_questions,
                "conclusion": conclusion
            }

        return summary
    
    def _generate_detailed_qacg_comparisons(self, target_data: List[Dict], target_system: str, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed QACG comparison data by reusing generated baseline data."""
        self.logger.info("Organizing detailed QACG comparison data...")

        detailed_comparisons = {
            "target_system": target_system,
            "total_questions": len(target_data),
            "qacg_pairs": []
        }

        sample_size = min(len(target_data), self.config.max_questions)

        baseline_data_by_name = {}
        for baseline_name, result in baseline_results.items():
            baseline_data_by_name[baseline_name] = result.get("baseline_data", [])

        for i, target_qa in enumerate(target_data[:sample_size]):
            question = target_qa["question"]

            qacg_pair = {
                "question_id": i + 1,
                "question": question,
                "groundtruth": target_qa.get("groundtruth", target_qa.get("expected_answer", "")),

                "target_system": {
                    "name": target_system,
                    "answer": target_qa.get("rag_answer", ""),
                    "context": target_qa.get("context", []),
                    "metadata": target_qa.get("metadata", {})
                },

                "baselines": {}
            }

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
                    baseline_qacg = {
                        "name": f"Baseline_{baseline_name}",
                        "answer": f"Unable to generate baseline response for {baseline_name} quality",
                        "context": [f"Unable to generate baseline evidence for {baseline_name} quality"],
                        "quality_level": baseline_name.lower(),
                        "description": self._get_baseline_description(baseline_name),
                        "generation_instruction": self.baseline_prompts[baseline_name]["instruction"],
                        "metadata": {"error": "baseline_generation_failed"}
                    }

                qacg_pair["baselines"][baseline_name] = baseline_qacg

            detailed_comparisons["qacg_pairs"].append(qacg_pair)

        return detailed_comparisons
    
    def _get_baseline_description(self, baseline_name: str) -> str:
        """Get baseline quality level description."""
        descriptions = {
            "Good": "High-quality baseline: provides detailed and accurate responses with complete key information and clear logic",
            "Medium": "Medium-quality baseline: provides basically correct responses with some missing details",
            "Bad": "Low-quality baseline: responses have reduced accuracy with obvious errors or omissions"
        }
        return descriptions.get(baseline_name, "Unknown baseline")

    def _analyze_failures(self, pairwise_results: List[Dict]) -> Dict[str, Any]:
        """Analyze failure reasons and patterns from comparison results."""
        failure_reasons = []

        for result in pairwise_results:
            for qr in result["question_results"]:
                passage_judgment = qr.get("passage_judgment", {})
                reason = passage_judgment.get("reason", "")
                if reason:
                    failure_reasons.append(reason)

        reason_counts = defaultdict(int)
        for reason in failure_reasons:
            keywords = ["accuracy", "completeness", "relevance", "evidence", "logic", "error", "missing", "unclear"]
            for keyword in keywords:
                if keyword in reason.lower():
                    reason_counts[keyword] += 1

        return {
            "total_reasons": len(failure_reasons),
            "keyword_counts": dict(reason_counts),
            "top_reasons": sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "llm_model": self.config.llm_model,
            "max_questions": self.config.max_questions,
            "early_stop_elo_diff": self.config.early_stop_elo_diff,
            "early_stop_ci_threshold": self.config.early_stop_ci_threshold,
            "initial_elo": self.config.initial_elo,
            "k_factor": self.config.k_factor
        }
    
    def _save_tournament_result(self, result: Dict[str, Any]):
        """Save tournament results to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / "tournament_result.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        self._save_tournament_report(result, output_dir)

        self.logger.info(f"Tournament results saved to: {output_dir}")

    def _save_baseline_result(self, result: Dict[str, Any]):
        """Save baseline comparison results to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / "baseline_comparison.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        if "detailed_qacg_comparisons" in result:
            with open(output_dir / "qacg_detailed_comparisons.json", 'w', encoding='utf-8') as f:
                json.dump(result["detailed_qacg_comparisons"], f, ensure_ascii=False, indent=2, default=str)
            self.logger.info(f"Detailed QACG comparison data saved to: {output_dir / 'qacg_detailed_comparisons.json'}")

        self._save_baseline_report(result, output_dir)

        self.logger.info(f"Baseline comparison results saved to: {output_dir}")
    
    def _save_tournament_report(self, result: Dict[str, Any], output_dir: Path):
        """Save tournament report  supporting Swiss rounds and dynamic Elo pairing."""
        tournament_type = result.get("tournament_type", "swiss_tournament")

        with open(output_dir / "tournament_report.md", 'w', encoding='utf-8') as f:
            if tournament_type == "swiss_tournament":
                f.write("# DICE Tournament Report (Swiss System)\n\n")
            elif tournament_type == "full_round_robin":
                f.write("# DICE Tournament Report (Full Round-Robin)\n\n")
            else:
                f.write("# DICE Tournament Report (Dynamic Elo Pairing)\n\n")

            f.write("## Final Rankings (Dynamic Elo)\n\n")
            final_ranking = result["final_ranking"]
            final_elo_scores = result["final_elo_scores"]

            for i, system in enumerate(final_ranking, 1):
                elo_score = final_elo_scores[system]
                f.write(f"{i}. **{system}**: {elo_score:.1f}\n")

            if tournament_type == "swiss_tournament":
                f.write("\n## Swiss Tournament Progress\n\n")
                swiss_results = result["swiss_results"]
                match_records = swiss_results.get("match_records", [])
                total_rounds = swiss_results.get("total_rounds", 4)

                f.write(f"Total matches: {len(match_records)} ({total_rounds} rounds, 4 matches per round)\n\n")

                f.write("### Match Summary by Round\n")
                current_round = 1
                for i, match in enumerate(match_records):
                    if match.get('round', 1) != current_round:
                        current_round = match.get('round', 1)
                        f.write(f"\n#### Round {current_round}\n")

                    f.write(f"**Match {match['match_num']}**: {match['system_a']} (ELO: {match['old_elo_a']:.1f}) vs {match['system_b']} (ELO: {match['old_elo_b']:.1f})\n")
                    f.write(f"- Winner: {match['winner']}\n")
                    f.write(f"- Elo change: {match['system_a']} ({match['old_elo_a']:.1f}→{match['new_elo_a']:.1f}), {match['system_b']} ({match['old_elo_b']:.1f}→{match['new_elo_b']:.1f})\n\n")

                f.write("## Swiss System Explanation\n\n")
                f.write("- **Round pairing**: 4 rounds total, 4 matches per round, each team plays once per round\n")
                f.write("- **Smart pairing**: Each round selects teams with closest Elo ratings who haven't yet played\n")
                f.write("- **Dynamic updates**: Elo scores updated in real-time reflecting actual strength changes\n")
                f.write("- **No seed teams**: Initial Elo = 1500, purely learned from match results\n")
                f.write("- **Fairness**: Ensures each pair of teams plays exactly once\n\n")

            elif tournament_type == "full_round_robin":
                f.write("\n## Full Round-Robin Tournament Progress\n\n")
                rr = result["round_robin_results"]
                match_records = rr.get("match_records", [])
                f.write(f"Total matches: {len(match_records)} (all vs all, each pair plays once)\n\n")

                f.write("### Match Summary\n")
                for match in match_records:
                    f.write(f"**Match {match['match_num']}**: {match['system_a']} (ELO: {match['old_elo_a']:.1f}) vs {match['system_b']} (ELO: {match['old_elo_b']:.1f})\n")
                    f.write(f"- Winner: {match['winner']}\n")
                    f.write(f"- Elo change: {match['system_a']} ({match['old_elo_a']:.1f}→{match['new_elo_a']:.1f}), {match['system_b']} ({match['old_elo_b']:.1f}→{match['new_elo_b']:.1f})\n\n")

                f.write("## Full Round-Robin Explanation\n\n")
                f.write("- **Pairing method**: All teams compete against each other exactly once (total: N(N-1)/2 matches)\n")
                f.write("- **Scoring method**: Cumulative soft-win scoring with dynamic Elo updates\n")
                f.write("- **Coverage**: Complete pairing coverage avoids incomplete sampling bias\n\n")
            else:
                f.write("\n## Dynamic Pairing Progress\n\n")
                dynamic_results = result.get("dynamic_results")
                if dynamic_results:
                    match_records = dynamic_results.get("match_records", [])
                else:
                    match_records = []

                f.write(f"Total matches: {len(match_records)}\n\n")

                f.write("### Key Matches\n")
                for i, match in enumerate(match_records):
                    f.write(f"**Match {match['match_num']}**: {match['system_a']} (ELO: {match['old_elo_a']:.1f}) vs {match['system_b']} (ELO: {match['old_elo_b']:.1f})\n")
                    f.write(f"- Winner: {match['winner']}\n")
                    f.write(f"- Elo change: {match['system_a']} ({match['old_elo_a']:.1f}→{match['new_elo_a']:.1f}), {match['system_b']} ({match['old_elo_b']:.1f}→{match['new_elo_b']:.1f})\n\n")

                f.write("## Dynamic Elo Pairing System Explanation\n\n")
                f.write("- **Smart pairing**: Each round selects teams with closest Elo ratings who haven't yet played\n")
                f.write("- **Dynamic updates**: Elo scores updated in real-time reflecting actual strength changes\n")
                f.write("- **Efficiency**: Information gain maximized, redundant matches minimized\n")
                f.write("- **No seed teams**: Initial Elo = 1500, purely learned from match results\n")
                f.write("- **Convergence**: Ends when rankings stabilize or max matches reached\n\n")

            f.write("## Dynamic Failure Pattern Clustering Analysis\n\n")
            failure_clusters = result.get("failure_analysis", {})
            for cluster_id, cluster_data in failure_clusters.items():
                f.write(f"### {cluster_data['label']}\n")
                f.write(f"- Related systems: {', '.join(cluster_data['systems'][:5])}{'...' if len(cluster_data['systems']) > 5 else ''}\n")
                f.write(f"- Failure case count: {cluster_data['size']}\n")

                top_keywords = cluster_data.get('top_keywords', [])
                if top_keywords:
                    keyword_str = ', '.join([f'{k}({v} times)' for k, v in top_keywords[:3]])
                    f.write(f"- Keywords: {keyword_str}\n")
                f.write("\n")

            total_calls = result.get("total_llm_calls", 0)
            total_matches = len(match_records) if match_records else 0
            f.write(f"## Performance Statistics\n\n")
            f.write(f"- Total matches: {total_matches} (vs traditional 28, reduced by {(28-total_matches)/28*100:.1f}%)\n")
            f.write(f"- Total LLM calls: {total_calls}\n")
            f.write(f"- Estimated time: ~{total_calls/40:.1f} minutes (8×A100)\n")
            f.write(f"- Average matches per team: {total_matches*2/8:.1f}\n")

            ci_analysis = result.get("ci_analysis", {})
            if ci_analysis:
                f.write(f"\n## 95% Confidence Interval Analysis\n\n")
                f.write(f"- Mean score difference: {ci_analysis.get('mean_score_diff', 0):.2f}\n")
                f.write(f"- 95% CI: {ci_analysis.get('ci_95', 'N/A')}\n")
                f.write(f"- Statistical significance: {ci_analysis.get('significance', 'N/A')}\n")
    
    def _save_baseline_report(self, result: Dict[str, Any], output_dir: Path):
        """Save baseline comparison report."""
        with open(output_dir / "baseline_report.md", 'w', encoding='utf-8') as f:
            f.write("# DICE Baseline Comparison Report\n\n")

            target_system = result["target_system"]
            f.write(f"## Target System: {target_system}\n\n")

            f.write("## Baseline Comparison Results\n\n")
            summary = result["summary"]

            for baseline_name, comparison in summary["comparisons"].items():
                win_rate = comparison["win_rate"]
                conclusion = comparison["conclusion"]
                f.write(f"### vs {baseline_name} Baseline\n")
                f.write(f"- Win rate: {win_rate:.1%}\n")
                f.write(f"- Conclusion: {conclusion}\n\n")

            total_calls = result["total_llm_calls"]
            f.write(f"## Performance Statistics\n\n")
            f.write(f"- Total LLM calls: {total_calls}\n")
            f.write(f"- Estimated time: ~{total_calls/40:.1f} minutes\n")


def create_simplified_evaluator(config: SimplifiedDICEConfig = None) -> SimplifiedDICEEvaluator:
    """Create a simplified DICE evaluator instance."""
    return SimplifiedDICEEvaluator(config) 