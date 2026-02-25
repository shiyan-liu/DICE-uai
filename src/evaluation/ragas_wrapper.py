#!/usr/bin/env python3
"""RAGAS DICE core module: system scoring and ranking based on RAGAS framework."""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Import RAGAS evaluator
from src.utils.ragas_impl import RagasEvaluator, RagasConfig


@dataclass
class RagasDiceConfig:
    """RAGAS DICE configuration."""
    llm_model: str = "deepseek-chat"
    embeddings_model: str = "BAAI/bge-small-zh-v1.5"  # Smaller model to save memory
    metrics: List[str] = None
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    output_dir: str = "ragas_dice_output"
    max_workers: int = 1
    batch_size: int = 5

    def __post_init__(self):
        if self.metrics is None:
            # Three core dimensions from the RAGAS paper
            self.metrics = [
                "faithfulness",
                "answer_relevancy",
                "context_relevance"
            ]

        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


class RagasDiceEvaluator:
    """RAGAS DICE evaluator."""

    def __init__(self, config: RagasDiceConfig):
        self.config = config
        self.logger = logging.getLogger("RagasDice")
        self._setup_logger()

        # Create RAGAS config
        self.ragas_config = RagasConfig(
            llm_model=config.llm_model,
            embeddings_model=config.embeddings_model,
            metrics=config.metrics,
            api_key=config.api_key,
            base_url=config.base_url
        )

        # Create RAGAS evaluator
        self.ragas_evaluator = RagasEvaluator(self.ragas_config)

        self.logger.info(f"RAGAS DICE evaluator initialized, model: {config.llm_model}")

    def _setup_logger(self):
        """Set up logging."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def evaluate_single_system(self, qacg_file: str, system_name: str = None) -> Dict[str, Any]:
        """Evaluate a single system's QACG data and return result dict."""
        qacg_path = Path(qacg_file)
        if not qacg_path.exists():
            raise FileNotFoundError(f"QACG file not found: {qacg_file}")

        # Determine system name
        if system_name is None:
            system_name = qacg_path.stem.replace("qacg_", "")

        self.logger.info(f"Evaluating system: {system_name}")

        # Load QACG data
        with open(qacg_file, 'r', encoding='utf-8') as f:
            qacg_data = json.load(f)

        self.logger.info(f"Loaded {len(qacg_data)} QA pairs")

        # Batch evaluation
        all_scores = []
        total_items = len(qacg_data)

        self.logger.info(f"Config: {self.config.max_workers} workers, batch_size: {self.config.batch_size}")
        if self.config.max_workers > 1:
            self.logger.info(f"Concurrent mode enabled, ~{self.config.max_workers}x speedup")
        else:
            self.logger.info(f"Single-thread mode (safe mode)")

        # Process in batches
        for i in range(0, total_items, self.config.batch_size):
            batch = qacg_data[i:i+self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            total_batches = (total_items + self.config.batch_size - 1) // self.config.batch_size

            self.logger.info(f"Batch {batch_num}/{total_batches}: processing {len(batch)} QA pairs (items {i+1}-{min(i+self.config.batch_size, total_items)})")

            # Use concurrent evaluation
            batch_scores = self._evaluate_batch_concurrent(batch, i, system_name, total_items)
            all_scores.extend(batch_scores)

            # Batch summary
            completed = min(i + self.config.batch_size, total_items)
            progress = completed / total_items * 100

            # Batch statistics
            batch_success = len([s for s in batch_scores if "error" not in s])
            batch_avg_score = sum(s["composite_score"] for s in batch_scores if "error" not in s) / max(batch_success, 1)

            self.logger.info(f"Batch {batch_num} done: {batch_success}/{len(batch)} succeeded, avg={batch_avg_score:.4f}, progress={completed}/{total_items} ({progress:.1f}%)")

        # Calculate statistics
        system_result = self._calculate_system_statistics(system_name, all_scores)

        # Save detailed results
        detail_file = Path(self.config.output_dir) / f"{system_name}_ragas_details.json"
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump({
                "system_name": system_name,
                "total_questions": len(all_scores),
                "detailed_scores": all_scores,
                "statistics": system_result
            }, f, ensure_ascii=False, indent=2)

        self.logger.info(f"System {system_name} evaluation done, composite={system_result['composite_score']:.4f}, saved to {detail_file}")

        return system_result

    def _evaluate_single_question(self, qa_item: Dict[str, Any], question_idx: int, total_questions: int, system_name: str) -> Dict[str, Any]:
        """Evaluate a single QA pair."""
        try:
            question = qa_item.get("question", "")[:100]
            self.logger.info(f"Question {question_idx}/{total_questions}: {question}...")

            # Check if running in a worker thread
            import threading
            thread_id = threading.current_thread().ident
            is_main_thread = thread_id == threading.main_thread().ident

            if not is_main_thread:
                # In worker thread, create independent evaluator to avoid event loop conflicts
                from ragas_evaluator import RagasEvaluator, RagasConfig
                thread_config = RagasConfig(
                    llm_model=self.config.llm_model,
                    embeddings_model=self.config.embeddings_model,
                    metrics=self.config.metrics,
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
                thread_evaluator = RagasEvaluator(thread_config)
                scores = thread_evaluator.evaluate_single_qacg(qa_item)
                composite_score = thread_evaluator.calculate_composite_score(scores)
            else:
                # In main thread, use shared evaluator
                scores = self.ragas_evaluator.evaluate_single_qacg(qa_item)
                composite_score = self.ragas_evaluator.calculate_composite_score(scores)

            result = {
                "question": qa_item.get("question", ""),
                "scores": scores,
                "composite_score": composite_score,
                "question_idx": question_idx
            }

            # Print question evaluation result
            self._print_question_result(result, system_name, question_idx, total_questions)

            return result

        except Exception as e:
            import traceback
            error_msg = f"QA evaluation failed: {e}"
            self.logger.error(f"Question {question_idx}/{total_questions}: {error_msg}")

            # Special hint for event loop errors
            if "event loop" in str(e).lower() or "asyncio" in str(e).lower():
                self.logger.error("Async event loop conflict detected, consider --safe_mode or fewer --max_workers")

            # Add default scores
            result = {
                "question": qa_item.get("question", ""),
                "scores": {metric: 0.0 for metric in self.config.metrics},
                "composite_score": 0.0,
                "error": str(e),
                "question_idx": question_idx
            }

            return result

    def _print_question_result(self, result: Dict[str, Any], system_name: str, question_idx: int, total_questions: int):
        """Print evaluation result for a single question."""
        question = result["question"][:80] + "..." if len(result["question"]) > 80 else result["question"]
        composite_score = result["composite_score"]
        scores = result["scores"]

        # Build metric score string
        metric_strs = []
        for metric, score in scores.items():
            if score is not None:
                metric_strs.append(f"{metric}={score:.3f}")
            else:
                metric_strs.append(f"{metric}=N/A")

        metrics_display = ", ".join(metric_strs)

        self.logger.info(f"[{system_name}] Q {question_idx}/{total_questions} done: {question}")
        self.logger.info(f"  composite={composite_score:.4f}, metrics: {metrics_display}")

        # Log progress every 10 questions
        if question_idx % 10 == 0:
            progress = question_idx / total_questions * 100
            self.logger.info(f"  [{system_name}] progress: {question_idx}/{total_questions} ({progress:.1f}%)")

    def _evaluate_batch_concurrent(self, batch: List[Dict[str, Any]], batch_start_idx: int, system_name: str, total_questions: int) -> List[Dict[str, Any]]:
        """Evaluate a batch of QA pairs concurrently."""
        if self.config.max_workers <= 1:
            # Single-thread mode
            batch_scores = []
            for i, qa_item in enumerate(batch):
                question_idx = batch_start_idx + i + 1
                result = self._evaluate_single_question(qa_item, question_idx, total_questions, system_name)
                batch_scores.append(result)
            return batch_scores

        # Multi-thread mode with error monitoring and auto-fallback
        batch_scores = [None] * len(batch)
        asyncio_errors = 0
        max_asyncio_errors = 3  # Max allowed async errors

        try:
            with ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(batch))) as executor:
                # Submit all tasks
                future_to_idx = {}
                for i, qa_item in enumerate(batch):
                    question_idx = batch_start_idx + i + 1
                    future = executor.submit(
                        self._evaluate_single_question,
                        qa_item,
                        question_idx,
                        total_questions,
                        system_name
                    )
                    future_to_idx[future] = i

                # Collect results
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        batch_scores[idx] = result
                    except Exception as e:
                        # Check for async-related errors
                        error_str = str(e).lower()
                        if "event loop" in error_str or "asyncio" in error_str or "bound to a different" in error_str:
                            asyncio_errors += 1
                            self.logger.error(f"Async error #{asyncio_errors}: {e}")

                            # Too many async errors, switch to safe mode
                            if asyncio_errors >= max_asyncio_errors:
                                self.logger.error(f"Too many async errors ({asyncio_errors}), switching to safe mode")
                                self.config.max_workers = 1
                                # Cancel remaining futures and switch to single-thread
                                for remaining_future in future_to_idx:
                                    if not remaining_future.done():
                                        remaining_future.cancel()

                                # Process remaining incomplete tasks
                                remaining_items = [batch[i] for i, score in enumerate(batch_scores) if score is None]
                                remaining_start = batch_start_idx + len([s for s in batch_scores if s is not None])

                                if remaining_items:
                                    self.logger.info(f"Processing remaining {len(remaining_items)} tasks in single-thread mode...")
                                    for i, qa_item in enumerate(remaining_items):
                                        question_idx = remaining_start + i + 1
                                        safe_result = self._evaluate_single_question(qa_item, question_idx, total_questions, system_name)
                                        batch_scores[remaining_start - batch_start_idx + i] = safe_result

                                break

                        # Create error result
                        question_idx = batch_start_idx + idx + 1
                        batch_scores[idx] = {
                            "question": batch[idx].get("question", ""),
                            "scores": {metric: 0.0 for metric in self.config.metrics},
                            "composite_score": 0.0,
                            "error": str(e),
                            "question_idx": question_idx
                        }

                        self.logger.error(f"Concurrent task failed: {e}")

        except Exception as e:
            self.logger.error(f"Critical concurrent error, switching to safe mode: {e}")
            self.config.max_workers = 1

            # Re-process entire batch in single-thread
            batch_scores = []
            for i, qa_item in enumerate(batch):
                question_idx = batch_start_idx + i + 1
                result = self._evaluate_single_question(qa_item, question_idx, total_questions, system_name)
                batch_scores.append(result)

        return batch_scores

    def _evaluate_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a batch of QA pairs (kept for compatibility)."""
        return self._evaluate_batch_concurrent(batch, 0, "unknown", len(batch))

    def _calculate_system_statistics(self, system_name: str, all_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate system statistics."""
        if not all_scores:
            return {
                "system_name": system_name,
                "composite_score": 0.0,
                "total_questions": 0,
                "metric_averages": {},
                "metric_std": {},
                "valid_questions": 0
            }

        # Extract valid scores
        valid_scores = [item for item in all_scores if "error" not in item]

        # Calculate per-metric averages
        metric_sums = {metric: 0.0 for metric in self.config.metrics}
        metric_counts = {metric: 0 for metric in self.config.metrics}
        composite_scores = []

        for item in valid_scores:
            scores = item["scores"]
            composite_scores.append(item["composite_score"])

            for metric in self.config.metrics:
                if metric in scores and scores[metric] is not None:
                    metric_sums[metric] += scores[metric]
                    metric_counts[metric] += 1

        # Calculate averages
        metric_averages = {}
        metric_std = {}

        for metric in self.config.metrics:
            if metric_counts[metric] > 0:
                metric_averages[metric] = metric_sums[metric] / metric_counts[metric]

                # Calculate standard deviation
                if len(valid_scores) > 1:
                    values = [item["scores"].get(metric, 0) for item in valid_scores
                             if metric in item["scores"] and item["scores"][metric] is not None]
                    if values:
                        metric_std[metric] = float(np.std(values))
                    else:
                        metric_std[metric] = 0.0
                else:
                    metric_std[metric] = 0.0
            else:
                metric_averages[metric] = 0.0
                metric_std[metric] = 0.0

        # Calculate composite score
        if composite_scores:
            overall_composite = sum(composite_scores) / len(composite_scores)
            composite_std = float(np.std(composite_scores)) if len(composite_scores) > 1 else 0.0
        else:
            overall_composite = 0.0
            composite_std = 0.0

        return {
            "system_name": system_name,
            "composite_score": overall_composite,
            "composite_std": composite_std,
            "total_questions": len(all_scores),
            "valid_questions": len(valid_scores),
            "metric_averages": metric_averages,
            "metric_std": metric_std,
            "success_rate": len(valid_scores) / len(all_scores) if all_scores else 0.0
        }

    def evaluate_multiple_systems(self, qacg_files: List[str]) -> Dict[str, Any]:
        """Evaluate multiple systems and generate ranking."""
        self.logger.info(f"Starting RAGAS DICE multi-system evaluation")
        self.logger.info(f"Systems to evaluate: {len(qacg_files)}")

        # List systems
        for i, qacg_file in enumerate(qacg_files, 1):
            system_name = Path(qacg_file).stem.replace("qacg_", "")
            self.logger.info(f"  {i}. {system_name}")

        # Evaluate each system
        system_results = []

        for i, qacg_file in enumerate(qacg_files, 1):
            system_name = Path(qacg_file).stem.replace("qacg_", "")

            self.logger.info(f"Evaluating system {i}/{len(qacg_files)}: {system_name}")

            try:
                result = self.evaluate_single_system(qacg_file, system_name)
                system_results.append(result)

                self.logger.info(f"System {system_name} done, score={result['composite_score']:.4f}")

            except Exception as e:
                self.logger.error(f"System {system_name} evaluation failed: {e}")
                # Add default result
                system_results.append({
                    "system_name": system_name,
                    "composite_score": 0.0,
                    "total_questions": 0,
                    "error": str(e)
                })

        # Generate ranking
        ranking_result = self._generate_ranking(system_results)

        # Save full results
        output_file = Path(self.config.output_dir) / "ragas_dice_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ranking_result, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Full results saved to: {output_file}")

        return ranking_result

    def _generate_ranking(self, system_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate system ranking."""
        # Sort by composite score
        valid_results = [r for r in system_results if "error" not in r]
        error_results = [r for r in system_results if "error" in r]

        # Sort
        ranked_systems = sorted(valid_results, key=lambda x: x["composite_score"], reverse=True)

        # Build ranking
        ranking = []
        for i, result in enumerate(ranked_systems, 1):
            ranking.append({
                "rank": i,
                "system_name": result["system_name"],
                "composite_score": result["composite_score"],
                "composite_std": result.get("composite_std", 0.0),
                "total_questions": result["total_questions"],
                "valid_questions": result.get("valid_questions", result["total_questions"]),
                "success_rate": result.get("success_rate", 1.0),
                "metric_averages": result.get("metric_averages", {})
            })

        # Append failed systems
        for result in error_results:
            ranking.append({
                "rank": len(ranked_systems) + 1,
                "system_name": result["system_name"],
                "composite_score": 0.0,
                "error": result["error"]
            })

        return {
            "evaluation_type": "RAGAS_DICE",
            "total_systems": len(system_results),
            "successful_systems": len(valid_results),
            "failed_systems": len(error_results),
            "ranking": ranking,
            "config": {
                "llm_model": self.config.llm_model,
                "metrics": self.config.metrics,
                "batch_size": self.config.batch_size
            }
        }
