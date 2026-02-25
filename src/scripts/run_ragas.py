#!/usr/bin/env python3
"""
DICE RAGAS main entry point.
RAGAS-based system scoring and ranking.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evaluation.ragas_wrapper import RagasDiceEvaluator, RagasDiceConfig


def log_and_print(message):
    """Output to both console and log file."""
    print(message)
    logging.info(message)


def setup_logging(output_dir: str):
    """Set up logging to console and file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_file = Path(output_dir) / "ragas_dice.log"
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if log_file.exists() and log_file.stat().st_size > 0:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('\n')

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"RAGAS DICE evaluation started - {timestamp}")


def discover_qacg_files(input_dir: str) -> List[str]:
    """Discover QACG files in directory."""
    qacg_dir = Path(input_dir)
    if not qacg_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    qacg_files = list(qacg_dir.glob("qacg_*.json"))
    qacg_files.sort()

    return [str(f) for f in qacg_files]


def print_ranking_summary(ranking_result: Dict[str, Any]):
    """Print ranking summary."""
    ranking = ranking_result["ranking"]

    log_and_print(f"\nRAGAS DICE System Ranking:")

    for item in ranking:
        rank = item["rank"]
        system_name = item["system_name"]
        score = item["composite_score"]

        if "error" in item:
            log_and_print(f"  {rank}. {system_name}: evaluation failed")
            log_and_print(f"      Error: {item['error']}")
        else:
            std = item.get("composite_std", 0.0)
            success_rate = item.get("success_rate", 1.0)
            valid_q = item.get("valid_questions", 0)
            total_q = item.get("total_questions", 0)

            log_and_print(f"  {rank}. {system_name}: {score:.4f} +/- {std:.4f}")
            log_and_print(f"      Valid questions: {valid_q}/{total_q} ({success_rate:.1%})")

            if "metric_averages" in item:
                metrics_str = []
                for metric, avg_score in item["metric_averages"].items():
                    metrics_str.append(f"{metric}={avg_score:.3f}")
                log_and_print(f"      Metrics: {', '.join(metrics_str)}")

    log_and_print(f"\nEvaluation stats:")
    log_and_print(f"  - Total systems: {ranking_result['total_systems']}")
    log_and_print(f"  - Successful: {ranking_result['successful_systems']}")
    log_and_print(f"  - Failed: {ranking_result['failed_systems']}")

    config = ranking_result.get("config", {})
    log_and_print(f"\nConfig:")
    log_and_print(f"  - Model: {config.get('llm_model', 'N/A')}")
    log_and_print(f"  - Metrics: {', '.join(config.get('metrics', []))}")
    log_and_print(f"  - Batch size: {config.get('batch_size', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="DICE RAGAS evaluation system")

    parser.add_argument("--input_dir", default="qacg_output",
                       help="QACG file directory (default: qacg_output)")
    parser.add_argument("--output_dir", default="ragas_dice_output",
                       help="Output directory (default: ragas_dice_output)")

    parser.add_argument("--llm_model", default="deepseek-chat",
                       help="LLM model name (default: deepseek-chat)")
    parser.add_argument("--embeddings_model", default="BAAI/bge-small-zh-v1.5",
                       help="Embedding model name (small model to save memory)")

    parser.add_argument("--metrics", nargs="+",
                       default=["faithfulness", "answer_relevancy", "context_relevance"],
                       help="RAGAS evaluation metrics")

    parser.add_argument("--max_workers", type=int, default=1,
                       help="Max concurrent workers (default: 1, recommend 2-4)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (default: 1, recommend 3-10)")

    parser.add_argument("--target_system",
                       help="Evaluate only this system (name without qacg_ prefix)")

    parser.add_argument("--safe_mode", action="store_true",
                       help="Safe mode: disable concurrency, batch size 1")
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode: verbose logging")

    args = parser.parse_args()

    if args.safe_mode:
        args.max_workers = 1
        args.batch_size = 1
        log_and_print("Safe mode enabled: single-thread, batch size 1")

    setup_logging(args.output_dir)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        log_and_print("Debug mode enabled")

    try:
        log_and_print("DICE RAGAS evaluation system")
        log_and_print(f"Input: {args.input_dir}")
        log_and_print(f"Output: {args.output_dir}")
        log_and_print(f"LLM: {args.llm_model}")
        log_and_print(f"Metrics: {', '.join(args.metrics)}")

        log_and_print(f"\nScanning QACG files...")
        qacg_files = discover_qacg_files(args.input_dir)

        if not qacg_files:
            log_and_print(f"No qacg_*.json files found in {args.input_dir}")
            return

        if args.target_system:
            target_files = [f for f in qacg_files if args.target_system in f]
            if not target_files:
                log_and_print(f"System {args.target_system} not found")
                return
            qacg_files = target_files
            log_and_print(f"Evaluating only: {args.target_system}")

        log_and_print(f"Found {len(qacg_files)} QACG files:")
        for i, f in enumerate(qacg_files, 1):
            system_name = Path(f).stem.replace("qacg_", "")
            log_and_print(f"  {i}. {system_name}")

        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            log_and_print("Warning: DEEPSEEK_API_KEY not set, using default key")
            api_key = "xxxxxxx"

        config = RagasDiceConfig(
            llm_model=args.llm_model,
            embeddings_model=args.embeddings_model,
            metrics=args.metrics,
            api_key=api_key,
            base_url="https://api.deepseek.com",
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            batch_size=args.batch_size
        )

        log_and_print(f"\nPerformance config:")
        log_and_print(f"  - Workers: {args.max_workers}")
        log_and_print(f"  - Batch size: {args.batch_size}")

        if args.max_workers > 1:
            estimated_speedup = min(args.max_workers, 3)
            log_and_print(f"  - Mode: concurrent (~{estimated_speedup:.1f}x speedup)")
            log_and_print(f"  - Note: may hit API rate limits; use --safe_mode if needed")
        else:
            log_and_print(f"  - Mode: sequential (safe but slower)")

        evaluator = RagasDiceEvaluator(config)

        if len(qacg_files) == 1:
            log_and_print(f"\nSingle-system evaluation mode")
            qacg_file = qacg_files[0]
            system_name = Path(qacg_file).stem.replace("qacg_", "")

            result = evaluator.evaluate_single_system(qacg_file, system_name)

            log_and_print(f"\nSystem {system_name} evaluation complete:")
            log_and_print(f"  Score: {result['composite_score']:.4f} +/- {result.get('composite_std', 0.0):.4f}")
            log_and_print(f"  Valid questions: {result.get('valid_questions', 0)}/{result['total_questions']}")
            log_and_print(f"  Success rate: {result.get('success_rate', 1.0):.1%}")

            if "metric_averages" in result:
                log_and_print(f"  Per-metric scores:")
                for metric, score in result["metric_averages"].items():
                    std = result.get("metric_std", {}).get(metric, 0.0)
                    log_and_print(f"    - {metric}: {score:.4f} +/- {std:.4f}")

        else:
            log_and_print(f"\nMulti-system ranking mode")
            ranking_result = evaluator.evaluate_multiple_systems(qacg_files)
            print_ranking_summary(ranking_result)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"RAGAS DICE evaluation completed - {timestamp}")
        log_and_print(f"\nResults saved to: {args.output_dir}")

    except KeyboardInterrupt:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"User interrupted - {timestamp}")
        log_and_print("\nUser interrupted")
        sys.exit(1)

    except Exception as e:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.error(f"Error - {timestamp}")
        log_and_print(f"Error: {e}")
        logging.exception("Details:")
        sys.exit(1)


if __name__ == "__main__":
    main()
