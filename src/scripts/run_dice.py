#!/usr/bin/env python3
"""
DICE main entry point.
Supports three scenarios:
A. Eight-system tournament (Swiss-style)
B. Single system vs virtual baseline
C. Full round-robin pairwise comparison
"""

import argparse
import logging
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evaluation.dice_engine import SimplifiedDICEEvaluator, SimplifiedDICEConfig


def log_and_print(message):
    """Output to both console and log file."""
    print(message)
    logging.info(message)


def setup_logging():
    """Set up logging to console and file."""
    import os
    from datetime import datetime

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_file = "dice.log"
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('\n')

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"DICE evaluation started - {timestamp}")


def scenario_a_tournament(args):
    """Scenario A: Eight-system tournament."""
    log_and_print("Scenario A: Eight-system tournament")

    qacg_dir = Path(args.input_dir)
    qacg_files = list(qacg_dir.glob("qacg_*.json"))

    if len(qacg_files) < 4:
        log_and_print(f"Need at least 4 QACG files, found {len(qacg_files)}")
        return

    qacg_files = qacg_files[:8]
    log_and_print(f"QACG files:")
    for f in qacg_files:
        log_and_print(f"  - {f.name}")

    import os
    config = SimplifiedDICEConfig(
        llm_model=args.llm_model,
        max_questions=args.max_questions,
        output_dir=args.output_dir,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        enable_deep_thinking=not args.no_deep_thinking
    )

    log_and_print(f"Config: {args.max_workers} workers, batch size: {args.batch_size}")

    evaluator = SimplifiedDICEEvaluator(config)
    result = evaluator.scenario_a_tournament([str(f) for f in qacg_files])

    log_and_print("\nTournament results (dynamic Elo pairing):")
    final_ranking = result["final_ranking"]
    final_elo_scores = result["final_elo_scores"]

    for i, system in enumerate(final_ranking, 1):
        elo_score = final_elo_scores[system]
        log_and_print(f"  {i}. {system}: {elo_score:.1f}")

    tournament_type = result.get("tournament_type", "swiss_tournament")

    if tournament_type == "swiss_tournament":
        swiss_results = result["swiss_results"]
        total_matches = len(swiss_results["match_records"])
        total_rounds = swiss_results.get("total_rounds", 4)
        efficiency = (28 - total_matches) / 28 * 100

        log_and_print(f"\nSwiss tournament summary:")
        log_and_print(f"  - Total matches: {total_matches} ({total_rounds} rounds)")
        log_and_print(f"  - Efficiency gain: {efficiency:.1f}% fewer matches")
        log_and_print(f"  - Avg matches per team: {total_matches*2/8:.1f}")
    else:
        dynamic_results = result["dynamic_results"]
        total_matches = len(dynamic_results["match_records"])
        efficiency = (28 - total_matches) / 28 * 100

        log_and_print(f"\nDynamic Elo pairing summary:")
        log_and_print(f"  - Total matches: {total_matches} (vs 28 for round-robin)")
        log_and_print(f"  - Efficiency gain: {efficiency:.1f}% fewer matches")
        log_and_print(f"  - Avg matches per team: {total_matches*2/8:.1f}")

    total_calls = result["total_llm_calls"]
    log_and_print(f"\nStats: {total_calls} LLM calls, ~{total_calls/40:.1f}min (8xA100)")
    log_and_print(f"Results saved to: {args.output_dir}")


def scenario_c_all_pairs(args):
    """Scenario C: Full round-robin pairwise comparison."""
    log_and_print("Scenario C: Full round-robin pairwise comparison")

    qacg_dir = Path(args.input_dir)
    qacg_files = list(qacg_dir.glob("qacg_*.json"))

    if len(qacg_files) < 2:
        log_and_print(f"Need at least 2 QACG files, found {len(qacg_files)}")
        return

    if not getattr(args, "use_all", False) and len(qacg_files) > 8:
        qacg_files = qacg_files[:8]
        log_and_print("More than 8 systems; using first 8 for comparability. Use --use_all to override.")

    log_and_print(f"QACG files:")
    for f in qacg_files:
        log_and_print(f"  - {f.name}")

    import os
    config = SimplifiedDICEConfig(
        llm_model=args.llm_model,
        max_questions=args.max_questions,
        output_dir=args.output_dir,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        enable_deep_thinking=not args.no_deep_thinking
    )

    log_and_print(f"Config: {args.max_workers} workers, batch size: {args.batch_size}")

    evaluator = SimplifiedDICEEvaluator(config)
    result = evaluator.scenario_c_full_round_robin([str(f) for f in qacg_files])

    log_and_print("\nRound-robin results:")
    final_ranking = result["final_ranking"]
    final_elo_scores = result["final_elo_scores"]
    for i, system in enumerate(final_ranking, 1):
        elo_score = final_elo_scores[system]
        log_and_print(f"  {i}. {system}: {elo_score:.1f}")

    rr = result.get("round_robin_results", {})
    total_matches = len(rr.get("match_records", []))
    n = len(final_ranking)
    expected = n * (n - 1) // 2
    log_and_print(f"\nRound-robin summary:")
    log_and_print(f"  - Total matches: {total_matches} (expected: {expected})")
    log_and_print(f"  - Avg matches per team: {(total_matches*2)/max(n,1):.1f}")

    total_calls = result["total_llm_calls"]
    log_and_print(f"\nStats: {total_calls} LLM calls, ~{total_calls/40:.1f}min (8xA100)")
    log_and_print(f"Results saved to: {args.output_dir}")

def scenario_b_baseline(args):
    """Scenario B: Single system vs virtual baseline."""
    log_and_print("Scenario B: Single system vs virtual baseline")

    target_file = Path(args.target_file)
    if not target_file.exists():
        log_and_print(f"Target file not found: {target_file}")
        return

    log_and_print(f"Target system: {target_file.name}")
    log_and_print(f"Config: {args.max_workers} workers, batch size: {args.batch_size}")

    import os
    config = SimplifiedDICEConfig(
        llm_model=args.llm_model,
        max_questions=args.max_questions,
        output_dir=args.output_dir,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        enable_deep_thinking=not args.no_deep_thinking
    )

    evaluator = SimplifiedDICEEvaluator(config)
    result = evaluator.scenario_b_baseline_comparison(str(target_file), args.target_system,
                                                     "dice_simplified_output/tournament_report.md")

    log_and_print(f"\n{result['target_system']} vs virtual baselines:")
    summary = result["summary"]

    for baseline_name, comparison in summary["comparisons"].items():
        win_rate = comparison["win_rate"]
        conclusion = comparison["conclusion"]
        log_and_print(f"  - vs {baseline_name}: {win_rate:.1%} - {conclusion}")

    total_calls = result["total_llm_calls"]
    log_and_print(f"\nStats: {total_calls} LLM calls, ~{total_calls/40:.1f}min")
    log_and_print(f"Results saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DICE evaluation system")

    parser.add_argument("--llm_model", default="deepseek-chat", help="LLM model name")
    parser.add_argument("--max_questions", type=int, default=70, help="Max questions")
    parser.add_argument("--output_dir", default="dice_simplified_output", help="Output directory")

    parser.add_argument("--no-deep-thinking", action="store_true",
                        help="Disable deep thinking mode, use direct output")

    parser.add_argument("--max_workers", type=int, default=1,
                        help="Max concurrent workers (dual GPU: 4-6, single GPU: 2)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (dual GPU: 8-12, single GPU: 4-6)")

    subparsers = parser.add_subparsers(dest="scenario", help="Scenario selection")

    parser_a = subparsers.add_parser("tournament", help="Scenario A: Eight-system tournament")
    parser_a.add_argument("--input_dir", default="qacg_output", help="QACG file directory")

    parser_b = subparsers.add_parser("baseline", help="Scenario B: Single system vs baseline")
    parser_b.add_argument("target_file", help="Target system QACG file")
    parser_b.add_argument("--target_system", help="Target system name (optional)")

    parser_c = subparsers.add_parser("allpairs", help="Scenario C: Full round-robin")
    parser_c.add_argument("--input_dir", default="qacg_output", help="QACG file directory")
    parser_c.add_argument("--use_all", action="store_true", help="Use all systems (default: max 8)")

    args = parser.parse_args()

    if not args.scenario:
        parser.print_help()
        return

    setup_logging()

    try:
        if args.scenario == "tournament":
            scenario_a_tournament(args)
        elif args.scenario == "baseline":
            scenario_b_baseline(args)
        elif args.scenario == "allpairs":
            scenario_c_all_pairs(args)

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"DICE evaluation completed - {timestamp}")

    except KeyboardInterrupt:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"User interrupted - {timestamp}")
        log_and_print("\nUser interrupted")
        sys.exit(1)
    except Exception as e:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.error(f"Error - {timestamp}")
        log_and_print(f"Error: {e}")
        logging.exception("Details:")
        sys.exit(1)


if __name__ == "__main__":
    main()
