#!/usr/bin/env python3
"""
DICE accuracy evaluation script.
Validates DICE credibility by comparing against human-annotated gold standard.
"""

import argparse
import logging
import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evaluation.dice_engine import SimplifiedDICEConfig
from src.utils.ragas_impl import RagasConfig
from src.evaluation.validator import UnifiedValidationEvaluator

def main():
    parser = argparse.ArgumentParser(description="Multi-RAG system accuracy validation")
    parser.add_argument("--qacg_files", nargs="+", required=True,
                       help="QACG file paths")
    parser.add_argument("--num_samples", type=int, default=200,
                       help="Number of evaluation samples")
    parser.add_argument("--annotation_file", type=str,
                       default="dice_human_annotations.json",
                       help="Human annotation file path")
    parser.add_argument("--output_dir", type=str, default="dice_validation_output",
                       help="Output directory")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--llm_model", type=str, default="deepseek-chat",
                       help="LLM model")
    parser.add_argument("--tournament_result_file", type=str,
                       default="dice_simplified_output/tournament_result.json",
                       help="Tournament result file for reusing existing judgments")
    parser.add_argument("--ragas", action="store_true",
                       help="Use RAGAS method (default: DICE)")
    parser.add_argument("--ragas_metrics", nargs="+",
                       default=["answer_relevancy", "context_precision", "context_recall", "faithfulness", "answer_correctness"],
                       help="RAGAS evaluation metrics")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.ragas:
        ragas_config = RagasConfig(
            llm_model=args.llm_model,
            metrics=args.ragas_metrics,
            api_key=os.environ.get("DEEPSEEK_API_KEY", "xxxxxxx"),
            base_url="https://api.deepseek.com"
        )
        evaluator = UnifiedValidationEvaluator(
            evaluation_method="ragas",
            ragas_config=ragas_config
        )
        evaluation_method = "RAGAS"
    else:
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

    print(f"{evaluation_method} system validation")
    print(f"QACG files: {len(args.qacg_files)}")
    print(f"Samples: {args.num_samples}")
    print(f"Method: {evaluation_method}")

    try:
        # Step 1: Sample evaluation pairs
        print("\nStep 1: Sampling evaluation pairs...")
        evaluation_pairs = evaluator.sample_evaluation_pairs(
            args.qacg_files, args.num_samples, args.random_seed
        )

        pairs_file = output_dir / "evaluation_pairs.json"
        with open(pairs_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_pairs, f, ensure_ascii=False, indent=2)
        print(f"Sampling complete, saved to: {pairs_file}")

        # Step 1.5: Check or create annotation file
        print(f"\nStep 1.5: Checking annotation file: {args.annotation_file}")
        annotation_file_path = Path(args.annotation_file)

        if not annotation_file_path.exists():
            print("Annotation file not found, creating template...")
            evaluator._create_annotation_template(args.annotation_file)
            print(f"Template created: {args.annotation_file}")
            print("Instructions:")
            print("   - Each pair_id needs 3 expert votes")
            print("   - Options: 'A wins', 'B wins', 'Tie'")
            print("   - Judge based on retrieval and answer quality")
            print("Note: Complete annotations before generating validation report.")
            print("Continuing with evaluation...\n")
        else:
            print(f"Annotation file found: {args.annotation_file}")

        # Step 2: Check or run evaluation
        results_file = output_dir / f"{evaluation_method.lower()}_results.json"
        evaluation_results = None

        print(f"\nStep 2: Checking {evaluation_method} results...")
        if results_file.exists():
            print(f"Found existing results: {results_file}")

            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    evaluation_results = json.load(f)
                print(f"Loaded {len(evaluation_results)} results")

                if len(evaluation_results) != len(evaluation_pairs):
                    print(f"Result count ({len(evaluation_results)}) != pair count ({len(evaluation_pairs)}), re-running...")
                    evaluation_results = None
                else:
                    print("Result count matches, using existing results")
            except Exception as e:
                print(f"Failed to load results: {e}")
                print("Re-running evaluation...")
                evaluation_results = None

        if evaluation_results is None:
            print(f"\nRunning {evaluation_method} evaluation...")
            evaluation_results = evaluator.run_evaluation(evaluation_pairs)

            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            print(f"{evaluation_method} evaluation complete, saved to: {results_file}")

        # Step 3: Try to load annotations and generate report
        print(f"\nStep 3: Checking annotation status...")

        try:
            gold_labels = evaluator.load_human_annotations(args.annotation_file)

            if len(gold_labels) == 0:
                print("Annotation file exists but has no valid annotations")
                print("Complete annotations and re-run for validation report")
                print(f"Evaluation results saved to: {results_file}")
                return

            print(f"Loaded {len(gold_labels)} human annotations")

            print("\nStep 4: Calculating agreement metrics...")
            agreement_metrics = evaluator.calculate_agreement(evaluation_results, gold_labels)

            print("Step 5: Calculating Elo correlation...")
            correlation_metrics = evaluator.calculate_elo_correlation(evaluation_results, gold_labels)

            print("Step 6: Generating validation report...")
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            report_file = output_dir / f"validation_report_{timestamp}.json"
            evaluator.generate_validation_report(
                agreement_metrics, correlation_metrics, evaluation_results, gold_labels, str(report_file)
            )

            print(f"\nValidation report saved to: {report_file}")

        except Exception as e:
            print(f"Cannot generate validation report: {e}")
            print("Possible reasons:")
            print("   1. Annotations not yet completed")
            print("   2. Incorrect annotation format")
            print("   3. Empty expert_votes field")
            print(f"\nEvaluation complete, results saved to: {results_file}")
            print("Re-run after completing annotations to generate validation report")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
