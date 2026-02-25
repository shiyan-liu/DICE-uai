
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score
from scipy.stats import kendalltau

from src.evaluation.dice_engine import SimplifiedDICEEvaluator, SimplifiedDICEConfig
from src.utils.ragas_impl import RagasEvaluator, RagasConfig

class DICEValidationEvaluator:
    """DICE validation evaluator for measuring DICE accuracy."""

    def __init__(self, config: SimplifiedDICEConfig, tournament_result_file: str = None):
        self.config = config
        self.logger = logging.getLogger("DICEValidation")
        self.dice_evaluator = SimplifiedDICEEvaluator(config)
        self.tournament_result_file = tournament_result_file
        self.tournament_results = None

        self._setup_logger()

        if self.tournament_result_file and Path(self.tournament_result_file).exists():
            self._load_tournament_results()

    def _setup_logger(self):
        """Set up logger."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _load_tournament_results(self):
        """Load tournament result file."""
        try:
            self.logger.info(f"Loading tournament results: {self.tournament_result_file}")
            with open(self.tournament_result_file, 'r', encoding='utf-8') as f:
                self.tournament_results = json.load(f)
            records = self.tournament_results.get('swiss_results', {}).get('match_records', [])
            self.logger.info(f"Loaded tournament results with {len(records)} match records")
        except Exception as e:
            self.logger.error(f"Failed to load tournament results: {e}")
            self.tournament_results = None

    def _find_tournament_match(self, system_a: str, system_b: str, question: str) -> Dict[str, Any]:
        """Find matching result in tournament records."""
        if not self.tournament_results:
            return None

        match_records = self.tournament_results.get('swiss_results', {}).get('match_records', [])

        for match in match_records:
            match_system_a = match.get('system_a', '')
            match_system_b = match.get('system_b', '')

            if ((match_system_a == system_a and match_system_b == system_b) or
                (match_system_a == system_b and match_system_b == system_a)):

                comparison = match.get('comparison', {})
                question_results = comparison.get('question_results', [])

                for q_result in question_results:
                    if q_result.get('question', '') == question:
                        return q_result

        return None

    def sample_evaluation_pairs(self, qacg_files: List[str], num_samples: int = 200,
                               random_seed: int = 42) -> List[Dict[str, Any]]:
        """Sample evaluation pairs."""
        import random
        random.seed(random_seed)

        all_pairs = []
        for file_path in qacg_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_pairs.extend(data)

        if len(all_pairs) < num_samples:
            self.logger.warning(f"Available pairs ({len(all_pairs)}) < requested ({num_samples})")
            return all_pairs

        return random.sample(all_pairs, num_samples)

    def run_dice_evaluation(self, evaluation_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run DICE evaluation."""
        results = []

        for i, pair in enumerate(evaluation_pairs):
            try:
                qa_a = pair.get('qa_a', {})
                qa_b = pair.get('qa_b', {})

                question = qa_a.get('question', '')
                system_a = pair.get('system_a', '')
                system_b = pair.get('system_b', '')

                tournament_match = self._find_tournament_match(system_a, system_b, question)

                if tournament_match:
                    self.logger.info(f"Using tournament result: {system_a} vs {system_b} - {question[:50]}...")

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
                        'source': 'tournament'
                    }
                else:
                    self.logger.info(f"No tournament result, running inference: {system_a} vs {system_b} - {question[:50]}...")

                    target_qa_a = {
                        'answer': qa_a.get('rag_answer', ''),
                        'context': qa_a.get('context', [])
                    }

                    target_qa_b = {
                        'answer': qa_b.get('rag_answer', ''),
                        'context': qa_b.get('context', [])
                    }

                    judgment = self.dice_evaluator.pairwise_judge.judge_pair(
                        question=question,
                        qa_a=target_qa_a,
                        qa_b=target_qa_b,
                        granularity="passage"
                    )

                    passage_judgment = judgment.get('passage_judgment', {})
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
                        'source': 'new_inference'
                    }

                results.append(result)

                if (i + 1) % 10 == 0:
                    self.logger.info(f"Completed {i + 1}/{len(evaluation_pairs)} evaluations")

            except Exception as e:
                self.logger.error(f"Error evaluating sample {i}: {e}")
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
                    'dice_explanation': f'Evaluation error: {str(e)}',
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
        """Load human annotations."""
        annotations = {}
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if 'index' in item and 'human_annotation' in item:
                        annotations[item['index']] = item['human_annotation']
        except Exception as e:
            self.logger.error(f"Failed to load annotation file: {e}")
        return annotations

    def calculate_agreement(self, results: List[Dict[str, Any]], gold_labels: Dict[int, str]) -> Dict[str, float]:
        """Calculate agreement metrics."""
        dice_scores = []
        human_scores = []

        for result in results:
            if result['index'] in gold_labels:
                dice_scores.append(result['dice_score'])
                human_annotation = gold_labels[result['index']]
                if human_annotation.lower() in ['a', 'system_a', 'good', 'correct', 'accurate']:
                    human_scores.append(1.0)
                elif human_annotation.lower() in ['b', 'system_b', 'bad', 'incorrect', 'inaccurate']:
                    human_scores.append(-1.0)
                else:
                    human_scores.append(0.0)

        if len(dice_scores) == 0:
            return {'correlation': 0.0, 'kappa': 0.0}

        correlation = np.corrcoef(dice_scores, human_scores)[0, 1] if len(dice_scores) > 1 else 0.0

        dice_binary = [1 if score > 0 else 0 for score in dice_scores]
        human_binary = [1 if score > 0 else 0 for score in human_scores]
        kappa = cohen_kappa_score(dice_binary, human_binary) if len(dice_scores) > 1 else 0.0

        return {
            'correlation': correlation,
            'kappa': kappa,
            'sample_size': len(dice_scores)
        }

    def calculate_elo_correlation(self, results: List[Dict[str, Any]], gold_labels: Dict[int, str]) -> Dict[str, float]:
        """Calculate ELO correlation."""
        return self.calculate_agreement(results, gold_labels)

    def analyze_disagreement_cases(self, results: List[Dict[str, Any]], gold_labels: Dict[int, str]) -> List[Dict[str, Any]]:
        """Analyze disagreement cases."""
        disagreement_cases = []

        for result in results:
            if result['index'] in gold_labels:
                dice_score = result['dice_score']
                human_annotation = gold_labels[result['index']]

                dice_a_better = dice_score > 0
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
        """Print disagreement analysis."""
        if not disagreement_cases:
            self.logger.info("No disagreement cases found")
            return

        self.logger.info(f"Found {len(disagreement_cases)} disagreement cases:")

        for case in disagreement_cases[:5]:
            self.logger.info(f"Case {case['index']}: DICE={case['dice_score']:.3f}, human={case['human_annotation']}")
            self.logger.info(f"Question: {case['question'][:100]}...")

    def generate_validation_report(self, results: List[Dict[str, Any]], gold_labels: Dict[int, str]) -> Dict[str, Any]:
        """Generate validation report."""
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


class RagasValidationEvaluator:
    """RAGAS validation evaluator."""

    def __init__(self, config: RagasConfig):
        self.config = config
        self.logger = logging.getLogger("RagasValidation")
        self.ragas_evaluator = RagasEvaluator(config)
        self._setup_logger()

    def _setup_logger(self):
        """Set up logger."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def run_ragas_evaluation(self, evaluation_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate all sampled comparison pairs using RAGAS."""
        self.logger.info(f"Starting RAGAS evaluation on {len(evaluation_pairs)} pairs")

        ragas_results = []
        for i, pair in enumerate(evaluation_pairs):
            self.logger.info(f"Progress: {i+1}/{len(evaluation_pairs)} - {pair['system_a']} vs {pair['system_b']}")

            qa_a = pair["qa_a"]
            qa_b = pair["qa_b"]

            result = self.ragas_evaluator._pairwise_comparison(
                [qa_a], [qa_b],
                pair["system_a"], pair["system_b"],
                max_questions=1
            )

            if result["question_results"]:
                question_result = result["question_results"][0]
                passage_judgment = question_result.get("passage_judgment", {})
                ragas_details = question_result.get("ragas_details", {})

                judgment = passage_judgment.get("label", "Tie")
                score = passage_judgment.get("score", 0.5)
                reason = passage_judgment.get("reason", "")

                self.logger.info(f"Pair #{i+1}: {judgment}, confidence={score:.4f}")

                ragas_result = {
                    "pair_id": i,
                    "question": pair["question"],
                    "system_a": pair["system_a"],
                    "system_b": pair["system_b"],
                    "dice_judgment": judgment,
                    "dice_score": score,
                    "dice_reason": reason,
                    "dice_margin_score": passage_judgment.get("margin_score", 0.0),
                    "combined_delta": ragas_details.get("composite_a", 0) - ragas_details.get("composite_b", 0),
                    "ragas_scores_a": ragas_details.get("scores_a", {}),
                    "ragas_scores_b": ragas_details.get("scores_b", {}),
                    "original_pair": pair
                }
            else:
                self.logger.warning(f"Pair #{i+1} failed")

                ragas_result = {
                    "pair_id": i,
                    "question": pair["question"],
                    "system_a": pair["system_a"],
                    "system_b": pair["system_b"],
                    "dice_judgment": "Tie",
                    "dice_score": 0.5,
                    "dice_reason": "RAGAS evaluation failed",
                    "dice_margin_score": 0.0,
                    "combined_delta": 0.0,
                    "ragas_scores_a": {},
                    "ragas_scores_b": {},
                    "original_pair": pair
                }

            ragas_results.append(ragas_result)

        judgments = [r["dice_judgment"] for r in ragas_results]
        a_wins = judgments.count("A wins")
        b_wins = judgments.count("B wins")
        ties = judgments.count("Tie")

        self.logger.info(f"RAGAS evaluation done: A wins={a_wins} ({a_wins/len(ragas_results)*100:.1f}%), "
                        f"B wins={b_wins} ({b_wins/len(ragas_results)*100:.1f}%), "
                        f"Tie={ties} ({ties/len(ragas_results)*100:.1f}%)")

        return ragas_results

    def load_human_annotations(self, annotation_file: str) -> Dict[int, str]:
        """Load human annotations via DICEValidationEvaluator."""
        from src.evaluation.dice_engine import SimplifiedDICEConfig
        temp_dice_evaluator = DICEValidationEvaluator(SimplifiedDICEConfig())
        return temp_dice_evaluator.load_human_annotations(annotation_file)

    def calculate_agreement(self, results, gold_labels):
        """Proxy agreement calculation."""
        from src.evaluation.dice_engine import SimplifiedDICEConfig
        temp_dice_evaluator = DICEValidationEvaluator(SimplifiedDICEConfig())
        return temp_dice_evaluator.calculate_agreement(results, gold_labels)

    def calculate_elo_correlation(self, results, gold_labels):
        """Proxy Elo correlation calculation."""
        from src.evaluation.dice_engine import SimplifiedDICEConfig
        temp_dice_evaluator = DICEValidationEvaluator(SimplifiedDICEConfig())
        return temp_dice_evaluator.calculate_elo_correlation(results, gold_labels)

    def analyze_disagreement_cases(self, results, gold_labels):
        """Proxy disagreement analysis."""
        from src.evaluation.dice_engine import SimplifiedDICEConfig
        temp_dice_evaluator = DICEValidationEvaluator(SimplifiedDICEConfig())
        return temp_dice_evaluator.analyze_disagreement_cases(results, gold_labels)

    def print_disagreement_analysis(self, disagreement_cases):
        """Proxy disagreement printing."""
        from src.evaluation.dice_engine import SimplifiedDICEConfig
        temp_dice_evaluator = DICEValidationEvaluator(SimplifiedDICEConfig())
        return temp_dice_evaluator.print_disagreement_analysis(disagreement_cases)

    def generate_validation_report(self, agreement_metrics, correlation_metrics, results, gold_labels, output_file):
        """Proxy report generation."""
        from src.evaluation.dice_engine import SimplifiedDICEConfig
        temp_dice_evaluator = DICEValidationEvaluator(SimplifiedDICEConfig())
        return temp_dice_evaluator.generate_validation_report(agreement_metrics, correlation_metrics, results, gold_labels, output_file)


class UnifiedValidationEvaluator:
    """Unified validation evaluator supporting both DICE and RAGAS."""

    def __init__(self, evaluation_method: str = "dice", dice_config: SimplifiedDICEConfig = None,
                 ragas_config: RagasConfig = None, tournament_result_file: str = None):
        self.evaluation_method = evaluation_method.lower()
        self.logger = logging.getLogger("UnifiedValidation")

        if self.evaluation_method == "dice":
            if dice_config is None:
                raise ValueError("dice_config is required for DICE method")
            self.evaluator = DICEValidationEvaluator(dice_config, tournament_result_file)
        elif self.evaluation_method == "ragas":
            if ragas_config is None:
                raise ValueError("ragas_config is required for RAGAS method")
            self.evaluator = RagasValidationEvaluator(ragas_config)
        else:
            raise ValueError(f"Unsupported evaluation method: {evaluation_method}")

        self._setup_logger()
        self.logger.info(f"Initialized unified evaluator with method: {self.evaluation_method.upper()}")

    def _setup_logger(self):
        """Set up logger."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _derive_dice_label(self, result: Dict[str, Any]) -> str:
        """Derive DICE label from result, handling different score scales."""
        explicit_label = result.get("dice_judgment")
        if explicit_label in {"A wins", "B wins", "Tie"}:
            return explicit_label

        score = result.get("dice_score")
        if isinstance(score, (int, float)):
            if 0.0 <= score <= 1.0:
                if score > 0.55:
                    return "A wins"
                if score < 0.45:
                    return "B wins"
                return "Tie"
            if score > 0.1:
                return "A wins"
            if score < -0.1:
                return "B wins"
            return "Tie"

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
        """Sample (q, cA, aA, cB, aB) pairs from QACG files for annotation."""
        self.logger.info(f"Sampling {num_samples} evaluation pairs")
        import random
        random.seed(random_seed)

        all_systems_data = {}
        for file_path in qacg_files:
            system_name = Path(file_path).stem.replace("qacg_", "")
            with open(file_path, 'r', encoding='utf-8') as f:
                all_systems_data[system_name] = json.load(f)

        systems = list(all_systems_data.keys())
        if len(systems) < 2:
            raise ValueError(f"Need at least 2 systems, got {len(systems)}")

        self.logger.info(f"Loaded {len(systems)} systems: {systems}")

        min_length = min(len(data) for data in all_systems_data.values())
        self.logger.info(f"Each system has {min_length} questions")

        all_combinations = []
        for i, system_a in enumerate(systems):
            for j, system_b in enumerate(systems):
                if i < j:
                    for q_idx in range(min_length):
                        qa_a = all_systems_data[system_a][q_idx]
                        qa_b = all_systems_data[system_b][q_idx]

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

        self.logger.info(f"Total possible combinations: {len(all_combinations)}")

        if len(all_combinations) < num_samples:
            self.logger.warning(f"Available combinations ({len(all_combinations)}) < requested ({num_samples})")
            sampled_pairs = all_combinations
        else:
            sampled_pairs = random.sample(all_combinations, num_samples)

        self.logger.info(f"Sampled {len(sampled_pairs)} evaluation pairs")
        return sampled_pairs

    def run_evaluation(self, evaluation_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the selected evaluation method."""
        if self.evaluation_method == "dice":
            return self.run_dice_evaluation(evaluation_pairs)
        elif self.evaluation_method == "ragas":
            return self.evaluator.run_ragas_evaluation(evaluation_pairs)

    def load_human_annotations(self, annotation_file: str) -> Dict[int, str]:
        """Load human annotations."""
        return self.evaluator.load_human_annotations(annotation_file)

    def calculate_agreement(self, results: List[Dict[str, Any]],
                          gold_labels: Dict[int, str]) -> Dict[str, float]:
        """Calculate agreement metrics."""
        return self.evaluator.calculate_agreement(results, gold_labels)

    def calculate_elo_correlation(self, results: List[Dict[str, Any]],
                                gold_labels: Dict[int, str]) -> Dict[str, float]:
        """Calculate Elo correlation."""
        return self.evaluator.calculate_elo_correlation(results, gold_labels)

    def analyze_disagreement_cases(self, results: List[Dict[str, Any]],
                                  gold_labels: Dict[int, str]) -> List[Dict[str, Any]]:
        """Analyze disagreement cases."""
        return self.evaluator.analyze_disagreement_cases(results, gold_labels)

    def print_disagreement_analysis(self, disagreement_cases: List[Dict[str, Any]]):
        """Print disagreement analysis."""
        return self.evaluator.print_disagreement_analysis(disagreement_cases)

    def generate_validation_report(self, agreement_metrics: Dict[str, Any],
                                 correlation_metrics: Dict[str, Any],
                                 results: List[Dict[str, Any]],
                                 gold_labels: Dict[int, str],
                                 output_file: str):
        """Generate validation report."""
        return self.evaluator.generate_validation_report(
            agreement_metrics, correlation_metrics, results, gold_labels, output_file
        )

    def run_dice_evaluation(self, evaluation_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate all sampled comparison pairs using DICE."""
        self.logger.info(f"Starting DICE evaluation on {len(evaluation_pairs)} pairs")

        dice_results = []
        for i, pair in enumerate(evaluation_pairs):
            self.logger.info(f"Evaluating pair {i+1}/{len(evaluation_pairs)}")

            qa_a = pair["qa_a"]
            qa_b = pair["qa_b"]

            result = self.evaluator.dice_evaluator._pairwise_comparison(
                [qa_a], [qa_b],
                pair["system_a"], pair["system_b"],
                max_questions=1
            )

            if result["question_results"]:
                question_result = result["question_results"][0]
                passage_judgment = question_result.get("passage_judgment", {})

                dice_result = {
                    "pair_id": i,
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
                dice_result = {
                    "pair_id": i,
                    "question": pair["question"],
                    "system_a": pair["system_a"],
                    "system_b": pair["system_b"],
                    "dice_judgment": "Tie",
                    "dice_score": 0.5,
                    "dice_reason": "Evaluation failed",
                    "dice_margin_score": 0.0,
                    "combined_delta": 0.0,
                    "original_pair": pair
                }

            dice_results.append(dice_result)

        return dice_results

    def _create_annotation_template(self, annotation_file: str):
        """Create human annotation template file."""
        self.logger.info(f"Creating annotation template: {annotation_file}")

        template = {
            "instructions": "3 experts independently annotate each pair_id. Fill expert_votes with 'A wins', 'B wins', or 'Tie'.",
            "annotation_guide": {
                "A wins": "System A is clearly better than System B",
                "B wins": "System B is clearly better than System A",
                "Tie": "Both systems perform comparably"
            },
            "annotations": [
                {
                    "pair_id": 0,
                    "question": "Example question",
                    "system_a": "system_a_name",
                    "answer_a": "System A answer",
                    "system_b": "system_b_name",
                    "answer_b": "System B answer",
                    "expert_votes": ["A wins", "B wins", "A wins"]
                }
            ]
        }

        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)

    def _generate_conclusion(self, agreement_metrics: Dict[str, Any],
                           correlation_metrics: Dict[str, Any]) -> str:
        """Generate conclusion string."""
        kappa = agreement_metrics["kappa"]
        tau = correlation_metrics["kendall_tau"]

        num_systems = len(correlation_metrics.get("dice_ranking", []))
        if num_systems == 2:
            if tau == -1.0:
                conclusion = f"2-system validation: DICE and human ranking fully reversed (tau=-1.0)."
                if kappa >= 0.6:
                    conclusion += f" But kappa ({kappa:.3f}) shows acceptable overall agreement."
                else:
                    conclusion += f" kappa ({kappa:.3f}) is also low; consider checking judgment logic."
                return conclusion
            elif tau == 1.0:
                return f"2-system validation: DICE and human ranking fully agree (tau=1.0), kappa={kappa:.3f}."

        if kappa >= 0.85 and tau >= 0.9:
            return "DICE validation passed. Both kappa and Kendall-tau meet thresholds."
        elif kappa >= 0.85:
            return "DICE partially passed. Kappa meets threshold but ranking correlation is insufficient."
        elif tau >= 0.9:
            return "DICE partially passed. Ranking correlation meets threshold but agreement is insufficient."
        else:
            return "DICE validation failed. Both kappa and Kendall-tau below thresholds."

    def _print_validation_summary(self, report: Dict[str, Any]):
        """Print validation summary."""
        summary = report["validation_summary"]

        print(f"\nDICE Validation Results")
        print(f"kappa (target>=0.85): {summary['kappa_score']:.3f}")
        print(f"Accuracy: {summary['accuracy']:.3f}")
        print(f"Kendall-tau (target>=0.9): {summary['kendall_tau']:.3f}")
        print(f"Status: {'PASSED' if summary['validation_passed'] else 'FAILED'}")
        print(f"\n{report['conclusion']}")
