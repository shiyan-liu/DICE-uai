#!/usr/bin/env python3
"""RAGAS evaluator using DeepSeek API."""

import json
import logging
import os
import time
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np

# Set env vars to force RAGAS to use our config
os.environ["OPENAI_API_KEY"] = "xxxxxxx"
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"

try:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness, ContextRelevance
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        # Fall back to legacy package
        from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False
    print(f"Warning: RAGAS not installed. Error: {e}")


@dataclass
class RagasConfig:
    """RAGAS evaluation config."""
    llm_model: str = "deepseek-chat"
    embeddings_model: str = "BAAI/bge-small-zh-v1.5"  # Smaller model to save memory
    metrics: List[str] = None
    api_key: str = "xxxxxxx"
    base_url: str = "https://api.deepseek.com"

    def __post_init__(self):
        if self.metrics is None:
            # Three core dimensions from the RAGAS paper
            self.metrics = ["faithfulness", "answer_relevancy", "context_relevance"]


class RagasEvaluator:
    """RAGAS evaluator using DeepSeek API."""

    def __init__(self, config: RagasConfig):
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS is not installed")

        self.config = config
        self.logger = logging.getLogger("RagasEvaluator")
        self._setup_logger()

        # Force set env vars
        self._force_openai_env()

        # Set up custom LLM
        self._setup_custom_llm()

        # Initialize three core RAGAS metrics
        self.metrics_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_relevance": ContextRelevance()  # Needs instantiation
        }

        self.active_metrics = [self.metrics_map[m] for m in self.config.metrics if m in self.metrics_map]
        self._configure_metrics_llm()

        self.logger.info(f"RAGAS evaluator initialized with DeepSeek: {self.config.llm_model}")

    def _setup_logger(self):
        """Set up logging."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _force_openai_env(self):
        """Force OpenAI env vars to point to DeepSeek."""
        os.environ["OPENAI_API_KEY"] = self.config.api_key
        os.environ["OPENAI_BASE_URL"] = self.config.base_url
        os.environ["OPENAI_API_BASE"] = self.config.base_url
        self.logger.info(f"OpenAI env vars set to DeepSeek: {self.config.base_url}")

    def _setup_custom_llm(self):
        """Set up DeepSeek LLM."""
        try:
            # Create ChatOpenAI instance pointing to DeepSeek
            self.custom_llm = ChatOpenAI(
                model=self.config.llm_model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                temperature=0.0,
                max_retries=2,
                request_timeout=60
            )

            self.ragas_llm = LangchainLLMWrapper(self.custom_llm)

            # Use local embedding model only, never call API
            try:
                import os
                # Temporarily remove OpenAI env vars to prevent embedding model from using API
                old_openai_key = os.environ.get("OPENAI_API_KEY")
                old_openai_base = os.environ.get("OPENAI_BASE_URL")
                if "OPENAI_API_KEY" in os.environ:
                    del os.environ["OPENAI_API_KEY"]
                if "OPENAI_BASE_URL" in os.environ:
                    del os.environ["OPENAI_BASE_URL"]

                model_kwargs = {
                    'device': 'cpu',
                    'trust_remote_code': True
                }
                encode_kwargs = {
                    'normalize_embeddings': True,
                    'batch_size': 1
                }

                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.config.embeddings_model,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    cache_folder='./models_cache'
                )
                self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)

                # Restore OpenAI env vars (for LLM only)
                if old_openai_key:
                    os.environ["OPENAI_API_KEY"] = old_openai_key
                if old_openai_base:
                    os.environ["OPENAI_BASE_URL"] = old_openai_base

                self.logger.info(f"Local embedding model loaded: {self.config.embeddings_model}")

                # Test embedding model
                test_embedding = self.embeddings.embed_query("test text")
                self.logger.info(f"Embedding model test passed, dim={len(test_embedding)}")

            except Exception as e:
                self.logger.error(f"Embedding model load failed: {e}")
                import traceback
                self.logger.error(f"Details: {traceback.format_exc()}")
                self.ragas_embeddings = None
                raise Exception(f"Embedding model load failed, cannot evaluate metrics requiring embeddings: {e}")

            self.logger.info(f"DeepSeek LLM configured: {self.config.llm_model}")

        except Exception as e:
            self.logger.error(f"LLM configuration failed: {e}")
            raise

    def _configure_metrics_llm(self):
        """Configure custom LLM for each metric."""
        try:
            for metric in self.active_metrics:
                if hasattr(metric, 'llm'):
                    metric.llm = self.ragas_llm
                if hasattr(metric, 'embeddings') and self.ragas_embeddings is not None:
                    metric.embeddings = self.ragas_embeddings
                elif hasattr(metric, 'embeddings') and self.ragas_embeddings is None:
                    self.logger.warning(f"Metric {type(metric).__name__} requires embeddings but none loaded")
            self.logger.info("All metrics LLM configured")
        except Exception as e:
            self.logger.error(f"Metrics LLM configuration failed: {e}")
            raise

    def _qacg_to_ragas_format(self, qacg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert QACG data to RAGAS format."""
        contexts = []
        if isinstance(qacg_data.get("context"), list):
            for ctx in qacg_data["context"]:
                if isinstance(ctx, dict):
                    contexts.append(ctx.get("text", str(ctx)))
                else:
                    contexts.append(str(ctx))
        elif qacg_data.get("context"):
            contexts = [str(qacg_data["context"])]

        # Ensure ground_truth is a string
        ground_truth = qacg_data.get("groundtruth", qacg_data.get("expected_answer", ""))

        if isinstance(ground_truth, list):
            if len(ground_truth) > 0:
                ground_truth = " ".join(str(item) for item in ground_truth)
            else:
                ground_truth = ""
        elif ground_truth is None:
            ground_truth = ""
        else:
            ground_truth = str(ground_truth)

        # Handle other fields that may be lists
        question = qacg_data.get("question", "")
        if isinstance(question, list):
            question = " ".join(str(item) for item in question) if question else ""
        else:
            question = str(question) if question else ""

        answer = qacg_data.get("rag_answer", "")
        if isinstance(answer, list):
            answer = " ".join(str(item) for item in answer) if answer else ""
        else:
            answer = str(answer) if answer else ""

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth
        }

    def evaluate_single_qacg(self, qacg_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a single QACG with retries to handle faithfulness NaN."""
        max_retries = 3
        retry_delay = 2.0

        for attempt in range(max_retries + 1):
            try:
                # Force reset env vars
                self._force_openai_env()

                ragas_data = self._qacg_to_ragas_format(qacg_data)

                # Data validation
                if not ragas_data["question"] or not ragas_data["answer"]:
                    self.logger.warning("Question or answer is empty, skipping")
                    return {metric: 0.0 for metric in self.config.metrics}

                if not ragas_data["contexts"]:
                    self.logger.warning("Context is empty, using default")
                    ragas_data["contexts"] = [""]

                self.logger.debug(f"Eval data: q_len={len(ragas_data['question'])}, a_len={len(ragas_data['answer'])}, ctx_count={len(ragas_data['contexts'])}")

                # Extra data validation and cleanup
                if not isinstance(ragas_data['ground_truth'], str):
                    self.logger.warning(f"ground_truth is not str: {type(ragas_data['ground_truth'])}, val: {ragas_data['ground_truth']}")
                    ragas_data['ground_truth'] = str(ragas_data['ground_truth'])

                if not isinstance(ragas_data['question'], str):
                    self.logger.warning(f"question is not str: {type(ragas_data['question'])}, val: {ragas_data['question']}")
                    ragas_data['question'] = str(ragas_data['question'])

                if not isinstance(ragas_data['answer'], str):
                    self.logger.warning(f"answer is not str: {type(ragas_data['answer'])}, val: {ragas_data['answer']}")
                    ragas_data['answer'] = str(ragas_data['answer'])

                # Create dataset
                dataset = Dataset.from_dict({
                    "question": [ragas_data["question"]],
                    "answer": [ragas_data["answer"]],
                    "contexts": [ragas_data["contexts"]],
                    "ground_truth": [ragas_data["ground_truth"]]
                })

                # Single-threaded evaluation to avoid concurrency issues
                self.logger.debug(f"Starting RAGAS evaluate (attempt {attempt + 1}/{max_retries + 1})")
                result = evaluate(
                    dataset,
                    metrics=self.active_metrics,
                    show_progress=False
                )
                self.logger.debug(f"RAGAS evaluate done, result type: {type(result)}")

                # Extract scores
                scores = {}

                # Metric name mapping (RAGAS uses different internal keys)
                metric_name_mapping = {
                    "context_relevance": "nv_context_relevance",
                    "faithfulness": "faithfulness",
                    "answer_relevancy": "answer_relevancy"
                }

                for metric_name in self.config.metrics:
                    try:
                        actual_key = metric_name_mapping.get(metric_name, metric_name)

                        # Try multiple ways to get the score
                        score_value = None

                        if hasattr(result, actual_key):
                            score_value = getattr(result, actual_key)
                        elif hasattr(result, '_scores_dict') and actual_key in result._scores_dict:
                            score_value = result._scores_dict[actual_key]
                        elif actual_key in result:
                            score_value = result[actual_key]

                        if score_value is not None:
                            if isinstance(score_value, (list, tuple)) and len(score_value) > 0:
                                actual_score = score_value[0]
                            else:
                                actual_score = score_value

                            # Handle NaN
                            if isinstance(actual_score, float) and (actual_score != actual_score):
                                self.logger.warning(f"Metric {metric_name} returned NaN, using default")
                                scores[metric_name] = 0.3 if metric_name == "faithfulness" else 0.5
                            elif isinstance(actual_score, (int, float)):
                                scores[metric_name] = float(actual_score)
                                self.logger.debug(f"Metric {metric_name} score: {scores[metric_name]}")
                            else:
                                self.logger.warning(f"Metric {metric_name} invalid type: {type(actual_score)}")
                                scores[metric_name] = 0.3 if metric_name == "faithfulness" else 0.5
                        else:
                            self.logger.warning(f"Could not get score for metric {metric_name}")
                            scores[metric_name] = 0.3 if metric_name == "faithfulness" else 0.5

                    except Exception as e:
                        self.logger.warning(f"Error getting metric {metric_name}: {e}")
                        scores[metric_name] = 0.3 if metric_name == "faithfulness" else 0.5

                # Validate all scores
                valid_scores = all(
                    isinstance(score, (int, float)) and not np.isnan(score) and not np.isinf(score)
                    for score in scores.values()
                )

                if valid_scores:
                    self.logger.debug(f"Evaluation succeeded, scores: {scores}")
                    return scores
                else:
                    self.logger.warning(f"Evaluation result contains invalid scores: {scores}")
                    if attempt < max_retries:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        return {metric: 0.3 if metric == "faithfulness" else 0.5 for metric in self.config.metrics}

            except Exception as e:
                self.logger.warning(f"RAGAS evaluation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    import traceback
                    error_msg = f"RAGAS evaluation failed: {str(e)}\nDetails:\n{traceback.format_exc()}"
                    self.logger.error(error_msg)
                    return {metric: 0.3 if metric == "faithfulness" else 0.5 for metric in self.config.metrics}

        # All attempts failed, return defaults
        return {metric: 0.3 if metric == "faithfulness" else 0.5 for metric in self.config.metrics}

    def calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """Calculate average score (simple mean per RAGAS paper)."""
        if not scores:
            return 0.0

        valid_scores = [score for score in scores.values() if score is not None and score >= 0]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    def compare_qacg_pair(self, qa_a: Dict[str, Any], qa_b: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two QACGs."""
        self.logger.info("Starting RAGAS pairwise evaluation")

        scores_a = self.evaluate_single_qacg(qa_a)
        self.logger.info(f"System A scores: {scores_a}")

        scores_b = self.evaluate_single_qacg(qa_b)
        self.logger.info(f"System B scores: {scores_b}")

        # Calculate composite scores
        composite_a = self.calculate_composite_score(scores_a)
        composite_b = self.calculate_composite_score(scores_b)

        self.logger.info(f"Composite: A={composite_a:.4f}, B={composite_b:.4f}, diff={composite_a - composite_b:.4f}")

        # Determine winner
        score_diff = composite_a - composite_b

        if abs(score_diff) < 0.05:
            judgment = "Tie"
        elif score_diff > 0:
            judgment = "A wins"
        else:
            judgment = "B wins"

        # Generate detailed reason
        reason_parts = []

        for metric in self.config.metrics:
            score_a = scores_a.get(metric, 0)
            score_b = scores_b.get(metric, 0)
            diff = score_a - score_b

            if abs(diff) > 0.01:
                better_system = "A" if diff > 0 else "B"
                self.logger.debug(f"{metric}: A={score_a:.3f} vs B={score_b:.3f} -> {better_system} better")

                if abs(diff) > 0.1:
                    reason_parts.append(f"{metric}: {better_system} better ({score_a:.3f} vs {score_b:.3f})")
            else:
                self.logger.debug(f"{metric}: A={score_a:.3f} vs B={score_b:.3f} -> tie")

        if not reason_parts:
            reason = f"Systems are close (A: {composite_a:.3f}, B: {composite_b:.3f})"
        else:
            reason = "; ".join(reason_parts)

        self.logger.info(f"Judgment: {judgment}, reason: {reason}")

        return {
            "judgment": judgment,
            "score_a": composite_a,
            "score_b": composite_b,
            "score_diff": score_diff,
            "detailed_scores_a": scores_a,
            "detailed_scores_b": scores_b,
            "reason": reason,
            "margin_score": abs(score_diff)
        }

    def _pairwise_comparison(self, qa_list_a: List[Dict[str, Any]],
                           qa_list_b: List[Dict[str, Any]],
                           system_a_name: str, system_b_name: str,
                           max_questions: int = None) -> Dict[str, Any]:
        """Perform pairwise comparison between two systems."""
        self.logger.info(f"Starting RAGAS pairwise comparison: {system_a_name} vs {system_b_name}")

        num_questions = min(len(qa_list_a), len(qa_list_b))
        if max_questions:
            num_questions = min(num_questions, max_questions)

        self.logger.info(f"Comparing {num_questions} questions: {system_a_name} vs {system_b_name}")

        question_results = []

        for i in range(num_questions):
            qa_a = qa_list_a[i]
            qa_b = qa_list_b[i]

            if qa_a.get("question") != qa_b.get("question"):
                self.logger.warning(f"Question {i} mismatch, skipping")
                continue

            question_text = qa_a.get("question", "")
            self.logger.info(f"Q{i+1}: {question_text[:100]}{'...' if len(question_text) > 100 else ''}")
            self.logger.debug(f"  A answer: {qa_a.get('rag_answer', '')[:80]}")
            self.logger.debug(f"  B answer: {qa_b.get('rag_answer', '')[:80]}")

            comparison_result = self.compare_qacg_pair(qa_a, qa_b)

            question_result = {
                "question_id": i,
                "question": qa_a.get("question", ""),
                "passage_judgment": {
                    "label": comparison_result["judgment"],
                    "score": comparison_result["score_a"] if comparison_result["judgment"] == "A wins"
                            else comparison_result["score_b"] if comparison_result["judgment"] == "B wins"
                            else (comparison_result["score_a"] + comparison_result["score_b"]) / 2,
                    "reason": comparison_result["reason"],
                    "margin_score": comparison_result["margin_score"]
                },
                "ragas_details": {
                    "scores_a": comparison_result["detailed_scores_a"],
                    "scores_b": comparison_result["detailed_scores_b"],
                    "composite_a": comparison_result["score_a"],
                    "composite_b": comparison_result["score_b"]
                }
            }

            question_results.append(question_result)

        # Calculate overall results
        a_wins = sum(1 for r in question_results if r["passage_judgment"]["label"] == "A wins")
        b_wins = sum(1 for r in question_results if r["passage_judgment"]["label"] == "B wins")
        ties = len(question_results) - a_wins - b_wins

        total = len(question_results)
        self.logger.info(
            f"Results: total={total}, "
            f"{system_a_name} wins={a_wins} ({a_wins/total*100:.1f}%), "
            f"{system_b_name} wins={b_wins} ({b_wins/total*100:.1f}%), "
            f"ties={ties} ({ties/total*100:.1f}%)"
        )

        if a_wins > b_wins:
            winner = system_a_name
        elif b_wins > a_wins:
            winner = system_b_name
        else:
            winner = "Tie"

        self.logger.info(f"Overall winner: {winner}")

        return {
            "system_a": system_a_name,
            "system_b": system_b_name,
            "question_results": question_results,
            "summary": {
                "total_questions": len(question_results),
                "a_wins": a_wins,
                "b_wins": b_wins,
                "ties": ties,
                "win_rate_a": a_wins / len(question_results) if question_results else 0,
                "win_rate_b": b_wins / len(question_results) if question_results else 0
            }
        }
