"""
QACG Tuple Generator
Generates Question-Answer-Context-Groundtruth tuples for each RAG system.
"""

import json
import logging
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
import os
import pandas as pd

from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from src.systems.llamaindex_impl import LlamaIndexRAGSystem
from src.systems.base import RAGConfig


class QACGGenerator:
    """QACG tuple generator."""

    def __init__(self, llm_model: str = "qwen2.5:7b"):
        """
        Initialize the generator.

        Args:
            llm_model: LLM model used for question generation.
        """
        self.logger = logging.getLogger(__name__)

        # Initialize the LLM for question generation
        if llm_model.startswith("openai"):
            self.question_llm = OpenAI(model=llm_model.replace("openai-", ""))
        else:
            self.question_llm = Ollama(model=llm_model, request_timeout=120.0)

        # Question generation templates (Chinese prompts for Chinese LLM)
        self.question_templates = [
            "根据以下文本内容，生成一个具体的问题：\n{context}\n\n请生成一个可以从上述内容中找到明确答案的问题：",
            "基于这段文字，提出一个关键问题：\n{context}\n\n问题应该针对文本中的核心信息：",
            "阅读下面的内容，设计一个问题：\n{context}\n\n问题要求能够通过文本内容回答：",
            "请根据以下信息提出一个问题：\n{context}\n\n确保问题的答案在文本中可以找到：",
            "分析这段文字，生成相关问题：\n{context}\n\n问题应该测试对文本内容的理解："
        ]

    def load_knowledge_base(self, jsonl_path: str) -> Dict[str, str]:
        """Load knowledge base from a JSONL file. Returns a dict mapping doc ID to content."""
        knowledge_base = {}

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        doc = json.loads(line)
                        content_id = doc.get('content_id', f'doc_{line_num}')
                        content = doc.get('content', '')
                        if content:
                            knowledge_base[content_id] = content
                    except json.JSONDecodeError:
                        self.logger.warning(f"Skipping unparseable line {line_num + 1}")

        self.logger.info(f"Loaded {len(knowledge_base)} documents")
        return knowledge_base

    def sample_documents(self, knowledge_base: Dict[str, str], sample_size: int = 50) -> Dict[str, str]:
        """Sample documents from the knowledge base."""
        if len(knowledge_base) <= sample_size:
            return knowledge_base

        # Random sampling
        doc_ids = list(knowledge_base.keys())
        sampled_ids = random.sample(doc_ids, sample_size)

        sampled_docs = {doc_id: knowledge_base[doc_id] for doc_id in sampled_ids}
        self.logger.info(f"Sampled {len(sampled_docs)} documents for QACG generation")

        return sampled_docs

    def generate_question_from_context(self, context: str) -> str:
        """Generate a question from the given context."""
        # Randomly select a template
        template = random.choice(self.question_templates)
        prompt = template.format(context=context[:1000])  # Limit context length

        try:
            if hasattr(self.question_llm, 'complete'):
                response = self.question_llm.complete(prompt)
                question = str(response).strip()
            else:
                # For Ollama and other models
                response = self.question_llm.generate([prompt])
                question = str(response).strip()

            # Clean up question format
            question = question.replace("问题：", "").replace("Question:", "").strip()
            if not question.endswith('?') and not question.endswith('？'):
                question += '？'

            return question
        except Exception as e:
            self.logger.error(f"Question generation failed: {e}")
            # Fall back to rule-based generation
            return self._generate_rule_based_question(context)

    def _generate_rule_based_question(self, context: str) -> str:
        """Generate a rule-based question as fallback."""
        # Simple rules: extract key info to generate questions (Chinese output for Chinese LLM)
        if "时间" in context or "日期" in context:
            return "这件事发生在什么时间？"
        elif "原因" in context or "因为" in context:
            return "造成这种情况的原因是什么？"
        elif "结果" in context or "影响" in context:
            return "这件事产生了什么影响或结果？"
        elif "地点" in context or "地区" in context:
            return "这件事发生在哪里？"
        else:
            return "根据文本内容，主要讲述了什么？"

    def load_test_questions(self, qa_file_path: str = "dataset/70条测试数据QA.txt") -> List[Dict[str, Any]]:
        """Load test questions from a QA file."""
        try:
            with open(qa_file_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            self.logger.info(f"Loaded {len(test_data)} test questions")
            return test_data
        except Exception as e:
            self.logger.error(f"Failed to load test data: {e}")
            return []

    def generate_qacg_for_system(self,
                                 rag_system: LlamaIndexRAGSystem,
                                 knowledge_base: Dict[str, str],
                                 num_questions: int = 70) -> List[Dict[str, Any]]:
        """Generate QACG tuples for a specific RAG system using test data."""
        self.logger.info(f"Generating QACG tuples for system {rag_system.config.system_name}")

        # Load test data
        test_questions = self.load_test_questions()
        if not test_questions:
            self.logger.error("Failed to load test data, falling back to generation mode")
            return self._generate_qacg_fallback(rag_system, knowledge_base, num_questions)

        # Use specified number of questions
        questions_to_use = test_questions[:num_questions]
        self.logger.info(f"Using {len(questions_to_use)} test questions")

        qacg_list = []

        for i, test_item in enumerate(questions_to_use):
            try:
                question = test_item["question"]
                expected_answer = test_item["answer"]

                # Query the RAG system
                rag_response = rag_system.query(question)
                rag_answer = rag_response.answer
                evidence = rag_response.evidence

                # Build QACG tuple
                qacg = {
                    "question": question,
                    "rag_answer": rag_answer,
                    "expected_answer": expected_answer,
                    "context": evidence,
                    "groundtruth": test_item.get("relevant_content", expected_answer),
                    "metadata": {
                        "system_name": rag_system.config.system_name,
                        "embedding_model": rag_system.config.embedding_model,
                        "llm_model": rag_system.config.llm_model,
                        "chunking_strategy": rag_system.config.chunking_strategy,
                        "retrieval_top_k": rag_system.config.retrieval_top_k,
                        "question_id": f"test_q_{i+1}",
                        "task_name": test_item.get("task_name", "unknown"),
                        "relevant_passage": test_item.get("relevant_passage", ""),
                        "generated_at": str(pd.Timestamp.now())
                    }
                }

                qacg_list.append(qacg)
                self.logger.info(f"Processed question {i+1}/{len(questions_to_use)}: {question[:50]}...")

            except Exception as e:
                self.logger.error(f"Error processing question {i+1}: {e}")
                continue

        self.logger.info(f"Successfully processed {len(qacg_list)} test questions")
        return qacg_list

    def _generate_qacg_fallback(self,
                               rag_system: LlamaIndexRAGSystem,
                               knowledge_base: Dict[str, str],
                               num_questions: int) -> List[Dict[str, Any]]:
        """Fallback to question generation mode when test data is unavailable."""
        self.logger.info("Using fallback mode for question generation")

        # Sample documents as context
        sampled_docs = self.sample_documents(knowledge_base, min(50, len(knowledge_base)))
        doc_contents = list(sampled_docs.values())

        qacg_list = []

        for i in range(num_questions):
            try:
                # Randomly select a document as context
                context = random.choice(doc_contents)

                # Truncate context to appropriate length
                context = context[:800] if len(context) > 800 else context

                # Generate question
                question = self.generate_question_from_context(context)

                # Query the RAG system
                rag_response = rag_system.query(question)
                answer = rag_response.answer
                evidence = rag_response.evidence

                # Build QACG tuple
                qacg = {
                    "question": question,
                    "answer": answer,
                    "context": evidence,
                    "groundtruth": context,
                    "metadata": {
                        "system_name": rag_system.config.system_name,
                        "embedding_model": rag_system.config.embedding_model,
                        "llm_model": rag_system.config.llm_model,
                        "chunking_strategy": rag_system.config.chunking_strategy,
                        "retrieval_top_k": rag_system.config.retrieval_top_k,
                        "question_id": f"fallback_q_{i+1}",
                        "generated_at": str(pd.Timestamp.now())
                    }
                }

                qacg_list.append(qacg)
                self.logger.info(f"Generated QACG {i+1}/{num_questions}")

            except Exception as e:
                self.logger.error(f"Error generating QACG {i+1}: {e}")
                continue

        self.logger.info(f"Generated {len(qacg_list)} QACG tuples")
        return qacg_list

    def create_rag_systems(self) -> List[LlamaIndexRAGSystem]:
        """Create 8 RAG system configurations (2x2x2)."""
        embedding_models = ["bge-large-zh", "bge-small-zh"]
        chunking_strategies = ["chunk_256", "chunk_512"]  # Length-based chunking strategies
        llm_models = ["qwen2.5", "qwen2.5-mini"]

        systems = []

        for embedding in embedding_models:
            for chunking in chunking_strategies:
                for llm in llm_models:
                    config = RAGConfig(
                        system_name=f"{embedding}_{chunking}_{llm}",
                        chunking_strategy=chunking,
                        chunk_size=512,
                        chunk_overlap=50,
                        embedding_model=embedding,
                        llm_model=llm,
                        retrieval_top_k=3,
                        temperature=0.1
                    )

                    system = LlamaIndexRAGSystem(config)
                    systems.append(system)

        self.logger.info(f"Created {len(systems)} RAG systems")
        return systems

    def save_qacg_results(self, qacg_data: List[Dict[str, Any]], output_path: str):
        """Save QACG results to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qacg_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"QACG results saved to: {output_path}")

    def generate_all_qacg(self,
                          jsonl_path: str,
                          output_dir: str = "qacg_output",
                          num_questions: int = 70):
        """Generate QACG tuples for all RAG systems."""
        self.logger.info("Starting QACG generation for all RAG systems")

        # Load raw knowledge base
        raw_knowledge_base = self.load_knowledge_base(jsonl_path)
        self.logger.info(f"Knowledge base: {jsonl_path}, {len(raw_knowledge_base)} docs")
        self.logger.info(f"Doc length range: {min(len(v) for v in raw_knowledge_base.values())} - {max(len(v) for v in raw_knowledge_base.values())} chars")

        # Create RAG systems
        rag_systems = self.create_rag_systems()
        self.logger.info(f"Created {len(rag_systems)} RAG system configurations")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Process each system independently
        system_stats = []

        for i, system in enumerate(rag_systems, 1):
            try:
                self.logger.info(f"Processing system {i}/{len(rag_systems)}: {system.config.system_name}")

                # Each system independently loads and processes the knowledge base
                self.logger.info(f"Processing knowledge base for {system.config.system_name}")

                # Use system-specific processing strategy
                processed_knowledge_base = self._process_knowledge_base_for_system(
                    system, raw_knowledge_base.copy()
                )

                # Record system processing stats
                stats = {
                    'system_name': system.config.system_name,
                    'chunking_strategy': system.config.chunking_strategy,
                    'embedding_model': system.config.embedding_model,
                    'llm_model': system.config.llm_model,
                    'input_docs': len(processed_knowledge_base),
                    'chunks_generated': len(system.nodes) if hasattr(system, 'nodes') and system.nodes else 0
                }
                system_stats.append(stats)

                # Generate QACG
                self.logger.info(f"Generating {num_questions} QACG tuples for {system.config.system_name}")
                qacg_data = self.generate_qacg_for_system(
                    system, processed_knowledge_base, num_questions
                )

                # Save results
                output_path = os.path.join(
                    output_dir,
                    f"qacg_{system.config.system_name}.json"
                )
                self.save_qacg_results(qacg_data, output_path)

                self.logger.info(f"System {system.config.system_name} done, saved to: {output_path}")

            except Exception as e:
                self.logger.error(f"Error processing system {system.config.system_name}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue

        # Final stats
        self.logger.info("All RAG systems processed - independence verification report")

        if system_stats:
            # Group stats by chunking strategy
            chunking_stats = {}
            for stat in system_stats:
                strategy = stat['chunking_strategy']
                if strategy not in chunking_stats:
                    chunking_stats[strategy] = []
                chunking_stats[strategy].append(stat['chunks_generated'])

            self.logger.info("Chunking strategy independence check:")
            for strategy, chunks_list in chunking_stats.items():
                self.logger.info(f"  {strategy}: {chunks_list} (chunk counts)")
                if len(set(chunks_list)) > 1:
                    self.logger.info(f"    Different embedding models produced different chunk counts")
                else:
                    self.logger.info(f"    All embedding models produced the same chunk count")

            # Verify different strategies produced different results
            all_chunks = [stat['chunks_generated'] for stat in system_stats]
            unique_chunks = len(set(all_chunks))
            self.logger.info(f"Overall independence: {unique_chunks}/{len(system_stats)} distinct chunk counts")

            if unique_chunks > 1:
                self.logger.info("Confirmed: different RAG configurations produced different results")
            else:
                self.logger.warning("Warning: all systems produced the same chunk count, check configurations")

        self.logger.info("QACG generation complete for all RAG systems")

    def _process_knowledge_base_for_system(self, system: LlamaIndexRAGSystem,
                                         raw_knowledge_base: Dict[str, str]) -> Dict[str, str]:
        """Process knowledge base independently for a specific system."""
        self.logger.info(f"Independent KB processing for {system.config.system_name}")
        self.logger.info(f"Raw KB docs: {len(raw_knowledge_base)}, chunking: {system.config.chunking_strategy}, "
                         f"chunk_size: {system.config.chunk_size}, overlap: {system.config.chunk_overlap}, "
                         f"embedding: {system.config.embedding_model}")

        # Check for cached knowledge base
        cache_dir = f"./knowledge_cache/{system.config.system_name}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "processed_kb.json")

        # Check cache
        if os.path.exists(cache_file):
            self.logger.info(f"Checking cache: {cache_file}")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_kb = json.load(f)

                # Validate cache config match
                if self._validate_cache_config(cached_kb.get('config', {}), system.config):
                    self.logger.info("Cache config matches")

                    # Check for cached chunks data
                    if 'chunks' in cached_kb and cached_kb['chunks']:
                        self.logger.info(f"Found {len(cached_kb['chunks'])} cached chunks, skipping re-chunking")

                        # Build vector store directly from cached chunks
                        self._load_cached_chunks_and_build_vector_store(system, cached_kb['chunks'])

                        return cached_kb['knowledge_base']
                    else:
                        self.logger.info("No chunks in cache, reprocessing")
                        system.process_knowledge_base(cached_kb['knowledge_base'])

                        return cached_kb['knowledge_base']
                else:
                    self.logger.info("Cache config mismatch, reprocessing")
            except Exception as e:
                self.logger.warning(f"Failed to read cache: {e}")

        # No cache or config mismatch, reprocess
        self.logger.info(f"Processing KB: each JSONL line as independent doc, "
                         f"strategy={system.config.chunking_strategy}, "
                         f"embedding={system.config.embedding_model}")

        # Each system independently chunks and embeds based on its own config
        processing_result = system.process_knowledge_base(raw_knowledge_base)

        # Log processing stats and prepare cache data
        chunks_data = []
        if hasattr(system, 'nodes') and system.nodes:
            chunk_count = len(system.nodes)
            self.logger.info(f"System {system.config.system_name}: {len(raw_knowledge_base)} docs -> "
                             f"{chunk_count} chunks ({chunk_count/len(raw_knowledge_base):.2f} per doc)")

            # Show sample chunks
            for i, node in enumerate(system.nodes[:3]):
                self.logger.info(f"  Chunk {i+1} (len={len(node.text)}): {node.text[:50]}...")

            # Serialize nodes for caching
            self.logger.info("Preparing chunks for cache...")
            for i, node in enumerate(system.nodes):
                chunk_data = {
                    "chunk_id": f"{node.metadata.get('doc_id', 'unknown')}_{i}",
                    "content": node.text,
                    "source_doc_id": node.metadata.get("doc_id", "unknown"),
                    "metadata": dict(node.metadata)
                }
                chunks_data.append(chunk_data)

            self.logger.info(f"Prepared {len(chunks_data)} chunks for cache")

        # Cache processing results (including chunks data)
        cache_data = {
            'config': {
                'chunking_strategy': system.config.chunking_strategy,
                'chunk_size': system.config.chunk_size,
                'chunk_overlap': system.config.chunk_overlap,
                'embedding_model': system.config.embedding_model,
                'system_name': system.config.system_name
            },
            'knowledge_base': raw_knowledge_base,
            'chunks': chunks_data,
            'processing_stats': {
                'input_doc_count': len(raw_knowledge_base),
                'chunk_count': len(chunks_data),
                'processed_at': str(pd.Timestamp.now())
            }
        }

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"KB processed and cached: {cache_file}")

        return raw_knowledge_base

    def _load_cached_chunks_and_build_vector_store(self, system: LlamaIndexRAGSystem, cached_chunks: List[Dict]) -> None:
        """Build vector store directly from cached chunks."""
        try:
            self.logger.info(f"Building vector store from cached chunks for {system.config.system_name}")

            # Convert cached chunks to system format and build vector store
            system.build_vector_store(cached_chunks, [])

            # Mark system as indexed
            system.is_indexed = True

            self.logger.info("Vector store built from cached chunks")

        except Exception as e:
            self.logger.error(f"Failed to build vector store from cache: {e}")
            self.logger.warning("Falling back to standard processing")
            raise e

    def _validate_cache_config(self, cached_config: Dict, current_config) -> bool:
        """Validate whether cached config matches current config."""
        key_fields = ['chunking_strategy', 'chunk_size', 'chunk_overlap', 'embedding_model']

        for field in key_fields:
            if cached_config.get(field) != getattr(current_config, field):
                return False

        return True
