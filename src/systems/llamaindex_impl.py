"""LlamaIndex-based RAG system implementation."""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    ServiceContext,
    SimpleDirectoryReader,
    Settings
)
from llama_index.core.node_parser import (
    SimpleNodeParser,
    SentenceSplitter,
    SemanticSplitterNodeParser
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
import chromadb

from .base import BaseRAGSystem, RAGConfig, RetrievalResult, RAGResponse


class LlamaIndexRAGSystem(BaseRAGSystem):
    """LlamaIndex-based RAG system."""
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.index = None
        self.query_engine = None
        self.documents = []
        self.nodes = []
        
        # Initialize embedding model
        self._setup_embedding_model()

        # Initialize LLM
        self._setup_llm_model()

        # Initialize node parser
        self._setup_node_parser()

        # Set global config
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm_model
        Settings.node_parser = self.node_parser
        
    def _setup_embedding_model(self):
        """Set up embedding model with GPU support."""
        import torch
        import os

        # Set HF mirror endpoint
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {device}")

        if self.config.embedding_model == "bge-large-zh":
            self.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-large-zh-v1.5",
                cache_folder="./models",
                device=device,
                max_length=512,
                trust_remote_code=True,
            )
        elif self.config.embedding_model == "bge-small-zh":
            self.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-zh-v1.5",
                cache_folder="./models",
                device=device,
                max_length=512,
                trust_remote_code=True,
            )
        else:
            # Default: use small model
            self.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder="./models",
                device=device,
                max_length=384,
                trust_remote_code=True,
            )
    
    def _setup_llm_model(self):
        """Set up LLM model."""
        if self.config.llm_model == "qwen2.5":
            # Ollama-hosted Qwen2.5 7B
            self.llm_model = Ollama(
                model="qwen2.5:7b",
                request_timeout=120.0,
                temperature=self.config.temperature
            )
        elif self.config.llm_model == "qwen2.5-mini":
            # Ollama-hosted Qwen2.5 0.5B
            self.llm_model = Ollama(
                model="qwen2.5:0.5b",
                request_timeout=120.0,
                temperature=self.config.temperature
            )
        elif self.config.llm_model.startswith("openai"):
            # OpenAI model
            from llama_index.llms.openai import OpenAI
            self.llm_model = OpenAI(
                model=self.config.llm_model.replace("openai-", ""),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        else:
            # Default: Qwen2.5 7B
            self.llm_model = Ollama(
                model="qwen2.5:7b",
                request_timeout=120.0,
                temperature=self.config.temperature
            )
    
    def _setup_node_parser(self):
        """Set up document chunker."""
        if self.config.chunking_strategy == "chunk_256":
            # 256-char chunks
            self.node_parser = SentenceSplitter(
                chunk_size=256,
                chunk_overlap=self.config.chunk_overlap
            )
        elif self.config.chunking_strategy == "chunk_512":
            # 512-char chunks
            self.node_parser = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=self.config.chunk_overlap
            )
        elif self.config.chunking_strategy == "sentence":
            # Sentence-level chunking
            self.node_parser = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        elif self.config.chunking_strategy == "semantic":
            # Chinese-aware sentence splitter for semantic chunking
            def chinese_sentence_splitter(text: str) -> List[str]:
                """Split Chinese text by punctuation with length-based fallback."""
                import jieba
                import re

                # Split on strong delimiters only (!, ?, ;, newline)
                sentences = re.split(r'[！？；\n]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]

                # For very long sentences (>400 chars), split further on period
                final_sentences = []
                for sentence in sentences:
                    if len(sentence) > 400:
                        sub_parts = re.split(r'[。]', sentence)
                        sub_parts = [part.strip() for part in sub_parts if part.strip() and len(part) > 30]
                        if len(sub_parts) > 1:
                            final_sentences.extend(sub_parts)
                        else:
                            final_sentences.append(sentence)
                    else:
                        final_sentences.append(sentence)

                return final_sentences

            self.node_parser = SemanticSplitterNodeParser(
                buffer_size=4,
                breakpoint_percentile_threshold=85,
                embed_model=self.embed_model,
                sentence_splitter=chinese_sentence_splitter
            )
        else:
            # Default: simple chunking
            self.node_parser = SimpleNodeParser.from_defaults(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
    
    def chunk_documents(self, documents: Dict[str, str]) -> List[Dict[str, Any]]:
        """Chunk documents using LlamaIndex."""
        self.logger.info(f"Chunking documents, strategy: {self.config.chunking_strategy}")

        # Convert to LlamaIndex Document format
        llama_documents = []
        for doc_id, content in documents.items():
            doc = Document(
                text=content,
                metadata={"doc_id": doc_id, "source": doc_id}
            )
            llama_documents.append(doc)

        self.documents = llama_documents

        # Chunk using node parser
        self.nodes = self.node_parser.get_nodes_from_documents(llama_documents)

        # Convert to standard format
        chunks = []
        for i, node in enumerate(self.nodes):
            chunk = {
                "chunk_id": f"{node.metadata.get('doc_id', 'unknown')}_{i}",
                "content": node.text,
                "source_doc_id": node.metadata.get("doc_id", "unknown"),
                "metadata": dict(node.metadata)
            }
            chunks.append(chunk)

        self.logger.info(f"Chunking complete: {len(chunks)} chunks generated")
        return chunks
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """No-op: LlamaIndex handles embeddings internally during index build."""
        self.logger.info(f"Embeddings will be computed during index build using {self.config.embedding_model}")
        return []
    
    def _get_cache_path(self) -> str:
        """Get vector store cache path."""
        config_hash = hash(f"{self.config.chunking_strategy}_{self.config.chunk_size}_{self.config.chunk_overlap}_{self.config.embedding_model}")
        return f"./chroma_db/{self.config.system_name}_{abs(config_hash) % 10000}"

    def _check_vector_store_cache(self) -> bool:
        """Check if a valid vector store cache exists."""
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            return False

        try:
            chroma_client = chromadb.PersistentClient(path=cache_path)
            collections = chroma_client.list_collections()
            if not collections:
                return False

            collection = collections[0]
            count = collection.count()
            self.logger.info(f"Found cached vector store with {count} vectors")
            return count > 0

        except Exception as e:
            self.logger.warning(f"Error checking vector store cache: {e}")
            return False

    def _load_vector_store_cache(self):
        """Load cached vector store."""
        cache_path = self._get_cache_path()
        self.logger.info(f"Loading cached vector store: {cache_path}")

        try:
            chroma_client = chromadb.PersistentClient(path=cache_path)
            collections = chroma_client.list_collections()

            if collections:
                chroma_collection = collections[0]
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                # Rebuild index from cache
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context,
                    embed_model=self.embed_model
                )

                # Create query engine
                self.query_engine = self.index.as_query_engine(
                    similarity_top_k=self.config.retrieval_top_k,
                    response_mode="compact",
                    llm=self.llm_model
                )

                self.logger.info("Successfully loaded cached vector store")
                return True

        except Exception as e:
            self.logger.error(f"Failed to load vector store cache: {e}")
            return False
    
    def _convert_cached_chunks_to_nodes(self, cached_chunks: List[Dict[str, Any]]):
        """Convert cached chunks to LlamaIndex Document and Node format."""
        from llama_index.core import Document
        from llama_index.core.schema import TextNode

        self.logger.info(f"Converting {len(cached_chunks)} cached chunks to LlamaIndex format")

        # Group by source_doc_id to create Documents
        doc_contents = {}
        for chunk in cached_chunks:
            source_doc_id = chunk.get('source_doc_id', 'unknown')
            if source_doc_id not in doc_contents:
                doc_contents[source_doc_id] = []
            doc_contents[source_doc_id].append(chunk['content'])

        # Create Documents
        self.documents = []
        for doc_id, contents in doc_contents.items():
            doc_text = '\n'.join(contents)
            doc = Document(
                text=doc_text,
                metadata={"doc_id": doc_id, "source": doc_id}
            )
            self.documents.append(doc)

        # Create Nodes
        self.nodes = []
        for chunk in cached_chunks:
            node = TextNode(
                text=chunk['content'],
                metadata=chunk.get('metadata', {})
            )
            # Ensure required metadata fields exist
            if 'doc_id' not in node.metadata:
                node.metadata['doc_id'] = chunk.get('source_doc_id', 'unknown')
            self.nodes.append(node)

        self.logger.info(f"Conversion complete: {len(self.documents)} docs, {len(self.nodes)} nodes")
    
    def build_vector_store(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Build vector store and query engine with caching support."""
        self.logger.info(f"Building vector store for system {self.config.system_name}")
        self.logger.info(f"Config - Chunking: {self.config.chunking_strategy}, "
                        f"Embedding: {self.config.embedding_model}, "
                        f"LLM: {self.config.llm_model}")

        # Check cache
        if self._check_vector_store_cache():
            if self._load_vector_store_cache():
                self.logger.info("Using cached vector store")
                return
            else:
                self.logger.warning("Cache load failed, rebuilding vector store")

        # Convert cached chunk dicts to LlamaIndex format if needed
        if chunks and len(chunks) > 0 and isinstance(chunks[0], dict) and 'content' in chunks[0]:
            self.logger.info("Detected cached chunks, converting to LlamaIndex format")
            self._convert_cached_chunks_to_nodes(chunks)

        # Create new vector store
        cache_path = self._get_cache_path()
        os.makedirs(cache_path, exist_ok=True)

        # Clean up old vector store if exists
        import shutil
        if os.path.exists(cache_path) and os.listdir(cache_path):
            self.logger.info(f"Cleaning old vector store: {cache_path}")
            shutil.rmtree(cache_path)
            os.makedirs(cache_path, exist_ok=True)

        chroma_client = chromadb.PersistentClient(path=cache_path)
        collection_name = f"kb_{self.config.system_name}_{hash(str(self.config.__dict__)) % 10000}"

        # Delete existing collection with same name
        try:
            chroma_client.delete_collection(name=collection_name)
        except:
            pass

        chroma_collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={
                "system_name": self.config.system_name,
                "chunking_strategy": self.config.chunking_strategy,
                "embedding_model": self.config.embedding_model,
                "llm_model": self.config.llm_model,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap
            }
        )

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.logger.info(f"Processing {len(self.documents)} documents, {len(self.nodes)} chunks")

        # Log first few chunks for verification
        for i, node in enumerate(self.nodes[:3]):
            self.logger.info(f"Chunk {i+1} (len={len(node.text)}): {node.text[:100]}...")

        # Build vector index (GPU-accelerated embedding applied automatically)
        self.logger.info("Creating vector index...")
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            node_parser=self.node_parser,
            show_progress=True
        )

        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.config.retrieval_top_k,
            response_mode="compact",
            llm=self.llm_model
        )

        self.logger.info(f"Vector store built and cached for {self.config.system_name}")
        self.logger.info(f"Cache path: {cache_path}, collection: {collection_name}")
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve relevant documents from the index."""
        if self.index is None:
            raise ValueError("Index not built. Call build_vector_store first")

        retriever = self.index.as_retriever(
            similarity_top_k=self.config.retrieval_top_k
        )

        retrieved_nodes = retriever.retrieve(query)

        results = []
        for i, node in enumerate(retrieved_nodes):
            result = RetrievalResult(
                content=node.text,
                score=node.score if hasattr(node, 'score') else 1.0,
                chunk_id=node.metadata.get("doc_id", f"chunk_{i}"),
                source_doc_id=node.metadata.get("doc_id", f"doc_{i}")
            )
            results.append(result)

        return results

    def generate_answer(self, query: str, retrieved_chunks: List[RetrievalResult]) -> str:
        """Generate answer using retrieved chunks."""
        if not retrieved_chunks:
            return "No relevant information found to answer this question."

        try:
            context_pieces = []
            for i, chunk in enumerate(retrieved_chunks):
                context_pieces.append(f"[Document {i+1}] {chunk.content}")

            context_str = "\n\n".join(context_pieces)

            # Chinese LLM prompt intentionally preserved for Chinese language support
            prompt = f"""你是一个专业的AI助手，请基于以下信息回答问题。

可用信息：
{context_str}

用户问题：{query}

回答要求：
1. 严格基于上述信息回答，不得添加任何外部知识或编造内容
2. 仔细分析问题与提供信息的关联度和完整性
3. 如果信息完全相关且充足，请给出完整、准确的回答
4. 如果信息部分相关但不完整，请基于已知信息回答，并说明信息局限性
5. 如果信息不相关或严重不足，请明确说明"信息不足"
6. 利用推理能力从多个文档片段中综合信息，避免重复
7. 保持回答的准确性和完整性，避免过度推测

推理步骤：
- 识别问题中的关键概念和实体
- 在提供的信息中寻找相关内容片段
- 综合多个片段信息，避免矛盾
- 优先使用最直接相关的信息

请回答："""

            if hasattr(self.llm_model, 'complete'):
                response = self.llm_model.complete(prompt)
                answer = str(response).strip()
            else:
                from llama_index.core.base.llms.types import ChatMessage
                messages = [ChatMessage(role="user", content=prompt)]
                response = self.llm_model.chat(messages)
                answer = str(response.message.content).strip()

            if answer.startswith(("根据", "基于", "据", "从")):
                for prefix in ["，", "：", ":", ",", "。"]:
                    if prefix in answer:
                        answer = answer.split(prefix, 1)[1].strip()
                        break

            self.logger.info(f"Answer generated using {len(retrieved_chunks)} document chunks")
            return answer

        except Exception as e:
            self.logger.error(f"Failed to generate answer with LLM: {e}")
            context = "\n".join([f"- {chunk.content}" for chunk in retrieved_chunks[:3]])
            return f"Based on retrieved information:\n{context}\n\nFor the question '{query}', unable to generate complete answer due to technical issues, but the above information may be helpful."

    def query(self, question: str) -> RAGResponse:
        """Query the RAG system."""
        try:
            retrieved_chunks = self.retrieve(question)

            answer = self.generate_answer(question, retrieved_chunks)

            evidence = [chunk.content for chunk in retrieved_chunks]

            return RAGResponse(
                question=question,
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                system_config=self.config,
                evidence=[chunk.content for chunk in retrieved_chunks],
                metadata={
                    "retrieval_count": len(retrieved_chunks),
                    "confidence": 0.8
                }
            )
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return RAGResponse(
                question=question,
                answer=f"Query failed: {str(e)}",
                retrieved_chunks=[],
                system_config=self.config,
                evidence=[],
                metadata={"error": str(e)}
            ) 