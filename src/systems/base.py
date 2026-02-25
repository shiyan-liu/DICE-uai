"""
Base abstract class for RAG systems.
Defines the common interface for RAG systems.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class RAGConfig:
    """RAG system config."""
    system_name: str
    chunking_strategy: str  # "chunk_256", "chunk_512", "sliding_window", "paragraph"
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-3.5-turbo"
    retrieval_top_k: int = 5
    retrieval_strategy: str = "vector_similarity"  # "vector_similarity", "hybrid", "bm25"
    temperature: float = 0.1
    max_tokens: int = 1000

@dataclass
class RetrievalResult:
    """Retrieval result."""
    content: str
    score: float
    chunk_id: str
    source_doc_id: str

@dataclass
class RAGResponse:
    """RAG system response."""
    question: str
    answer: str
    retrieved_chunks: List[RetrievalResult]
    system_config: RAGConfig
    evidence: List[str] = None
    metadata: Dict[str, Any] = None

class BaseRAGSystem(ABC):
    """Base abstract class for RAG systems."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.system_name}")
        self.knowledge_base = {}
        self.vector_store = None
        self.is_indexed = False

    @abstractmethod
    def chunk_documents(self, documents: Dict[str, str]) -> List[Dict[str, Any]]:
        """Chunk documents into smaller pieces."""
        pass

    @abstractmethod
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """Create embedding vectors for chunks."""
        pass

    @abstractmethod
    def build_vector_store(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Build vector store from chunks and embeddings."""
        pass

    @abstractmethod
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        pass

    @abstractmethod
    def generate_answer(self, query: str, retrieved_chunks: List[RetrievalResult]) -> str:
        """Generate answer based on retrieved documents."""
        pass

    def process_knowledge_base(self, documents: Dict[str, str]) -> str:
        """Process knowledge base and create index."""
        self.logger.info(f"Processing knowledge base for system: {self.config.system_name}")

        chunks = self.chunk_documents(documents)
        self.logger.info(f"Chunking complete: {len(chunks)} chunks")

        embeddings = self.create_embeddings(chunks)
        self.logger.info("Embedding creation complete")

        self.build_vector_store(chunks, embeddings)
        self.logger.info("Vector store built")

        self.is_indexed = True
        return f"knowledge_base_{self.config.system_name}"

    def query(self, question: str) -> RAGResponse:
        """Execute RAG query."""
        if not self.is_indexed:
            raise ValueError("Knowledge base not indexed yet. Call process_knowledge_base first.")

        retrieved_chunks = self.retrieve(question)
        answer = self.generate_answer(question, retrieved_chunks)

        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            system_config=self.config,
            metadata={
                "num_retrieved": len(retrieved_chunks),
                "avg_score": sum(chunk.score for chunk in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0
            }
        )

    def save_processed_kb(self, save_path: str):
        """Save processed knowledge base."""
        pass

    def load_processed_kb(self, load_path: str):
        """Load processed knowledge base."""
        pass
