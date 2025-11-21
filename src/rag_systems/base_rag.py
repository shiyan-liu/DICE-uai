"""
RAG系统基础抽象类
定义了RAG系统的通用接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class RAGConfig:
    """RAG系统配置"""
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
    """检索结果"""
    content: str
    score: float
    chunk_id: str
    source_doc_id: str

@dataclass
class RAGResponse:
    """RAG系统响应"""
    question: str
    answer: str
    retrieved_chunks: List[RetrievalResult]
    system_config: RAGConfig
    evidence: List[str] = None  # 添加evidence字段，存储检索到的文档内容
    metadata: Dict[str, Any] = None

class BaseRAGSystem(ABC):
    """RAG系统基础抽象类"""
    
    def __init__(self, config: RAGConfig):
        """
        初始化RAG系统
        
        Args:
            config: RAG系统配置
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.system_name}")
        self.knowledge_base = {}
        self.vector_store = None
        self.is_indexed = False
    
    @abstractmethod
    def chunk_documents(self, documents: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        对文档进行分块处理
        
        Args:
            documents: 文档字典 {doc_id: content}
            
        Returns:
            List[Dict]: 分块结果，每个块包含 {chunk_id, content, source_doc_id, metadata}
        """
        pass
    
    @abstractmethod
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """
        为分块创建嵌入向量
        
        Args:
            chunks: 分块列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        pass
    
    @abstractmethod
    def build_vector_store(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        构建向量存储
        
        Args:
            chunks: 分块列表
            embeddings: 嵌入向量列表
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        检索相关文档
        
        Args:
            query: 查询问题
            
        Returns:
            List[RetrievalResult]: 检索结果列表
        """
        pass
    
    @abstractmethod
    def generate_answer(self, query: str, retrieved_chunks: List[RetrievalResult]) -> str:
        """
        基于检索到的文档生成答案
        
        Args:
            query: 查询问题
            retrieved_chunks: 检索到的文档块
            
        Returns:
            str: 生成的答案
        """
        pass
    
    def process_knowledge_base(self, documents: Dict[str, str]) -> str:
        """
        处理知识库，创建专属的索引
        
        Args:
            documents: 原始文档
            
        Returns:
            str: 处理后的知识库路径或标识符
        """
        self.logger.info(f"开始处理知识库，系统: {self.config.system_name}")
        
        # 1. 文档分块
        chunks = self.chunk_documents(documents)
        self.logger.info(f"文档分块完成，共 {len(chunks)} 个块")
        
        # 2. 创建嵌入
        embeddings = self.create_embeddings(chunks)
        self.logger.info("嵌入向量创建完成")
        
        # 3. 构建向量存储
        self.build_vector_store(chunks, embeddings)
        self.logger.info("向量存储构建完成")
        
        self.is_indexed = True
        return f"knowledge_base_{self.config.system_name}"
    
    def query(self, question: str) -> RAGResponse:
        """
        执行RAG查询
        
        Args:
            question: 用户问题
            
        Returns:
            RAGResponse: RAG响应
        """
        if not self.is_indexed:
            raise ValueError("知识库尚未建立索引，请先调用 process_knowledge_base")
        
        # 1. 检索相关文档
        retrieved_chunks = self.retrieve(question)
        
        # 2. 生成答案
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
        """保存处理后的知识库"""
        # 实现保存逻辑（可以是向量数据库、文件等）
        pass
    
    def load_processed_kb(self, load_path: str):
        """加载处理后的知识库"""
        # 实现加载逻辑
        pass 