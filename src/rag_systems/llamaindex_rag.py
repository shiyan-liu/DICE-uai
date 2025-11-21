"""
基于LlamaIndex的RAG系统实现
支持多种embedding模型、chunking策略和LLM模型
"""

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

from .base_rag import BaseRAGSystem, RAGConfig, RetrievalResult, RAGResponse


class LlamaIndexRAGSystem(BaseRAGSystem):
    """基于LlamaIndex的RAG系统实现"""
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.index = None
        self.query_engine = None
        self.documents = []
        self.nodes = []
        
        # 初始化embedding模型
        self._setup_embedding_model()
        
        # 初始化LLM模型
        self._setup_llm_model()
        
        # 初始化node parser
        self._setup_node_parser()
        
        # 设置全局配置
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm_model
        Settings.node_parser = self.node_parser
        
    def _setup_embedding_model(self):
        """设置embedding模型（支持GPU加速）"""
        import torch
        import os
        
        # 设置镜像环境变量
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"使用设备: {device}")
        
        if self.config.embedding_model == "bge-large-zh":
            self.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-large-zh-v1.5",
                cache_folder="./models",
                device=device,
                max_length=512,
                trust_remote_code=True,  # 允许加载自定义代码
            )
        elif self.config.embedding_model == "bge-small-zh":
            self.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-zh-v1.5",
                cache_folder="./models",
                device=device,
                max_length=512,
                trust_remote_code=True,  # 允许加载自定义代码
            )
        else:
            # 默认使用小模型
            self.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder="./models",
                device=device,
                max_length=384,
                trust_remote_code=True,  # 允许加载自定义代码
            )
    
    def _setup_llm_model(self):
        """设置LLM模型"""
        if self.config.llm_model == "qwen2.5":
            # 使用Ollama本地部署的Qwen2.5
            self.llm_model = Ollama(
                model="qwen2.5:7b",
                request_timeout=120.0,
                temperature=self.config.temperature
            )
        elif self.config.llm_model == "qwen2.5-mini":
            # 使用Ollama本地部署的Qwen2.5-0.5B (超小模型)
            self.llm_model = Ollama(
                model="qwen2.5:0.5b",
                request_timeout=120.0,
                temperature=self.config.temperature
            )
        elif self.config.llm_model.startswith("openai"):
            # 如果需要使用OpenAI模型
            from llama_index.llms.openai import OpenAI
            self.llm_model = OpenAI(
                model=self.config.llm_model.replace("openai-", ""),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        else:
            # 默认使用Qwen2.5
            self.llm_model = Ollama(
                model="qwen2.5:7b",
                request_timeout=120.0,
                temperature=self.config.temperature
            )
    
    def _setup_node_parser(self):
        """设置文档分块器"""
        if self.config.chunking_strategy == "chunk_256":
            # 256字符长度分块
            self.node_parser = SentenceSplitter(
                chunk_size=256,
                chunk_overlap=self.config.chunk_overlap
            )
        elif self.config.chunking_strategy == "chunk_512":
            # 512字符长度分块  
            self.node_parser = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=self.config.chunk_overlap
            )
        elif self.config.chunking_strategy == "sentence":
            # 保留原有sentence分块选项
            self.node_parser = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        elif self.config.chunking_strategy == "semantic":
            # 为中文文本设置合适的分词器
            def chinese_sentence_splitter(text: str) -> List[str]:
                """中文分句器，结合jieba分词和标点符号"""
                import jieba
                import re
                
                # 🔧 修改：减少句号分割，只使用强分句符号
                # 移除句号，只保留感叹号、问号、分号等强分句符号
                sentences = re.split(r'[！？；\n]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                # 🔧 修改：对于极长的句子（>400字符），才进行进一步分割
                final_sentences = []
                for sentence in sentences:
                    if len(sentence) > 400:  # 进一步提高阈值到400
                        # 🔧 修改：只按照句号分割（作为二级分割）
                        sub_parts = re.split(r'[。]', sentence)
                        sub_parts = [part.strip() for part in sub_parts if part.strip() and len(part) > 30]
                        if len(sub_parts) > 1:  # 只有真正分割出多个部分才使用
                            final_sentences.extend(sub_parts)
                        else:
                            final_sentences.append(sentence)
                    else:
                        final_sentences.append(sentence)
                
                return final_sentences
            
            self.node_parser = SemanticSplitterNodeParser(
                buffer_size=4,  # 🔧 修改：从3增加到4，更多上下文
                breakpoint_percentile_threshold=85,  # 🔧 修改：从80提升到85，更少分割点
                embed_model=self.embed_model,
                sentence_splitter=chinese_sentence_splitter
            )
        else:
            # 默认使用简单分块
            self.node_parser = SimpleNodeParser.from_defaults(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
    
    def chunk_documents(self, documents: Dict[str, str]) -> List[Dict[str, Any]]:
        """使用LlamaIndex进行文档分块"""
        self.logger.info(f"使用LlamaIndex进行文档分块，策略: {self.config.chunking_strategy}")
        
        # 转换为LlamaIndex Document格式
        llama_documents = []
        for doc_id, content in documents.items():
            doc = Document(
                text=content,
                metadata={"doc_id": doc_id, "source": doc_id}
            )
            llama_documents.append(doc)
        
        self.documents = llama_documents
        
        # 使用node parser进行分块
        self.nodes = self.node_parser.get_nodes_from_documents(llama_documents)
        
        # 转换为标准格式
        chunks = []
        for i, node in enumerate(self.nodes):
            chunk = {
                "chunk_id": f"{node.metadata.get('doc_id', 'unknown')}_{i}",
                "content": node.text,
                "source_doc_id": node.metadata.get("doc_id", "unknown"),
                "metadata": dict(node.metadata)
            }
            chunks.append(chunk)
        
        self.logger.info(f"分块完成，共生成 {len(chunks)} 个chunks")
        return chunks
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """
        LlamaIndex会在build_vector_store中自动处理embedding
        
        注释说明：
        - LlamaIndex的VectorStoreIndex.from_documents()会自动调用embed_model
        - 每个文档块都会被转换为embedding向量并存储在向量数据库中
        - GPU加速在_setup_embedding_model中配置，自动应用到这个过程
        - 返回空列表是因为embedding是内部处理的，外部不需要直接访问
        """
        self.logger.info("📊 LlamaIndex将在构建索引时自动处理embedding（支持GPU加速）")
        self.logger.info(f"🎯 将使用 {self.config.embedding_model} 模型进行向量化")
        return []
    
    def _get_cache_path(self) -> str:
        """获取向量存储缓存路径"""
        config_hash = hash(f"{self.config.chunking_strategy}_{self.config.chunk_size}_{self.config.chunk_overlap}_{self.config.embedding_model}")
        return f"./chroma_db/{self.config.system_name}_{abs(config_hash) % 10000}"
    
    def _check_vector_store_cache(self) -> bool:
        """检查向量存储缓存是否存在且有效"""
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            return False
        
        # 检查缓存是否有数据
        try:
            chroma_client = chromadb.PersistentClient(path=cache_path)
            collections = chroma_client.list_collections()
            if not collections:
                return False
            
            # 检查第一个集合是否有数据
            collection = collections[0]
            count = collection.count()
            self.logger.info(f"发现缓存的向量存储，包含 {count} 个向量")
            return count > 0
            
        except Exception as e:
            self.logger.warning(f"检查向量存储缓存时出错: {e}")
            return False
    
    def _load_vector_store_cache(self):
        """加载缓存的向量存储"""
        cache_path = self._get_cache_path()
        self.logger.info(f"加载缓存的向量存储: {cache_path}")
        
        try:
            chroma_client = chromadb.PersistentClient(path=cache_path)
            collections = chroma_client.list_collections()
            
            if collections:
                chroma_collection = collections[0]  # 使用第一个集合
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # 重新创建索引
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context,
                    embed_model=self.embed_model
                )
                
                # 创建查询引擎
                self.query_engine = self.index.as_query_engine(
                    similarity_top_k=self.config.retrieval_top_k,
                    response_mode="compact",
                    llm=self.llm_model
                )
                
                self.logger.info(f"成功加载缓存的向量存储")
                return True
                
        except Exception as e:
            self.logger.error(f"加载向量存储缓存失败: {e}")
            return False
    
    def _convert_cached_chunks_to_nodes(self, cached_chunks: List[Dict[str, Any]]):
        """
        将缓存的chunks数据转换为LlamaIndex的Document和Node格式
        
        Args:
            cached_chunks: 缓存的chunks数据
        """
        from llama_index.core import Document
        from llama_index.core.schema import TextNode
        
        self.logger.info(f"📦 转换 {len(cached_chunks)} 个缓存chunks为LlamaIndex格式")
        
        # 转换为Documents（按source_doc_id分组）
        doc_contents = {}
        for chunk in cached_chunks:
            source_doc_id = chunk.get('source_doc_id', 'unknown')
            if source_doc_id not in doc_contents:
                doc_contents[source_doc_id] = []
            doc_contents[source_doc_id].append(chunk['content'])
        
        # 创建Documents
        self.documents = []
        for doc_id, contents in doc_contents.items():
            doc_text = '\n'.join(contents)
            doc = Document(
                text=doc_text,
                metadata={"doc_id": doc_id, "source": doc_id}
            )
            self.documents.append(doc)
        
        # 创建Nodes
        self.nodes = []
        for chunk in cached_chunks:
            node = TextNode(
                text=chunk['content'],
                metadata=chunk.get('metadata', {})
            )
            # 确保metadata中有必要的字段
            if 'doc_id' not in node.metadata:
                node.metadata['doc_id'] = chunk.get('source_doc_id', 'unknown')
            self.nodes.append(node)
        
        self.logger.info(f"✅ 转换完成: {len(self.documents)} 个文档, {len(self.nodes)} 个节点")
    
    def build_vector_store(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """构建向量存储和查询引擎（支持缓存）"""
        self.logger.info(f"为系统 {self.config.system_name} 构建独立的向量存储")
        self.logger.info(f"使用策略 - Chunking: {self.config.chunking_strategy}, "
                        f"Embedding: {self.config.embedding_model}, "
                        f"LLM: {self.config.llm_model}")
        
        # 检查缓存
        if self._check_vector_store_cache():
            if self._load_vector_store_cache():
                self.logger.info("✅ 成功使用缓存的向量存储")
                return
            else:
                self.logger.warning("⚠️ 加载缓存失败，重新构建向量存储")
        
        # 如果传入的是缓存的chunks数据，需要先转换为LlamaIndex格式
        if chunks and len(chunks) > 0 and isinstance(chunks[0], dict) and 'content' in chunks[0]:
            self.logger.info(f"🔄 检测到缓存的chunks数据，转换为LlamaIndex格式")
            self._convert_cached_chunks_to_nodes(chunks)
        
        # 创建新的向量存储
        cache_path = self._get_cache_path()
        os.makedirs(cache_path, exist_ok=True)
        
        # 清理旧的向量存储（如果存在）
        import shutil
        if os.path.exists(cache_path) and os.listdir(cache_path):
            self.logger.info(f"清理旧的向量存储: {cache_path}")
            shutil.rmtree(cache_path)
            os.makedirs(cache_path, exist_ok=True)
        
        chroma_client = chromadb.PersistentClient(path=cache_path)
        collection_name = f"kb_{self.config.system_name}_{hash(str(self.config.__dict__)) % 10000}"
        
        # 删除已存在的同名集合
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
        
        # 记录处理的文档数量和chunking详情
        self.logger.info(f"处理 {len(self.documents)} 个原始文档")
        self.logger.info(f"生成 {len(self.nodes)} 个文档块")
        
        # 打印前几个块的信息用于验证
        for i, node in enumerate(self.nodes[:3]):
            self.logger.info(f"块 {i+1} (长度: {len(node.text)}): {node.text[:100]}...")
        
        # 创建索引（GPU加速的embedding会在这里自动使用）
        self.logger.info("🚀 开始创建向量索引（使用GPU加速的embedding）")
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            node_parser=self.node_parser,
            show_progress=True
        )
        
        # 创建查询引擎
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.config.retrieval_top_k,
            response_mode="compact",
            llm=self.llm_model
        )
        
        self.logger.info(f"✅ 系统 {self.config.system_name} 向量存储构建完成并缓存")
        self.logger.info(f"📁 存储路径: {cache_path}")
        self.logger.info(f"🏷️ 集合名称: {collection_name}")
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """检索相关文档"""
        if self.index is None:
            raise ValueError("索引未构建，请先调用build_vector_store")
        
        # 使用索引进行检索
        retriever = self.index.as_retriever(
            similarity_top_k=self.config.retrieval_top_k
        )
        
        retrieved_nodes = retriever.retrieve(query)
        
        # 转换为标准格式
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
        """使用检索到的chunks生成答案"""
        if not retrieved_chunks:
            return "没有找到相关信息来回答这个问题。"
        
        try:
            # 将检索到的chunks转换为上下文字符串
            context_pieces = []
            for i, chunk in enumerate(retrieved_chunks):
                context_pieces.append(f"[文档{i+1}] {chunk.content}")
            
            context_str = "\n\n".join(context_pieces)
            
            # 统一的智能提示词，既能发挥large模型优势，又能约束mini模型
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

            # 使用LLM生成答案
            if hasattr(self.llm_model, 'complete'):
                response = self.llm_model.complete(prompt)
                answer = str(response).strip()
            else:
                # 对于Ollama等其他模型
                from llama_index.core.base.llms.types import ChatMessage
                messages = [ChatMessage(role="user", content=prompt)]
                response = self.llm_model.chat(messages)
                answer = str(response.message.content).strip()
            
            # 清理答案，移除可能的前言
            if answer.startswith(("根据", "基于", "据", "从")):
                # 尝试找到实际答案开始的位置
                for prefix in ["，", "：", ":", ",", "。"]:
                    if prefix in answer:
                        answer = answer.split(prefix, 1)[1].strip()
                        break
            
            self.logger.info(f"成功生成答案，使用了 {len(retrieved_chunks)} 个文档块")
            return answer
            
        except Exception as e:
            self.logger.error(f"使用LLM生成答案时出错: {e}")
            # 降级为简单拼接
            context = "\n".join([f"- {chunk.content}" for chunk in retrieved_chunks[:3]])
            return f"基于检索到的相关信息：\n{context}\n\n对于问题 '{query}'，由于技术问题无法生成完整答案，但上述信息可能对您有帮助。"
    
    def query(self, question: str) -> RAGResponse:
        """查询RAG系统"""
        try:
            # 检索
            retrieved_chunks = self.retrieve(question)
            
            # 生成答案
            answer = self.generate_answer(question, retrieved_chunks)
            
            # 构建证据列表
            evidence = [chunk.content for chunk in retrieved_chunks]
            
            return RAGResponse(
                question=question,
                answer=answer,
                retrieved_chunks=retrieved_chunks,  # 使用正确的字段名
                system_config=self.config,          # 使用正确的字段名
                evidence=[chunk.content for chunk in retrieved_chunks],  # 添加evidence字段
                metadata={
                    "retrieval_count": len(retrieved_chunks),
                    "confidence": 0.8
                }
            )
        except Exception as e:
            self.logger.error(f"查询失败: {e}")
            return RAGResponse(
                question=question,
                answer=f"查询失败: {str(e)}",
                retrieved_chunks=[],
                system_config=self.config,
                evidence=[],  # 添加空的evidence列表
                metadata={"error": str(e)}
            ) 