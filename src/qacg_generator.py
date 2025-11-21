"""
QACG四元组生成器
用于为每个RAG系统生成Question-Answer-Context-Groundtruth四元组数据
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

from .rag_systems.llamaindex_rag import LlamaIndexRAGSystem
from .rag_systems.base_rag import RAGConfig


class QACGGenerator:
    """QACG四元组生成器"""
    
    def __init__(self, llm_model: str = "qwen2.5:7b"):
        """
        初始化生成器
        
        Args:
            llm_model: 用于生成问题的LLM模型
        """
        self.logger = logging.getLogger(__name__)
        
        # 初始化用于生成问题的LLM
        if llm_model.startswith("openai"):
            self.question_llm = OpenAI(model=llm_model.replace("openai-", ""))
        else:
            self.question_llm = Ollama(model=llm_model, request_timeout=120.0)
        
        # 问题生成模板
        self.question_templates = [
            "根据以下文本内容，生成一个具体的问题：\n{context}\n\n请生成一个可以从上述内容中找到明确答案的问题：",
            "基于这段文字，提出一个关键问题：\n{context}\n\n问题应该针对文本中的核心信息：",
            "阅读下面的内容，设计一个问题：\n{context}\n\n问题要求能够通过文本内容回答：",
            "请根据以下信息提出一个问题：\n{context}\n\n确保问题的答案在文本中可以找到：",
            "分析这段文字，生成相关问题：\n{context}\n\n问题应该测试对文本内容的理解："
        ]
    
    def load_knowledge_base(self, jsonl_path: str) -> Dict[str, str]:
        """
        加载知识库
        
        Args:
            jsonl_path: JSONL文件路径
            
        Returns:
            Dict[str, str]: 文档ID到内容的映射
        """
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
                        self.logger.warning(f"跳过无法解析的行 {line_num + 1}")
        
        self.logger.info(f"加载了 {len(knowledge_base)} 个文档")
        return knowledge_base
    
    def sample_documents(self, knowledge_base: Dict[str, str], sample_size: int = 50) -> Dict[str, str]:
        """
        从知识库中采样文档
        
        Args:
            knowledge_base: 完整知识库
            sample_size: 采样大小
            
        Returns:
            Dict[str, str]: 采样后的文档
        """
        if len(knowledge_base) <= sample_size:
            return knowledge_base
        
        # 随机采样
        doc_ids = list(knowledge_base.keys())
        sampled_ids = random.sample(doc_ids, sample_size)
        
        sampled_docs = {doc_id: knowledge_base[doc_id] for doc_id in sampled_ids}
        self.logger.info(f"采样了 {len(sampled_docs)} 个文档用于生成QACG")
        
        return sampled_docs
    
    def generate_question_from_context(self, context: str) -> str:
        """
        从上下文生成问题
        
        Args:
            context: 上下文文本
            
        Returns:
            str: 生成的问题
        """
        # 随机选择一个模板
        template = random.choice(self.question_templates)
        prompt = template.format(context=context[:1000])  # 限制上下文长度
        
        try:
            if hasattr(self.question_llm, 'complete'):
                response = self.question_llm.complete(prompt)
                question = str(response).strip()
            else:
                # 对于Ollama等其他模型
                response = self.question_llm.generate([prompt])
                question = str(response).strip()
            
            # 清理问题格式
            question = question.replace("问题：", "").replace("Question:", "").strip()
            if not question.endswith('?') and not question.endswith('？'):
                question += '？'
            
            return question
        except Exception as e:
            self.logger.error(f"生成问题失败: {e}")
            # 降级为规则生成
            return self._generate_rule_based_question(context)
    
    def _generate_rule_based_question(self, context: str) -> str:
        """
        基于规则生成问题（降级方案）
        
        Args:
            context: 上下文文本
            
        Returns:
            str: 生成的问题
        """
        # 简单的规则：提取关键信息生成问题
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
    
    def load_test_questions(self, qa_file_path: str = "dice/70条测试数据QA.txt") -> List[Dict[str, Any]]:
        """
        加载给定的70条测试数据
        
        Args:
            qa_file_path: QA测试数据文件路径
            
        Returns:
            List[Dict]: 测试问题列表
        """
        try:
            with open(qa_file_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            self.logger.info(f"成功加载 {len(test_data)} 条测试数据")
            return test_data
        except Exception as e:
            self.logger.error(f"加载测试数据失败: {e}")
            return []

    def generate_qacg_for_system(self, 
                                 rag_system: LlamaIndexRAGSystem, 
                                 knowledge_base: Dict[str, str],
                                 num_questions: int = 70) -> List[Dict[str, Any]]:
        """
        为特定RAG系统使用给定的70条测试数据生成QACG四元组
        
        Args:
            rag_system: RAG系统实例
            knowledge_base: 知识库
            num_questions: 使用的问题数量（默认70）
            
        Returns:
            List[Dict]: QACG四元组列表
        """
        self.logger.info(f"为系统 {rag_system.config.system_name} 使用给定测试数据生成QACG四元组")
        
        # 加载给定的70条测试数据
        test_questions = self.load_test_questions()
        if not test_questions:
            self.logger.error("无法加载测试数据，回退到生成模式")
            return self._generate_qacg_fallback(rag_system, knowledge_base, num_questions)
        
        # 使用指定数量的问题
        questions_to_use = test_questions[:num_questions]
        self.logger.info(f"使用前 {len(questions_to_use)} 条测试问题")
        
        qacg_list = []
        
        for i, test_item in enumerate(questions_to_use):
            try:
                question = test_item["question"]
                expected_answer = test_item["answer"]
                
                # 使用RAG系统生成答案
                rag_response = rag_system.query(question)
                rag_answer = rag_response.answer
                evidence = rag_response.evidence
                
                # 构建QACG四元组
                qacg = {
                    "question": question,
                    "rag_answer": rag_answer,  # RAG生成的答案
                    "expected_answer": expected_answer,  # 预期答案
                    "context": evidence,  # RAG检索到的上下文
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
                self.logger.info(f"处理第 {i+1}/{len(questions_to_use)} 个测试问题: {question[:50]}...")
                
            except Exception as e:
                self.logger.error(f"处理第 {i+1} 个测试问题时出错: {e}")
                continue
        
        self.logger.info(f"成功处理 {len(qacg_list)} 个测试问题")
        return qacg_list
    
    def _generate_qacg_fallback(self, 
                               rag_system: LlamaIndexRAGSystem, 
                               knowledge_base: Dict[str, str],
                               num_questions: int) -> List[Dict[str, Any]]:
        """
        回退到原始的问题生成模式（当无法加载测试数据时使用）
        """
        self.logger.info("使用回退模式生成问题")
        
        # 采样文档作为上下文
        sampled_docs = self.sample_documents(knowledge_base, min(50, len(knowledge_base)))
        doc_contents = list(sampled_docs.values())
        
        qacg_list = []
        
        for i in range(num_questions):
            try:
                # 随机选择一个文档作为上下文
                context = random.choice(doc_contents)
                
                # 截取适当长度的上下文
                context = context[:800] if len(context) > 800 else context
                
                # 生成问题
                question = self.generate_question_from_context(context)
                
                # 使用RAG系统生成答案
                rag_response = rag_system.query(question)
                answer = rag_response.answer
                evidence = rag_response.evidence
                
                # 构建QACG四元组
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
                self.logger.info(f"生成第 {i+1}/{num_questions} 个QACG")
                
            except Exception as e:
                self.logger.error(f"生成第 {i+1} 个QACG时出错: {e}")
                continue
        
        self.logger.info(f"成功生成 {len(qacg_list)} 个QACG四元组")
        return qacg_list
    
    def create_rag_systems(self) -> List[LlamaIndexRAGSystem]:
        """
        创建8种RAG系统配置 (2x2x2)
        
        Returns:
            List[LlamaIndexRAGSystem]: RAG系统列表
        """
        embedding_models = ["bge-large-zh", "bge-small-zh"]
        chunking_strategies = ["chunk_256", "chunk_512"]  # 改为基于长度的分块策略
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
        
        self.logger.info(f"创建了 {len(systems)} 个RAG系统")
        return systems
    
    def save_qacg_results(self, qacg_data: List[Dict[str, Any]], output_path: str):
        """
        保存QACG结果到JSON文件
        
        Args:
            qacg_data: QACG数据
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qacg_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"QACG结果已保存到: {output_path}")
    
    def generate_all_qacg(self, 
                          jsonl_path: str, 
                          output_dir: str = "qacg_output",
                          num_questions: int = 70):
        """
        为所有RAG系统生成QACG四元组
        确保每个系统独立处理JSONL中的每行文本
        
        Args:
            jsonl_path: 知识库JSONL文件路径
            output_dir: 输出目录
            num_questions: 每个系统生成的问题数量
        """
        self.logger.info("🚀 开始为所有RAG系统生成QACG四元组")
        self.logger.info("=" * 80)
        
        # 预先加载原始知识库，记录基本信息
        raw_knowledge_base = self.load_knowledge_base(jsonl_path)
        self.logger.info(f"📂 加载原始知识库: {jsonl_path}")
        self.logger.info(f"📄 总文档数: {len(raw_knowledge_base)}")
        self.logger.info(f"📏 文档长度范围: {min(len(v) for v in raw_knowledge_base.values())} - {max(len(v) for v in raw_knowledge_base.values())} 字符")
        
        # 创建RAG系统
        rag_systems = self.create_rag_systems()
        self.logger.info(f"🏗️  创建了 {len(rag_systems)} 个RAG系统配置")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 为每个系统独立处理知识库并生成QACG
        system_stats = []
        
        for i, system in enumerate(rag_systems, 1):
            try:
                self.logger.info("=" * 80)
                self.logger.info(f"🔄 处理系统 {i}/{len(rag_systems)}: {system.config.system_name}")
                self.logger.info("=" * 80)
                
                # 每个系统独立加载和处理知识库
                self.logger.info(f"📚 为系统 {system.config.system_name} 独立处理知识库")
                self.logger.info("🔑 关键特性: 每个系统根据其配置独立分块和嵌入相同的原始数据")
                
                # 使用系统特定的处理策略处理知识库
                processed_knowledge_base = self._process_knowledge_base_for_system(
                    system, raw_knowledge_base.copy()  # 传递副本确保独立性
                )
                
                # 记录系统处理统计
                stats = {
                    'system_name': system.config.system_name,
                    'chunking_strategy': system.config.chunking_strategy,
                    'embedding_model': system.config.embedding_model,
                    'llm_model': system.config.llm_model,
                    'input_docs': len(processed_knowledge_base),
                    'chunks_generated': len(system.nodes) if hasattr(system, 'nodes') and system.nodes else 0
                }
                system_stats.append(stats)
                
                # 生成QACG
                self.logger.info(f"❓ 开始为系统 {system.config.system_name} 生成 {num_questions} 个QACG四元组")
                qacg_data = self.generate_qacg_for_system(
                    system, processed_knowledge_base, num_questions
                )
                
                # 保存结果
                output_path = os.path.join(
                    output_dir, 
                    f"qacg_{system.config.system_name}.json"
                )
                self.save_qacg_results(qacg_data, output_path)
                
                self.logger.info(f"✅ 系统 {system.config.system_name} 处理完成")
                self.logger.info(f"💾 结果保存至: {output_path}")
                
            except Exception as e:
                self.logger.error(f"❌ 处理系统 {system.config.system_name} 时出错: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue
        
        # 输出最终统计信息
        self.logger.info("=" * 80)
        self.logger.info("📊 所有RAG系统处理完成 - 独立性验证报告")
        self.logger.info("=" * 80)
        
        if system_stats:
            # 按chunking策略分组统计
            chunking_stats = {}
            for stat in system_stats:
                strategy = stat['chunking_strategy']
                if strategy not in chunking_stats:
                    chunking_stats[strategy] = []
                chunking_stats[strategy].append(stat['chunks_generated'])
            
            self.logger.info("🔍 分块策略独立性验证:")
            for strategy, chunks_list in chunking_stats.items():
                self.logger.info(f"   {strategy}: {chunks_list} (chunks数量)")
                if len(set(chunks_list)) > 1:
                    self.logger.info(f"     ✅ 不同embedding模型产生了不同的chunk数量")
                else:
                    self.logger.info(f"     ⚠️  所有embedding模型产生了相同的chunk数量")
            
            # 验证不同策略产生了不同的结果
            all_chunks = [stat['chunks_generated'] for stat in system_stats]
            unique_chunks = len(set(all_chunks))
            self.logger.info(f"🎯 总体独立性: {unique_chunks}/{len(system_stats)} 种不同的chunk数量")
            
            if unique_chunks > 1:
                self.logger.info("✅ 确认：不同RAG系统配置产生了不同的处理结果")
            else:
                self.logger.warning("⚠️  警告：所有系统产生了相同的chunk数量，请检查配置差异")
                
        self.logger.info("🎉 所有RAG系统的QACG生成完成") 
    
    def _process_knowledge_base_for_system(self, system: LlamaIndexRAGSystem, 
                                         raw_knowledge_base: Dict[str, str]) -> Dict[str, str]:
        """
        为特定系统处理知识库 - 确保独立处理
        
        Args:
            system: RAG系统实例
            raw_knowledge_base: 原始知识库数据(JSONL文件的每行作为一个文档)
            
        Returns:
            Dict[str, str]: 处理后的知识库
        """
        self.logger.info(f"🔄 为系统 {system.config.system_name} 执行独立的知识库处理")
        self.logger.info(f"📊 原始知识库文档数量: {len(raw_knowledge_base)}")
        self.logger.info(f"⚙️  系统配置:")
        self.logger.info(f"   - Chunking策略: {system.config.chunking_strategy}")
        self.logger.info(f"   - Chunk大小: {system.config.chunk_size}")
        self.logger.info(f"   - Chunk重叠: {system.config.chunk_overlap}")
        self.logger.info(f"   - Embedding模型: {system.config.embedding_model}")
        
        # 检查是否已有处理过的知识库缓存
        cache_dir = f"./knowledge_cache/{system.config.system_name}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "processed_kb.json")
        
        # 检查缓存
        if os.path.exists(cache_file):
            self.logger.info(f"📁 检查缓存文件: {cache_file}")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_kb = json.load(f)
                
                # 验证缓存配置是否匹配
                if self._validate_cache_config(cached_kb.get('config', {}), system.config):
                    self.logger.info(f"✅ 缓存配置匹配，加载缓存的知识库")
                    
                    # 检查是否有缓存的chunks数据
                    if 'chunks' in cached_kb and cached_kb['chunks']:
                        self.logger.info(f"📦 发现缓存的chunks数据: {len(cached_kb['chunks'])} 个chunks")
                        self.logger.info(f"🚀 直接使用缓存的chunks，跳过重新分块")
                        
                        # 直接使用缓存的chunks数据构建向量存储
                        self._load_cached_chunks_and_build_vector_store(system, cached_kb['chunks'])
                        
                        return cached_kb['knowledge_base']
                    else:
                        self.logger.info(f"⚠️ 缓存中没有chunks数据，需要重新处理")
                        # 重新处理知识库以建立索引
                        self.logger.info(f"🏗️  重新建立向量索引 (缓存中无chunks数据)")
                        system.process_knowledge_base(cached_kb['knowledge_base'])
                        
                        return cached_kb['knowledge_base']
                else:
                    self.logger.info(f"❌ 缓存配置不匹配，将重新处理")
            except Exception as e:
                self.logger.warning(f"⚠️  读取缓存失败: {e}")
        
        # 如果没有缓存或配置不匹配，重新处理
        self.logger.info(f"🔄 开始独立处理知识库")
        self.logger.info(f"📝 处理策略详情:")
        self.logger.info(f"   - 每个JSONL行将作为独立文档处理")
        self.logger.info(f"   - 使用 {system.config.chunking_strategy} 分块策略")
        self.logger.info(f"   - 使用 {system.config.embedding_model} 嵌入模型")
        
        # 为当前系统独立处理知识库
        # 这里的关键是：每个系统都会根据自己的配置独立分块和嵌入
        processing_result = system.process_knowledge_base(raw_knowledge_base)
        
        # 记录处理结果统计信息并准备缓存数据
        chunks_data = []
        if hasattr(system, 'nodes') and system.nodes:
            chunk_count = len(system.nodes)
            self.logger.info(f"📈 系统 {system.config.system_name} 处理统计:")
            self.logger.info(f"   - 输入文档数: {len(raw_knowledge_base)}")
            self.logger.info(f"   - 生成chunks数: {chunk_count}")
            self.logger.info(f"   - 平均每文档chunks: {chunk_count/len(raw_knowledge_base):.2f}")
            
            # 显示前几个chunk的样本
            for i, node in enumerate(system.nodes[:3]):
                self.logger.info(f"   - Chunk {i+1} (长度 {len(node.text)}): {node.text[:50]}...")
            
            # 将nodes转换为可序列化的格式用于缓存
            self.logger.info("📦 准备缓存chunks数据...")
            for i, node in enumerate(system.nodes):
                chunk_data = {
                    "chunk_id": f"{node.metadata.get('doc_id', 'unknown')}_{i}",
                    "content": node.text,
                    "source_doc_id": node.metadata.get("doc_id", "unknown"),
                    "metadata": dict(node.metadata)
                }
                chunks_data.append(chunk_data)
            
            self.logger.info(f"✅ 已准备 {len(chunks_data)} 个chunks用于缓存")
        
        # 缓存处理结果（包含chunks数据）
        cache_data = {
            'config': {
                'chunking_strategy': system.config.chunking_strategy,
                'chunk_size': system.config.chunk_size,
                'chunk_overlap': system.config.chunk_overlap,
                'embedding_model': system.config.embedding_model,
                'system_name': system.config.system_name
            },
            'knowledge_base': raw_knowledge_base,
            'chunks': chunks_data,  # 新增：保存chunks数据
            'processing_stats': {
                'input_doc_count': len(raw_knowledge_base),
                'chunk_count': len(chunks_data),
                'processed_at': str(pd.Timestamp.now())
            }
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 知识库处理完成并缓存: {cache_file}")
        self.logger.info(f"✅ 系统 {system.config.system_name} 知识库独立处理完成")
        
        return raw_knowledge_base
    
    def _load_cached_chunks_and_build_vector_store(self, system: LlamaIndexRAGSystem, cached_chunks: List[Dict]) -> None:
        """
        直接使用缓存的chunks数据构建向量存储
        
        Args:
            system: RAG系统实例
            cached_chunks: 缓存的chunks数据
        """
        try:
            self.logger.info(f"🔄 为系统 {system.config.system_name} 使用缓存chunks构建向量存储")
            
            # 将缓存的chunks转换为系统需要的格式
            # 直接调用向量存储构建，跳过分块步骤
            system.build_vector_store(cached_chunks, [])  # embeddings参数为空，LlamaIndex会自动处理
            
            # 设置系统状态为已索引
            system.is_indexed = True
            
            self.logger.info(f"✅ 成功使用缓存chunks构建向量存储")
            
        except Exception as e:
            self.logger.error(f"❌ 使用缓存chunks构建向量存储失败: {e}")
            self.logger.warning(f"⚠️ 回退到标准处理流程")
            # 如果失败，回退到标准流程
            raise e

    def _validate_cache_config(self, cached_config: Dict, current_config) -> bool:
        """验证缓存配置是否与当前配置匹配"""
        key_fields = ['chunking_strategy', 'chunk_size', 'chunk_overlap', 'embedding_model']
        
        for field in key_fields:
            if cached_config.get(field) != getattr(current_config, field):
                return False
        
        return True 