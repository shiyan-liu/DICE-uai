"""
支持多RAG系统的数据加载器
为不同的RAG系统创建专属的知识库处理流程
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..rag_systems.rag_manager import RAGSystemManager, create_default_systems
from ..rag_systems.base_rag import RAGConfig
from ..core.dice_evaluator import RAGPair

class MultiRAGDataLoader:
    """支持多RAG系统的数据加载器"""
    
    def __init__(self, storage_dir: str = "rag_storage"):
        """
        初始化多RAG数据加载器
        
        Args:
            storage_dir: RAG系统存储目录
        """
        self.rag_manager = RAGSystemManager(storage_dir)
        self.logger = logging.getLogger(__name__)
        self.original_documents = {}
    
    def load_test_data(self, test_file_path: str) -> List[Dict[str, Any]]:
        """加载测试数据"""
        self.logger.info(f"加载测试数据: {test_file_path}")
        
        if test_file_path.endswith('.txt'):
            return self._load_txt_data(test_file_path)
        elif test_file_path.endswith('.jsonl'):
            return self._load_jsonl_data(test_file_path)
        else:
            raise ValueError(f"不支持的文件格式: {test_file_path}")
    
    def _load_txt_data(self, file_path: str) -> List[Dict[str, Any]]:
        """加载TXT格式的测试数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        try:
            data = json.loads(content)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            self.logger.error(f"无法解析JSON格式的文件: {file_path}")
            raise
    
    def _load_jsonl_data(self, file_path: str) -> List[Dict[str, Any]]:
        """加载JSONL格式的数据"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        self.logger.warning(f"跳过无法解析的行: {line}")
        return data
    
    def load_knowledge_base(self, kb_file_path: str) -> Dict[str, str]:
        """加载原始知识库"""
        self.logger.info(f"加载知识库: {kb_file_path}")
        
        knowledge_base = {}
        
        if kb_file_path.endswith('.jsonl'):
            with open(kb_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            doc = json.loads(line)
                            content_id = doc.get('content_id', '')
                            content = doc.get('content', '')
                            if content_id and content:
                                knowledge_base[content_id] = content
                        except json.JSONDecodeError:
                            continue
        
        self.original_documents = knowledge_base
        self.logger.info(f"加载了 {len(knowledge_base)} 个原始文档")
        return knowledge_base
    
    def setup_rag_systems(self, 
                         documents: Dict[str, str], 
                         system_configs: Optional[List[RAGConfig]] = None) -> List[str]:
        """
        设置RAG系统
        
        Args:
            documents: 原始文档
            system_configs: 系统配置列表（None时使用默认配置）
            
        Returns:
            List[str]: 创建的系统ID列表
        """
        if system_configs is None:
            # 使用默认配置创建多个系统
            system_ids = create_default_systems(self.rag_manager, documents)
        else:
            # 使用自定义配置
            system_ids = []
            for config in system_configs:
                system_id = self.rag_manager.create_system_from_config(config)
                self.rag_manager.process_knowledge_base_for_system(system_id, documents)
                system_ids.append(system_id)
        
        self.logger.info(f"设置了 {len(system_ids)} 个RAG系统: {system_ids}")
        return system_ids
    
    def create_system_comparison_pairs(self, 
                                     test_data: List[Dict[str, Any]], 
                                     system_a_id: str, 
                                     system_b_id: str) -> List[RAGPair]:
        """
        创建两个系统的比较对
        
        Args:
            test_data: 测试数据
            system_a_id: 系统A的ID
            system_b_id: 系统B的ID
            
        Returns:
            List[RAGPair]: RAG对比对列表
        """
        questions = [item.get('question', '') for item in test_data if item.get('question')]
        
        return self.rag_manager.create_rag_pairs_from_systems(
            questions, system_a_id, system_b_id
        )
    
    def create_all_pairwise_comparisons(self, 
                                      test_data: List[Dict[str, Any]], 
                                      system_ids: List[str]) -> Dict[Tuple[str, str], List[RAGPair]]:
        """
        创建所有系统的两两比较
        
        Args:
            test_data: 测试数据
            system_ids: 系统ID列表
            
        Returns:
            Dict[Tuple[str, str], List[RAGPair]]: 系统对比对映射
        """
        comparisons = {}
        
        for i in range(len(system_ids)):
            for j in range(i + 1, len(system_ids)):
                system_a = system_ids[i]
                system_b = system_ids[j]
                
                self.logger.info(f"创建比较对: {system_a} vs {system_b}")
                
                rag_pairs = self.create_system_comparison_pairs(
                    test_data, system_a, system_b
                )
                
                comparisons[(system_a, system_b)] = rag_pairs
        
        return comparisons
    
    def get_system_info_summary(self) -> Dict[str, Dict[str, Any]]:
        """获取所有系统的信息摘要"""
        systems_info = {}
        
        for system_id in self.rag_manager.list_systems():
            systems_info[system_id] = self.rag_manager.get_system_info(system_id)
        
        return systems_info
    
    def save_system_responses(self, 
                            questions: List[str], 
                            output_file: str,
                            system_ids: Optional[List[str]] = None):
        """
        保存所有系统的响应到文件
        
        Args:
            questions: 问题列表
            output_file: 输出文件路径
            system_ids: 要查询的系统ID列表（None时查询所有系统）
        """
        if system_ids is None:
            system_ids = self.rag_manager.list_systems()
        
        responses_data = []
        
        for question in questions:
            question_responses = {'question': question, 'systems': {}}
            
            for system_id in system_ids:
                try:
                    response = self.rag_manager.query_system(system_id, question)
                    question_responses['systems'][system_id] = {
                        'answer': response.answer,
                        'retrieved_chunks': [
                            {
                                'content': chunk.content,
                                'score': chunk.score,
                                'chunk_id': chunk.chunk_id
                            } for chunk in response.retrieved_chunks
                        ],
                        'metadata': response.metadata
                    }
                except Exception as e:
                    self.logger.error(f"查询系统 {system_id} 失败: {e}")
                    question_responses['systems'][system_id] = {'error': str(e)}
            
            responses_data.append(question_responses)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for data in responses_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        self.logger.info(f"系统响应已保存到: {output_file}")
    
    def prepare_multi_system_evaluation(self, 
                                      test_file: str, 
                                      kb_file: str,
                                      system_configs: Optional[List[RAGConfig]] = None,
                                      num_samples: Optional[int] = None) -> Tuple[Dict[Tuple[str, str], List[RAGPair]], List[str]]:
        """
        准备多系统评估的完整流程
        
        Args:
            test_file: 测试数据文件
            kb_file: 知识库文件
            system_configs: 系统配置列表
            num_samples: 限制样本数量
            
        Returns:
            Tuple[Dict, List]: (系统比较对映射, 系统ID列表)
        """
        # 1. 加载数据
        test_data = self.load_test_data(test_file)
        documents = self.load_knowledge_base(kb_file)
        
        if num_samples and len(test_data) > num_samples:
            test_data = test_data[:num_samples]
            self.logger.info(f"限制测试数据为 {num_samples} 条")
        
        # 2. 设置RAG系统
        system_ids = self.setup_rag_systems(documents, system_configs)
        
        # 3. 创建所有两两比较
        comparisons = self.create_all_pairwise_comparisons(test_data, system_ids)
        
        return comparisons, system_ids
    
    def clear_all_systems(self):
        """清理所有系统和存储"""
        for system_id in list(self.rag_manager.list_systems()):
            self.rag_manager.remove_system(system_id)
        self.rag_manager.clear_storage()
        self.logger.info("所有系统已清理")

def load_multi_rag_data(test_file: str = "dice/70条测试数据QA.txt",
                       kb_file: str = "dice/知识源.jsonl",
                       num_samples: int = 10) -> Tuple[Dict[Tuple[str, str], List[RAGPair]], List[str]]:
    """
    快速加载多RAG系统数据的便捷函数
    
    Args:
        test_file: 测试数据文件
        kb_file: 知识库文件
        num_samples: 样本数量限制
        
    Returns:
        Tuple[Dict, List]: (系统比较对映射, 系统ID列表)
    """
    loader = MultiRAGDataLoader()
    return loader.prepare_multi_system_evaluation(
        test_file, kb_file, num_samples=num_samples
    ) 