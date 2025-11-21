#!/usr/bin/env python3
"""
RAGAS DICE 核心模块
基于RAGAS框架的系统评分和排名
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# 导入RAGAS评估器
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from ragas_evaluator import RagasEvaluator, RagasConfig


@dataclass
class RagasDiceConfig:
    """RAGAS DICE配置"""
    llm_model: str = "deepseek-chat"
    embeddings_model: str = "BAAI/bge-small-zh-v1.5"  # 使用更小的模型节省内存
    metrics: List[str] = None
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    output_dir: str = "ragas_dice_output"
    max_workers: int = 1
    batch_size: int = 5
    
    def __post_init__(self):
        if self.metrics is None:
            # 基于RAGAS原论文的三个核心维度
            self.metrics = [
                "faithfulness",
                "answer_relevancy",
                "context_relevance"
            ]
        
        # 确保输出目录存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


class RagasDiceEvaluator:
    """RAGAS DICE评估器"""
    
    def __init__(self, config: RagasDiceConfig):
        self.config = config
        self.logger = logging.getLogger("RagasDice")
        self._setup_logger()
        
        # 创建RAGAS配置
        self.ragas_config = RagasConfig(
            llm_model=config.llm_model,
            embeddings_model=config.embeddings_model,
            metrics=config.metrics,
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        # 创建RAGAS评估器
        self.ragas_evaluator = RagasEvaluator(self.ragas_config)
        
        self.logger.info(f"RAGAS DICE评估器初始化完成，使用模型: {config.llm_model}")
    
    def _setup_logger(self):
        """设置日志"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def evaluate_single_system(self, qacg_file: str, system_name: str = None) -> Dict[str, Any]:
        """
        评估单个系统的QACG数据
        
        Args:
            qacg_file: QACG文件路径
            system_name: 系统名称（可选）
            
        Returns:
            评估结果字典
        """
        qacg_path = Path(qacg_file)
        if not qacg_path.exists():
            raise FileNotFoundError(f"QACG文件不存在: {qacg_file}")
        
        # 确定系统名称
        if system_name is None:
            system_name = qacg_path.stem.replace("qacg_", "")
        
        self.logger.info(f"🔍 开始评估系统: {system_name}")
        
        # 加载QACG数据
        with open(qacg_file, 'r', encoding='utf-8') as f:
            qacg_data = json.load(f)
        
        self.logger.info(f"📊 加载了 {len(qacg_data)} 个问答对")
        
        # 批量评估
        all_scores = []
        total_items = len(qacg_data)
        
        self.logger.info(f"⚙️ 评估配置: {self.config.max_workers} 个工作线程, 批大小: {self.config.batch_size}")
        if self.config.max_workers > 1:
            self.logger.info(f"🚀 启用并发模式，预计加速 {self.config.max_workers}x")
        else:
            self.logger.info(f"🔄 使用单线程模式（安全模式）")
        
        # 按批次处理
        for i in range(0, total_items, self.config.batch_size):
            batch = qacg_data[i:i+self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            total_batches = (total_items + self.config.batch_size - 1) // self.config.batch_size
            
            self.logger.info(f"\n{'='*20} 批次 {batch_num}/{total_batches} {'='*20}")
            self.logger.info(f"⏳ 开始处理 {len(batch)} 个问答对 (题目 {i+1}-{min(i+self.config.batch_size, total_items)})")
            
            # 使用新的并发评估方法
            batch_scores = self._evaluate_batch_concurrent(batch, i, system_name, total_items)
            all_scores.extend(batch_scores)
            
            # 批次完成总结
            completed = min(i + self.config.batch_size, total_items)
            progress = completed / total_items * 100
            
            # 计算批次统计
            batch_success = len([s for s in batch_scores if "error" not in s])
            batch_avg_score = sum(s["composite_score"] for s in batch_scores if "error" not in s) / max(batch_success, 1)
            
            self.logger.info(f"✅ 批次 {batch_num} 完成:")
            self.logger.info(f"    📊 成功: {batch_success}/{len(batch)} 题")
            self.logger.info(f"    📈 批次平均分: {batch_avg_score:.4f}")
            self.logger.info(f"    🎯 总进度: {completed}/{total_items} ({progress:.1f}%)")
            self.logger.info(f"{'='*50}")
        
        # 计算统计信息
        system_result = self._calculate_system_statistics(system_name, all_scores)
        
        # 保存详细结果
        detail_file = Path(self.config.output_dir) / f"{system_name}_ragas_details.json"
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump({
                "system_name": system_name,
                "total_questions": len(all_scores),
                "detailed_scores": all_scores,
                "statistics": system_result
            }, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"✅ 系统 {system_name} 评估完成")
        self.logger.info(f"📊 综合得分: {system_result['composite_score']:.4f}")
        self.logger.info(f"💾 详细结果保存至: {detail_file}")
        
        return system_result
    
    def _evaluate_single_question(self, qa_item: Dict[str, Any], question_idx: int, total_questions: int, system_name: str) -> Dict[str, Any]:
        """评估单个问答对"""
        try:
            question = qa_item.get("question", "")[:100]  # 截取前100字符显示
            self.logger.info(f"📝 问题 {question_idx}/{total_questions}: {question}...")
            
            # 检查是否在多线程环境中
            import threading
            thread_id = threading.current_thread().ident
            is_main_thread = thread_id == threading.main_thread().ident
            
            if not is_main_thread:
                # 在工作线程中，创建独立的评估器实例以避免事件循环冲突
                from ragas_evaluator import RagasEvaluator, RagasConfig
                thread_config = RagasConfig(
                    llm_model=self.config.llm_model,
                    embeddings_model=self.config.embeddings_model,
                    metrics=self.config.metrics,
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
                thread_evaluator = RagasEvaluator(thread_config)
                scores = thread_evaluator.evaluate_single_qacg(qa_item)
                composite_score = thread_evaluator.calculate_composite_score(scores)
            else:
                # 在主线程中，使用共享的评估器
                scores = self.ragas_evaluator.evaluate_single_qacg(qa_item)
                composite_score = self.ragas_evaluator.calculate_composite_score(scores)
            
            result = {
                "question": qa_item.get("question", ""),
                "scores": scores,
                "composite_score": composite_score,
                "question_idx": question_idx
            }
            
            # 打印详细的题目评估结果
            self._print_question_result(result, system_name, question_idx, total_questions)
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = f"评估问答对失败: {e}"
            self.logger.error(f"❌ 问题 {question_idx}/{total_questions}: {error_msg}")
            
            # 如果是事件循环错误，给出特殊提示
            if "event loop" in str(e).lower() or "asyncio" in str(e).lower():
                self.logger.error("⚠️ 检测到异步事件循环冲突，建议使用 --safe_mode 或减少 --max_workers")
            
            # 添加默认得分
            result = {
                "question": qa_item.get("question", ""),
                "scores": {metric: 0.0 for metric in self.config.metrics},
                "composite_score": 0.0,
                "error": str(e),
                "question_idx": question_idx
            }
            
            return result
    
    def _print_question_result(self, result: Dict[str, Any], system_name: str, question_idx: int, total_questions: int):
        """打印单个问题的评估结果"""
        question = result["question"][:80] + "..." if len(result["question"]) > 80 else result["question"]
        composite_score = result["composite_score"]
        scores = result["scores"]
        
        # 构建指标得分字符串
        metric_strs = []
        for metric, score in scores.items():
            if score is not None:
                metric_strs.append(f"{metric}={score:.3f}")
            else:
                metric_strs.append(f"{metric}=N/A")
        
        metrics_display = ", ".join(metric_strs)
        
        # 打印到控制台和日志
        result_msg = f"✅ [{system_name}] 问题 {question_idx}/{total_questions} 完成"
        self.logger.info(result_msg)
        self.logger.info(f"    📝 问题: {question}")
        self.logger.info(f"    📊 综合得分: {composite_score:.4f}")
        self.logger.info(f"    🔍 各指标: {metrics_display}")
        
        # 添加分隔线（每10题）
        if question_idx % 10 == 0:
            progress = question_idx / total_questions * 100
            self.logger.info(f"    📈 [{system_name}] 进度: {question_idx}/{total_questions} ({progress:.1f}%)")
            self.logger.info(f"    {'─' * 60}")
    
    def _evaluate_batch_concurrent(self, batch: List[Dict[str, Any]], batch_start_idx: int, system_name: str, total_questions: int) -> List[Dict[str, Any]]:
        """并发评估一批问答对"""
        if self.config.max_workers <= 1:
            # 单线程模式
            batch_scores = []
            for i, qa_item in enumerate(batch):
                question_idx = batch_start_idx + i + 1
                result = self._evaluate_single_question(qa_item, question_idx, total_questions, system_name)
                batch_scores.append(result)
            return batch_scores
        
        # 多线程模式 - 添加错误监控和自动降级
        batch_scores = [None] * len(batch)
        asyncio_errors = 0
        max_asyncio_errors = 3  # 最多允许3个异步错误
        
        try:
            with ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(batch))) as executor:
                # 提交所有任务
                future_to_idx = {}
                for i, qa_item in enumerate(batch):
                    question_idx = batch_start_idx + i + 1
                    future = executor.submit(
                        self._evaluate_single_question, 
                        qa_item, 
                        question_idx, 
                        total_questions, 
                        system_name
                    )
                    future_to_idx[future] = i
                
                # 收集结果
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        batch_scores[idx] = result
                    except Exception as e:
                        # 检查是否是异步相关错误
                        error_str = str(e).lower()
                        if "event loop" in error_str or "asyncio" in error_str or "bound to a different" in error_str:
                            asyncio_errors += 1
                            self.logger.error(f"⚠️ 异步错误 #{asyncio_errors}: {e}")
                            
                            # 如果异步错误过多，立即切换到安全模式
                            if asyncio_errors >= max_asyncio_errors:
                                self.logger.error(f"🚨 异步错误过多 ({asyncio_errors}次)，自动切换到安全模式")
                                self.config.max_workers = 1
                                # 取消剩余的Future并切换到单线程处理
                                for remaining_future in future_to_idx:
                                    if not remaining_future.done():
                                        remaining_future.cancel()
                                
                                # 处理剩余的未完成任务
                                remaining_items = [batch[i] for i, score in enumerate(batch_scores) if score is None]
                                remaining_start = batch_start_idx + len([s for s in batch_scores if s is not None])
                                
                                if remaining_items:
                                    self.logger.info(f"🔄 单线程模式处理剩余 {len(remaining_items)} 个任务...")
                                    for i, qa_item in enumerate(remaining_items):
                                        question_idx = remaining_start + i + 1
                                        safe_result = self._evaluate_single_question(qa_item, question_idx, total_questions, system_name)
                                        batch_scores[remaining_start - batch_start_idx + i] = safe_result
                                
                                break
                        
                        # 创建错误结果
                        question_idx = batch_start_idx + idx + 1
                        batch_scores[idx] = {
                            "question": batch[idx].get("question", ""),
                            "scores": {metric: 0.0 for metric in self.config.metrics},
                            "composite_score": 0.0,
                            "error": str(e),
                            "question_idx": question_idx
                        }
                        
                        self.logger.error(f"并发任务失败: {e}")
        
        except Exception as e:
            self.logger.error(f"🚨 并发执行严重错误，切换到安全模式: {e}")
            self.config.max_workers = 1
            
            # 重新单线程处理整个批次
            batch_scores = []
            for i, qa_item in enumerate(batch):
                question_idx = batch_start_idx + i + 1
                result = self._evaluate_single_question(qa_item, question_idx, total_questions, system_name)
                batch_scores.append(result)
        
        return batch_scores
    
    def _evaluate_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """评估一批问答对（为兼容性保留的方法）"""
        return self._evaluate_batch_concurrent(batch, 0, "unknown", len(batch))
    
    def _calculate_system_statistics(self, system_name: str, all_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算系统统计信息"""
        if not all_scores:
            return {
                "system_name": system_name,
                "composite_score": 0.0,
                "total_questions": 0,
                "metric_averages": {},
                "metric_std": {},
                "valid_questions": 0
            }
        
        # 提取所有有效得分
        valid_scores = [item for item in all_scores if "error" not in item]
        
        # 计算各指标的平均值
        metric_sums = {metric: 0.0 for metric in self.config.metrics}
        metric_counts = {metric: 0 for metric in self.config.metrics}
        composite_scores = []
        
        for item in valid_scores:
            scores = item["scores"]
            composite_scores.append(item["composite_score"])
            
            for metric in self.config.metrics:
                if metric in scores and scores[metric] is not None:
                    metric_sums[metric] += scores[metric]
                    metric_counts[metric] += 1
        
        # 计算平均值
        metric_averages = {}
        metric_std = {}
        
        for metric in self.config.metrics:
            if metric_counts[metric] > 0:
                metric_averages[metric] = metric_sums[metric] / metric_counts[metric]
                
                # 计算标准差
                if len(valid_scores) > 1:
                    values = [item["scores"].get(metric, 0) for item in valid_scores 
                             if metric in item["scores"] and item["scores"][metric] is not None]
                    if values:
                        metric_std[metric] = float(np.std(values))
                    else:
                        metric_std[metric] = 0.0
                else:
                    metric_std[metric] = 0.0
            else:
                metric_averages[metric] = 0.0
                metric_std[metric] = 0.0
        
        # 计算综合得分
        if composite_scores:
            overall_composite = sum(composite_scores) / len(composite_scores)
            composite_std = float(np.std(composite_scores)) if len(composite_scores) > 1 else 0.0
        else:
            overall_composite = 0.0
            composite_std = 0.0
        
        return {
            "system_name": system_name,
            "composite_score": overall_composite,
            "composite_std": composite_std,
            "total_questions": len(all_scores),
            "valid_questions": len(valid_scores),
            "metric_averages": metric_averages,
            "metric_std": metric_std,
            "success_rate": len(valid_scores) / len(all_scores) if all_scores else 0.0
        }
    
    def evaluate_multiple_systems(self, qacg_files: List[str]) -> Dict[str, Any]:
        """
        评估多个系统并生成排名
        
        Args:
            qacg_files: QACG文件路径列表
            
        Returns:
            评估和排名结果
        """
        self.logger.info(f"🚀 开始RAGAS DICE多系统评估")
        self.logger.info(f"📁 待评估系统数量: {len(qacg_files)}")
        
        # 显示系统列表
        for i, qacg_file in enumerate(qacg_files, 1):
            system_name = Path(qacg_file).stem.replace("qacg_", "")
            self.logger.info(f"  {i}. {system_name}")
        
        # 评估每个系统
        system_results = []
        
        for i, qacg_file in enumerate(qacg_files, 1):
            system_name = Path(qacg_file).stem.replace("qacg_", "")
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"🔍 评估系统 {i}/{len(qacg_files)}: {system_name}")
            self.logger.info(f"{'='*80}")
            
            try:
                result = self.evaluate_single_system(qacg_file, system_name)
                system_results.append(result)
                
                self.logger.info(f"✅ 系统 {system_name} 评估完成")
                self.logger.info(f"📊 得分: {result['composite_score']:.4f}")
                
            except Exception as e:
                self.logger.error(f"❌ 系统 {system_name} 评估失败: {e}")
                # 添加默认结果
                system_results.append({
                    "system_name": system_name,
                    "composite_score": 0.0,
                    "total_questions": 0,
                    "error": str(e)
                })
        
        # 生成排名
        ranking_result = self._generate_ranking(system_results)
        
        # 保存完整结果
        output_file = Path(self.config.output_dir) / "ragas_dice_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ranking_result, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 完整结果保存至: {output_file}")
        
        return ranking_result
    
    def _generate_ranking(self, system_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成系统排名"""
        # 按综合得分排序
        valid_results = [r for r in system_results if "error" not in r]
        error_results = [r for r in system_results if "error" in r]
        
        # 排序
        ranked_systems = sorted(valid_results, key=lambda x: x["composite_score"], reverse=True)
        
        # 生成排名信息
        ranking = []
        for i, result in enumerate(ranked_systems, 1):
            ranking.append({
                "rank": i,
                "system_name": result["system_name"],
                "composite_score": result["composite_score"],
                "composite_std": result.get("composite_std", 0.0),
                "total_questions": result["total_questions"],
                "valid_questions": result.get("valid_questions", result["total_questions"]),
                "success_rate": result.get("success_rate", 1.0),
                "metric_averages": result.get("metric_averages", {})
            })
        
        # 添加失败的系统
        for result in error_results:
            ranking.append({
                "rank": len(ranked_systems) + 1,
                "system_name": result["system_name"],
                "composite_score": 0.0,
                "error": result["error"]
            })
        
        return {
            "evaluation_type": "RAGAS_DICE",
            "total_systems": len(system_results),
            "successful_systems": len(valid_results),
            "failed_systems": len(error_results),
            "ranking": ranking,
            "config": {
                "llm_model": self.config.llm_model,
                "metrics": self.config.metrics,
                "batch_size": self.config.batch_size
            }
        }
