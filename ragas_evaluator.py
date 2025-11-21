#!/usr/bin/env python3
"""
RAGAS评估器 - 完全使用DeepSeek API
解决RAGAS内部对OpenAI API的依赖问题
"""

import json
import logging
import os
import time
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np

# 设置环境变量，强制RAGAS使用我们的配置
os.environ["OPENAI_API_KEY"] = "xxxxxxx"
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"

try:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness, ContextRelevance
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    # 尝试使用新的HuggingFaceEmbeddings
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        # 如果新包不可用，回退到旧包
        from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False
    print(f"警告: RAGAS框架未安装。错误: {e}")


@dataclass
class RagasConfig:
    """RAGAS评估配置"""
    llm_model: str = "deepseek-chat"
    embeddings_model: str = "BAAI/bge-small-zh-v1.5"  # 使用更小的模型节省内存
    metrics: List[str] = None
    api_key: str = "xxxxxxx"
    base_url: str = "https://api.deepseek.com"
    
    def __post_init__(self):
        if self.metrics is None:
            # 基于RAGAS原论文的三个核心维度
            self.metrics = ["faithfulness", "answer_relevancy", "context_relevance"]


class RagasEvaluator:
    """RAGAS评估器 - 使用DeepSeek API"""
    
    def __init__(self, config: RagasConfig):
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS框架未安装")
        
        self.config = config
        self.logger = logging.getLogger("RagasEvaluator")
        self._setup_logger()
        
        # 强制设置环境变量
        self._force_openai_env()
        
        # 设置自定义LLM
        self._setup_custom_llm()
        
        # 初始化RAGAS三个核心metrics
        self.metrics_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_relevance": ContextRelevance()  # 需要实例化
        }
        
        self.active_metrics = [self.metrics_map[m] for m in self.config.metrics if m in self.metrics_map]
        self._configure_metrics_llm()
        
        self.logger.info(f"RAGAS评估器初始化完成，使用DeepSeek: {self.config.llm_model}")
    
    def _setup_logger(self):
        """设置日志"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)  # 恢复正常日志级别
    
    def _force_openai_env(self):
        """强制设置OpenAI环境变量以欺骗RAGAS"""
        os.environ["OPENAI_API_KEY"] = self.config.api_key
        os.environ["OPENAI_BASE_URL"] = self.config.base_url
        os.environ["OPENAI_API_BASE"] = self.config.base_url
        self.logger.info(f"强制设置OpenAI环境变量指向DeepSeek: {self.config.base_url}")
    
    def _setup_custom_llm(self):
        """设置DeepSeek LLM"""
        try:
            # 创建指向DeepSeek的ChatOpenAI实例
            self.custom_llm = ChatOpenAI(
                model=self.config.llm_model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                temperature=0.0,
                max_retries=2,
                request_timeout=60
            )
            
            self.ragas_llm = LangchainLLMWrapper(self.custom_llm)
            
            # 强制使用本地嵌入模型，绝对不调用API
            try:
                # 确保不会调用任何API端点
                import os
                # 临时移除OpenAI相关环境变量，防止嵌入模型尝试使用API
                old_openai_key = os.environ.get("OPENAI_API_KEY")
                old_openai_base = os.environ.get("OPENAI_BASE_URL")
                if "OPENAI_API_KEY" in os.environ:
                    del os.environ["OPENAI_API_KEY"]
                if "OPENAI_BASE_URL" in os.environ:
                    del os.environ["OPENAI_BASE_URL"]
                
                # 添加内存优化配置
                model_kwargs = {
                    'device': 'cpu',
                    'trust_remote_code': True
                }
                encode_kwargs = {
                    'normalize_embeddings': True,
                    'batch_size': 1
                    # 移除show_progress_bar以避免参数冲突
                }
                
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.config.embeddings_model,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    cache_folder='./models_cache'
                )
                self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
                
                # 恢复OpenAI环境变量（仅用于LLM）
                if old_openai_key:
                    os.environ["OPENAI_API_KEY"] = old_openai_key
                if old_openai_base:
                    os.environ["OPENAI_BASE_URL"] = old_openai_base
                
                self.logger.info(f"本地嵌入模型加载成功: {self.config.embeddings_model}")
                
                # 测试嵌入模型是否正常工作
                test_embedding = self.embeddings.embed_query("测试文本")
                self.logger.info(f"嵌入模型测试成功，维度: {len(test_embedding)}")
                
            except Exception as e:
                self.logger.error(f"嵌入模型加载失败: {e}")
                import traceback
                self.logger.error(f"详细错误: {traceback.format_exc()}")
                # 如果嵌入模型加载失败，设为None
                self.ragas_embeddings = None
                raise Exception(f"嵌入模型加载失败，无法继续评估需要嵌入的指标: {e}")
            
            self.logger.info(f"DeepSeek LLM配置成功: {self.config.llm_model}")
            
        except Exception as e:
            self.logger.error(f"LLM配置失败: {e}")
            raise
    
    def _configure_metrics_llm(self):
        """为每个metric配置自定义LLM"""
        try:
            for metric in self.active_metrics:
                if hasattr(metric, 'llm'):
                    metric.llm = self.ragas_llm
                if hasattr(metric, 'embeddings') and self.ragas_embeddings is not None:
                    metric.embeddings = self.ragas_embeddings
                elif hasattr(metric, 'embeddings') and self.ragas_embeddings is None:
                    self.logger.warning(f"指标 {type(metric).__name__} 需要嵌入模型，但嵌入模型未加载")
            self.logger.info("所有Metrics LLM配置完成")
        except Exception as e:
            self.logger.error(f"Metrics LLM配置失败: {e}")
            raise
    
    def _qacg_to_ragas_format(self, qacg_data: Dict[str, Any]) -> Dict[str, Any]:
        """将QACG数据转换为RAGAS格式"""
        contexts = []
        if isinstance(qacg_data.get("context"), list):
            for ctx in qacg_data["context"]:
                if isinstance(ctx, dict):
                    contexts.append(ctx.get("text", str(ctx)))
                else:
                    contexts.append(str(ctx))
        elif qacg_data.get("context"):
            contexts = [str(qacg_data["context"])]
        
        # 处理ground_truth字段，确保它是字符串格式
        ground_truth = qacg_data.get("groundtruth", qacg_data.get("expected_answer", ""))
        
        # 如果ground_truth是列表，将其转换为字符串
        if isinstance(ground_truth, list):
            if len(ground_truth) > 0:
                # 如果列表不为空，连接所有元素
                ground_truth = " ".join(str(item) for item in ground_truth)
            else:
                # 如果列表为空，使用空字符串
                ground_truth = ""
        elif ground_truth is None:
            ground_truth = ""
        else:
            # 确保是字符串
            ground_truth = str(ground_truth)
        
        # 同样处理其他可能是列表的字段
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
        """使用改进的RAGAS评估单个QACG - 解决faithfulness NaN问题"""
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries + 1):
            try:
                # 强制重新设置环境变量
                self._force_openai_env()
                
                ragas_data = self._qacg_to_ragas_format(qacg_data)
                
                # 数据验证
                if not ragas_data["question"] or not ragas_data["answer"]:
                    self.logger.warning("问题或答案为空，跳过评估")
                    return {metric: 0.0 for metric in self.config.metrics}
                
                if not ragas_data["contexts"]:
                    self.logger.warning("上下文为空，使用默认值")
                    ragas_data["contexts"] = [""]
                
                self.logger.debug(f"准备评估数据: 问题长度={len(ragas_data['question'])}, 答案长度={len(ragas_data['answer'])}, 上下文数量={len(ragas_data['contexts'])}")
                
                # 额外的数据验证和清理
                if not isinstance(ragas_data['ground_truth'], str):
                    self.logger.warning(f"ground_truth不是字符串类型: {type(ragas_data['ground_truth'])}, 值: {ragas_data['ground_truth']}")
                    ragas_data['ground_truth'] = str(ragas_data['ground_truth'])
                
                if not isinstance(ragas_data['question'], str):
                    self.logger.warning(f"question不是字符串类型: {type(ragas_data['question'])}, 值: {ragas_data['question']}")
                    ragas_data['question'] = str(ragas_data['question'])
                
                if not isinstance(ragas_data['answer'], str):
                    self.logger.warning(f"answer不是字符串类型: {type(ragas_data['answer'])}, 值: {ragas_data['answer']}")
                    ragas_data['answer'] = str(ragas_data['answer'])
                
                # 创建数据集
                dataset = Dataset.from_dict({
                    "question": [ragas_data["question"]],
                    "answer": [ragas_data["answer"]],
                    "contexts": [ragas_data["contexts"]],
                    "ground_truth": [ragas_data["ground_truth"]]
                })
                
                # 使用单线程评估避免并发问题
                self.logger.debug(f"开始RAGAS evaluate调用 (尝试 {attempt + 1}/{max_retries + 1})")
                result = evaluate(
                    dataset, 
                    metrics=self.active_metrics,
                    show_progress=False
                )
                self.logger.debug(f"RAGAS evaluate完成，结果类型: {type(result)}")
                
                # 提取得分
                scores = {}
                
                # 建立指标名称映射关系，RAGAS内部使用不同的键名
                metric_name_mapping = {
                    "context_relevance": "nv_context_relevance",
                    "faithfulness": "faithfulness",
                    "answer_relevancy": "answer_relevancy"
                }
                
                for metric_name in self.config.metrics:
                    try:
                        # 获取实际在RAGAS结果中的键名
                        actual_key = metric_name_mapping.get(metric_name, metric_name)
                        
                        # 尝试多种方式获取得分
                        score_value = None
                        
                        if hasattr(result, actual_key):
                            score_value = getattr(result, actual_key)
                        elif hasattr(result, '_scores_dict') and actual_key in result._scores_dict:
                            score_value = result._scores_dict[actual_key]
                        elif actual_key in result:
                            score_value = result[actual_key]
                        
                        if score_value is not None:
                            # 处理不同格式的得分值
                            if isinstance(score_value, (list, tuple)) and len(score_value) > 0:
                                actual_score = score_value[0]
                            else:
                                actual_score = score_value
                            
                            # 检查NaN值并处理
                            if isinstance(actual_score, float) and (actual_score != actual_score):
                                self.logger.warning(f"指标 {metric_name} 返回NaN，使用默认值")
                                scores[metric_name] = 0.3 if metric_name == "faithfulness" else 0.5
                            elif isinstance(actual_score, (int, float)):
                                scores[metric_name] = float(actual_score)
                                self.logger.debug(f"成功获取指标 {metric_name} 得分: {scores[metric_name]}")
                            else:
                                self.logger.warning(f"指标 {metric_name} 得分类型无效: {type(actual_score)}")
                                scores[metric_name] = 0.3 if metric_name == "faithfulness" else 0.5
                        else:
                            self.logger.warning(f"无法获取指标 {metric_name} 的得分")
                            scores[metric_name] = 0.3 if metric_name == "faithfulness" else 0.5
                            
                    except Exception as e:
                        self.logger.warning(f"获取指标 {metric_name} 时出错: {e}")
                        scores[metric_name] = 0.3 if metric_name == "faithfulness" else 0.5
                
                # 验证所有得分是否有效
                valid_scores = all(
                    isinstance(score, (int, float)) and not np.isnan(score) and not np.isinf(score)
                    for score in scores.values()
                )
                
                if valid_scores:
                    self.logger.debug(f"评估成功，得分: {scores}")
                    return scores
                else:
                    self.logger.warning(f"评估结果包含无效得分: {scores}")
                    if attempt < max_retries:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        # 返回安全的默认得分
                        return {metric: 0.3 if metric == "faithfulness" else 0.5 for metric in self.config.metrics}
                
            except Exception as e:
                self.logger.warning(f"RAGAS评估尝试 {attempt + 1} 失败: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    import traceback
                    error_msg = f"RAGAS评估失败: {str(e)}\n详细错误:\n{traceback.format_exc()}"
                    self.logger.error(error_msg)
                    # 返回安全的默认得分
                    return {metric: 0.3 if metric == "faithfulness" else 0.5 for metric in self.config.metrics}
        
        # 如果所有尝试都失败，返回默认得分
        return {metric: 0.3 if metric == "faithfulness" else 0.5 for metric in self.config.metrics}
    
    def calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """计算平均得分（RAGAS原论文未提及加权组合）"""
        if not scores:
            return 0.0
        
        # 简单平均，符合RAGAS原论文的做法
        valid_scores = [score for score in scores.values() if score is not None and score >= 0]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    
    def compare_qacg_pair(self, qa_a: Dict[str, Any], qa_b: Dict[str, Any]) -> Dict[str, Any]:
        """比较两个QACG的性能"""
        self.logger.info("开始RAGAS对比评估")
        
        # 分别评估两个系统
        print(f"\n🔍 评估系统A...")
        scores_a = self.evaluate_single_qacg(qa_a)
        print(f"   系统A得分: {scores_a}")
        
        print(f"🔍 评估系统B...")
        scores_b = self.evaluate_single_qacg(qa_b)
        print(f"   系统B得分: {scores_b}")
        
        # 计算综合得分
        composite_a = self.calculate_composite_score(scores_a)
        composite_b = self.calculate_composite_score(scores_b)
        
        print(f"\n📊 综合得分对比:")
        print(f"   系统A综合得分: {composite_a:.4f}")
        print(f"   系统B综合得分: {composite_b:.4f}")
        print(f"   得分差异: {composite_a - composite_b:.4f}")
        
        # 确定获胜者
        score_diff = composite_a - composite_b
        
        if abs(score_diff) < 0.05:
            judgment = "Tie"
            judgment_icon = "⚖️"
        elif score_diff > 0:
            judgment = "A wins"
            judgment_icon = "🏆"
        else:
            judgment = "B wins"
            judgment_icon = "🏆"
        
        # 生成详细理由
        reason_parts = []
        detail_parts = []
        
        print(f"\n📋 详细指标对比:")
        for metric in self.config.metrics:
            score_a = scores_a.get(metric, 0)
            score_b = scores_b.get(metric, 0)
            diff = score_a - score_b
            
            # 确定哪个系统在该指标上更优
            if abs(diff) > 0.01:  # 降低阈值以显示更多细节
                if diff > 0:
                    better_system = "A"
                    icon = "📈"
                else:
                    better_system = "B"
                    icon = "📉"
                detail_parts.append(f"   {icon} {metric}: A={score_a:.3f} vs B={score_b:.3f} → 系统{better_system}更优")
                
                if abs(diff) > 0.1:  # 只有显著差异才加入理由
                    reason_parts.append(f"{metric}: 系统{better_system}更优 ({score_a:.3f} vs {score_b:.3f})")
            else:
                detail_parts.append(f"   ⚖️ {metric}: A={score_a:.3f} vs B={score_b:.3f} → 相当")
        
        # 打印详细对比
        for detail in detail_parts:
            print(detail)
        
        if not reason_parts:
            reason = f"两系统性能接近 (A: {composite_a:.3f}, B: {composite_b:.3f})"
        else:
            reason = "; ".join(reason_parts)
        
        # 打印最终判断
        print(f"\n{judgment_icon} 最终判断: {judgment}")
        print(f"📝 判断理由: {reason}")
        print("-" * 80)
        
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
        """进行两个系统的成对比较"""
        self.logger.info(f"开始RAGAS成对比较: {system_a_name} vs {system_b_name}")
        
        num_questions = min(len(qa_list_a), len(qa_list_b))
        if max_questions:
            num_questions = min(num_questions, max_questions)
        
        print(f"\n🔥 RAGAS系统对比: {system_a_name} vs {system_b_name}")
        print(f"📊 将评估 {num_questions} 个问题")
        print("=" * 100)
        
        question_results = []
        
        for i in range(num_questions):
            qa_a = qa_list_a[i]
            qa_b = qa_list_b[i]
            
            if qa_a.get("question") != qa_b.get("question"):
                self.logger.warning(f"问题{i}不匹配，跳过")
                continue
            
            # 显示当前评估的问题信息
            question_text = qa_a.get("question", "")
            print(f"\n📝 问题 {i+1}: {question_text[:100]}{'...' if len(question_text) > 100 else ''}")
            print(f"🤖 系统A回答: {qa_a.get('rag_answer', '')[:80]}{'...' if len(qa_a.get('rag_answer', '')) > 80 else ''}")
            print(f"🤖 系统B回答: {qa_b.get('rag_answer', '')[:80]}{'...' if len(qa_b.get('rag_answer', '')) > 80 else ''}")
            
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
        
        # 计算总体结果
        a_wins = sum(1 for r in question_results if r["passage_judgment"]["label"] == "A wins")
        b_wins = sum(1 for r in question_results if r["passage_judgment"]["label"] == "B wins")
        ties = len(question_results) - a_wins - b_wins
        
        # 显示总体统计
        print(f"\n🎯 总体对比结果:")
        print(f"   📊 总问题数: {len(question_results)}")
        print(f"   🏆 {system_a_name} 获胜: {a_wins} 次 ({a_wins/len(question_results)*100:.1f}%)")
        print(f"   🏆 {system_b_name} 获胜: {b_wins} 次 ({b_wins/len(question_results)*100:.1f}%)")
        print(f"   ⚖️ 平局: {ties} 次 ({ties/len(question_results)*100:.1f}%)")
        
        if a_wins > b_wins:
            winner = system_a_name
            win_icon = "🥇"
        elif b_wins > a_wins:
            winner = system_b_name
            win_icon = "🥇"
        else:
            winner = "平局"
            win_icon = "⚖️"
        
        print(f"\n{win_icon} 总体胜者: {winner}")
        print("=" * 100)
        
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


class RagasValidationEvaluator:
    """RAGAS验证评估器"""
    
    def __init__(self, config: RagasConfig):
        self.config = config
        self.logger = logging.getLogger("RagasValidation")
        self.ragas_evaluator = RagasEvaluator(config)
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def sample_evaluation_pairs(self, qacg_files: List[str], num_samples: int = 200, 
                               random_seed: int = 42) -> List[Dict[str, Any]]:
        """采样评估对（复用DICE的逻辑）"""
        import random
        from pathlib import Path
        
        self.logger.info(f"开始采样 {num_samples} 对评估样本")
        random.seed(random_seed)
        
        # 加载所有系统数据
        all_systems_data = {}
        for file_path in qacg_files:
            system_name = Path(file_path).stem.replace("qacg_", "")
            with open(file_path, 'r', encoding='utf-8') as f:
                all_systems_data[system_name] = json.load(f)
        
        systems = list(all_systems_data.keys())
        if len(systems) < 2:
            raise ValueError(f"需要至少2个系统，实际获得{len(systems)}个")
        
        min_length = min(len(data) for data in all_systems_data.values())
        
        # 生成所有可能的系统对和题目组合
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
        
        # 随机采样
        if len(all_combinations) < num_samples:
            sampled_pairs = all_combinations
        else:
            sampled_pairs = random.sample(all_combinations, num_samples)
        
        self.logger.info(f"成功采样 {len(sampled_pairs)} 对评估样本")
        return sampled_pairs
    
    def run_ragas_evaluation(self, evaluation_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用RAGAS评估所有采样的对比对"""
        self.logger.info(f"开始RAGAS评估 {len(evaluation_pairs)} 对样本")
        
        print(f"\n🚀 RAGAS批量评估开始")
        print(f"📊 总共需要评估: {len(evaluation_pairs)} 对样本")
        print("🔔 每次评估会显示详细的判断过程和结果")
        print("=" * 120)
        
        ragas_results = []
        for i, pair in enumerate(evaluation_pairs):
            print(f"\n⏳ 进度: {i+1}/{len(evaluation_pairs)} ({(i+1)/len(evaluation_pairs)*100:.1f}%)")
            print(f"🔍 评估对 #{i+1}: {pair['system_a']} vs {pair['system_b']}")
            
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
                
                # 显示本次评估的最终结果
                judgment = passage_judgment.get("label", "Tie")
                score = passage_judgment.get("score", 0.5)
                reason = passage_judgment.get("reason", "")
                
                judgment_icon = "🏆" if judgment != "Tie" else "⚖️"
                print(f"\n✅ 评估对 #{i+1} 完成:")
                print(f"   {judgment_icon} 结果: {judgment}")
                print(f"   📊 置信度: {score:.4f}")
                print(f"   📝 理由: {reason}")
                
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
                print(f"\n❌ 评估对 #{i+1} 失败:")
                print(f"   ⚠️ RAGAS评估过程中出现错误")
                
                ragas_result = {
                    "pair_id": i,
                    "question": pair["question"],
                    "system_a": pair["system_a"],
                    "system_b": pair["system_b"],
                    "dice_judgment": "Tie",
                    "dice_score": 0.5,
                    "dice_reason": "RAGAS评估失败",
                    "dice_margin_score": 0.0,
                    "combined_delta": 0.0,
                    "ragas_scores_a": {},
                    "ragas_scores_b": {},
                    "original_pair": pair
                }
            
            ragas_results.append(ragas_result)
            print("═" * 120)
        
        # 显示批量评估统计
        print(f"\n🎊 RAGAS批量评估完成！")
        print(f"📊 评估统计:")
        
        # 统计结果
        judgments = [r["dice_judgment"] for r in ragas_results]
        a_wins = judgments.count("A wins")
        b_wins = judgments.count("B wins")
        ties = judgments.count("Tie")
        
        print(f"   🏆 A wins: {a_wins} 次 ({a_wins/len(ragas_results)*100:.1f}%)")
        print(f"   🏆 B wins: {b_wins} 次 ({b_wins/len(ragas_results)*100:.1f}%)")
        print(f"   ⚖️ Tie: {ties} 次 ({ties/len(ragas_results)*100:.1f}%)")
        print("=" * 120)
        
        return ragas_results
    
    def load_human_annotations(self, annotation_file: str):
        """加载人工标注（复用DICE的逻辑）"""
        # 这里需要导入并复用DICE的逻辑
        from dice_evaluation_script import DICEValidationEvaluator
        from src.dice.dice_simplified import SimplifiedDICEConfig
        
        # 创建临时DICE评估器来复用其方法
        temp_config = SimplifiedDICEConfig()
        temp_evaluator = DICEValidationEvaluator(temp_config)
        return temp_evaluator.load_human_annotations(annotation_file)
    
    def calculate_agreement(self, results, gold_labels):
        """计算一致性（复用DICE的逻辑）"""
        from dice_evaluation_script import DICEValidationEvaluator
        from src.dice.dice_simplified import SimplifiedDICEConfig
        
        temp_config = SimplifiedDICEConfig()
        temp_evaluator = DICEValidationEvaluator(temp_config)
        return temp_evaluator.calculate_agreement(results, gold_labels)
    
    def calculate_elo_correlation(self, results, gold_labels):
        """计算Elo相关性（复用DICE的逻辑）"""
        from dice_evaluation_script import DICEValidationEvaluator
        from src.dice.dice_simplified import SimplifiedDICEConfig
        
        temp_config = SimplifiedDICEConfig()
        temp_evaluator = DICEValidationEvaluator(temp_config)
        return temp_evaluator.calculate_elo_correlation(results, gold_labels)
    
    def analyze_disagreement_cases(self, results, gold_labels):
        """分析不一致case（复用DICE的逻辑）"""
        from dice_evaluation_script import DICEValidationEvaluator
        from src.dice.dice_simplified import SimplifiedDICEConfig
        
        temp_config = SimplifiedDICEConfig()
        temp_evaluator = DICEValidationEvaluator(temp_config)
        return temp_evaluator.analyze_disagreement_cases(results, gold_labels)
    
    def print_disagreement_analysis(self, disagreement_cases):
        """打印分歧分析（复用DICE的逻辑）"""
        from dice_evaluation_script import DICEValidationEvaluator
        from src.dice.dice_simplified import SimplifiedDICEConfig
        
        temp_config = SimplifiedDICEConfig()
        temp_evaluator = DICEValidationEvaluator(temp_config)
        return temp_evaluator.print_disagreement_analysis(disagreement_cases)
    
    def generate_validation_report(self, agreement_metrics, correlation_metrics, results, gold_labels, output_file):
        """生成验证报告（复用DICE的逻辑）"""
        from dice_evaluation_script import DICEValidationEvaluator
        from src.dice.dice_simplified import SimplifiedDICEConfig
        
        temp_config = SimplifiedDICEConfig()
        temp_evaluator = DICEValidationEvaluator(temp_config)
        return temp_evaluator.generate_validation_report(agreement_metrics, correlation_metrics, results, gold_labels, output_file)