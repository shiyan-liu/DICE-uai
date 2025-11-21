#!/usr/bin/env python3
"""
DICE RAGAS 主入口
基于RAGAS框架的系统评分和排名系统
"""

import argparse
import logging
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from src.ragas.ragas_dice_core import RagasDiceEvaluator, RagasDiceConfig


def log_and_print(message):
    """同时输出到控制台和日志文件"""
    print(message)
    logging.info(message)


def setup_logging(output_dir: str):
    """设置日志 - 同时输出到控制台和文件"""
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除现有处理器（避免重复）
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器 - 追加模式
    log_file = Path(output_dir) / "ragas_dice.log"
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 在日志文件中添加运行分隔符
    if log_file.exists() and log_file.stat().st_size > 0:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('\n')
    
    # 记录运行开始
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    separator_msg = f"{'='*80}\n🚀 RAGAS DICE评估开始 - {timestamp}\n{'='*80}"
    logging.info(separator_msg)


def discover_qacg_files(input_dir: str) -> List[str]:
    """自动发现QACG文件"""
    qacg_dir = Path(input_dir)
    if not qacg_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    qacg_files = list(qacg_dir.glob("qacg_*.json"))
    qacg_files.sort()  # 按名称排序
    
    return [str(f) for f in qacg_files]


def print_ranking_summary(ranking_result: Dict[str, Any]):
    """打印排名摘要"""
    ranking = ranking_result["ranking"]
    
    log_and_print(f"\n🏆 RAGAS DICE 系统排名:")
    log_and_print(f"{'='*80}")
    
    for item in ranking:
        rank = item["rank"]
        system_name = item["system_name"]
        score = item["composite_score"]
        
        # 添加奖牌图标
        medal = ""
        if rank == 1:
            medal = "🥇"
        elif rank == 2:
            medal = "🥈"
        elif rank == 3:
            medal = "🥉"
        
        # 检查是否有错误
        if "error" in item:
            log_and_print(f"  {rank}. {system_name}: ❌ 评估失败")
            log_and_print(f"      错误: {item['error']}")
        else:
            std = item.get("composite_std", 0.0)
            success_rate = item.get("success_rate", 1.0)
            valid_q = item.get("valid_questions", 0)
            total_q = item.get("total_questions", 0)
            
            log_and_print(f"  {rank}. {system_name}: {score:.4f} ± {std:.4f} {medal}")
            log_and_print(f"      有效问题: {valid_q}/{total_q} ({success_rate:.1%})")
            
            # 显示各指标得分
            if "metric_averages" in item:
                metrics_str = []
                for metric, avg_score in item["metric_averages"].items():
                    metrics_str.append(f"{metric}={avg_score:.3f}")
                log_and_print(f"      指标详情: {', '.join(metrics_str)}")
    
    # 显示统计信息
    log_and_print(f"\n📊 评估统计:")
    log_and_print(f"  - 总系统数: {ranking_result['total_systems']}")
    log_and_print(f"  - 成功评估: {ranking_result['successful_systems']}")
    log_and_print(f"  - 评估失败: {ranking_result['failed_systems']}")
    
    # 显示配置信息
    config = ranking_result.get("config", {})
    log_and_print(f"\n⚙️ 评估配置:")
    log_and_print(f"  - 模型: {config.get('llm_model', 'N/A')}")
    log_and_print(f"  - 指标: {', '.join(config.get('metrics', []))}")
    log_and_print(f"  - 批大小: {config.get('batch_size', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="DICE RAGAS 评估系统")
    
    # 输入输出参数
    parser.add_argument("--input_dir", default="qacg_output", 
                       help="QACG文件目录 (默认: qacg_output)")
    parser.add_argument("--output_dir", default="ragas_dice_output", 
                       help="输出目录 (默认: ragas_dice_output)")
    
    # 模型配置
    parser.add_argument("--llm_model", default="deepseek-chat", 
                       help="LLM模型名称 (默认: deepseek-chat)")
    parser.add_argument("--embeddings_model", default="BAAI/bge-small-zh-v1.5",
                       help="嵌入模型名称（使用小模型节省内存）")
    
    # RAGAS指标配置
    parser.add_argument("--metrics", nargs="+", 
                       default=["faithfulness", "answer_relevancy", "context_relevance"],
                       help="RAGAS核心评估指标列表（基于原论文）")
    
    # 性能配置
    parser.add_argument("--max_workers", type=int, default=1,
                       help="最大并发worker数量 (默认: 3, 推荐2-4)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="每批处理的问题数量 (默认: 5, 推荐3-10)")
    
    # 特定系统评估
    parser.add_argument("--target_system", 
                       help="只评估指定的系统（系统名称，不含qacg_前缀）")
    
    # 调试和安全配置
    parser.add_argument("--safe_mode", action="store_true",
                       help="安全模式：禁用并发，批大小设为1")
    parser.add_argument("--debug", action="store_true", 
                       help="调试模式：输出更详细的日志")
    
    args = parser.parse_args()
    
    # 处理安全模式
    if args.safe_mode:
        args.max_workers = 1
        args.batch_size = 1
        log_and_print("⚠️ 安全模式已启用：单线程 + 批大小1")
    
    # 设置日志
    setup_logging(args.output_dir)
    
    # 处理调试模式
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        log_and_print("🐛 调试模式已启用：输出详细日志")
    
    try:
        log_and_print("🚀 DICE RAGAS 评估系统")
        log_and_print(f"📁 输入目录: {args.input_dir}")
        log_and_print(f"📂 输出目录: {args.output_dir}")
        log_and_print(f"🤖 LLM模型: {args.llm_model}")
        log_and_print(f"🔍 评估指标: {', '.join(args.metrics)}")
        
        # 发现QACG文件
        log_and_print(f"\n🔍 扫描QACG文件...")
        qacg_files = discover_qacg_files(args.input_dir)
        
        if not qacg_files:
            log_and_print(f"❌ 在目录 {args.input_dir} 中未找到任何qacg_*.json文件")
            return
        
        # 过滤目标系统
        if args.target_system:
            target_files = [f for f in qacg_files if args.target_system in f]
            if not target_files:
                log_and_print(f"❌ 未找到系统 {args.target_system} 的QACG文件")
                return
            qacg_files = target_files
            log_and_print(f"🎯 只评估指定系统: {args.target_system}")
        
        log_and_print(f"📊 找到 {len(qacg_files)} 个QACG文件:")
        for i, f in enumerate(qacg_files, 1):
            system_name = Path(f).stem.replace("qacg_", "")
            log_and_print(f"  {i}. {system_name}")
        
        # 检查API密钥
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            log_and_print("⚠️ 警告: 未设置DEEPSEEK_API_KEY环境变量，将使用默认密钥")
            api_key = "xxxxxxx"
        
        # 创建配置
        config = RagasDiceConfig(
            llm_model=args.llm_model,
            embeddings_model=args.embeddings_model,
            metrics=args.metrics,
            api_key=api_key,
            base_url="https://api.deepseek.com",
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            batch_size=args.batch_size
        )
        
        # 显示性能配置
        log_and_print(f"\n⚙️ 性能配置:")
        log_and_print(f"  - 并发workers: {args.max_workers}")
        log_and_print(f"  - 批处理大小: {args.batch_size}")
        
        if args.max_workers > 1:
            estimated_speedup = min(args.max_workers, 3)  # 实际加速比通常小于线程数
            log_and_print(f"  - 模式: 🚀 并发模式 (预期加速 ~{estimated_speedup:.1f}x)")
            log_and_print(f"  - 警告: ⚠️ 并发可能导致API限制，如遇问题请使用 --safe_mode")
        else:
            log_and_print(f"  - 模式: 🐌 单线程模式 (安全但较慢)")
        
        # 显示并发建议
        if len(qacg_files) == 1:
            total_questions = 0
            try:
                with open(qacg_files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    total_questions = len(data)
            except:
                total_questions = 0
            
            if total_questions > 0:
                estimated_time_single = total_questions * 3  # 假设每题3秒
                estimated_time_concurrent = estimated_time_single / max(args.max_workers, 1)
                log_and_print(f"  - 预估时间: {estimated_time_concurrent/60:.1f} 分钟 ({total_questions} 题)")
                if args.max_workers == 1 and total_questions > 20:
                    log_and_print(f"  - 💡 建议: 题目较多，考虑使用并发 --max_workers 3")
        
        # 创建评估器并开始评估
        evaluator = RagasDiceEvaluator(config)
        
        if len(qacg_files) == 1:
            # 单系统评估
            log_and_print(f"\n🎯 单系统评估模式")
            qacg_file = qacg_files[0]
            system_name = Path(qacg_file).stem.replace("qacg_", "")
            
            result = evaluator.evaluate_single_system(qacg_file, system_name)
            
            log_and_print(f"\n✅ 系统 {system_name} 评估完成:")
            log_and_print(f"  📊 综合得分: {result['composite_score']:.4f} ± {result.get('composite_std', 0.0):.4f}")
            log_and_print(f"  📝 有效问题: {result.get('valid_questions', 0)}/{result['total_questions']}")
            log_and_print(f"  📈 成功率: {result.get('success_rate', 1.0):.1%}")
            
            # 显示各指标得分
            if "metric_averages" in result:
                log_and_print(f"  🔍 各指标得分:")
                for metric, score in result["metric_averages"].items():
                    std = result.get("metric_std", {}).get(metric, 0.0)
                    log_and_print(f"    - {metric}: {score:.4f} ± {std:.4f}")
        
        else:
            # 多系统评估和排名
            log_and_print(f"\n🏆 多系统排名模式")
            ranking_result = evaluator.evaluate_multiple_systems(qacg_files)
            
            # 打印排名摘要
            print_ranking_summary(ranking_result)
        
        # 记录成功完成
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        completion_msg = f"✅ RAGAS DICE评估成功完成 - {timestamp}\n{'='*80}"
        logging.info(completion_msg)
        log_and_print(f"\n💾 详细结果保存在: {args.output_dir}")
        
    except KeyboardInterrupt:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        interrupt_msg = f"⚡ 用户中断执行 - {timestamp}\n{'='*80}"
        logging.info(interrupt_msg)
        log_and_print("\n⚡ 用户中断执行")
        sys.exit(1)
        
    except Exception as e:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        error_msg = f"❌ 执行出错 - {timestamp}"
        logging.error(error_msg)
        log_and_print(f"❌ 执行出错: {e}")
        logging.exception("详细错误信息:")
        logging.info("="*80)
        sys.exit(1)


if __name__ == "__main__":
    main()
