#!/usr/bin/env python3
"""
DICE主程序
用于执行完整的DICE评估流程
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dice import DICEEvaluator, DICEConfig, create_dice_evaluator

def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dice_evaluation.log', encoding='utf-8')
        ]
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DICE RAG系统评估器")
    
    # 输入参数
    parser.add_argument("--input_dir", type=str, default="qacg_output",
                        help="输入数据目录（包含QACG生成的JSON文件）")
    parser.add_argument("--output_dir", type=str, default="dice_output",
                        help="输出目录")
    
    # 模型参数
    parser.add_argument("--llm_model", type=str, default="qwen2.5:7b",
                        help="用于判决的LLM模型")
    parser.add_argument("--judge_temperature", type=float, default=0.1,
                        help="判决温度")
    
    # 粒度控制
    parser.add_argument("--enable_token", action="store_true", default=True,
                        help="启用token粒度")
    parser.add_argument("--enable_sentence", action="store_true", default=True,
                        help="启用sentence粒度")
    parser.add_argument("--enable_passage", action="store_true", default=True,
                        help="启用passage粒度")
    parser.add_argument("--enable_kg", action="store_true", default=True,
                        help="启用KG粒度")
    
    # 评估控制
    parser.add_argument("--max_questions", type=int, default=None,
                        help="最大评估问题数（用于测试）")
    parser.add_argument("--pairwise_only", action="store_true",
                        help="仅执行成对比较，不生成全局矩阵")
    parser.add_argument("--detailed", action="store_true",
                        help="输出详细信息，包括QACG四元组和四个维度的A、B值")
    
    # 其他参数
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger("DICE.Main")
    
    logger.info("🎯 DICE评估器启动")
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    
    # 创建DICE配置
    config = DICEConfig(
        llm_model=args.llm_model,
        judge_temperature=args.judge_temperature,
        enable_token=args.enable_token,
        enable_sentence=args.enable_sentence,
        enable_passage=args.enable_passage,
        enable_kg=args.enable_kg,
        output_dir=args.output_dir,
        detailed_output=args.detailed  # 添加详细输出选项
    )
    
    # 创建评估器
    evaluator = DICEEvaluator(config)
    
    # 查找输入文件
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return
    
    # 获取所有JSON文件
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        logger.error(f"在{input_dir}中未找到JSON文件")
        return
    
    logger.info(f"找到 {len(json_files)} 个数据文件")
    for file in json_files:
        logger.info(f"  - {file.name}")
    
    try:
        # 执行评估
        if args.pairwise_only and len(json_files) >= 2:
            # 仅执行两个系统的成对比较
            logger.info("执行成对比较模式")
            
            file_a, file_b = json_files[0], json_files[1]
            data_a = evaluator.load_qacg_data(str(file_a))
            data_b = evaluator.load_qacg_data(str(file_b))
            
            # 限制问题数量
            if args.max_questions:
                data_a = data_a[:args.max_questions]
                data_b = data_b[:args.max_questions]
            
            # 执行成对比较
            results = []
            min_len = min(len(data_a), len(data_b))
            
            for i in range(min_len):
                if data_a[i]["question"] == data_b[i]["question"]:
                    logger.info(f"评估问题 {i+1}/{min_len}: {data_a[i]['question'][:50]}...")
                    result = evaluator.evaluate_pair(data_a[i], data_b[i])
                    results.append(result)
                    
                    # 输出中间结果
                    winner = result["fusion_result"]["winner"]
                    elo_delta = result["combined_delta"]
                    logger.info(f"  结果: {winner}, Elo差: {elo_delta:.1f}")
            
            # 汇总结果
            summary = evaluator._summarize_pair_results(results)
            logger.info("🏆 成对比较结果:")
            logger.info(f"  总问题数: {summary['total_questions']}")
            logger.info(f"  {file_a.stem} 胜率: {summary['win_rate_a']:.1%}")
            logger.info(f"  {file_b.stem} 胜率: {summary['win_rate_b']:.1%}")
            logger.info(f"  平局率: {summary['tie_rate']:.1%}")
            logger.info(f"  平均Elo差: {summary['avg_elo_delta']:.2f}")
            
            # 保存结果
            pairwise_result = {
                "system_a": file_a.stem,
                "system_b": file_b.stem,
                "results": results,
                "summary": summary,
                "config": config.__dict__
            }
            
            output_path = Path(args.output_dir) / "pairwise_result.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                # 使用自定义编码器处理特殊类型
                from src.dice.dice_core import NumpyJSONEncoder
                json.dump(pairwise_result, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
            
            logger.info(f"结果已保存到: {output_path}")
        
        else:
            # 执行全局评估
            logger.info("执行全局评估模式")
            file_paths = [str(f) for f in json_files]
            
            # 如果指定了问题数量限制，需要预处理数据
            if args.max_questions:
                logger.info(f"限制评估问题数量为: {args.max_questions}")
                # 这里可以添加数据预处理逻辑
            
            results = evaluator.evaluate_all_pairs(file_paths)
            
            # 输出结果摘要
            logger.info("🏆 全局评估结果:")
            elo_matrix = results["elo_matrix"]
            for i, system in enumerate(elo_matrix["ranking"], 1):
                score = elo_matrix["elo_scores"][system]
                logger.info(f"  {i}. {system}: {score:.1f}")
            
            logger.info(f"评估完成！详细结果已保存到: {args.output_dir}")
    
    except KeyboardInterrupt:
        logger.info("评估被用户中断")
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}", exc_info=True)
        raise

def demo_mode():
    """演示模式：使用示例数据进行快速测试"""
    logger = logging.getLogger("DICE.Demo")
    logger.info("🚀 DICE演示模式")
    
    # 创建简化配置
    config = DICEConfig(
        llm_model="qwen2.5:7b",  # 使用可用的模型版本
        output_dir="dice_demo_output",
        enable_kg=False  # 演示模式关闭KG粒度以加快速度
    )
    
    evaluator = DICEEvaluator(config)
    
    # 创建示例数据
    qa_a = {
        "question": "特朗普将钢铁关税提高到多少？",
        "rag_answer": "特朗普将钢铁关税从25%提高到50%。",
        "context": ["特朗普宣布将钢铁进口关税从25%提高至50%"]
    }
    
    qa_b = {
        "question": "特朗普将钢铁关税提高到多少？", 
        "rag_answer": "钢铁关税提高到50%。",
        "context": ["钢铁关税调整至50%"]
    }
    
    # 执行评估
    result = evaluator.evaluate_pair(qa_a, qa_b)
    
    # 输出结果
    logger.info("演示结果:")
    logger.info(f"  获胜者: {result['fusion_result']['winner']}")
    logger.info(f"  Elo差: {result['combined_delta']:.2f}")
    logger.info(f"  置信度: {result['fusion_result']['confidence']:.2f}")
    
    # 显示各粒度结果
    for granularity, judgment in result["granularity_results"].items():
        logger.info(f"  {granularity}粒度: {judgment['label']} - {judgment['reason']}")

if __name__ == "__main__":
    # 检查是否为演示模式
    if len(sys.argv) == 2 and sys.argv[1] == "--demo":
        setup_logging("INFO")
        demo_mode()
    else:
        main() 