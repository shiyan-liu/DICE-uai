#!/usr/bin/env python3
"""
DICE准确率评估脚本
用于验证DICE系统的可信度，通过与人工标注的"金标准"进行对比
"""

import argparse
import logging
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evaluation.dice_engine import SimplifiedDICEConfig
from src.utils.ragas_impl import RagasConfig
from src.evaluation.validator import UnifiedValidationEvaluator

def main():
    parser = argparse.ArgumentParser(description="多RAG系统准确率验证评估")
    parser.add_argument("--qacg_files", nargs="+", required=True,
                       help="QACG文件路径列表")
    parser.add_argument("--num_samples", type=int, default=200,
                       help="采样评估对数量")
    parser.add_argument("--annotation_file", type=str, 
                       default="dice_human_annotations.json",
                       help="人工标注文件路径")
    parser.add_argument("--output_dir", type=str, default="dice_validation_output",
                       help="输出目录")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--llm_model", type=str, default="deepseek-chat",
                       help="LLM模型")
    parser.add_argument("--tournament_result_file", type=str, 
                       default="dice_simplified_output/tournament_result.json",
                       help="tournament结果文件路径，用于复用已有判断")
    parser.add_argument("--ragas", action="store_true",
                       help="使用RAGAS方法进行评估（默认使用DICE方法）")
    parser.add_argument("--ragas_metrics", nargs="+", 
                       default=["answer_relevancy", "context_precision", "context_recall", "faithfulness", "answer_correctness"],
                       help="RAGAS评估指标列表")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 根据评估方法创建配置和评估器
    if args.ragas:
        # RAGAS配置 - 使用DeepSeek
        ragas_config = RagasConfig(
            llm_model=args.llm_model,
            metrics=args.ragas_metrics,
            api_key=os.environ.get("DEEPSEEK_API_KEY", "xxxxxxx"),  # 使用DeepSeek API
            base_url="https://api.deepseek.com"
        )
        evaluator = UnifiedValidationEvaluator(
            evaluation_method="ragas",
            ragas_config=ragas_config
        )
        evaluation_method = "RAGAS"
    else:
        # DICE配置
        dice_config = SimplifiedDICEConfig(
            llm_model=args.llm_model,
            output_dir=str(output_dir),
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com"
        )
        evaluator = UnifiedValidationEvaluator(
            evaluation_method="dice",
            dice_config=dice_config,
            tournament_result_file=args.tournament_result_file
        )
        evaluation_method = "DICE"
    
    print(f"🔬 {evaluation_method}系统验证评估")
    print(f"📁 QACG文件数量: {len(args.qacg_files)}")
    print(f"📊 采样数量: {args.num_samples}")
    print(f"🔧 评估方法: {evaluation_method}")
    
    try:
        # 步骤1: 采样评估对
        print("\n📋 步骤1: 采样评估对...")
        evaluation_pairs = evaluator.sample_evaluation_pairs(
            args.qacg_files, args.num_samples, args.random_seed
        )
        
        # 保存采样结果
        pairs_file = output_dir / "evaluation_pairs.json"
        with open(pairs_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_pairs, f, ensure_ascii=False, indent=2)
        print(f"✅ 采样完成，保存至: {pairs_file}")
        
        # 步骤1.5: 检查或创建人工标注文件
        print(f"\n📝 步骤1.5: 检查人工标注文件: {args.annotation_file}")
        annotation_file_path = Path(args.annotation_file)
        
        if not annotation_file_path.exists():
            print("⚠️  人工标注文件不存在，创建标注模板...")
            evaluator._create_annotation_template(args.annotation_file)
            print(f"✅ 已创建标注模板: {args.annotation_file}")
            print("💡 标注说明:")
            print("   - 每个pair_id需要3位专家独立投票")
            print("   - 投票选项: 'A wins'、'B wins'、'Tie'")
            print("   - 请根据检索质量和回答质量进行判断")
            print("⚠️  如需生成验证报告，请先完成标注后再运行")
            print("✅ 程序将继续执行DICE评估...\n")
        else:
            print(f"✅ 人工标注文件已存在: {args.annotation_file}")
        
        # 步骤2: 检查或运行DICE评估
        results_file = output_dir / f"{evaluation_method.lower()}_results.json"
        evaluation_results = None
        
        print(f"\n🤖 步骤2: 检查{evaluation_method}评估结果文件...")
        if results_file.exists():
            print(f"✅ 发现已有评估结果文件: {results_file}")
            print("📂 加载已有评估结果，跳过重新评估...")
            
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    evaluation_results = json.load(f)
                print(f"✅ 成功加载 {len(evaluation_results)} 个评估结果")
                
                # 验证评估结果是否与当前采样对匹配
                if len(evaluation_results) != len(evaluation_pairs):
                    print(f"⚠️  评估结果数量({len(evaluation_results)})与采样对数量({len(evaluation_pairs)})不匹配")
                    print("🔄 将重新运行评估...")
                    evaluation_results = None
                else:
                    print("✅ 评估结果数量匹配，将使用已有结果")
            except Exception as e:
                print(f"❌ 加载评估结果失败: {e}")
                print("🔄 将重新运行评估...")
                evaluation_results = None
        
        # 如果没有加载到有效的评估结果，则运行评估
        if evaluation_results is None:
            print(f"\n🤖 运行{evaluation_method}系统评估...")
            evaluation_results = evaluator.run_evaluation(evaluation_pairs)
            
            # 保存评估结果
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            print(f"✅ {evaluation_method}评估完成，保存至: {results_file}")
        
        # 步骤3: 尝试加载人工标注并生成报告
        print(f"\n📊 步骤3: 检查人工标注完成情况...")
        
        try:
            # 尝试加载人工标注
            gold_labels = evaluator.load_human_annotations(args.annotation_file)
            
            if len(gold_labels) == 0:
                print("⚠️  人工标注文件存在但没有有效标注")
                print("💡 请完成标注后重新运行以生成验证报告")
                print(f"✅ DICE评估结果已保存至: {results_file}")
                return
            
            print(f"✅ 成功加载 {len(gold_labels)} 个人工标注")
            
            # 计算一致性指标
            print("\n📊 步骤4: 计算一致性指标...")
            agreement_metrics = evaluator.calculate_agreement(evaluation_results, gold_labels)
            
            # 计算Elo相关性
            print("📊 步骤5: 计算Elo排序相关性...")
            correlation_metrics = evaluator.calculate_elo_correlation(evaluation_results, gold_labels)
            
            # 生成报告
            print("📝 步骤6: 生成验证报告...")
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            report_file = output_dir / f"validation_report_{timestamp}.json"
            evaluator.generate_validation_report(
                agreement_metrics, correlation_metrics, evaluation_results, gold_labels, str(report_file)
            )
            
            print(f"\n✅ 验证报告已保存至: {report_file}")
            
        except Exception as e:
            print(f"⚠️  无法生成验证报告: {e}")
            print("💡 这可能是因为:")
            print("   1. 人工标注文件尚未完成标注")
            print("   2. 标注格式不正确")
            print("   3. expert_votes字段为空")
            print(f"\n✅ DICE评估已完成，结果已保存至: {results_file}")
            print("📝 完成人工标注后，可重新运行脚本生成验证报告")
        
    except Exception as e:
        print(f"❌ 评估过程中出现错误: {e}")
        import traceback
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
