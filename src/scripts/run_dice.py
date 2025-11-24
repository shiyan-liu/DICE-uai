#!/usr/bin/env python3
"""
DICE 精简版主入口
支持三个场景：
A. 八系统锦标赛（瑞士轮，效率更高）
B. 单系统vs虚拟基线
C. 全对全两两配对（完整循环赛）
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# 添加项目路径到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evaluation.dice_engine import SimplifiedDICEEvaluator, SimplifiedDICEConfig


def log_and_print(message):
    """同时输出到控制台和日志文件"""
    print(message)
    logging.info(message)


def setup_logging():
    """设置日志 - 同时输出到控制台和文件"""
    import os
    from datetime import datetime
    
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
    log_file = "dice.log"
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 在日志文件中添加运行分隔符
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        # 如果文件存在且不为空，添加空行分隔符
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('\n')
    
    # 记录运行开始
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    separator_msg = f"{'='*80}\n🚀 DICE评估开始 - {timestamp}\n{'='*80}"
    logging.info(separator_msg)


def scenario_a_tournament(args):
    """场景A: 八系统锦标赛"""
    log_and_print("🏆 DICE精简版 - 场景A: 八系统锦标赛")
    
    # 自动发现QACG文件
    qacg_dir = Path(args.input_dir)
    qacg_files = list(qacg_dir.glob("qacg_*.json"))
    
    if len(qacg_files) < 4:
        log_and_print(f"❌ 需要至少4个QACG文件，找到{len(qacg_files)}个")
        return
    
    # 选择前8个文件
    qacg_files = qacg_files[:8]
    log_and_print(f"📁 使用的QACG文件:")
    for f in qacg_files:
        log_and_print(f"  - {f.name}")
    
    # 配置
    import os
    config = SimplifiedDICEConfig(
        llm_model=args.llm_model,
        max_questions=args.max_questions,
        output_dir=args.output_dir,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        enable_deep_thinking=not args.no_deep_thinking  # 根据命令行参数控制
    )
    
    # 显示并发配置
    log_and_print(f"⚙️ 并发配置: {args.max_workers} workers, 批大小: {args.batch_size}")
    if args.max_workers == 1:
        log_and_print("   模式: 串行处理（兼容模式）")
    else:
        log_and_print(f"   模式: 并发处理（估计加速 {args.max_workers}x）")
    
    # 执行锦标赛
    evaluator = SimplifiedDICEEvaluator(config)
    result = evaluator.scenario_a_tournament([str(f) for f in qacg_files])
    
    # 输出结果摘要（动态Elo配对模式）
    log_and_print("\n🏆 锦标赛结果 (全部8个系统 - 动态Elo配对系统):")
    final_ranking = result["final_ranking"] 
    final_elo_scores = result["final_elo_scores"]
    
    for i, system in enumerate(final_ranking, 1):
        elo_score = final_elo_scores[system]
        # 标注前3强
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
        log_and_print(f"  {i}. {system}: {elo_score:.1f} {medal}")
    
    # 显示比赛概况
    tournament_type = result.get("tournament_type", "swiss_tournament")
    
    if tournament_type == "swiss_tournament":
        swiss_results = result["swiss_results"]
        total_matches = len(swiss_results["match_records"])
        total_rounds = swiss_results.get("total_rounds", 4)
        efficiency = (28 - total_matches) / 28 * 100  # 相比传统联赛的效率提升
        
        log_and_print(f"\n🔄 瑞士轮比赛概况:")
        log_and_print(f"  - 总比赛场次: {total_matches}场 ({total_rounds}轮，每轮4场)")
        log_and_print(f"  - 效率提升: 减少{efficiency:.1f}%的比赛场次")
        log_and_print(f"  - 每队平均对战: {total_matches*2/8:.1f}场")
    else:
        dynamic_results = result["dynamic_results"]
        total_matches = len(dynamic_results["match_records"])
        efficiency = (28 - total_matches) / 28 * 100  # 相比传统联赛的效率提升
        
        log_and_print(f"\n🔄 动态Elo配对概况:")
        log_and_print(f"  - 总比赛场次: {total_matches}场 (传统联赛28场)")
        log_and_print(f"  - 效率提升: 减少{efficiency:.1f}%的比赛场次")
        log_and_print(f"  - 每队平均对战: {total_matches*2/8:.1f}场")
    
    # 统计信息
    total_calls = result["total_llm_calls"]
    log_and_print(f"\n📊 性能统计:")
    log_and_print(f"  - 总LLM调用: {total_calls}次")
    log_and_print(f"  - 估计用时: ~{total_calls/40:.1f}分钟 (8×A100)")
    log_and_print(f"  - 结果保存: {args.output_dir}")


def scenario_c_all_pairs(args):
    """场景C: 全对全两两配对（完整循环赛）"""
    log_and_print("🏆 DICE精简版 - 场景C: 全对全两两配对（完整循环赛）")
    
    # 自动发现QACG文件
    qacg_dir = Path(args.input_dir)
    qacg_files = list(qacg_dir.glob("qacg_*.json"))
    
    if len(qacg_files) < 2:
        log_and_print(f"❌ 需要至少2个QACG文件，找到{len(qacg_files)}个")
        return
    
    # 若超过8个，为与场景A可比，默认取前8个；可通过参数覆盖
    if not getattr(args, "use_all", False) and len(qacg_files) > 8:
        qacg_files = qacg_files[:8]
        log_and_print("ℹ️ 超过8个系统，默认取前8个以便与场景A可比；使用 --use_all 可覆盖")
    
    log_and_print(f"📁 使用的QACG文件:")
    for f in qacg_files:
        log_and_print(f"  - {f.name}")
    
    # 配置
    import os
    config = SimplifiedDICEConfig(
        llm_model=args.llm_model,
        max_questions=args.max_questions,
        output_dir=args.output_dir,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        enable_deep_thinking=not args.no_deep_thinking
    )
    
    log_and_print(f"⚙️ 并发配置: {args.max_workers} workers, 批大小: {args.batch_size}")
    
    evaluator = SimplifiedDICEEvaluator(config)
    result = evaluator.scenario_c_full_round_robin([str(f) for f in qacg_files])
    
    # 输出结果摘要（完整循环赛）
    log_and_print("\n🏆 循环赛结果 (全部系统 - 全对全两两配对):")
    final_ranking = result["final_ranking"] 
    final_elo_scores = result["final_elo_scores"]
    for i, system in enumerate(final_ranking, 1):
        elo_score = final_elo_scores[system]
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
        log_and_print(f"  {i}. {system}: {elo_score:.1f} {medal}")
    
    # 比赛概况
    rr = result.get("round_robin_results", {})
    total_matches = len(rr.get("match_records", []))
    n = len(final_ranking)
    expected = n * (n - 1) // 2
    log_and_print(f"\n🔄 完整循环赛概况:")
    log_and_print(f"  - 总比赛场次: {total_matches}场（理论应为{expected}场）")
    log_and_print(f"  - 每队平均对战: {(total_matches*2)/max(n,1):.1f}场")
    
    # 统计信息
    total_calls = result["total_llm_calls"]
    log_and_print(f"\n📊 性能统计:")
    log_and_print(f"  - 总LLM调用: {total_calls}次")
    log_and_print(f"  - 估计用时: ~{total_calls/40:.1f}分钟 (8×A100)")
    log_and_print(f"  - 结果保存: {args.output_dir}")

def scenario_b_baseline(args):
    """场景B: 单系统vs虚拟基线"""
    log_and_print("🎯 DICE精简版 - 场景B: 单系统vs虚拟基线")
    
    # 检查目标文件
    target_file = Path(args.target_file)
    if not target_file.exists():
        log_and_print(f"❌ 目标文件不存在: {target_file}")
        return
    
    log_and_print(f"📁 目标系统文件: {target_file.name}")
    log_and_print(f"⚙️ 并发配置: {args.max_workers} workers, 批大小: {args.batch_size}")
    
    # 配置
    import os
    config = SimplifiedDICEConfig(
        llm_model=args.llm_model,
        max_questions=args.max_questions,
        output_dir=args.output_dir,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        enable_deep_thinking=not args.no_deep_thinking  # 根据命令行参数控制
    )
    
    # 执行基线对比
    evaluator = SimplifiedDICEEvaluator(config)
    result = evaluator.scenario_b_baseline_comparison(str(target_file), args.target_system, 
                                                     "dice_simplified_output/tournament_report.md")
    
    # 输出结果摘要
    log_and_print(f"\n🎯 {result['target_system']} vs 虚拟基线:")
    summary = result["summary"]
    
    for baseline_name, comparison in summary["comparisons"].items():
        win_rate = comparison["win_rate"]
        conclusion = comparison["conclusion"]
        log_and_print(f"  - vs {baseline_name}: {win_rate:.1%} - {conclusion}")
    
    # 统计信息
    total_calls = result["total_llm_calls"]
    log_and_print(f"\n📊 性能统计:")
    log_and_print(f"  - 总LLM调用: {total_calls}次")
    log_and_print(f"  - 估计用时: ~{total_calls/40:.1f}分钟")
    log_and_print(f"  - 结果保存: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DICE精简版评估系统")
    
    # 通用参数
    parser.add_argument("--llm_model", default="deepseek-chat", help="LLM模型名称")
    parser.add_argument("--max_questions", type=int, default=70, help="最大问题数量")
    parser.add_argument("--output_dir", default="dice_simplified_output", help="输出目录")
    
    # DeepSeek-R1模型配置
    parser.add_argument("--no-deep-thinking", action="store_true", 
                        help="禁用深度思考模式，使用直接输出模式")
    
    # 并发优化参数 - 双GPU优化
    parser.add_argument("--max_workers", type=int, default=1, 
                        help="最大并发worker数量 (双GPU推荐4-6, 单GPU推荐2)")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="每批处理的问题数量 (双GPU推荐8-12, 单GPU推荐4-6)")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="scenario", help="场景选择")
    
    # 场景A: 锦标赛
    parser_a = subparsers.add_parser("tournament", help="场景A: 八系统锦标赛")
    parser_a.add_argument("--input_dir", default="qacg_output", help="QACG文件目录")
    
    # 场景B: 基线对比
    parser_b = subparsers.add_parser("baseline", help="场景B: 单系统vs虚拟基线")
    parser_b.add_argument("target_file", help="目标系统的QACG文件")
    parser_b.add_argument("--target_system", help="目标系统名称（可选）")
    
    # 场景C: 全对全两两配对（完整循环赛）
    parser_c = subparsers.add_parser("allpairs", help="场景C: 全对全两两配对（完整循环赛）")
    parser_c.add_argument("--input_dir", default="qacg_output", help="QACG文件目录")
    parser_c.add_argument("--use_all", action="store_true", help="使用目录下所有系统（默认最多取8个以便对比）")
    
    # 解析参数
    args = parser.parse_args()
    
    if not args.scenario:
        parser.print_help()
        return
    
    # 设置日志
    setup_logging()
    
    # 执行对应场景
    try:
        if args.scenario == "tournament":
            scenario_a_tournament(args)
        elif args.scenario == "baseline":
            scenario_b_baseline(args)
        elif args.scenario == "allpairs":
            scenario_c_all_pairs(args)
        
        # 记录成功完成
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        completion_msg = f"✅ DICE评估成功完成 - {timestamp}\n{'='*80}"
        logging.info(completion_msg)
        
    except KeyboardInterrupt:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        interrupt_msg = f"⚡ 用户中断执行 - {timestamp}\n{'='*80}"
        logging.info(interrupt_msg)
        log_and_print("\n⚡ 用户中断执行")
        sys.exit(1)
    except Exception as e:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        error_msg = f"❌ 执行出错 - {timestamp}"
        logging.error(error_msg)
        log_and_print(f"❌ 执行出错: {e}")
        logging.exception("详细错误信息:")
        logging.info("="*80)
        sys.exit(1)


if __name__ == "__main__":
    main()
