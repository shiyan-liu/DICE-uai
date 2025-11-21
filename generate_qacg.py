"""
QACG四元组生成主脚本
执行此脚本为8个RAG系统生成QACG四元组数据
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 添加项目路径到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """检查必要的依赖是否安装"""
    missing_deps = []
    
    try:
        import llama_index
    except ImportError:
        missing_deps.append("llama-index")
    
    try:
        import sentence_transformers
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    try:
        import chromadb
    except ImportError:
        missing_deps.append("chromadb")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    if missing_deps:
        print("❌ 缺少以下依赖包:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\n请运行以下命令安装:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('qacg_generation.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def main():
    """主函数"""
    # 检查依赖
    if not check_dependencies():
        return 1
    
    # 动态导入（在依赖检查后）
    try:
        from src.qacg_generator import QACGGenerator
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保所有依赖已正确安装:")
        print("pip install -r requirements.txt")
        return 1
    
    parser = argparse.ArgumentParser(description='生成QACG四元组数据')
    parser.add_argument(
        '--jsonl_path', 
        type=str, 
        default='dice/知识源.jsonl',
        help='知识库JSONL文件路径'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='qacg_output',  
        help='输出目录'
    )
    parser.add_argument(
        '--num_questions', 
        type=int, 
        default=70,
        help='每个RAG系统生成的问题数量'
    )
    parser.add_argument(
        '--llm_model', 
        type=str, 
        default='qwen2.5:7b',
        help='用于生成问题的LLM模型 (如果使用OpenAI请设置为 openai-gpt-3.5-turbo)'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("开始生成QACG四元组数据")
    logger.info("=" * 60)
    logger.info(f"知识库路径: {args.jsonl_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"每个系统问题数量: {args.num_questions}")
    logger.info(f"问题生成模型: {args.llm_model}")
    
    # 检查输入文件
    if not os.path.exists(args.jsonl_path):
        logger.error(f"知识库文件不存在: {args.jsonl_path}")
        return 1
    
    # 检查Ollama服务
    if not args.llm_model.startswith('openai'):
        try:
            import requests
            response = requests.get('http://localhost:11434/api/version', timeout=5)
            if response.status_code != 200:
                logger.error("Ollama服务未运行，请先启动: ollama serve")
                return 1
            logger.info(f"✅ Ollama服务正常运行，版本: {response.json().get('version', 'unknown')}")
        except Exception as e:
            logger.error(f"无法连接Ollama服务: {e}")
            logger.error("请确保Ollama服务已启动: ollama serve")
            return 1
    
    try:
        # 创建生成器
        logger.info("初始化QACG生成器...")
        generator = QACGGenerator(llm_model=args.llm_model)
        
        # 生成QACG
        logger.info("开始生成QACG四元组...")
        generator.generate_all_qacg(
            jsonl_path=args.jsonl_path,
            output_dir=args.output_dir,
            num_questions=args.num_questions
        )
        
        logger.info("=" * 60)
        logger.info("QACG四元组生成完成")
        logger.info("=" * 60)
        logger.info(f"结果保存在: {args.output_dir}")
        
        # 列出生成的文件
        if os.path.exists(args.output_dir):
            files = [f for f in os.listdir(args.output_dir) if f.endswith('.json')]
            logger.info(f"生成了 {len(files)} 个文件:")
            for file in sorted(files):
                file_path = os.path.join(args.output_dir, file)
                file_size = os.path.getsize(file_path)
                logger.info(f"  - {file} ({file_size} bytes)")
        
        logger.info("\n🎉 生成成功！可以继续进行DICE评估")
        return 0
        
    except Exception as e:
        logger.error(f"生成过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 