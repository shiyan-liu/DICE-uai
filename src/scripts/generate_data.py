"""
QACG tuple generation script.
Generates QACG tuples for 8 RAG systems.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def check_dependencies():
    """Check that required dependencies are installed."""
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
        print("Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall with: pip install -r requirements.txt")
        return False

    return True

def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('qacg_generation.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main entry point."""
    if not check_dependencies():
        return 1

    try:
        from src.generation.qacg import QACGGenerator
    except ImportError as e:
        print(f"Import error: {e}")
        print("Install dependencies: pip install -r requirements.txt")
        return 1

    parser = argparse.ArgumentParser(description='Generate QACG tuples')
    parser.add_argument(
        '--jsonl_path',
        type=str,
        default='dataset/知识源.jsonl',
        help='Knowledge base JSONL file path'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='qacg_output',
        help='Output directory'
    )
    parser.add_argument(
        '--num_questions',
        type=int,
        default=70,
        help='Number of questions per RAG system'
    )
    parser.add_argument(
        '--llm_model',
        type=str,
        default='qwen2.5:7b',
        help='LLM model for question generation (e.g. openai-gpt-3.5-turbo)'
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting QACG tuple generation")
    logger.info(f"Knowledge base: {args.jsonl_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Questions per system: {args.num_questions}")
    logger.info(f"LLM model: {args.llm_model}")

    if not os.path.exists(args.jsonl_path):
        logger.error(f"Knowledge base file not found: {args.jsonl_path}")
        return 1

    if not args.llm_model.startswith('openai'):
        try:
            import requests
            response = requests.get('http://localhost:11434/api/version', timeout=5)
            if response.status_code != 200:
                logger.error("Ollama service not running. Start with: ollama serve")
                return 1
            logger.info(f"Ollama service OK, version: {response.json().get('version', 'unknown')}")
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            logger.error("Start Ollama first: ollama serve")
            return 1

    try:
        logger.info("Initializing QACG generator...")
        generator = QACGGenerator(llm_model=args.llm_model)

        logger.info("Generating QACG tuples...")
        generator.generate_all_qacg(
            jsonl_path=args.jsonl_path,
            output_dir=args.output_dir,
            num_questions=args.num_questions
        )

        logger.info("QACG generation complete")
        logger.info(f"Results saved to: {args.output_dir}")

        if os.path.exists(args.output_dir):
            files = [f for f in os.listdir(args.output_dir) if f.endswith('.json')]
            logger.info(f"Generated {len(files)} files:")
            for file in sorted(files):
                file_path = os.path.join(args.output_dir, file)
                file_size = os.path.getsize(file_path)
                logger.info(f"  - {file} ({file_size} bytes)")

        logger.info("Generation complete. Ready for DICE evaluation.")
        return 0

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
