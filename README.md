# DICE

![](docs/dice_framework.png)

Code for evaluating RAG systems with interpretable reasoning and probabilistic scoring.

This repository implements **DICE** (Discrete Interpretable Comparative Evaluation), a framework for responsible, explainable, and confidence-aware RAG evaluation.

## Features

- Evidence-coupled reasoning for transparent decision-making
- Probabilistic {A, B, Tie} scoring for confidence-aware judgments
- Efficient large-scale evaluation with Swiss-system tournament
- Reproducible benchmarks for multi-system comparisons

## Installation

1. Clone the repository:
   ```bash
   git clone xxx
   cd DICE
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Deploy Ollama for Local LLM:

   This project uses **Qwen2.5** models for evaluation. To run with local LLM instead of API calls:

   - Install Ollama: Download and install from [ollama.ai](https://ollama.ai)

   - Pull the required model:
     ```bash
     ollama pull qwen2.5:7b  # Main evaluation model (7B version)
     ollama pull qwen2.5:0.5b  # Optional: smaller variant for testing
     ```

   - Start Ollama service (in a separate terminal):
     ```bash
     ollama serve
     ```
     Service will be available at `http://localhost:11434`

   - Verify model installation:
     ```bash
     ollama list  # Should show qwen2.5:7b and optionally qwen2.5-0.5b
     ```

## Usage

Note: If using Ollama, ensure `ollama serve` is running in a separate terminal before executing any evaluation scripts.

### 1. Prepare Your Data

To evaluate your own RAG systems, save your generation results in JSON files with the prefix `qacg_` (e.g., `qacg_system_a.json`).

Required JSON format for each system:

```json
[
  {
    "question": "What is the capital of France?",
    "rag_answer": "The capital is Paris.",
    "context": ["Paris is the capital and most populous city of France."],
    "groundtruth": "Paris"
  },
  ...
]
```

Place these files in a directory (e.g., `my_systems/`).

### 2. Run Evaluation

#### Scenario A: Compare Multiple Systems (Tournament)

If you have 4 or more systems to compare, use the tournament mode. This runs a Swiss-system tournament to efficiently rank your systems.

```bash
python src/scripts/run_dice.py --scenario tournament --input_dir my_systems/
```

#### Scenario B: Evaluate a Single System (Baseline Comparison)

If you have a single system (or fewer than 4) and want to evaluate its absolute quality, compare it against built-in baselines (Good, Medium, Bad).

```bash
python src/scripts/run_dice.py --scenario baseline --target_file my_systems/qacg_my_system.json --target_system MySystemName
```

#### Scenario C: Compare All Pairs (Round Robin)

For a comprehensive comparison of all systems against each other (N*N), use the allpairs mode. Note that this requires more API calls.

```bash
python src/scripts/run_dice.py --scenario allpairs --input_dir my_systems/
```

To reproduce the original experiment with Qwen2.5 locally, ensure Ollama is serving and the models are loaded. The framework will automatically use the local Ollama service when configured.

### 3. Generate Synthetic Data (Optional)

If you don't have RAG outputs yet, you can generate benchmark data using our provided knowledge base in `dataset/`.

```bash
python src/scripts/generate_data.py --num_questions 20 --output_dir my_systems
```

### 4. Run RAGAS Evaluation (Optional)

To run standard RAGAS metrics alongside DICE:

```bash
python src/scripts/run_ragas.py --input_dir my_systems/ --output_dir ragas_results
```
