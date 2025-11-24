# DICE

![DICE Framework](docs/dice_framework.png)

Code for evaluating RAG systems with interpretable reasoning and probabilistic scoring.

This repository implements **DICE** (Discrete Interpretable Comparative Evaluation), a framework introduced in [ResponsibleFM @ NeurIPS 2025](https://openreview.net/forum?id=RNz4AfOfh3) for responsible, explainable, and confidence-aware RAG evaluation.

## Features

- Evidence-coupled reasoning for transparent decision-making  
- Probabilistic $\{A, B, Tie\}$ scoring for confidence-aware judgments  
- Efficient large-scale evaluation with Swiss-system tournament  
- Reproducible benchmarks for multi-system comparisons  

## Usage

1. Clone the repository:  
   ```bash
   git clone https://github.com/shiyan-liu/DICE.git
   cd DICE
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run evaluation**:

   **Generate QACG data** (Question, Answer, Context, Groundtruth):
   ```bash
   python src/scripts/generate_data.py --num_questions 20
   ```

   **Run DICE Evaluation** (Tournament Mode):
   ```bash
   python src/scripts/run_dice.py --scenario tournament
   ```

   **Run DICE Evaluation** (Baseline Comparison):
   ```bash
   python src/scripts/run_dice.py --scenario baseline --target_system bge-large-zh_chunk_256_qwen2.5
   ```

   **Run RAGAS Evaluation**:
   ```bash
   python src/scripts/run_ragas.py --input_dir qacg_output --output_dir ragas_dice_output
   ```

   **Validate Evaluation Accuracy** (requires human annotation):
   ```bash
   python src/scripts/validate_dice.py --qacg_files qacg_output/qacg_system_A.json qacg_output/qacg_system_B.json
   ```

## Reference

If you use DICE in your work, please cite:

```bibtex
@inproceedings{liu2025dice,
  title={DICE: Discrete Interpretable Comparison Evaluation with Probabilistic Scoring for Retrieval-Augmented Generation},
  author={Liu, Shiyan and Ma, Jian},
  booktitle={Socially Responsible and Trustworthy Foundation Models at NeurIPS 2025}
}
```
