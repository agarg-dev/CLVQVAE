# CLVQ-VAE: Cross-Layer Discrete Concept Discovery for Interpreting Language Models

This repository contains the implementation of CLVQ-VAE (Cross-Layer Vector Quantized Variational Autoencoder), a framework for discovering interpretable concepts in language models by mapping representations across layers through vector quantization.

## Overview

CLVQ-VAE addresses limitations in current interpretability methods by:
- Operating in discrete space with clear conceptual boundaries (unlike continuous SAEs)
- Analyzing cross-layer transformations to capture how concepts evolve
- Collapsing duplicated features in transformer residual streams into compact, interpretable concept vectors

## Repository Structure

```
├── data/                          # Dataset directory
│   ├── agnews/                    # AG News dataset
│   ├── eraser-movie/              # ERASER-Movie dataset
│   └── jigsaw/                    # Jigsaw Toxicity dataset
├── scripts/                       # Execution scripts
│   ├── main.sh                    # Main training pipeline
│   ├── faithfulness.sh            # Faithfulness evaluation
│   ├── llm_judge_evaluation.sh    # LLM-as-a-judge evaluation
│   └── analyze_agreement.sh       # Inter-judge agreement analysis
└── src/                           # Source code
    ├── models/                    # Model architectures
    ├── evaluation/                # Evaluation scripts
    ├── IG_backpropagation/        # Integrated Gradients for saliency
    └── embedding_extractor/       # Embedding extraction utilities
```

## Installation

```bash
# Install dependencies
pip install torch transformers neurox
pip install scikit-learn numpy pandas matplotlib seaborn wordcloud
```

## Dataset Preparation

Download the prepared datasets from this Dropbox link (https://www.dropbox.com/scl/fo/hre4iczg0dpz2vs5p2vcx/AGg_naici_1m2Vt3waLoTOg?rlkey=myzedcpmjywm7h8ksjogkgg7x&e=1&st=w4kojso4&dl=0) and place them in the `data/` directory.

## Usage

### Training

Run the main training pipeline:

```bash
bash scripts/main.sh
```

### Configuration

Key parameters in `main.sh`:
* `datasetName`: Dataset name ("eraser", "jigsaw", "agnews")
* `input_layer`: Lower layer index (e.g., 8 for BERT/RoBERTa)
* `output_layer`: Higher layer index (e.g., 12 for BERT/RoBERTa)
* `temperature`: Sampling temperature (default: 1.0)
* `top_k`: Number of top codebook vectors (default: 5)
* `initialization`: Codebook initialization ("spherical", "kmeans", or "random")
* `K`: Codebook size (default: 400)

### Recommended Layer Pairs

* BERT/RoBERTa (12 layers): 8→12
* LLaMA-2-7B (32 layers): 28→32
* Qwen2.5-3B (36 layers): 32→36

## Evaluation

Faithfulness evaluation:

```bash
bash scripts/faithfulness.sh
```

LLM-as-a-judge evaluation:

```bash
bash scripts/llm_judge_evaluation.sh
```

Inter-judge agreement analysis:

```bash
bash scripts/analyze_agreement.sh
```

## Supported Models

* RoBERTa-base (fine-tuned)
* BERT-base (fine-tuned)
* LLaMA-2-7B (zero-shot)
* Qwen2.5-3B-Instruct (zero-shot)

## Supported Datasets

* ERASER-Movie: Sentiment classification (binary)
* Jigsaw Toxicity: Toxicity detection (binary)
* AG News: News topic classification (4 classes)


## Citation

```bibtex
@inproceedings{garg2026clvqvae,
  title     = {Cross-Layer Discrete Concept Discovery for Interpreting Language Models},
  author    = {Garg, Ankur and Yu, Xuemin and Sajjad, Hassan and Ebrahimi Kahou, Samira},
  year      = {2025}
}
```

