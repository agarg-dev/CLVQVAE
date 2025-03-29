# CLVQ-VAE: Cross-Layer Discrete Concept Discovery for Interpreting Language Models

This repository contains the implementation of CLVQ-VAE (Cross-Layer Vector Quantized Variational Autoencoder), a framework designed to discover discrete concepts between neural network layers in language models.

## Overview

CLVQ-VAE maps representations from a lower layer to a higher layer through a discrete bottleneck, enabling the identification of discrete concepts that characterize transformations in a model's processing hierarchy. Our approach combines:

- An adaptive residual encoder that preserves input while learning minimal transformations
- A vector quantizer with EMA updates and controlled stochastic sampling
- A transformer decoder that reconstructs higher-layer representations

## Key Features

- **Temperature-based sampling** from top-k nearest codebook vectors during quantization
- **Scaled-spherical k-means++** initialization for codebook vectors
- **Cross-layer concept discovery** through reconstruction objectives between layers

## Repository Structure

```
CLVQVAE/
├── scripts/
│   ├── main.sh                # Main script for running the full pipeline
│   └── faithfulness.sh        # Script for faithfulness evaluation
├── src/
│   ├── models/                # Model architecture implementations
│   ├── main.py                # Main training and inference code
│   ├── extract_codebook.py    # Utility for extracting codebook vectors
│   ├── latent_explanation_for_salients.py # Generate explanations
│   ├── analyze_latent_concept_movie.py    # Concept analysis
│   └── faithfulness_evaluation.py         # Faithfulness evaluation
└── data/                      # Directory for datasets and embeddings
```

## Usage

### Setup

1. Clone the repository
2. Install the required dependencies
3. Prepare your data directory with embeddings

### Prepare your dataset:

The dataset information can be downloaded from this [Dropbox link](https://www.dropbox.com/scl/fo/hre4iczg0dpz2vs5p2vcx/AGg_naici_1m2Vt3waLoTOg?rlkey=myzedcpmjywm7h8ksjogkgg7x&st=w4kojso4&dl=0)

### Running the Full Pipeline

```bash
# Run the main pipeline
bash scripts/main.sh

# Run faithfulness evaluation
bash scripts/faithfulness.sh
```

## Configuration

The main script accepts several parameters that can be modified:

- `datasetName`: Name of the dataset
- `input_layer`/`output_layer`: Layer indices for cross-layer analysis
- `temperature`: Temperature parameter for sampling (default: 1.0)
- `top_k`: Number of top vectors to sample from (default: 5)
- `initialization`: Codebook initialization method (default: "spherical")
- `K`: Number of codebook vectors (default: 400)
