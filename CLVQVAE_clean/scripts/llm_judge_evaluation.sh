#!/bin/bash

# ============================================================================
# LLM Judge Evaluation Script
# Compares different VQ-VAE initialization methods using LLM-based evaluation
# ============================================================================

source $HOME/CLVQVAE/bin/activate

# ============================================================================
# API Configuration
# ============================================================================
# IMPORTANT: Set your API keys as environment variables in your ~/.bashrc
#
# Add to your ~/.bashrc:
#   export OPENAI_API_KEY='your-key-here'
#   export GOOGLE_API_KEY='your-key-here'
#   export ANTHROPIC_API_KEY='your-key-here'
# ============================================================================

if [ -z "$OPENAI_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "WARNING: No API keys found. Please set at least one of:"
    echo "  - OPENAI_API_KEY"
    echo "  - GOOGLE_API_KEY"
    echo "  - ANTHROPIC_API_KEY"
fi

# ============================================================================
# Configuration
# ============================================================================

# Dataset and Model Selection
datasetName="agnews"    # Options: "jigsaw", "eraser-movie", "agnews"
model_name="llama"      # Options: "bert", "roberta", "llama", "qwen"
llm_name="claude"       # Options: "claude", "openai", "gemini", "gemini_flash"

echo "Dataset: ${datasetName}"
echo "Model: ${model_name}"
echo "LLM Judge: ${llm_name}"

# Directory Structure
baseDir=..
srcDir="${baseDir}/src"
dataDir="${baseDir}/data"
TrainDatasetPath="${dataDir}/${datasetName}/train/train.txt"
DevDatasetPath="${dataDir}/${datasetName}/dev/dev.txt"
DevDatasetJson="${dataDir}/${datasetName}/dev/dev.json"

# ============================================================================
# Model-Specific Configuration
# ============================================================================

if [ "$model_name" == "llama" ]; then
    K=600
    INPUT_LAYER=28
    OUTPUT_LAYER=32
elif [ "$model_name" == "qwen" ]; then
    K=400
    INPUT_LAYER=32
    OUTPUT_LAYER=36
else
    # BERT and RoBERTa
    K=400
    INPUT_LAYER=8
    OUTPUT_LAYER=12
fi

# Shared VQ-VAE Parameters
temperature=1
top_k=5
seed=42
perplexity_weight=0
commitment_cost=0.1
lr=0.001

# ============================================================================
# Initialization Methods to Compare
# ============================================================================

CONFIG_1_NAME="random"
INIT_1="random"

CONFIG_2_NAME="kmean++"
INIT_2="kmean++"

CONFIG_3_NAME="spherical"
INIT_3="spherical"

# ============================================================================
# Build Configuration Paths
# ============================================================================

if [ "$model_name" == "llama" ] || [ "$model_name" == "qwen" ]; then
    suffix="_lr${lr}"
else
    suffix=""
fi

# Configuration 1: Random initialization
layerSuffix1="_encoder_temp${temperature}_k${top_k}_${INIT_1}_K${K}_seed20_perplexity${perplexity_weight}_commitment_cost_${commitment_cost}_nomask${suffix}"
layerConfig1="${INPUT_LAYER}_${OUTPUT_LAYER}${layerSuffix1}"

# Configuration 2: K-means++ initialization
layerSuffix2="_encoder_temp${temperature}_k${top_k}_${INIT_2}_K${K}_seed${seed}_perplexity${perplexity_weight}_commitment_cost_${commitment_cost}_nomask${suffix}"
layerConfig2="${INPUT_LAYER}_${OUTPUT_LAYER}${layerSuffix2}"

# Configuration 3: Spherical initialization
layerSuffix3="_encoder_temp${temperature}_k${top_k}_${INIT_3}_K${K}_seed${seed}_perplexity${perplexity_weight}_commitment_cost_${commitment_cost}_nomask${suffix}"
layerConfig3="${INPUT_LAYER}_${OUTPUT_LAYER}${layerSuffix3}"

# Output directories
vqcOutputDir1="${baseDir}/output/${datasetName}/${model_name}/${layerConfig1}"
vqcOutputDir2="${baseDir}/output/${datasetName}/${model_name}/${layerConfig2}"
vqcOutputDir3="${baseDir}/output/${datasetName}/${model_name}/${layerConfig3}"
llmJudgeOutputDir="${baseDir}/llm_judge_results/${datasetName}/${model_name}/initialization_comparison_${llm_name}"

mkdir -p ${llmJudgeOutputDir}

# ============================================================================
# Evaluation Parameters
# ============================================================================

num_samples=100
top_k_words=10
top_k_sentences=5

# ============================================================================
# Run Evaluation
# ============================================================================

echo ""
echo "===== Starting LLM Judge Evaluation ====="
echo "Comparing initialization methods:"
echo "  1. ${CONFIG_1_NAME}: ${layerConfig1}"
echo "  2. ${CONFIG_2_NAME}: ${layerConfig2}"
echo "  3. ${CONFIG_3_NAME}: ${layerConfig3}"
echo ""

python ${srcDir}/evaluation/llm_judge_evaluation_2.py \
    --config-names "${CONFIG_1_NAME}" "${CONFIG_2_NAME}" "${CONFIG_3_NAME}" \
    --config-paths "${vqcOutputDir1}" "${vqcOutputDir2}" "${vqcOutputDir3}" \
    --train-dataset-file "${TrainDatasetPath}" \
    --dev-dataset-file "${DevDatasetPath}" \
    --dev-dataset-json "${DevDatasetJson}" \
    --dataset-name "${datasetName}" \
    --output-dir "${llmJudgeOutputDir}" \
    --num-samples ${num_samples} \
    --top-k-words ${top_k_words} \
    --top-k-sentences ${top_k_sentences} \
    --llm ${llm_name}

echo ""
echo "===== Evaluation Complete ====="
echo "Results saved to: ${llmJudgeOutputDir}"
