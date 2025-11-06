#!/bin/bash
# ============================================================================
# Faithfulness Evaluation Script
# Measures explanation faithfulness using ablation-based metrics
# ============================================================================

# Environment Setup

source $HOME/CLVQVAE/bin/activate

# ============================================================================
# Configuration
# ============================================================================

# Dataset and Model Selection
datasetName="agnews"    # Options: "jigsaw", "eraser-movie", "agnews"
model_name="llama"      # Options: "bert", "roberta", "mistral-7b", "qwen", "llama"

echo "Dataset: ${datasetName}"
echo "Model: ${model_name}"

# Directory Structure
baseDir=..
srcDir="${baseDir}/src"
dataDir="${baseDir}/data"

# ============================================================================
# VQ-VAE Configuration
# ============================================================================

# Layer Configuration
input_layer=28
output_layer=32
analysis_layer=28

# Codebook Configuration
K=400
initialization="spherical"     # Options: "random", "kmean++", "spherical"
random_vector_seed=42

# Sampling Configuration
temperature=1
top_k=5

# Training Configuration
seed=20
commitment_cost=0.1
perplexity_weight=0

# Alpha Configuration
use_fixed_alpha=False
fixed_alpha=1

# ============================================================================
# Build Experiment Path
# ============================================================================

if [ "${use_fixed_alpha}" = "True" ]; then
    layerConfig="${input_layer}_${output_layer}_encoder_temp${temperature}_k${top_k}_${initialization}_K${K}_seed${seed}_perplexity${perplexity_weight}_fixed_alpha_${fixed_alpha}"
else
    layerConfig="${input_layer}_${output_layer}_encoder_temp${temperature}_k${top_k}_${initialization}_K${K}_seed${random_vector_seed}_perplexity${perplexity_weight}_commitment_cost_${commitment_cost}_nomask_networkseed${seed}"
fi

# ============================================================================
# Setup Model-Specific Paths
# ============================================================================

if [ "${model_name}" == "mistral-7b" ] || [ "${model_name}" == "qwen" ] || [ "${model_name}" == "llama" ]; then
    # Decoder models
    modelPath="${baseDir}/models/${model_name}"
    datasetPath="${dataDir}/${datasetName}/dev/${model_name}/dev.prompt.txt.tok.sent_len"
    embed_file="dev.prompt"
else
    # Encoder models
    modelPath="${baseDir}/models/glue-${datasetName}-${model_name}"
    datasetPath="${dataDir}/${datasetName}/dev/${model_name}/dev.txt.tok.sent_len"
    embed_file="dev"
fi

groundTruth="${dataDir}/${datasetName}/dev/dev.json"
eval_embedding="${dataDir}/${datasetName}/dev/${model_name}/embedding/layer${input_layer}/${embed_file}.txt.tok.sent_len-layer${input_layer}_min_0_max_1000000_del_1000000-dataset.json"

# VQ-VAE outputs
vqcOutputDir="${baseDir}/output/${datasetName}/${model_name}/${layerConfig}"
merged_explanation_file="${vqcOutputDir}/merged_explanations.csv"
codebookPath="${vqcOutputDir}/codebook_vectors.pt"

# Faithfulness results
faithfulnessOutputDir="${baseDir}/${model_name}/faithfulness_results"
mkdir -p ${faithfulnessOutputDir}

# ============================================================================
# Run Faithfulness Evaluation
# ============================================================================

echo ""
echo "===== Running Faithfulness Evaluation ====="
echo "Analysis Layer: ${analysis_layer}"
echo "Config: ${layerConfig}"
echo ""

# Select appropriate evaluation script based on model type
if [ "${model_name}" == "mistral-7b" ] || [ "${model_name}" == "qwen" ] || [ "${model_name}" == "llama" ]; then
    echo "Using decoder model evaluation..."
    python ${srcDir}/evaluation/faithfulness_evaluation_decoder.py \
        --dataset-path ${datasetPath} \
        --merged-explanation-file ${merged_explanation_file} \
        --model-name ${modelPath} \
        --codebook-vectors ${codebookPath} \
        --output-dir ${faithfulnessOutputDir} \
        --layer-idx ${analysis_layer} \
        --ground-truth-file ${groundTruth} \
        --eval-embedding ${eval_embedding} \
        --ablation-method "mean"
else
    echo "Using encoder model evaluation..."
    python ${srcDir}/evaluation/faithfulness_evaluation_encoder.py \
        --dataset-path ${datasetPath} \
        --merged-explanation-file ${merged_explanation_file} \
        --model-name ${modelPath} \
        --codebook-vectors ${codebookPath} \
        --output-dir ${faithfulnessOutputDir} \
        --layer-idx ${analysis_layer} \
        --ground-truth-file ${groundTruth}
fi

echo ""
echo "===== Evaluation Complete ====="
echo "Results saved to: ${faithfulnessOutputDir}"
