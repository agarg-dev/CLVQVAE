#!/bin/bash
# ============================================================================
# Generate IG Explanation Files
# Converts IG attribution scores to human-readable explanation files
# ============================================================================

# Environment Setup
source $HOME/CLVQVAE/bin/activate


# ============================================================================
# Configuration
# ============================================================================

dataset="agnews"        # Options: "jigsaw", "eraser-movie", "agnews"
model_name="qwen"       # Options: "bert", "roberta", "qwen", "llama"
layer=32                # Layer to generate explanations for
include_last_token=True # Whether to include the last token

scriptDir="../../../../src/generate_explanation_files"
inputDir="../../../../data/${dataset}/dev/${model_name}/IG_attributions"
outDir="../../../../data/${dataset}/dev/${model_name}/IG_attributions"

mkdir -p ${outDir}

# ============================================================================
# Generate Explanation Files
# ============================================================================

inputFile="${inputDir}/IG_explanation_layer_${layer}.csv"
saveFile="${outDir}/explanation_layer_${layer}.txt"

echo ""
echo "===== Generating IG Explanations ====="
echo "Dataset: ${dataset}"
echo "Model: ${model_name}"
echo "Layer: ${layer}"
echo "Input: ${inputFile}"
echo "Output: ${saveFile}"
echo ""

python ${scriptDir}/generate_IG_explanation_salient_words.py \
    ${inputFile} \
    ${saveFile} \
    top-1 \
    --include-last-token

echo ""
echo "===== Explanation Generation Complete ====="
echo "Explanations saved to: ${saveFile}"
echo ""

