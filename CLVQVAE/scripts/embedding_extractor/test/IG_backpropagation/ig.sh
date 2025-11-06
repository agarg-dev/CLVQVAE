#!/bin/bash
# ============================================================================
# Integrated Gradients (IG) Attribution Computation
# Computes feature importance using IG for sequence classification models
# ============================================================================

# Environment Setup
source $HOME/CLVQVAE/bin/activate


# ============================================================================
# Configuration
# ============================================================================

dataset="agnews"        # Options: "jigsaw", "eraser-movie", "agnews"
model_name="qwen"       # Options: "bert", "roberta", "qwen", "llama"
layer=32                # Layer to compute attributions for

scriptDir="../../../../src/IG_backpropagation"
outDir="../../../../data/${dataset}/dev/${model_name}/IG_attributions"

mkdir -p ${outDir}

# ============================================================================
# Setup Model-Specific Parameters
# ============================================================================

# Determine input file and model path based on model type
if [ "${model_name}" == "qwen" ] || [ "${model_name}" == "llama" ]; then
    # Decoder models use prompted input
    inputFile="../../../../data/${dataset}/dev/${model_name}/dev.prompt.txt.tok.sent_len"
    model="../../../../models/${model_name}"
else
    # Encoder models use standard input
    inputFile="../../../../data/${dataset}/dev/${model_name}/dev.txt.tok.sent_len"
    model="../../../../models/glue-${dataset}-${model_name}"
fi

saveFile="${outDir}/IG_explanation_layer_${layer}.csv"

# ============================================================================
# IG Computation
# ============================================================================

echo ""
echo "===== Computing Integrated Gradients ====="
echo "Dataset: ${dataset}"
echo "Model: ${model_name}"
echo "Layer: ${layer}"
echo "Input: ${inputFile}"
echo "Output: ${saveFile}"
echo ""

# Run appropriate IG script based on model type
if [ "${model_name}" == "roberta" ]; then
    echo "Using RoBERTa-specific IG computation..."
    python ${scriptDir}/ig_for_sequence_classification_roberta.py \
        ${inputFile} \
        ${model} \
        ${layer} \
        ${saveFile}
elif [ "${model_name}" == "qwen" ] || [ "${model_name}" == "llama" ]; then
    echo "Using decoder model IG computation..."
    python ${scriptDir}/ig_for_sequence_classification_mistral.py \
        ${inputFile} \
        ${model} \
        ${layer} \
        ${saveFile} \
        ${dataset}
else
    echo "Using BERT-specific IG computation..."
    python ${scriptDir}/ig_for_sequence_classification_bert.py \
        ${inputFile} \
        ${model} \
        ${layer} \
        ${saveFile}
fi

echo ""
echo "===== IG Computation Complete ====="
echo "Attributions saved to: ${saveFile}"
echo ""
