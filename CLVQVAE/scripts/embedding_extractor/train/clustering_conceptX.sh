#!/bin/bash
# ============================================================================
# Training Data Embedding Extraction
# Extracts layer-wise activations and creates filtered datasets
# ============================================================================

# Environment Setup
source $HOME/CLVQVAE/bin/activate

# ============================================================================
# Configuration
# ============================================================================

dataset="agnews"        # Options: "jigsaw", "eraser-movie", "agnews"
model_name="qwen"       # Options: "bert", "roberta", "qwen", "llama"
layer="36"              # Layer to extract from the model

scriptDir="../../../src/embedding_extractor"
modelPath="../../../models/${model_name}"
dataPath="../../../data/${dataset}/train/${model_name}"

# ============================================================================
# Setup Model-Specific Parameters
# ============================================================================

# Determine input file and sentence tag based on model type
if [ "${model_name}" == "qwen" ] || [ "${model_name}" == "llama" ]; then
    # Decoder models use prompted input, no special token
    input="train.prompt.txt"
    sentence_tag=""
elif [ "${model_name}" == "roberta" ]; then
    # RoBERTa uses <s> token
    input="train.txt"
    sentence_tag="<s>"
else
    # BERT uses [CLS] token
    input="train.txt"
    sentence_tag="[CLS]"
fi

working_file="${input}.tok.sent_len"
outputDir="${dataPath}/embedding/layer${layer}"

mkdir -p ${outputDir}

# ============================================================================
# Embedding Extraction Pipeline
# ============================================================================

echo ""
echo "===== Extracting Embeddings ====="
echo "Dataset: ${dataset}"
echo "Model: ${model_name}"
echo "Layer: ${layer}"
echo "Input: ${working_file}"
echo ""

# Step 1: Extract layer-wise activations
echo "Step 1: Extracting layer ${layer} activations..."
python ${scriptDir}/neurox_extraction.py \
    --model_desc ${modelPath} \
    --input_corpus ${dataPath}/${working_file} \
    --output_file ${outputDir}/${working_file}.activations.json \
    --output_type json \
    --decompose_layers \
    --filter_layers ${layer} \
    --batch_size 32 \
    --num_workers $SLURM_CPUS_PER_TASK

# Step 2: Create dataset with word and sentence indexes
echo "Step 2: Creating dataset file..."
python ${scriptDir}/create_data_single_layer.py \
    --text-file ${dataPath}/${working_file}.modified \
    --activation-file ${outputDir}/${working_file}.activations-layer${layer}.json \
    --output-prefix ${outputDir}/${working_file}-layer${layer} \
    --sentence-tag ${sentence_tag}

# Step 3: Apply frequency filtering
echo "Step 3: Applying frequency filtering..."
minfreq=5
maxfreq=20
delfreq=1000000

python ${scriptDir}/frequency_filter_data.py \
    --input-file ${outputDir}/${working_file}-layer${layer}-dataset.json \
    --frequency-file ${dataPath}/${working_file}.words_freq \
    --sentence-file ${outputDir}/${working_file}-layer${layer}-sentences.json \
    --minimum-frequency $minfreq \
    --maximum-frequency $maxfreq \
    --delete-frequency ${delfreq} \
    --output-file ${outputDir}/${working_file}-layer${layer} \
    --sentence-tag ${sentence_tag}

echo ""
echo "===== Extraction Complete ====="
echo "Output saved to: ${outputDir}"
