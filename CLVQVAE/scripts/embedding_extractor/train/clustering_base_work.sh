#!/bin/bash

# ============================================================================
# Training Data Preprocessing Pipeline
# Tokenizes, filters, and prepares text data for embedding extraction
# ============================================================================

# Environment Setup
source $HOME/CLVQVAE/bin/activate

# ============================================================================
# Configuration
# ============================================================================

dataset="agnews"
model="llama"           # Options: "bert", "roberta", "qwen", "llama"

scriptDir="../../../src/embedding_extractor"
inputPath="../../../data/${dataset}/train"
outputPath="${inputPath}/${model}"
input="train.txt"

mkdir -p ${outputPath}

# ============================================================================
# Setup Model-Specific Parameters
# ============================================================================

sentence_length=300
file_to_process="${inputPath}/${input}"

# Apply prompt template for decoder models
if [ "$model" == "qwen" ] || [ "$model" == "llama" ]; then
    promptFile="../../../data/${dataset}/${dataset}_prompt.txt"
    prompted_input="train.prompt.txt"

    echo "Applying prompt template for ${model}..."
    python ${scriptDir}/apply_prompt.py \
        --input-file ${inputPath}/${input} \
        --output-file ${outputPath}/${prompted_input} \
        --prompt-file ${promptFile}

    input=${prompted_input}
    file_to_process=${outputPath}/${input}
    sentence_length=400
fi

working_file="${input}.tok.sent_len"

# ============================================================================
# Preprocessing Pipeline
# ============================================================================

echo ""
echo "===== Starting Preprocessing Pipeline ====="
echo "Dataset: ${dataset}"
echo "Model: ${model}"
echo "Input: ${file_to_process}"
echo ""

# Step 1: Tokenize with Moses tokenizer
echo "Step 1: Tokenizing..."
perl ${scriptDir}/tokenizer/tokenizer.perl -l en -no-escape < ${file_to_process} > ${outputPath}/${input}.tok

# Step 2: Filter by sentence length
echo "Step 2: Filtering by sentence length (max: ${sentence_length})..."
python ${scriptDir}/sentence_length.py \
    --text-file ${outputPath}/${input}.tok \
    --length ${sentence_length} \
    --output-file ${outputPath}/${input}.tok.sent_len

# Step 3: Modify input format for model compatibility
echo "Step 3: Modifying input format..."
python ${scriptDir}/modify_input.py \
    --text-file ${outputPath}/${input}.tok.sent_len \
    --output-file ${outputPath}/${input}.tok.sent_len.modified

# Step 4: Calculate vocabulary frequency
echo "Step 4: Calculating vocabulary frequency..."
python ${scriptDir}/frequency_count.py \
    --input-file ${outputPath}/${working_file} \
    --output-file ${outputPath}/${working_file}.words_freq

echo ""
echo "===== Pipeline Complete ====="
echo "Output saved to: ${outputPath}"
