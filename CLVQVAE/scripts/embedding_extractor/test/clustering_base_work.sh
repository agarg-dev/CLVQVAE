#!/bin/bash
# ============================================================================
# Test/Dev Data Preprocessing
# Tokenizes, filters, and prepares input data for embedding extraction
# ============================================================================

# Environment Setup
source $HOME/CLVQVAE/bin/activate

# ============================================================================
# Configuration
# ============================================================================

dataset="agnews"        # Options: "jigsaw", "eraser-movie", "agnews"
model_name="qwen"       # Options: "bert", "roberta", "qwen", "llama"

scriptDir="../../../src/embedding_extractor"
inputPath="../../../data/${dataset}/dev"
outputPath="${inputPath}/${model_name}"
input="dev.txt"         # The name of the input file

mkdir -p ${outputPath}

# ============================================================================
# Setup Model-Specific Parameters
# ============================================================================

# Set default values
sentence_length=300
file_to_process="${inputPath}/${input}"

# Conditionally apply prompt if model is a decoder model
if [ "${model_name}" == "qwen" ] || [ "${model_name}" == "llama" ]; then
    promptFile="../../../data/${dataset}/${dataset}_prompt.txt"
    prompted_input="dev.prompt.txt"

    echo ""
    echo "===== Applying Prompt Template ====="
    echo "Model: ${model_name}"
    echo "Prompt file: ${promptFile}"
    echo ""

    # Run the script to apply the prompt template
    python ${scriptDir}/apply_prompt.py \
        --input-file ${inputPath}/${input} \
        --output-file ${outputPath}/${prompted_input} \
        --prompt-file ${promptFile}

    # Update variables for the rest of the pipeline
    input=${prompted_input}
    file_to_process=${outputPath}/${input}
    sentence_length=400
fi

working_file="${input}.tok.sent_len"

# ============================================================================
# Preprocessing Pipeline
# ============================================================================

echo ""
echo "===== Preprocessing Dev Data ====="
echo "Dataset: ${dataset}"
echo "Model: ${model_name}"
echo "Input: ${input}"
echo "Sentence length: ${sentence_length}"
echo ""

# Step 1: Tokenize text with moses tokenizer
echo "Step 1: Tokenizing text..."
perl ${scriptDir}/tokenizer/tokenizer.perl -l en -no-escape < ${file_to_process} > ${outputPath}/${input}.tok

# Step 2: Do sentence length filtering
echo "Step 2: Filtering by sentence length..."
python ${scriptDir}/sentence_length.py \
    --text-file ${outputPath}/${input}.tok \
    --length ${sentence_length} \
    --output-file ${outputPath}/${input}.tok.sent_len

# Step 3: Modify the input file to be compatible with the model
echo "Step 3: Modifying input for model compatibility..."
python ${scriptDir}/modify_input.py \
    --text-file ${outputPath}/${input}.tok.sent_len \
    --output-file ${outputPath}/${input}.tok.sent_len.modified

# Step 4: Calculate vocabulary size and word frequencies
echo "Step 4: Computing word frequencies..."
python ${scriptDir}/frequency_count.py \
    --input-file ${outputPath}/${working_file} \
    --output-file ${outputPath}/${working_file}.words_freq

echo ""
echo "===== Preprocessing Complete ====="
echo "Output saved to: ${outputPath}"
echo ""
