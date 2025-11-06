#!/bin/bash

# ============================================================================
# CLVQVAE Training and Inference Script
# ============================================================================

source $HOME/CLVQVAE/bin/activate

# ============================================================================
# Configuration
# ============================================================================

# Dataset and Model Selection
datasetName="eraser-movie"  # Options: "eraser-movie", "jigsaw", "agnews"
model_name="bert"           # Options: "bert", "roberta", "qwen", "llama"

echo "Dataset: ${datasetName}"
echo "Model: ${model_name}"

# Directory Structure
baseDir=..
srcDir="${baseDir}/src"
dataDir="${baseDir}/data/${datasetName}"

# ============================================================================
# Model-Specific Hyperparameters
# ============================================================================

if [ "${model_name}" = "qwen" ] || [ "${model_name}" = "llama" ]; then
    lr=1e-3
    batch_size=64
    if [ "${model_name}" = "llama" ]; then
        input_layer=28
        output_layer=32
    elif [ "${model_name}" = "qwen" ]; then
        input_layer=32
        output_layer=32
    fi
else
    # BERT and RoBERTa
    lr=5e-3
    batch_size=128
    input_layer=8
    output_layer=8
fi

# ============================================================================
# VQ-VAE Hyperparameters
# ============================================================================

# Codebook Configuration
K=400                          # Number of codebook vectors (options: 50, 100, 400, 800, 1200)
initialization="spherical"     # Options: "random", "kmean++", "spherical"
random_vector_seed=42          # Codebook initialization seed (options: 0, 42, 10, 20, 30)

# Sampling Configuration
temperature=1                  # Options: 0.5, 1, 2, 3
top_k=5                        # Options: 1, 5, 10, 50, 100

# Training Configuration
seed=20                        # Network weight initialization seed (options: 0, 42, 10, 20, 30)
num_epochs=100
commitment_cost=0.1
perplexity_weight=0

# Regularization
encoder_weight_decay=1e-4
decoder_weight_decay=1e-4

# Learning Rate Scheduler
scheduler_type="plateau"       # Options: "plateau", "cosine_warmup"

# Alpha Configuration
use_fixed_alpha=False
fixed_alpha=1

# ============================================================================
# Generate Experiment Name
# ============================================================================

if [ "${use_fixed_alpha}" = "True" ]; then
    echo "Config: init=${initialization}, K=${K}, layers=${input_layer}-${output_layer}, temp=${temperature}, top_k=${top_k}, seed=${random_vector_seed}, commit=${commitment_cost}, perplexity=${perplexity_weight}, fixed_alpha=${fixed_alpha}"
    layer="${input_layer}_${output_layer}_encoder_temp${temperature}_k${top_k}_${initialization}_K${K}_seed${random_vector_seed}_perplexity${perplexity_weight}_fixed_alpha_${fixed_alpha}"
else
    echo "Config: init=${initialization}, K=${K}, layers=${input_layer}-${output_layer}, temp=${temperature}, top_k=${top_k}, seed=${random_vector_seed}, commit=${commitment_cost}, perplexity=${perplexity_weight}, lr=${lr}, net_seed=${seed}"
    layer="${input_layer}_${output_layer}_encoder_temp${temperature}_k${top_k}_${initialization}_K${K}_seed${random_vector_seed}_perplexity${perplexity_weight}_commitment_cost_${commitment_cost}_nomask_networkseed${seed}"
fi

# ============================================================================
# Setup Directories
# ============================================================================

output_dir="${baseDir}/output/${datasetName}/${model_name}/${layer}"
concept_dir="${baseDir}/concepts/${datasetName}/${model_name}/${layer}"
codebook_dir="${baseDir}/codebooks/${datasetName}/${model_name}"

mkdir -p "${output_dir}"
mkdir -p "${concept_dir}"
mkdir -p "${codebook_dir}"
mkdir -p "${dataDir}"

# ============================================================================
# Setup File Paths
# ============================================================================

# Determine file naming convention based on model type
if [ "${model_name}" = "mistral-7b" ] || [ "${model_name}" = "qwen" ] || [ "${model_name}" = "llama" ]; then
    train_embed_file_name=train.prompt
    dev_embed_file_name=dev.prompt
else
    train_embed_file_name=train
    dev_embed_file_name=dev
fi

# Input embedding files
train_input="${dataDir}/train/${model_name}/embedding/layer${input_layer}/${train_embed_file_name}.txt.tok.sent_len-layer${input_layer}_min_5_max_20_del_1000000-dataset.json"
train_output="${dataDir}/train/${model_name}/embedding/layer${output_layer}/${train_embed_file_name}.txt.tok.sent_len-layer${output_layer}_min_5_max_20_del_1000000-dataset.json"
eval_input="${dataDir}/dev/${model_name}/embedding/layer${input_layer}/${dev_embed_file_name}.txt.tok.sent_len-layer${input_layer}_min_0_max_1000000_del_1000000-dataset.json"
eval_output="${dataDir}/dev/${model_name}/embedding/layer${output_layer}/${dev_embed_file_name}.txt.tok.sent_len-layer${output_layer}_min_0_max_1000000_del_1000000-dataset.json"

# Explanation files
explanation_file="${dataDir}/dev/${model_name}/IG_attributions/explanation_layer_${input_layer}.txt"
merged_explanation="${output_dir}/merged_explanations.csv"
text_data="${dataDir}/train/${model_name}/${train_embed_file_name}.txt.tok.sent_len"

# ============================================================================
# Training Pipeline
# ============================================================================

# Step 1: Train the VQC model
echo ""
echo "===== Step 1: Training VQC Model ====="
python ${srcDir}/main.py \
    --num_embeddings ${K} \
    --input_layer_embedding ${train_input} \
    --output_layer_embedding ${train_output} \
    --output_dir ${output_dir} \
    --mode train \
    --use_ema \
    --use_sampling \
    --top_k ${top_k} \
    --temperature ${temperature} \
    --use_adaptive_encoder \
    --initialization ${initialization} \
    --random_vector_seed ${random_vector_seed} \
    --codebook_dir ${codebook_dir} \
    --perplexity_weight ${perplexity_weight} \
    --input_layer ${input_layer} \
    --output_layer ${output_layer} \
    --commitment_cost ${commitment_cost} \
    --learning_rate ${lr} \
    --encoder_weight_decay ${encoder_weight_decay} \
    --decoder_weight_decay ${decoder_weight_decay} \
    --model_name ${model_name} \
    --batch_size ${batch_size} \
    --num_epochs ${num_epochs} \
    --scheduler_type ${scheduler_type} \
    --seed ${seed}

# Step 2: Run inference
echo ""
echo "===== Step 2: Running Inference ====="
python ${srcDir}/main.py \
    --model_path ${output_dir}/model.pt \
    --input_layer_embedding ${eval_input} \
    --output_layer_embedding ${eval_output} \
    --output_dir ${output_dir} \
    --mode inference

# Step 3: Extract codebook vectors
echo ""
echo "===== Step 3: Extracting Codebook Vectors ====="
python ${srcDir}/extract_codebook.py \
    --model_path ${output_dir}/model.pt \
    --output_path ${output_dir}/codebook_vectors.pt

# Step 4: Generate explanations
echo ""
echo "===== Step 4: Generating Explanations ====="
python ${srcDir}/latent_explanation_for_salients.py \
    --token_map ${output_dir}/token_to_index_map.json \
    --explanation ${explanation_file} \
    --output ${merged_explanation}

# Step 5: Analyze latent concepts (optional)
# echo ""
# echo "===== Step 5: Analyzing Latent Concepts ====="
python ${srcDir}/analyze_latent_concept.py \
    --vector_map ${output_dir}/vector_map.json \
    --input_data ${text_data} \
    --output_dir ${concept_dir}

echo ""
echo "===== Pipeline Complete ====="
echo "Results saved to: ${output_dir}"
