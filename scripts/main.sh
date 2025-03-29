#!/bin/zsh

# Base directory and paths
baseDir="$PWD"
srcDir="${baseDir}/src"
dataDir="${baseDir}/data"

# Dataset and model configuration
datasetName="movie"
input_layer=8
output_layer=12
temperature=1
top_k=5
initialization="spherical"
K=400  # Number of codebook vectors

# Create a descriptive layer name
layer="${input_layer}_${output_layer}_encoder_temp${temperature}_k${top_k}_${initialization}_K${K}"

# Output directories
output_dir="${baseDir}/output/${datasetName}_${layer}"
concept_dir="${baseDir}/concepts/${datasetName}_${layer}"

# Create necessary directories
mkdir -p "${output_dir}"
mkdir -p "${concept_dir}"
mkdir -p "${dataDir}"
mkdir -p "${dataDir}/embeddings"

# Input and output files
train_input="${dataDir}/embeddings/${datasetName}_train_layer${input_layer}.json"
train_output="${dataDir}/embeddings/${datasetName}_train_layer${output_layer}.json"
eval_input="${dataDir}/embeddings/${datasetName}_dev_layer${input_layer}.json"
eval_output="${dataDir}/embeddings/${datasetName}_dev_layer${output_layer}.json"
text_data="${dataDir}/embeddings/${datasetName}_train.txt"
explanation_file="${dataDir}/explanation_layer_${layer}.txt"
merged_explanation="${output_dir}/merged_explanations.csv"
codebook_output="${output_dir}/codebook_vectors.pt"

# ===== Step 1: Train the VQC model =====
echo "===== Training VQC model ====="
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
    --initialization ${initialization}

# ===== Step 2: Run inference with the trained model =====
echo "===== Running inference with VQC model ====="
python ${srcDir}/main.py \
    --model_path ${output_dir}/model.pt \
    --test_fcinput_data ${eval_input} \
    --test_fcoutput_data ${eval_output} \
    --output_dir ${output_dir} \
    --mode inference

# ===== Step 3: Extract codebook vectors =====
echo "===== Extracting codebook vectors ====="
python ${srcDir}/extract_codebook.py \
    --model_path ${output_dir}/model.pt \
    --output_path ${codebook_output}

# ===== Step 4: Generate explanations for salient tokens =====
echo "===== Generating explanations for salient tokens ====="
python ${srcDir}/latent_explanation_for_salients.py \
    --token_map ${output_dir}/token_to_index_map.json \
    --explanation ${explanation_file} \
    --output ${merged_explanation}

# ===== Step 5: Analyze the latent concepts =====
echo "===== Analyzing latent concepts ====="
python ${srcDir}/analyze_latent_concept_movie.py \
    --vector_map ${output_dir}/vector_map.json \
    --input_data ${text_data} \
    --output_dir ${concept_dir}

echo "===== All processing complete ====="