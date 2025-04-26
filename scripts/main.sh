#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=42gb
#SBATCH --gres=gpu:1
#SBATCH --partition=bigmem
#SBATCH --time=3:00:00
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log

export PYTHONUNBUFFERED=1

module load conda
conda activate CLVQVAE_same
# Base directory and paths
datasetName="eraser-movie"

baseDir=..
srcDir="${baseDir}/src"
dataDir="${baseDir}/data/${datasetName}"  # Path to the dataset

# Dataset and model configuration
input_layer=8
output_layer=12
temperature=1
top_k=5
initialization="kmean++"  # Options: "random", "kmean++", "spherical"
random_vector_seed=42
K=400  # Number of codebook vectors

echo "Layer Configs: initialization: ${initialization}, K:${K}, input_layer:${input_layer}, output_layer:${output_layer}, temperature:${temperature}, top_k:${top_k}, random_vector_seed:${random_vector_seed}"


# Create a descriptive layer name
layer="${input_layer}_${output_layer}_encoder_temp${temperature}_k${top_k}_${initialization}_K${K}_seed${random_vector_seed}"

# Output directories
output_dir="${baseDir}/output/${datasetName}/${layer}"
concept_dir="${baseDir}/concepts/${datasetName}/${layer}"

# Create necessary directories
mkdir -p "${output_dir}"
mkdir -p "${concept_dir}"
mkdir -p "${dataDir}"
# mkdir -p "${dataDir}/embeddings"

# Input and output files
train_input="${dataDir}/train/embedding/layer${input_layer}/movie_train.txt.tok.sent_len-layer${input_layer}_min_5_max_20_del_1000000-dataset.json"
train_output="${dataDir}/train/embedding/layer${output_layer}/movie_train.txt.tok.sent_len-layer${output_layer}_min_5_max_20_del_1000000-dataset.json"
eval_input="${dataDir}/dev/embedding/layer${input_layer}/movie_dev_subset.txt.tok.sent_len-layer${input_layer}_min_0_max_1000000_del_1000000-dataset.json"
eval_output="${dataDir}/dev/embedding/layer${output_layer}/movie_dev_subset.txt.tok.sent_len-layer${output_layer}_min_0_max_1000000_del_1000000-dataset.json"


explanation_file="${dataDir}/dev/IG_attributions/explanation_layer_${input_layer}.txt"
merged_explanation="${output_dir}/merged_explanations.csv"
text_data="${dataDir}/train/movie_train.txt.tok.sent_len"


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
    --initialization ${initialization} \
    --random_vector_seed ${random_vector_seed}

# ===== Step 2: Run inference with the trained model =====
echo "===== Running inference with VQC model ====="
python ${srcDir}/main.py \
    --model_path ${output_dir}/model.pt \
    --input_layer_embedding ${eval_input} \
    --output_layer_embedding ${eval_output} \
    --output_dir ${output_dir} \
    --mode inference

# ===== Step 3: Extract codebook vectors =====
echo "===== Extracting codebook vectors ====="
python ${srcDir}/extract_codebook.py \
    --model_path ${output_dir}/model.pt \
    --output_path ${output_dir}/codebook_vectors.pt

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