#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log

mkdir -p logs

export PYTHONUNBUFFERED=1

module load conda
conda activate CLVQVAE_same

datasetName="eraser-movie"  # Options: "jigsaw", "eraser-movie", "agnews"
model_name="roberta"  # Options: "bert", "roberta"
echo "datasetName: ${datasetName}"
echo "model_name: ${model_name}"

baseDir=..
srcDir="${baseDir}/src"
dataDir="${baseDir}/data/${datasetName}"  # Path to the dataset

# Dataset and model configuration
input_layer=8
output_layer=12
temperature=1
top_k=5
initialization="spherical"  # Options: "random", "kmean++", "spherical",  kmedoid++, spherical_kmedoid++
random_vector_seed=42  # Options: 0, 42, 10, 20, 30
K=400  # Number of codebook vectors
perplexity_weight=0
use_fixed_alpha=False  # Use fixed alpha value for the model
fixed_alpha=1  # Fixed alpha value for the model
no_cls="True"  # Set to "True" if you want to remove the CLS token from the input data
commitment_cost=0.1  # Commitment cost for the VQVAE model

if [ "${use_fixed_alpha}" = "True" ]; then
    echo "Layer Configs: initialization: ${initialization}, K:${K}, input_layer:${input_layer}, output_layer:${output_layer}, temperature:${temperature}, top_k:${top_k}, random_vector_seed:${random_vector_seed}, commitment_cost:${commitment_cost}, factor:0.5 perplexity:${perplexity_weight}, no_cls:${no_cls}, fixed_alpha:${fixed_alpha}"
# Create a descriptive layer name
    layer="${input_layer}_${output_layer}_encoder_temp${temperature}_k${top_k}_${initialization}_K${K}_seed${random_vector_seed}_perplexity${perplexity_weight}_no_cls_${no_cls}_fixed_alpha_${fixed_alpha}"
else
    echo "Layer Configs: initialization: ${initialization}, K:${K}, input_layer:${input_layer}, output_layer:${output_layer}, temperature:${temperature}, top_k:${top_k}, random_vector_seed:${random_vector_seed}, commitment_cost:${commitment_cost}, factor:0.5 perplexity:${perplexity_weight}, no_cls:${no_cls}, adaptive_alpha"
# Create a descriptive layer name
    layer="${input_layer}_${output_layer}_encoder_temp${temperature}_k${top_k}_${initialization}_K${K}_seed${random_vector_seed}_perplexity${perplexity_weight}_commitment_cost_${commitment_cost}"
fi

# Output directories
output_dir="${baseDir}/output/${datasetName}/${model_name}/${layer}"
concept_dir="${baseDir}/concepts/${datasetName}/${model_name}/${layer}"
codebook_dir="${baseDir}/codebooks/${datasetName}/${model_name}"

# Create necessary directories
mkdir -p "${output_dir}"
mkdir -p "${concept_dir}"
mkdir -p "${codebook_dir}"
mkdir -p "${dataDir}"


# Input and output files
if [ "${no_cls}" = "True" ]; then
    echo "Removing CLS token from input data"
    train_input="${dataDir}/train/${model_name}/embedding/layer${input_layer}_no_s/train.txt.tok.sent_len-layer${input_layer}_min_5_max_20_del_1000000-dataset.json"
    train_output="${dataDir}/train/${model_name}/embedding/layer${output_layer}_no_s/train.txt.tok.sent_len-layer${output_layer}_min_5_max_20_del_1000000-dataset.json"
    eval_input="${dataDir}/dev/${model_name}/embedding/layer${input_layer}_no_s/dev.txt.tok.sent_len-layer${input_layer}_min_0_max_1000000_del_1000000-dataset.json"
    eval_output="${dataDir}/dev/${model_name}/embedding/layer${output_layer}_no_s/dev.txt.tok.sent_len-layer${output_layer}_min_0_max_1000000_del_1000000-dataset.json"
else
    echo "Keeping CLS token in input data"
    train_input="${dataDir}/train/${model_name}/embedding/layer${input_layer}/train.txt.tok.sent_len-layer${input_layer}_min_5_max_20_del_1000000-dataset.json"
    train_output="${dataDir}/train/${model_name}/embedding/layer${output_layer}/train.txt.tok.sent_len-layer${output_layer}_min_5_max_20_del_1000000-dataset.json"
    eval_input="${dataDir}/dev/${model_name}/embedding/layer${input_layer}/dev.txt.tok.sent_len-layer${input_layer}_min_0_max_1000000_del_1000000-dataset.json"
    eval_output="${dataDir}/dev/${model_name}/embedding/layer${output_layer}/dev.txt.tok.sent_len-layer${output_layer}_min_0_max_1000000_del_1000000-dataset.json"
fi


explanation_file="${dataDir}/dev/${model_name}/IG_attributions/explanation_layer_${input_layer}.txt"
merged_explanation="${output_dir}/merged_explanations.csv"
text_data="${dataDir}/train/${model_name}/train.txt.tok.sent_len"



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
    --random_vector_seed ${random_vector_seed} \
    --codebook_dir ${codebook_dir} \
    --perplexity_weight ${perplexity_weight} \
    --input_layer ${input_layer} \
    --output_layer ${output_layer} \
    --commitment_cost ${commitment_cost}

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

# # ===== Step 5: Analyze the latent concepts =====
# echo "===== Analyzing latent concepts ====="
# python ${srcDir}/analyze_latent_concept_movie.py \
#     --vector_map ${output_dir}/vector_map.json \
#     --input_data ${text_data} \
#     --output_dir ${concept_dir}

# echo "===== All processing complete ====="