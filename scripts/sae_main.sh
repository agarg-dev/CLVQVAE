#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=logs/sae_output_%j.log
#SBATCH --error=logs/sae_error_%j.log

mkdir -p logs

export PYTHONUNBUFFERED=1

module load conda
conda activate CLVQVAE_same

datasetName="eraser-movie"  # Options: "jigsaw", "eraser-movie", "agnews"
model_name="bert"  # Options: "bert", "roberta"
echo "datasetName: ${datasetName}"
echo "model_name: ${model_name}"

baseDir=..
srcDir="${baseDir}/src"
dataDir="${baseDir}/data/${datasetName}"  # Path to the dataset

# SAE configuration
input_layer=8
output_layer=12
hidden_dim=4096  # SAE hidden dimension (should be larger than input dim)
sparsity_weight=0.0001  # L1 sparsity weight
no_cls="True"  # Set to "True" if you want to remove the CLS token from the input data

echo "SAE Configs: hidden_dim:${hidden_dim}, input_layer:${input_layer}, output_layer:${output_layer}, sparsity_weight:${sparsity_weight}, no_cls:${no_cls}"

# Create a descriptive layer name
layer="${input_layer}_${output_layer}_sae_hidden${hidden_dim}_sparsity${sparsity_weight}"

# Output directories
output_dir="${baseDir}/output/${datasetName}/${model_name}/${layer}"
concept_dir="${baseDir}/concepts/${datasetName}/${model_name}/${layer}"

# Create necessary directories
mkdir -p "${output_dir}"
mkdir -p "${concept_dir}"
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
merged_explanation="${output_dir}/sae_merged_explanations.csv"
text_data="${dataDir}/train/${model_name}/train.txt.tok.sent_len"

# ===== Step 1: Train the SAE model =====
echo "===== Training SAE model ====="
python ${srcDir}/sae_main.py \
    --hidden_dim ${hidden_dim} \
    --sparsity_weight ${sparsity_weight} \
    --input_layer_embedding ${train_input} \
    --output_layer_embedding ${train_output} \
    --output_dir ${output_dir} \
    --mode train \
    --input_layer ${input_layer} \
    --output_layer ${output_layer}

# ===== Step 2: Run inference with the trained SAE model =====
echo "===== Running inference with SAE model ====="
python ${srcDir}/sae_main.py \
    --model_path ${output_dir}/sae_model.pt \
    --input_layer_embedding ${eval_input} \
    --output_dir ${output_dir} \
    --mode inference

# ===== Step 3: Extract SAE decoder vectors =====
echo "===== Extracting SAE decoder vectors ====="
python ${srcDir}/extract_sae_vectors.py \
    --model_path ${output_dir}/sae_model.pt \
    --output_path ${output_dir}/sae_decoder_vectors.pt

# ===== Step 4: Generate explanations for salient tokens =====
echo "===== Generating SAE explanations for salient tokens ====="
python ${srcDir}/sae_explanation_script.py \
    --token_map ${output_dir}/token_to_neuron_map.json \
    --explanation ${explanation_file} \
    --output ${merged_explanation}

# # ===== Step 5: Analyze the SAE concepts =====
# echo "===== Analyzing SAE concepts ====="
# python ${srcDir}/analyze_sae_concept.py \
#     --neuron_map ${output_dir}/neuron_map.json \
#     --input_data ${text_data} \
#     --output_dir ${concept_dir}

echo "===== SAE processing complete ====="