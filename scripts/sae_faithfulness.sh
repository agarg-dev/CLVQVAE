#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --partition=cpu2021
#SBATCH --time=0:30:00
#SBATCH --output=logs/sae_output_%j.log
#SBATCH --error=logs/sae_error_%j.log

mkdir -p logs
export PYTHONUNBUFFERED=1

module load conda
conda activate CLVQVAE_same

datasetName="jigsaw" # Options: "jigsaw", "eraser-movie", agnews
model_name="bert"  # Options: "bert", "roberta"
echo "datasetName: ${datasetName}"
echo "model_name: ${model_name}"

# Base directories and paths
baseDir=..
srcDir="${baseDir}/src"
dataDir="${baseDir}/data"

# Model and dataset configuration
modelPath=../models/glue-${datasetName}-${model_name} 
datasetPath="${dataDir}/${datasetName}/dev/${model_name}/dev.txt.tok"
groundTruth="${dataDir}/${datasetName}/dev/dev.json"

# SAE configuration
input_layer=8
output_layer=12
analysis_layer=8
hidden_dim=4096  # SAE hidden dimension
sparsity_weight=0.0001  # L1 sparsity weight
echo "SAE Configs: hidden_dim:${hidden_dim}, input_layer:${input_layer}, output_layer:${output_layer}, sparsity_weight:${sparsity_weight}"

# Create a descriptive layer suffix
layerSuffix="_sae_hidden${hidden_dim}_sparsity${sparsity_weight}"
layerConfig="${input_layer}_${output_layer}${layerSuffix}"

# Output directories
saeOutputDir="${baseDir}/output/${datasetName}/${model_name}/${layerConfig}"
faithfulnessOutputDir="${baseDir}/${model_name}/sae_faithfulness_results"

# Input files for faithfulness evaluation
merged_explanation_file="${saeOutputDir}/sae_merged_explanations.csv"
saeVectorsPath="${saeOutputDir}/sae_decoder_vectors.pt"

# Create output directory
mkdir -p ${faithfulnessOutputDir}

# Run faithfulness evaluation
echo "Running SAE faithfulness measurement on model"
echo "Analysis Layer: ${analysis_layer}"
echo "SAE Layer Config: ${layerConfig}"

python ${srcDir}/evaluation/sae_faithfulness_evaluation.py \
  --dataset-path ${datasetPath} \
  --merged-explanation-file ${merged_explanation_file} \
  --model-name ${modelPath} \
  --sae-decoder-vectors ${saeVectorsPath} \
  --output-dir ${faithfulnessOutputDir} \
  --layer-idx ${analysis_layer} \
  --ground-truth-file ${groundTruth}

echo "Results saved to ${faithfulnessOutputDir}"