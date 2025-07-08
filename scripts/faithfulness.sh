#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --partition=cpu2023
#SBATCH --time=0:30:00
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log

mkdir -p logs
export PYTHONUNBUFFERED=1

module load conda
conda activate CLVQVAE_same

datasetName="eraser-movie" # Options: "jigsaw", "eraser-movie", agnews
model_name="roberta"  # Options: "bert", "roberta"
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

# Layer configuration
input_layer=8
output_layer=12
analysis_layer=8
temperature=1
top_k=5
initialization="spherical"  # Options: "random", "kmean++", "spherical", kmedoid++, spherical_kmedoid++
K=400  # Number of codebook vectors
seed=42
use_fixed_alpha=False  # Use fixed alpha value for the model
fixed_alpha=1  # Fixed alpha value for the model
no_cls="True"  # Set to "True" if you want to remove the CLS token from the input data
perplexity_weight=0
commitment_cost=0.1  # Commitment cost for the VQVAE model

if [ "${use_fixed_alpha}" = "True" ]; then
    echo "Layer Configs: initialization: ${initialization}, K:${K}, input_layer:${input_layer}, output_layer:${output_layer}, temperature:${temperature}, top_k:${top_k}, seed:${seed}, perplexity:${perplexity_weight} no_cls:${no_cls}, fixed_alpha:${fixed_alpha}"
    # Create a descriptive layer suffix
    layerSuffix="_encoder_temp${temperature}_k${top_k}_${initialization}_K${K}_seed${seed}_perplexity${perplexity_weight}_no_cls_${no_cls}_fixed_alpha_${fixed_alpha}"
    layerConfig="${input_layer}_${output_layer}${layerSuffix}"
else
    echo "Layer Configs: initialization: ${initialization}, K:${K}, input_layer:${input_layer}, output_layer:${output_layer}, temperature:${temperature}, top_k:${top_k}, seed:${seed}, perplexity:${perplexity_weight} no_cls:${no_cls}, commitment_cost_${commitment_cost}"
    # Create a descriptive layer suffix
    layerSuffix="_encoder_temp${temperature}_k${top_k}_${initialization}_K${K}_seed${seed}_perplexity${perplexity_weight}_commitment_cost_${commitment_cost}"
    layerConfig="${input_layer}_${output_layer}${layerSuffix}"
fi

# echo "Layer Configs: initialization: ${initialization}, K:${K}, input_layer:${input_layer}, output_layer:${output_layer}, temperature:${temperature}, top_k:${top_k}, seed:${seed}, perplexity:${perplexity_weight}"
# # Create a descriptive layer suffix
# layerSuffix="_encoder_temp${temperature}_k${top_k}_${initialization}_K${K}_seed${seed}_perplexity${perplexity_weight}"
# layerConfig="${input_layer}_${output_layer}${layerSuffix}"

# Output directories
vqcOutputDir="${baseDir}/output/${datasetName}/${model_name}/${layerConfig}"
faithfulnessOutputDir="${baseDir}/${model_name}/faithfulness_results"

# Input files for faithfulness evaluation
merged_explanation_file="${vqcOutputDir}/merged_explanations.csv"
codebookPath="${vqcOutputDir}/codebook_vectors.pt"

# Create output directory
mkdir -p ${faithfulnessOutputDir}



# Run faithfulness evaluation
echo "Running faithfulness measurement on model"
echo "Analysis Layer: ${analysis_layer}"
echo "VQC Layer Config: ${layerConfig}"


python ${srcDir}/evaluation/faithfulness_evaluation.py \
  --dataset-path ${datasetPath} \
  --merged-explanation-file ${merged_explanation_file} \
  --model-name ${modelPath} \
  --codebook-vectors ${codebookPath} \
  --output-dir ${faithfulnessOutputDir} \
  --layer-idx ${analysis_layer} \
  --ground-truth-file ${groundTruth}

echo "Results saved to ${faithfulnessOutputDir}"