#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --partition=cpu2023
#SBATCH --time=1:00:00
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log

export PYTHONUNBUFFERED=1

module load conda
conda activate CLVQVAE_same
# Base directories and paths
baseDir=..
srcDir="${baseDir}/src"
dataDir="${baseDir}/data"

# Model and dataset configuration
datasetName="eraser-movie"
modelPath=../models/glue-${datasetName}-roberta 
datasetPath="${dataDir}/${datasetName}/dev/movie_dev_subset.txt.tok"
groundTruth="${dataDir}/${datasetName}/dev/movie_dev_subset.json"

# Layer configuration
input_layer=8
output_layer=12
analysis_layer=8
temperature=1
top_k=5
initialization="kmean++"  # Options: "random", "kmean++", "spherical"
K=400  # Number of codebook vectors
echo "Layer Configs: initialization: ${initialization}, K:${K}, input_layer:${input_layer}, output_layer:${output_layer}, temperature:${temperature}, top_k:${top_k}"

# Create a descriptive layer suffix
layerSuffix="_encoder_temp${temperature}_k${top_k}_${initialization}_K${K}"
layerConfig="${input_layer}_${output_layer}${layerSuffix}"

# Output directories
vqcOutputDir="${baseDir}/output/${datasetName}/${layerConfig}"
faithfulnessOutputDir="${baseDir}/faithfulness_results"

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