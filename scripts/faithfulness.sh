#!/bin/bash

# Base directories and paths
baseDir="$PWD"
srcDir="${baseDir}/src"
dataDir="${baseDir}/data"

# Model and dataset configuration
datasetName="movie"
modelPath="$HOME/models/pretrained-transformer"  # Path to your pretrained model
datasetPath="${dataDir}/${datasetName}_dev_subset.txt.tok"
groundTruth="${dataDir}/${datasetName}_dev_subset.json"

# Layer configuration
start_layer=8
end_layer=12
analysis_layer=8
temperature=1
top_k=5
initialization="spherical"
K=400  # Number of codebook vectors

# Create a descriptive layer suffix
layerSuffix="_encoder_temp${temperature}_k${top_k}_${initialization}_K${K}"
layerConfig="${start_layer}_${end_layer}${layerSuffix}"

# Output directories
vqcOutputDir="${baseDir}/output/${datasetName}_${layerConfig}"
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

python ${srcDir}/faithfulness_evaluation.py \
  --dataset-path ${datasetPath} \
  --merged-explanation-file ${merged_explanation_file} \
  --model-name ${modelPath} \
  --codebook-vectors ${codebookPath} \
  --output-dir ${faithfulnessOutputDir} \
  --layer-idx ${analysis_layer} \
  --ground-truth-file ${groundTruth}

echo "Results saved to ${faithfulnessOutputDir}"