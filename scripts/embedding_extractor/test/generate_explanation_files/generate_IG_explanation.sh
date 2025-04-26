#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --partition=cpu2023
#SBATCH --time=3:00:00
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
module load conda
conda activate CLVQVAE_same


dataset=eraser-movie
scriptDir=../../../../src/generate_explanation_files
inputDir=../../../../data/${dataset}/dev/IG_attributions
outDir=../../../../data/${dataset}/dev/IG_attributions
# outDir=../../../../data/${dataset}/dev/IG_explanation_files_mass_50

mkdir ${outDir}

layer=7
echo ${inputDir}/IG_explanation_layer_${layer}.csv
saveFile=${outDir}/explanation_layer_${layer}.txt
python ${scriptDir}/generate_IG_explanation_salient_words.py ${inputDir}/IG_explanation_layer_${layer}.csv ${saveFile} top-1

layer=11
echo ${inputDir}/IG_explanation_layer_${layer}.csv
saveFile=${outDir}/explanation_layer_${layer}.txt
python ${scriptDir}/generate_IG_explanation_salient_words.py ${inputDir}/IG_explanation_layer_${layer}.csv ${saveFile} top-1