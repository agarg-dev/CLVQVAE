#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100
#SBATCH --time=3:00:00
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
module load conda
conda activate CLVQVAE_same
pip install captum

dataset=eraser-movie
scriptDir=../../../../src/IG_backpropagation
inputFile=../../../../data/${dataset}/dev/movie_dev_subset.txt.tok.sent_len
model=../../../../models/glue-${dataset}-roberta

outDir=../../../../data/${dataset}/dev/IG_attributions
mkdir ${outDir}

layer=7
saveFile=${outDir}/IG_explanation_layer_${layer}.csv
python ${scriptDir}/ig_for_sequence_classification.py ${inputFile} ${model} $layer ${saveFile}

layer=11
saveFile=${outDir}/IG_explanation_layer_${layer}.csv
python ${scriptDir}/ig_for_sequence_classification.py ${inputFile} ${model} $layer ${saveFile}

