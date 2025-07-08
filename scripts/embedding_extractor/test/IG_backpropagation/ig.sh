#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100
#SBATCH --time=3:00:00
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log

mkdir -p logs
module load conda
conda activate CLVQVAE_same
pip install captum

dataset=agnews # Options: "jigsaw", "eraser-movie", "agnews"
model_name=bert #options: bert, roberta
scriptDir=../../../../src/IG_backpropagation
inputFile=../../../../data/${dataset}/dev/${model_name}/dev.txt.tok.sent_len
model=../../../../models/glue-${dataset}-${model_name}

outDir=../../../../data/${dataset}/dev/${model_name}/IG_attributions
mkdir ${outDir}

layer=12
saveFile=${outDir}/IG_explanation_layer_${layer}.csv

if [ "${model_name}" == "roberta" ]; then
    echo "Using RoBERTa model for IG computation"
    python ${scriptDir}/ig_for_sequence_classification_roberta.py ${inputFile} ${model} $layer ${saveFile}
else
    echo "Using BERT model for IG computation"
    python ${scriptDir}/ig_for_sequence_classification_bert.py ${inputFile} ${model} $layer ${saveFile}
fi

# layer=11
# saveFile=${outDir}/IG_explanation_layer_${layer}.csv
# python ${scriptDir}/ig_for_sequence_classification.py ${inputFile} ${model} $layer ${saveFile}

