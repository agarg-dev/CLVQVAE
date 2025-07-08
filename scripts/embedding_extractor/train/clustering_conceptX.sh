#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --partition=cpu2022
#SBATCH --time=5:00:00
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log

mkdir -p logs

module load conda
conda activate CLVQVAE_same

dataset=agnews #options: jigsaw, eraser-movie, agnews
model_name=bert #options: bert, roberta
no_cls="True"
# analyze latent concepts of layer 12
layer=12

# absolute_path=/home/ankur.garg1/CLVQVAE/data/${dataset}/train
scriptDir=../../../src/embedding_extractor
DataPath=../../../data/${dataset}/train/${model_name} # path to a sentence file
input=train.txt #name of the sentence file


# put model name or path to a finetuned model for "xxx"
# model=../../../models/glue-${dataset}-roberta
model=../../../models/glue-${dataset}-${model_name}
mkdir ${DataPath}/embedding

if [ "${no_cls}" = "True" ]; then
    echo "No CLS token in input data"
    layer_name=${layer}_no_s
else 
    echo "CLS token in input data"
    layer_name=${layer}
fi

outputDir=${DataPath}/embedding/layer${layer_name} #do not change this
mkdir ${outputDir}

working_file=$input.tok.sent_len #do not change this

echo "Preprocessing data for layer ${layer_name} of model ${model_name}"
# Extract layer-wise activations
 python ${scriptDir}/neurox_extraction.py \
      --model_desc ${model} \
      --input_corpus ${DataPath}/${working_file} \
      --output_file ${outputDir}/${working_file}.activations.json \
      --output_type json \
      --decompose_layers \
      --include_special_tokens \
      --filter_layers ${layer} \
      --input_type text

# Create a dataset file with word and sentence indexes
echo "Creating dataset file"
python ${scriptDir}/create_data_single_layer.py --text-file ${DataPath}/${working_file}.modified --activation-file ${outputDir}/${working_file}.activations-layer${layer}.json --output-prefix ${outputDir}/${working_file}-layer${layer}


echo "Creating filter file"
# Filter number of tokens to fit in the memory for clustering. Input file will be from step 4. minfreq sets the minimum frequency. If a word type appears is coming less than minfreq, it will be dropped. if a word comes
minfreq=5
maxfreq=20
delfreq=1000000
python ${scriptDir}/frequency_filter_data.py --input-file ${outputDir}/${working_file}-layer${layer}-dataset.json --frequency-file ${DataPath}/${working_file}.words_freq --sentence-file ${outputDir}/${working_file}-layer${layer}-sentences.json --minimum-frequency $minfreq --maximum-frequency $maxfreq --delete-frequency ${delfreq} --output-file ${outputDir}/${working_file}-layer${layer}

