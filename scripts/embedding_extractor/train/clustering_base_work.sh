#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --partition=cpu2023
#SBATCH --time=1:00:00
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log

module load conda
conda activate CLVQVAE_same

dataset=eraser-movie
# dataset=jigsaw
scriptDir=../../../src/embedding_extractor
inputPath=../../../data/${dataset}/train # path to a sentence file
input=movie_train.txt #name of the sentence file

mkdir -p ${dataset}/train

# maximum sentence length
sentence_length=300

working_file=$input.tok.sent_len #do not change this

#1. Tokenize text with moses tokenizer
perl ${scriptDir}/tokenizer/tokenizer.perl -l en -no-escape < ${inputPath}/$input > ${inputPath}/$input.tok

#2. Do sentence length filtering and keep sentences max length of 300
python ${scriptDir}/sentence_length.py --text-file ${inputPath}/$input.tok --length ${sentence_length} --output-file ${inputPath}/$input.tok.sent_len

#3. Modify the input file to be compatible with the model
python ${scriptDir}/modify_input.py --text-file ${inputPath}/$input.tok.sent_len --output-file ${inputPath}/$input.tok.sent_len.modified --sentence-tag "<s>"

#4. Calculate vocabulary size
python ${scriptDir}/frequency_count.py --input-file ${inputPath}/${working_file}.modified --output-file ${inputPath}/${working_file}.words_freq

