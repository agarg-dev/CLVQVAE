#!/bin/bash

scriptDir=../../../../src/IG_backpropagation
inputFile=../../../../data/eraser-movie/train/movie_train.txt.tok.sent_len
model=../../../../models/glue-${dataset}-roberta

outDir=../../../../data/eraser-movie/train/IG_attributions
mkdir ${outDir}

layer=8
saveFile=${outDir}/IG_explanation_layer_${layer}.csv
python ${scriptDir}/ig_for_sequence_classification.py ${inputFile} ${model} $layer ${saveFile}