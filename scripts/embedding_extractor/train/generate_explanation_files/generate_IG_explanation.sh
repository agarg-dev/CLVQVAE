#!/bin/bash

scriptDir=../../../../src/generate_explanation_files
inputDir=../../../../data/eraser-movie/train/IG_attributions
outDir=../../../../data/eraser-movie/train/IG_attributions
# outDir=../../../../data/eraser-movie/train/IG_explanation_files_mass_50

mkdir ${outDir}

layer=8
echo ${inputDir}/IG_explanation_layer_${layer}.csv
saveFile=${outDir}/explanation_layer_${layer}.txt
python ${scriptDir}/generate_IG_explanation_salient_words.py ${inputDir}/IG_explanation_layer_${layer}.csv ${saveFile} top-1