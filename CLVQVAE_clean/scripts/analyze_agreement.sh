#!/bin/bash
# ============================================================================
# Inter-LLM Agreement Analysis Script
# Analyzes agreement between different LLM judges across evaluations
# ============================================================================

# Environment Setup
source $HOME/CLVQVAE/bin/activate

# ============================================================================
# Configuration
# ============================================================================

baseDir=".."
srcDir="${baseDir}/src"
analysisScript="${srcDir}/evaluation/inter_llm_analysis.py"

# Comparison Type
# Options: "4-way_comparison", "8-12_comparison"
comparisonName="8-12_comparison"

# ============================================================================
# Run Analysis Across Datasets and Models
# ============================================================================

for datasetName in "jigsaw" "eraser-movie" "agnews"; do
    for model_name in "bert" "roberta" "llama" "qwen"; do

        resultsBaseDir="${baseDir}/llm_judge_logs_rating_based_initialization/${datasetName}/${model_name}"
        outputDir="${resultsBaseDir}/${comparisonName}_analysis"

        # Find all judgment files for the specified comparison
        judgmentFiles=()
        for llm_dir in ${resultsBaseDir}/${comparisonName}_*; do
            if [ -d "${llm_dir}" ]; then
                file="${llm_dir}/llm_judgments.jsonl"
                if [ -f "${file}" ]; then
                    judgmentFiles+=("${file}")
                fi
            fi
        done

        # Skip if no judgment files found
        if [ ${#judgmentFiles[@]} -eq 0 ]; then
            echo "No judgment files found for ${datasetName}/${model_name} with comparison '${comparisonName}'. Skipping."
            continue
        fi

        # Run analysis
        echo ""
        echo "===== Inter-LLM Analysis ====="
        echo "Dataset: ${datasetName}"
        echo "Model: ${model_name}"
        echo "Comparison: ${comparisonName}"
        echo "Found ${#judgmentFiles[@]} judgment files"
        echo ""

        python "${analysisScript}" \
            --judgments-files "${judgmentFiles[@]}" \
            --output-dir "${outputDir}" \
            --drop-highest-disagreement

    done
done

echo ""
echo "===== Analysis Complete ====="
