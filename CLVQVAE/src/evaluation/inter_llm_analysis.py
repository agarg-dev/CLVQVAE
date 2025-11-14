"""
Enhanced analysis script for calculating inter-LLM agreement for
model configuration evaluation using multiple LLM judges.
"""

import pandas as pd
import json
import argparse
import numpy as np
from pathlib import Path
from tabulate import tabulate
from scipy import stats
import itertools
import hashlib
import warnings
import simpledorff
from simpledorff.metrics import interval_metric

warnings.filterwarnings('ignore')


def load_judgments(judgments_file):
    """Load and parse judgments from a JSONL file."""
    ratings_data = []
    try:
        with open(judgments_file, 'r') as f:
            for line_idx, line in enumerate(f):
                try:
                    data = json.loads(line)
                    if not data.get("llm_response"):
                        continue
                    sentence = data.get('prompt_data', {}).get('sentence', '')
                    if not sentence:
                        continue
                    sample_hash_id = hashlib.md5(sentence.encode()).hexdigest()

                    # Handle new format where config names are direct keys
                    llm_response = data["llm_response"]
                    for config_name, rating_info in llm_response.items():
                        if isinstance(rating_info, dict) and 'rating' in rating_info and 'reason' in rating_info:
                            # Convert string ratings to int if needed
                            rating_value = rating_info['rating']
                            if isinstance(rating_value, str):
                                try:
                                    rating_value = int(rating_value)
                                except ValueError:
                                    continue

                            rating_data = {
                                'sample_id': sample_hash_id,
                                'configuration': config_name.strip().strip("'\""),
                                'rating': rating_value,
                                'reason': rating_info['reason']
                            }
                            ratings_data.append(rating_data)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line {line_idx} in {judgments_file}")
    except FileNotFoundError:
        print(f"Error: Judgments file not found at '{judgments_file}'.")
    return ratings_data

def calculate_krippendorff_alpha(config_data):
    """Calculate Krippendorff's Alpha for a configuration."""
    if not all(col in config_data.columns for col in ['sample_id', 'llm', 'rating']):
        return np.nan
    try:
        k_alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
            config_data, experiment_col='sample_id', annotator_col='llm', class_col='rating',
            metric_fn=interval_metric
        )
    except Exception:
        k_alpha = np.nan
    return k_alpha

def calculate_pearson_correlations(config_data):
    """Calculate average Pearson correlation for a configuration."""
    rating_pivot = config_data.pivot_table(index='sample_id', columns='llm', values='rating').dropna()
    if rating_pivot.shape[0] < 2 or rating_pivot.shape[1] < 2:
        return np.nan, 0, []
    correlations = [stats.pearsonr(rating_pivot.iloc[:, i], rating_pivot.iloc[:, j])[0]
                    for i, j in itertools.combinations(range(rating_pivot.shape[1]), 2)]
    correlations = [c for c in correlations if not np.isnan(c)]
    return np.mean(correlations) if correlations else np.nan, len(rating_pivot), list(rating_pivot.columns)


def calculate_judge_disagreement_scores(df_combined):
    """Calculate Pearson correlation-based disagreement score for each judge compared to all others."""
    llms = df_combined['llm'].unique()
    if len(llms) < 2:
        return {}

    judge_disagreement = {}

    for target_llm in llms:
        correlations_with_others = []

        for other_llm in llms:
            if target_llm == other_llm:
                continue

            # Get ratings from both judges for the same samples
            target_data = df_combined[df_combined['llm'] == target_llm][['sample_id', 'configuration', 'rating']]
            other_data = df_combined[df_combined['llm'] == other_llm][['sample_id', 'configuration', 'rating']]

            # Merge on sample_id and configuration to get paired ratings
            merged = pd.merge(target_data, other_data, on=['sample_id', 'configuration'], suffixes=('_target', '_other'))

            if len(merged) >= 2:  # Need at least 2 points for correlation
                try:
                    corr, _ = stats.pearsonr(merged['rating_target'], merged['rating_other'])
                    if not np.isnan(corr):
                        correlations_with_others.append(corr)
                except:
                    continue

        if correlations_with_others:
            # Convert average correlation to disagreement score (higher = more disagreement)
            avg_correlation = np.mean(correlations_with_others)
            judge_disagreement[target_llm] = 1 - avg_correlation

    return judge_disagreement

def identify_highest_disagreement_judge(judge_disagreement):
    """Identify the judge with the highest disagreement score."""
    if not judge_disagreement:
        return None, None

    highest_disagreement_judge = max(judge_disagreement.items(), key=lambda x: x[1])
    return highest_disagreement_judge[0], highest_disagreement_judge[1]

def print_judge_disagreement_analysis(judge_disagreement):
    """Print analysis of judge disagreement scores."""
    if not judge_disagreement:
        print("\n=== Judge Disagreement Analysis ===")
        print("Not enough judges for disagreement analysis.")
        return

    print("\n=== Judge Disagreement Analysis ===")
    print("Disagreement scores (1 - avg_pearson_correlation, higher = more disagreement):")

    sorted_judges = sorted(judge_disagreement.items(), key=lambda x: x[1])
    for judge, score in sorted_judges:
        avg_correlation = 1 - score
        print(f"  {judge}: {score:.3f} (avg correlation: {avg_correlation:.3f})")

    highest_judge, highest_score = identify_highest_disagreement_judge(judge_disagreement)
    if highest_judge:
        print(f"\nJudge with highest disagreement: {highest_judge} (score: {highest_score:.3f})")

def calculate_kendall_w(llm_rankings):
    """Calculate Kendall's W (coefficient of concordance) for ranking agreement."""
    if len(llm_rankings) < 2:
        return np.nan

    # Get common configurations across all LLMs
    all_configs = set()
    for rankings in llm_rankings.values():
        all_configs.update(rankings.index)

    common_configs = list(all_configs)
    for rankings in llm_rankings.values():
        common_configs = [config for config in common_configs if config in rankings.index]

    if len(common_configs) < 2:
        return np.nan

    # Create ranking matrix: rows = configurations, columns = LLMs
    ranking_matrix = []
    for config in common_configs:
        config_ranks = []
        for llm_name in llm_rankings.keys():
            if config in llm_rankings[llm_name].index:
                config_ranks.append(llm_rankings[llm_name][config])
            else:
                return np.nan  # Missing ranking
        ranking_matrix.append(config_ranks)

    ranking_matrix = np.array(ranking_matrix)
    m, n = ranking_matrix.shape  # m = configurations, n = LLMs

    if n < 2 or m < 2:
        return np.nan

    # Calculate rank sums for each configuration
    rank_sums = np.sum(ranking_matrix, axis=1)

    # Calculate mean rank sum
    mean_rank_sum = np.mean(rank_sums)

    # Calculate S (sum of squared deviations)
    S = np.sum((rank_sums - mean_rank_sum) ** 2)

    # Calculate Kendall's W
    W = 12 * S / (n**2 * (m**3 - m))

    return W

def calculate_ranking_agreement(df_combined):
    """Calculate configuration ranking agreement between LLMs."""
    llm_rankings = {llm: df_combined[df_combined['llm'] == llm].groupby('configuration')['rating'].mean().rank(ascending=False, method='average')
                    for llm in df_combined['llm'].unique()}
    if len(llm_rankings) < 2:
        return {}

    ranking_correlations = []
    print("\n=== Configuration Ranking Agreement (Overall Preference) ===")
    for llm1, llm2 in itertools.combinations(llm_rankings.keys(), 2):
        common_configs = list(set(llm_rankings[llm1].index) & set(llm_rankings[llm2].index))
        if len(common_configs) >= 2:
            rank1, rank2 = llm_rankings[llm1][common_configs], llm_rankings[llm2][common_configs]
            corr, p_val = stats.spearmanr(rank1, rank2)
            if not np.isnan(corr):
                ranking_correlations.append(corr)
                print(f"  {llm1} vs {llm2}: Spearman ρ = {corr:.3f} (p={p_val:.3f})")

    avg_agreement = np.mean(ranking_correlations) if ranking_correlations else np.nan
    if not np.isnan(avg_agreement):
        print(f"\n  Average Ranking Agreement (Spearman ρ): {avg_agreement:.3f}")

    # Calculate Kendall's W (coefficient of concordance)
    kendall_w = calculate_kendall_w(llm_rankings)
    if not np.isnan(kendall_w):
        print(f"  Kendall's W (coefficient of concordance): {kendall_w:.3f}")

    return {"avg_ranking_spearman": avg_agreement, "kendall_w": kendall_w}

def calculate_inter_rater_agreement(df_combined, configurations):
    """Calculate various inter-rater agreement metrics."""
    agreement_results = {}
    all_llms = df_combined['llm'].unique()
    print(f"\nCalculating agreement across {len(all_llms)} LLMs: {', '.join(all_llms)}")
    print("\n=== Overall Inter-LLM Agreement (Sentence-Level) ===")

    for config in configurations:
        config_data = df_combined[df_combined['configuration'] == config]
        if config_data['llm'].nunique() < 2: continue

        k_alpha = calculate_krippendorff_alpha(config_data)
        if not np.isnan(k_alpha):
            agreement_results[f"{config}_krippendorff_alpha"] = k_alpha
            print(f"  {config}: Krippendorff's α = {k_alpha:.3f}")

        avg_corr, n_samples, llm_list = calculate_pearson_correlations(config_data)
        if not np.isnan(avg_corr):
            agreement_results[f"{config}_pearson_r"] = avg_corr
            print(f"  {config}: Avg Pearson r = {avg_corr:.3f} (n_samples={n_samples}, LLMs: {', '.join(llm_list)})")

    agreement_results.update(calculate_ranking_agreement(df_combined))
    return agreement_results


def create_comprehensive_summary(df_combined, configurations, agreement_results):
    """Create a comprehensive summary table."""
    summary_stats = []
    for config in configurations:
        config_data = df_combined[df_combined['configuration'] == config]
        if config_data.empty: continue
        stats_row = {
            'Configuration': config,
            'Avg Rating': config_data['rating'].mean(),
            'Std Dev': config_data['rating'].std(),
            'Count': len(config_data),
            **{f'Rating {i} %': (config_data['rating'] == i).mean() * 100 for i in range(3, 0, -1)},
            'Avg Pearson r': agreement_results.get(f"{config}_pearson_r"),
            'Kripp. α': agreement_results.get(f"{config}_krippendorff_alpha")
        }
        summary_stats.append(stats_row)

    summary_df = pd.DataFrame(summary_stats).sort_values('Avg Rating', ascending=False).round(3)
    print("\n=== Comprehensive Configuration Summary ===")
    print(tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False))
    return summary_df

def get_agreement_level(alpha_val):
    if pd.isna(alpha_val) or alpha_val < 0.667: return "Unreliable"
    if alpha_val < 0.800: return "Tentative"
    return "Reliable"

def print_final_results(summary_df):
    """Print final results with winner configuration."""
    print("\n🏆 FINAL RESULTS:")
    if summary_df.empty:
        print("No results to display.")
        return
    winner = summary_df.iloc[0]
    print(f"Best Configuration: {winner['Configuration']}")
    print(f"  Average Rating: {winner['Avg Rating']:.3f}")
    print(f"  Percentage of 'Good' (3) ratings: {winner['Rating 3 %']:.1f}%")
    if 'Kripp. α' in winner and not pd.isna(winner['Kripp. α']):
        agreement = get_agreement_level(winner['Kripp. α'])
        print(f"  Inter-LLM Agreement (Kripp. α): {winner['Kripp. α']:.3f} ({agreement})")

def save_results(output_dir, summary_df, agreement_results, loaded_llms):
    """Save detailed results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "inter_llm_analysis_results.json"
    results = {
        'summary': summary_df.to_dict('records'),
        'agreement_metrics': agreement_results,
        'analyzed_llms': loaded_llms,
        'total_ratings': int(summary_df['Count'].sum())
    }
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_file}")

def analyze_multiple_llm_results(judgments_files, output_dir, drop_highest_disagreement=False):
    """Analyze results from multiple LLM judges from a list of files."""
    all_ratings, loaded_llms = [], []
    for f_path_str in judgments_files:
        f_path = Path(f_path_str)
        # Extract LLM name from the parent directory name
        llm_name = f_path.parent.name.split('_')[-1]
        print(f"Loading judgments from {llm_name} at {f_path}...")
        ratings = load_judgments(f_path)
        if ratings:
            for r in ratings: r['llm'] = llm_name
            all_ratings.extend(ratings)
            loaded_llms.append(llm_name)

    if not all_ratings:
        print("Error: No valid ratings data found from any LLM judge.")
        return

    df_combined = pd.DataFrame(all_ratings)

    # Calculate judge disagreement scores
    judge_disagreement = calculate_judge_disagreement_scores(df_combined)
    print_judge_disagreement_analysis(judge_disagreement)

    # Optionally drop the judge with highest disagreement
    if drop_highest_disagreement and judge_disagreement:
        highest_disagreement_judge, highest_score = identify_highest_disagreement_judge(judge_disagreement)
        if highest_disagreement_judge:
            print(f"\n=== Dropping Judge with Highest Disagreement ===")
            print(f"Removing {highest_disagreement_judge} (disagreement score: {highest_score:.3f})")

            # Filter out the highest disagreement judge
            df_combined = df_combined[df_combined['llm'] != highest_disagreement_judge]
            loaded_llms = [llm for llm in loaded_llms if llm != highest_disagreement_judge]

            print(f"Remaining judges: {', '.join(loaded_llms)}")
            print("\n=== Re-running Analysis with Remaining Judges ===")

            # Recalculate disagreement scores with remaining judges
            judge_disagreement_after = calculate_judge_disagreement_scores(df_combined)
            print_judge_disagreement_analysis(judge_disagreement_after)

    configurations = sorted(df_combined['configuration'].unique())
    agreement_results = calculate_inter_rater_agreement(df_combined, configurations)
    summary_df = create_comprehensive_summary(df_combined, configurations, agreement_results)
    print_final_results(summary_df)

    # Include disagreement analysis in saved results
    if judge_disagreement:
        agreement_results['judge_disagreement_scores'] = judge_disagreement
        if drop_highest_disagreement:
            agreement_results['dropped_judge'] = highest_disagreement_judge if 'highest_disagreement_judge' in locals() else None

    save_results(Path(output_dir), summary_df, agreement_results, loaded_llms)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze inter-LLM agreement for model configuration evaluation.")
    parser.add_argument('--judgments-files', nargs='+', required=True, help='A list of paths to the llm_judgments.jsonl files.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the final analysis JSON file.')
    parser.add_argument('--drop-highest-disagreement', action='store_true', help='Drop the judge with highest disagreement and re-run analysis.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    analyze_multiple_llm_results(args.judgments_files, args.output_dir, args.drop_highest_disagreement)
