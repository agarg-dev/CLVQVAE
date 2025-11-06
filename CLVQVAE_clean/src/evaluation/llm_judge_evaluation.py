



#RATING CODE


import os
import json
import pandas as pd
import random
import argparse
from tqdm import tqdm
from openai import OpenAI
import anthropic
import google.generativeai as genai
import time      # To pause between retries
import functools # To help the decorator function correctly


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

DATASET_LABEL_MAPS = {
    "agnews": {
        "LABEL_0": "World News",
        "LABEL_1": "Sports News",
        "LABEL_2": "Business News",
        "LABEL_3": "Science/Technology News"
    },
    "eraser-movie": {
        "0": "Negative Sentiment Review",
        "1": "Positive Sentiment Review"
    },
    "jigsaw": {
        "0": "Non-toxic Comment",
        "1": "Toxic Comment"
    }
}


#  Dataset-specific guidance for the LLM judge
EVALUATION_GUIDANCE = {
    "jigsaw": """
**Guidance for Toxicity Detection:**
Be lenient with borderline cases since non-toxic sentences can be confused with toxic ones. Context matters greatly - strong emotions, passionate language, or criticism doesn't automatically mean toxicity. For 'Toxic' predictions: Look for patterns suggesting harmful intent, but accept that detection is challenging. For 'Non-toxic' predictions: Accept concepts suggesting civil discourse, even if emotionally charged or critical. Sarcasm and irony can be easily misinterpreted.
""",
    "eraser-movie": """
**Guidance for Sentiment Analysis:**
Movie reviews are often nuanced and mixed. For positive predictions: Accept concepts suggesting overall appreciation, enjoyment, or recommendation, even if some criticisms are present. For negative predictions: Accept concepts suggesting overall disappointment or criticism, even if some positive aspects are mentioned. Focus on the dominant sentiment direction rather than requiring pure positive/negative language.
""",
    "agnews": """
**Guidance for Topic Classification:**
News topics frequently overlap - a tech company's earnings (Business + Science/Tech), sports business deals (Sports + Business), or international conflicts affecting markets (World + Business) are common. Accept concepts that show connection to the predicted category even if they could reasonably fit multiple categories. Look for: World News (countries, politics, conflicts, international themes), Sports (teams, games, players, athletic activities), Business (companies, markets, financial concepts), Science/Tech (technology, research, innovations, technical concepts).
"""
}

EXAMPLE_FORMAT = {
     3: '{'
         '"Config_A": {"rating": 3, "reason": "Contains words that strongly support the predicted label and form a coherent concept"}, '
         '"Config_B": {"rating": 2, "reason": "Somewhat supports the prediction but contains mixed concepts"}, '
         '"Config_C": {"rating": 1, "reason": "Does not support the prediction, contains irrelevant words"}'
         '}',
     4: '{'
         '"Config_A": {"rating": 3, "reason": "Contains words that strongly support the predicted label and form a coherent concept"}, '
         '"Config_B": {"rating": 2, "reason": "Somewhat supports the prediction but is too general"}, '
         '"Config_C": {"rating": 1, "reason": "Does not support the prediction, contains irrelevant words"}, '
         '"Config_D": {"rating": 1, "reason": "Contains random words that contradict the expected label"}'
         '}'
}



# =============================================================================
# LLM API FUNCTIONS
# =============================================================================

def retry(retries=3, delay=10, backoff=2):
    """
    A decorator for retrying a function call with exponential backoff.
    It will retry on any exception.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries - 1:
                        print(f"Final attempt failed for {func.__name__}. Error: {e}")
                        return None

                    print(f"Attempt {i+1}/{retries} failed for {func.__name__}. Error: {e}. Retrying in {current_delay:.2f} seconds...")
                    time.sleep(current_delay)

                    current_delay = (current_delay * backoff) + random.uniform(0, 1)
        return wrapper
    return decorator


@retry(retries=3, delay=5, backoff=2)
def call_llm_judge(prompt, model_name="gpt-4o-mini"):
    """Call OpenAI GPT model for evaluation."""
    try:
        client = OpenAI(api_key='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')

        final_prompt = prompt + """

CRITICAL: You must respond with ONLY valid JSON in the exact format requested above. Do not include any explanatory text before or after the JSON. Your entire response should be parseable JSON.
"""

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": final_prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"An API error occurred: {e}")
        raise e

@retry(retries=3, delay=5, backoff=2)
def call_claude_haiku_judge(prompt, model_name="claude-3-5-haiku-latest"):
#claude-3-haiku-20240307
    """Call Claude Haiku model for evaluation."""
    try:
        client = anthropic.Anthropic(
            api_key='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',
            timeout=60.0
        )

        json_prompt = prompt + """

CRITICAL: You must respond with ONLY valid JSON in the exact format requested above. Do not include any explanatory text before or after the JSON. Your entire response should be parseable JSON.
"""

        response = client.messages.create(
            model=model_name,
            max_tokens=4096,
            system="You are a helpful assistant that always responds with valid JSON format exactly as requested. Never include explanatory text - only the JSON object.",
            messages=[{"role": "user", "content": json_prompt}]
        )

        response_text = response.content[0].text.strip()

        # Extract JSON if there's extra text
        if not response_text.startswith('{'):
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != 0:
                response_text = response_text[start:end]

        return json.loads(response_text)

    except json.JSONDecodeError as e:
        print(f"Error: Claude response was not valid JSON: {e}")
        print(f"Response was: {response_text[:200]}...")
        raise e
    except Exception as e:
        print(f"Error with Claude API: {e}")
        raise e

@retry(retries=3, delay=5, backoff=2)
def call_gemini_judge(prompt, model_name="gemini-1.5-flash-latest"):
    """Call Google Gemini model for evaluation."""
    try:
        genai.configure(api_key='AIzaSyBG7O0O-m-iZdqFRWMe--NVQhEBcaBVmHM')
        model = genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "application/json"}
        )

        final_prompt = prompt + """

CRITICAL: You must respond with ONLY valid JSON in the exact format requested above. Do not include any explanatory text before or after the JSON. Your entire response should be parseable JSON.
"""

        response = model.generate_content(final_prompt)
        return json.loads(response.text)

    except Exception as e:
        print(f"An API error occurred with Gemini: {e}")
        raise e


# =============================================================================
# PROMPT BUILDING
# =============================================================================

def build_rating_prompt(base_data, all_configs_info, dataset_name):
    """Build the evaluation prompt for rating multiple concepts."""
    sentence = base_data['sentence']
    prediction = base_data['prediction']
    label_meaning = base_data['label_meaning']

    guidance_text = EVALUATION_GUIDANCE.get(dataset_name, "")
    num_configs = len(all_configs_info)

    prompt = f"""
You are an expert AI and Linguistics researcher. Your task is to evaluate how well each "Concept Representation" explains a model's prediction for a given sentence.

**Context:**
- **Sentence:** "{sentence}"
- **Model's Prediction:** The model classified this as '{prediction}' (Meaning: **{label_meaning}**).

**Your Task:**
For each "Concept Representation" below, rate how well it provides a plausible reason for the model's prediction. A concept representation is a group of words or sentences that together represent a meaningful concept.

**Key Question:** If a model only focused on this "Concept Representation", how well would it support making a prediction of '{label_meaning}'?

**Important Guidelines:**
- **Similar representations with significant overlap should receive the same rating** - if two concepts contain many of the same words or convey similar meanings, they should be rated equally.
- **Words are not inherently better than sentences** - concept sentences may be more detailed, but focus on the final sentiment/meaning inferred from the concept rather than the level of detail.
- **Be flexible with pattern matching** - as long as the overall concept or general theme can be identified and reasonably supports the prediction, it should be considered a good concept even if not perfectly precise.

{guidance_text}

**Rating Rubric:**
- **3 (Good):** The concept representation shows a general connection to the predicted label '{label_meaning}' - even if not perfectly precise, the overall theme or pattern is recognizable and plausibly supportive.
- **2 (Fair):** The concept representation has some connection to the prediction but may be broad, mixed, or only partially relevant.
- **1 (Poor):** The concept representation shows little to no connection to the prediction, is mostly irrelevant, or clearly contradicts the expected label.

**Concept Representations to Evaluate:**
---
"""
    # Append each configuration's details to the prompt
    for config in all_configs_info:
        config_name = config['name']
        prompt += f'\n**Concept from Configuration: "{config_name}"**\n'
        if config['type'] == 'words':
            words_str = ", ".join(config['data'])
            prompt += f"- **Concept Words:** {words_str}\n"
        elif config['type'] == 'sentences':
            sentences_str = "\n".join([f"  - \"{s}\"" for s in config['data']])
            prompt += f"- **Representative Sentences:**\n{sentences_str}\n"

    example_str = EXAMPLE_FORMAT.get(num_configs)
    if example_str:
        example_section = f"Example Format for {num_configs} configurations:\n```json\n{example_str}\n```"
    else:
        example_section = ""

    prompt += f"""
---
**Output Instructions:**
Respond with a single valid JSON object. Use each configuration name as a key, with the value being an object containing:
- `rating`: An integer from 1-3 based on the rubric above
- `reason`: A brief explanation justifying your rating

All {num_configs} configurations must be included. Do not include any text outside the JSON.

{example_section}
"""
    return prompt

# =============================================================================
# CONCEPT REPRESENTATION EXTRACTION
# =============================================================================


def get_concept_representation(salient_token_key, config_data, all_train_sentences, k_words, k_sentences, min_len, max_len):
    """
    Extract concept representation for a given salient token,
    filtering sentences by length with a fallback to the original list.
    """
    token_map = config_data.get('token_map', {})
    vector_map = config_data.get('vector_map', {})
    concept_idx = str(token_map.get(salient_token_key, "NA"))

    if concept_idx == "NA" or concept_idx not in vector_map:
        return None, None

    concept_tokens = vector_map.get(concept_idx, [])
    if not concept_tokens:
        return None, None


    cls_tokens = [token for token in concept_tokens if '[CLS]' in token or '<s>' in token or '</s>' in token]
    is_cls_concept = len(cls_tokens) > round(len(concept_tokens) / 2)

    if is_cls_concept and cls_tokens:
        sentences_list = []
        for token in cls_tokens:
            try:
                sentence_idx = int(token.split("_")[-1])
                if sentence_idx < len(all_train_sentences):
                    sentences_list.append(all_train_sentences[sentence_idx])
            except (ValueError, IndexError):
                continue

        # --- MODIFICATION WITH FALLBACK START ---
        # Filter the collected sentences based on word count
        filtered_sentences = [
            s for s in sentences_list
            if min_len <= len(s.split()) <= max_len
        ]

        # Fallback Logic: If filtering removed all sentences, use the original list.
        # Otherwise, use the list of sentences that fit the length criteria.
        final_sentences_to_use = filtered_sentences if filtered_sentences else sentences_list

        k = min(k_sentences, len(final_sentences_to_use))
        return "sentences", final_sentences_to_use[:k]
        # --- MODIFICATION WITH FALLBACK END ---

    else:
        words = [
            token.split("_")[0] for token in concept_tokens
            if not ('[CLS]' in token or '<s>' in token or '</s>' in token)
        ]
        if not words:
            return None, None

        word_freq = pd.Series(words).value_counts()
        return "words", word_freq.head(k_words).index.tolist()


def load_configuration_data(config_paths):
    """
    Load data for all configurations, dynamically handling VQ-VAE and SAE file names.
    """
    all_config_data = {}

    for config_name, path in config_paths.items():
        print(f"Loading '{config_name}' from: {path}")
        try:
            # Detect model type based on config name to load correct files
            is_sae = 'sae' in config_name.lower()

            if is_sae:
                print(f" -> Detected SAE configuration. Using SAE file names.")
                vector_map_file = "neuron_map.json"
                token_map_file = "token_to_neuron_map.json"
                explanations_file = "sae_merged_explanations.csv"
            else:
                print(f" -> Detected VQ-VAE/other configuration. Using default file names.")
                vector_map_file = "vector_map.json"
                token_map_file = "token_to_index_map.json"
                explanations_file = "merged_explanations.csv"

            all_config_data[config_name] = {
                # Load SAE's 'neuron_map' into the 'vector_map' key for compatibility
                "vector_map": json.load(open(os.path.join(path, vector_map_file))),
                "token_map": json.load(open(os.path.join(path, token_map_file))),
                "explanations": pd.read_csv(os.path.join(path, explanations_file))
            }
        except FileNotFoundError as e:
            print(f"Error: Could not find a required file in '{path}'. Details: {e}")
            print("Please ensure the model has been trained and the output files exist.")
            return None

    return all_config_data



def load_dataset_sentences(dataset_file):
    """Load sentences from dataset file."""
    try:
        with open(dataset_file, 'r') as f:
            sentences = [line.strip() for line in f.readlines()]
        return sentences
    except (FileNotFoundError, TypeError):
        print(f"Error: Dataset file not found at '{dataset_file}'.")
        return None


def perform_stratified_sampling(reference_explanations, dev_json_dataset, num_samples, label_map):
    """Perform stratified sampling based on prediction correctness."""
    try:
        print(f"Loading true labels from: {dev_json_dataset}")
        with open(dev_json_dataset, 'r') as f:
            true_labels_data = json.load(f)
        true_labels_map = {i: item['label'] for i, item in enumerate(true_labels_data)}
    except FileNotFoundError:
        print(f"\n[Warning] True labels file not found at '{dev_json_dataset}'.")
        print("Falling back to simple random sampling.\n")
        return reference_explanations.sample(n=num_samples, random_state=42)

    reference_explanations['true_label'] = reference_explanations['sentence_index'].map(true_labels_map).astype(str)
    is_binary = len(label_map) == 2

    def get_category(row):
        pred = str(row['prediction'])
        true = str(row['true_label'])
        if is_binary:
            if pred == '1' and true == '1': return 'TP'
            if pred == '1' and true == '0': return 'FP'
            if pred == '0' and true == '0': return 'TN'
            if pred == '0' and true == '1': return 'FN'
        else:
            return 'Correct' if pred == true else 'Incorrect'
        return 'N/A'

    reference_explanations['category'] = reference_explanations.apply(get_category, axis=1)

    group_names = ["TP", "FP", "TN", "FN"] if is_binary else ["Correct", "Incorrect"]
    print(f"Performing stratified sampling ({', '.join(group_names)})...")

    # Calculate samples per group to distribute as evenly as possible
    samples_per_group = num_samples // len(group_names)
    remainder = num_samples % len(group_names)

    sampled_indices = []
    for i, name in enumerate(group_names):
        group_df = reference_explanations[reference_explanations['category'] == name]

        # Add remainder to the first few groups
        num_to_sample = samples_per_group + 1 if i < remainder else samples_per_group

        # Ensure we don't try to sample more than available
        num_to_sample = min(len(group_df), num_to_sample)

        sampled_indices.extend(group_df.sample(n=num_to_sample, random_state=42).index.tolist())

    # Initial sampled DataFrame
    sampled_explanations = reference_explanations.loc[sampled_indices]

    # Fill up to num_samples if any group was smaller than its target
    if len(sampled_explanations) < num_samples:
        remaining_count = num_samples - len(sampled_explanations)
        # Find indices that have not been sampled yet
        remaining_indices = reference_explanations.index.difference(sampled_explanations.index)

        if len(remaining_indices) > 0:
             # Sample from the remaining pool to fill the deficit
             num_to_fill = min(remaining_count, len(remaining_indices))
             additional_samples = reference_explanations.loc[remaining_indices].sample(n=num_to_fill, random_state=42)
             sampled_explanations = pd.concat([sampled_explanations, additional_samples])

    # Shuffle the final result and reset the index
    sampled_explanations = sampled_explanations.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Sampled {len(sampled_explanations)} explanations with stratification.")
    return sampled_explanations

# =============================================================================
# LLM EVALUATION
# =============================================================================

def get_llm_judge_function(llm_name):
    """Return the appropriate LLM judge function based on the specified LLM."""
    if llm_name == "claude":
        return call_claude_haiku_judge
    elif llm_name == "openai":
        return lambda prompt: call_llm_judge(prompt, model_name="gpt-5-mini")
    elif llm_name == "gemini_flash":
        return lambda prompt: call_gemini_judge(prompt, model_name="gemini-2.0-flash-lite")
    elif llm_name == "gemini":
        return lambda prompt: call_gemini_judge(prompt, model_name="gemini-2.0-flash")
    else:
        raise ValueError(f"Unsupported LLM: {llm_name}")

def is_valid_rating_response(response, config_names):
    """Check if the LLM response for rating is valid."""
    if not response or not isinstance(response, dict):
        return False

    response_names = set()
    for config_name in config_names:
        if config_name not in response:
            return False

        config_rating = response[config_name]
        if not isinstance(config_rating, dict):
            return False

        if 'rating' not in config_rating or 'reason' not in config_rating:
            return False

        rating_value = config_rating['rating']
        # Handle both string and numeric ratings
        if isinstance(rating_value, str):
            try:
                rating_value = int(rating_value)
            except ValueError:
                return False
        elif not isinstance(rating_value, int):
            return False

        if rating_value < 1 or rating_value > 3:
            return False

        response_names.add(config_name)

    return response_names == set(config_names)

def evaluate_samples(sampled_explanations, all_config_data, configurations, all_train_sentences, all_dev_sentences,
                    label_map, args, output_file):
    """Evaluate and rate concepts for all sampled explanations and save results."""
    llm_judge = get_llm_judge_function(args.llm)

    with open(output_file, 'w') as f_out:
        for _, row in tqdm(sampled_explanations.iterrows(), total=len(sampled_explanations), desc="Evaluating Samples"):
            sentence_idx = int(row['sentence_index'])
            position_idx = int(row['position_index'])
            prediction_key = str(row['prediction']).strip()
            salient_token_key = f"{row['token']}_{position_idx}_{sentence_idx}"

            # Check if sentence index is valid
            if sentence_idx >= len(all_dev_sentences):
                print(f"\n[Warning] Sentence index {sentence_idx} out of bounds for dev sentences. Skipping sample.")
                continue

            base_prompt_data = {
                "sentence": all_dev_sentences[sentence_idx],
                "prediction": prediction_key,
                "label_meaning": label_map.get(prediction_key, "Meaning Not Found"),
            }

            all_configs_for_sample = []
            empty_configs = []

            for config_name in configurations:
                concept_type, concept_data = get_concept_representation(
                    salient_token_key, all_config_data[config_name], all_train_sentences,
                    args.top_k_words, args.top_k_sentences, args.min_sentence_len, args.max_sentence_len
                )
                if concept_type and concept_data:
                    all_configs_for_sample.append({
                        "name": config_name, "type": concept_type, "data": concept_data
                    })
                else:
                    print(f"\n[Info] No concept representation found for config '{config_name}' in sample {sentence_idx}. Will assign rating 1.")
                    empty_configs.append(config_name)

            # If all configurations are empty, skip the sample
            if len(all_configs_for_sample) == 0:
                print(f"\n[Warning] Skipping sample {sentence_idx} - all configurations have empty concept representations.")
                continue

            # Only call LLM judge if we have at least one valid configuration
            llm_result = None
            if all_configs_for_sample:
                prompt = build_rating_prompt(base_prompt_data, all_configs_for_sample, args.dataset_name)
                llm_result = llm_judge(prompt)

            config_names = [c['name'] for c in all_configs_for_sample]

            # Process LLM result and add empty configurations with rating 1
            if llm_result and is_valid_rating_response(llm_result, config_names):
                # Add empty configurations with rating 1
                for empty_config in empty_configs:
                    llm_result[empty_config] = {
                        "rating": 1,
                        "reason": "Empty concept cluster provides no meaningful explanation"
                    }

                # Include all configurations in the prompt data for consistency
                all_configs_including_empty = all_configs_for_sample + [
                    {"name": config, "type": "empty", "data": []} for config in empty_configs
                ]

                judgment_record = {
                    "prompt_data": {**base_prompt_data, "configs": all_configs_including_empty},
                    "llm_response": llm_result,
                    "ground_truth_label": str(row.get('true_label', 'N/A')),
                    "category": row.get('category', 'N/A')
                }
                f_out.write(json.dumps(judgment_record) + "\n")
            else:
                print(f"\n[Warning] LLM judge failed or returned invalid JSON for sentence index {sentence_idx}. Skipping.")

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Use an LLM to judge the quality of learned concepts.")

    parser.add_argument('--config-names', nargs='+', required=True, help='A list of names for the configurations.')
    parser.add_argument('--config-paths', nargs='+', required=True, help='A list of paths to the output directories.')
    parser.add_argument('--train-dataset-file', type=str, required=True, help='Path to the training dataset text file.')
    parser.add_argument('--dev-dataset-file', type=str, required=True, help='Path to the development dataset text file.')
    parser.add_argument('--dev-dataset-json', type=str, required=True, help='Path to the development dataset JSON file.')
    parser.add_argument('--dataset-name', type=str, required=True, help='Name of the dataset (e.g., agnews).')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the LLM judgments.')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of samples to evaluate.')
    parser.add_argument('--top-k-words', type=int, default=10, help='Number of top frequent words to show.')
    parser.add_argument('--top-k-sentences', type=int, default=5, help='Number of top representative sentences.')
    parser.add_argument('--min-sentence-len', type=int, default=5, help='Minimum number of words for a concept sentence.')
    parser.add_argument('--max-sentence-len', type=int, default=30, help='Maximum number of words for a concept sentence.')
    parser.add_argument('--llm', type=str, default='claude', choices=['openai', 'claude', 'gemini', 'gemini_flash'], help='Which LLM to use.')

    return parser.parse_args()

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function."""
    args = parse_args()

    if len(args.config_names) != len(args.config_paths):
        raise ValueError("Error: The number of --config-names must match the number of --config-paths.")

    if args.dataset_name not in DATASET_LABEL_MAPS:
        raise ValueError(f"Error: Label meanings for '{args.dataset_name}' not defined.")

    label_map = DATASET_LABEL_MAPS[args.dataset_name]
    config_paths = dict(zip(args.config_names, args.config_paths))

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "llm_judgments.jsonl")

    print("Loading data...")
    all_config_data = load_configuration_data(config_paths)
    all_train_sentences = load_dataset_sentences(args.train_dataset_file)
    all_dev_sentences = load_dataset_sentences(args.dev_dataset_file)

    if not all([all_config_data, all_train_sentences, all_dev_sentences]):
        print("Aborting due to data loading errors.")
        return

    reference_explanations = all_config_data[args.config_names[0]]["explanations"]
    sampled_explanations = perform_stratified_sampling(
        reference_explanations, args.dev_dataset_json, args.num_samples, label_map
    )

    print(f"\nStarting evaluation on {len(sampled_explanations)} samples...")
    evaluate_samples(
        sampled_explanations, all_config_data, list(config_paths.keys()),
        all_train_sentences, all_dev_sentences, label_map, args, output_file
    )

    print(f"\nEvaluation complete. Judgments saved to {output_file}")


if __name__ == '__main__':
    main()










# #TOURNAMENT BRACKET CODE


# import os
# import json
# import pandas as pd
# import random
# import argparse
# from tqdm import tqdm
# from openai import OpenAI
# import anthropic
# import google.generativeai as genai
# import time      # To pause between retries
# import functools # To help the decorator function correctly
# import math      # For tournament bracket calculations


# # =============================================================================
# # CONFIGURATION AND CONSTANTS
# # =============================================================================

# DATASET_LABEL_MAPS = {
#     "agnews": {
#         "LABEL_0": "World News",
#         "LABEL_1": "Sports News",
#         "LABEL_2": "Business News",
#         "LABEL_3": "Science/Technology News"
#     },
#     "eraser-movie": {
#         "0": "Negative Sentiment Review",
#         "1": "Positive Sentiment Review"
#     },
#     "jigsaw": {
#         "0": "Non-toxic Comment",
#         "1": "Toxic Comment"
#     }
# }


# #  Dataset-specific guidance for the LLM judge
# EVALUATION_GUIDANCE = {
#     "jigsaw": """
# **Guidance for Toxicity Detection:**
# For 'Toxic' predictions: Look for explicit toxicity markers (slurs, threats, hate speech). For 'Non-toxic' predictions: General negativity or criticism can be valid if it doesn't cross into severe toxicity. Consider context - negative words may express legitimate emotion rather than true toxicity. 
# """,
#     "eraser-movie": """
# **Guidance for Sentiment Analysis:**
# Positive predictions should align with positive sentiment words (excellent, love, great). Negative predictions should contain negative sentiment words (terrible, hate, disappointing). There is no notion of very positive or very negative words so same rating should be given to words like "good" and "excellent" or "bad" and "terrible".
# """,
#     "agnews": """
# **Guidance for Topic Classification:**
# Concepts should contain topic-specific keywords: World News (countries, politics, conflicts etc), Sports (teams, games, players, etc), Business (companies, markets, financial, etc), Science/Tech (technology, research, innovations, etc). 
# """
# }

# TOURNAMENT_EXAMPLE_FORMAT = '{"winner": "Config_A", "Config_A": {"score": 3, "reason": "Contains words that strongly support the predicted label and form a coherent concept"}, "Config_B": {"score": 2, "reason": "Somewhat supports the prediction but contains mixed concepts"}}'



# # =============================================================================
# # LLM API FUNCTIONS
# # =============================================================================

# def retry(retries=3, delay=10, backoff=2):
#     """
#     A decorator for retrying a function call with exponential backoff.
#     It will retry on any exception.
#     """
#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             current_delay = delay
#             for i in range(retries):
#                 try:
#                     return func(*args, **kwargs)
#                 except Exception as e:
#                     if i == retries - 1:
#                         print(f"Final attempt failed for {func.__name__}. Error: {e}")
#                         return None

#                     print(f"Attempt {i+1}/{retries} failed for {func.__name__}. Error: {e}. Retrying in {current_delay:.2f} seconds...")
#                     time.sleep(current_delay)

#                     current_delay = (current_delay * backoff) + random.uniform(0, 1)
#         return wrapper
#     return decorator


# @retry(retries=3, delay=5, backoff=2)
# def call_llm_judge(prompt, model_name="gpt-4o-mini"):
#     """Call OpenAI GPT model for evaluation."""
#     try:
#         client = OpenAI(api_key='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')

#         final_prompt = prompt + """

# CRITICAL: You must respond with ONLY valid JSON in the exact format requested above. Do not include any explanatory text before or after the JSON. Your entire response should be parseable JSON.
# """

#         response = client.chat.completions.create(
#             model=model_name,
#             messages=[{"role": "user", "content": final_prompt}],
#             response_format={"type": "json_object"}
#         )
#         return json.loads(response.choices[0].message.content)
#     except Exception as e:
#         print(f"An API error occurred: {e}")
#         raise e

# @retry(retries=3, delay=5, backoff=2)
# def call_claude_haiku_judge(prompt, model_name="claude-3-haiku-20240307"):
#     """Call Claude Haiku model for evaluation."""
#     try:
#         client = anthropic.Anthropic(
#             api_key='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',
#             timeout=60.0
#         )

#         json_prompt = prompt + """

# CRITICAL: You must respond with ONLY valid JSON in the exact format requested above. Do not include any explanatory text before or after the JSON. Your entire response should be parseable JSON.
# """

#         response = client.messages.create(
#             model=model_name,
#             max_tokens=4096,
#             system="You are a helpful assistant that always responds with valid JSON format exactly as requested. Never include explanatory text - only the JSON object.",
#             messages=[{"role": "user", "content": json_prompt}]
#         )

#         response_text = response.content[0].text.strip()

#         # Extract JSON if there's extra text
#         if not response_text.startswith('{'):
#             start = response_text.find('{')
#             end = response_text.rfind('}') + 1
#             if start != -1 and end != 0:
#                 response_text = response_text[start:end]

#         return json.loads(response_text)

#     except json.JSONDecodeError as e:
#         print(f"Error: Claude response was not valid JSON: {e}")
#         print(f"Response was: {response_text[:200]}...")
#         raise e
#     except Exception as e:
#         print(f"Error with Claude API: {e}")
#         raise e

# @retry(retries=3, delay=5, backoff=2)
# def call_gemini_judge(prompt, model_name="gemini-1.5-flash-latest"):
#     """Call Google Gemini model for evaluation."""
#     try:
#         genai.configure(api_key='AIzaSyBG7O0O-m-iZdqFRWMe--NVQhEBcaBVmHM')
#         model = genai.GenerativeModel(
#             model_name,
#             generation_config={"response_mime_type": "application/json"}
#         )

#         final_prompt = prompt + """

# CRITICAL: You must respond with ONLY valid JSON in the exact format requested above. Do not include any explanatory text before or after the JSON. Your entire response should be parseable JSON.
# """

#         response = model.generate_content(final_prompt)
#         return json.loads(response.text)

#     except Exception as e:
#         print(f"An API error occurred with Gemini: {e}")
#         raise e


# # =============================================================================
# # PROMPT BUILDING
# # =============================================================================

# def build_tournament_prompt(base_data, config_a, config_b, dataset_name):
#     """Build the evaluation prompt for head-to-head tournament comparison."""
#     sentence = base_data['sentence']
#     prediction = base_data['prediction']
#     label_meaning = base_data['label_meaning']

#     guidance_text = EVALUATION_GUIDANCE.get(dataset_name, "")

#     prompt = f"""
# You are an expert AI and Linguistics researcher. Your task is to compare two "Concept Representations" to determine which better explains a model's prediction for a given sentence.

# **Context:**
# - **Sentence:** "{sentence}"
# - **Model's Prediction:** The model classified this as '{prediction}' (Meaning: **{label_meaning}**).

# **Your Task:**
# Compare the two concept representations below and determine which provides a better explanation for the model's prediction. Ask yourself: **"If a model only focused on this concept, would it make a prediction of '{label_meaning}'?"**.

# {guidance_text}

# **Scoring Rubric (1-3 scale):**
# - **3 (Good):** The concept representation is coherent and contains words/sentences that support the predicted label '{label_meaning}'.
# - **2 (Fair):** The concept representation is somewhat coherent and supports the prediction but is either too general or lacks clear focus.
# - **1 (Poor):** The concept representation is not coherent, does not support the prediction, contains irrelevant/random words, or contradicts the expected label.

# **Concept Representations to Compare:**
# ---

# **Concept from Configuration: "{config_a['name']}"**
# """

#     if config_a['type'] == 'words':
#         words_str = ", ".join(config_a['data'])
#         prompt += f"- **Concept Words:** {words_str}\n"
#     elif config_a['type'] == 'sentences':
#         sentences_str = "\n".join([f"  - \"{s}\"" for s in config_a['data']])
#         prompt += f"- **Representative Sentences:**\n{sentences_str}\n"

#     prompt += f"""
# **Concept from Configuration: "{config_b['name']}"**
# """

#     if config_b['type'] == 'words':
#         words_str = ", ".join(config_b['data'])
#         prompt += f"- **Concept Words:** {words_str}\n"
#     elif config_b['type'] == 'sentences':
#         sentences_str = "\n".join([f"  - \"{s}\"" for s in config_b['data']])
#         prompt += f"- **Representative Sentences:**\n{sentences_str}\n"

#     prompt += f"""
# ---
# **Output Instructions:**
# Respond with a single valid JSON object containing:
# - `winner`: The name of the configuration that provides the better explanation ("{config_a['name']}" or "{config_b['name']}")
# - `{config_a['name']}`: Object with `score` (1-3) and `reason` (brief explanation)
# - `{config_b['name']}`: Object with `score` (1-3) and `reason` (brief explanation)

# Example Format:
# ```json
# {TOURNAMENT_EXAMPLE_FORMAT}
# ```

# Do not include any text outside the JSON.
# """
#     return prompt

# # =============================================================================
# # CONCEPT REPRESENTATION EXTRACTION
# # =============================================================================


# def get_concept_representation(salient_token_key, config_data, all_train_sentences, k_words, k_sentences, min_len, max_len):
#     """
#     Extract concept representation for a given salient token,
#     filtering sentences by length with a fallback to the original list.
#     """
#     token_map = config_data.get('token_map', {})
#     vector_map = config_data.get('vector_map', {})
#     concept_idx = str(token_map.get(salient_token_key, "NA"))

#     if concept_idx == "NA" or concept_idx not in vector_map:
#         return None, None

#     concept_tokens = vector_map.get(concept_idx, [])
#     if not concept_tokens:
#         return None, None

#     # cls_tokens = [token for token in concept_tokens if '[CLS]' in token or '<s>' in token or '</s>' in token]
#     # is_cls_concept = (
#     #     ('[CLS]' in salient_token_key or '<s>' in salient_token_key or '</s>' in salient_token_key) or
#     #     (len(cls_tokens) > round(len(concept_tokens) / 2))
#     # )

#     cls_tokens = [token for token in concept_tokens if '[CLS]' in token or '<s>' in token or '</s>' in token]
#     is_cls_concept = len(cls_tokens) > round(len(concept_tokens) / 2)

#     if is_cls_concept and cls_tokens:
#         sentences_list = []
#         for token in cls_tokens:
#             try:
#                 sentence_idx = int(token.split("_")[-1])
#                 if sentence_idx < len(all_train_sentences):
#                     sentences_list.append(all_train_sentences[sentence_idx])
#             except (ValueError, IndexError):
#                 continue

#         # --- MODIFICATION WITH FALLBACK START ---
#         # Filter the collected sentences based on word count
#         filtered_sentences = [
#             s for s in sentences_list
#             if min_len <= len(s.split()) <= max_len
#         ]

#         # Fallback Logic: If filtering removed all sentences, use the original list.
#         # Otherwise, use the list of sentences that fit the length criteria.
#         final_sentences_to_use = filtered_sentences if filtered_sentences else sentences_list

#         k = min(k_sentences, len(final_sentences_to_use))
#         return "sentences", final_sentences_to_use[:k]
#         # --- MODIFICATION WITH FALLBACK END ---

#     else:
#         words = [
#             token.split("_")[0] for token in concept_tokens
#             if not ('[CLS]' in token or '<s>' in token or '</s>' in token)
#         ]
#         if not words:
#             return None, None

#         word_freq = pd.Series(words).value_counts()
#         return "words", word_freq.head(k_words).index.tolist()


# def load_configuration_data(config_paths):
#     """
#     Load data for all configurations, dynamically handling VQ-VAE and SAE file names.
#     """
#     all_config_data = {}

#     for config_name, path in config_paths.items():
#         print(f"Loading '{config_name}' from: {path}")
#         try:
#             # Detect model type based on config name to load correct files
#             is_sae = 'sae' in config_name.lower()

#             if is_sae:
#                 print(f" -> Detected SAE configuration. Using SAE file names.")
#                 vector_map_file = "neuron_map.json"
#                 token_map_file = "token_to_neuron_map.json"
#                 explanations_file = "sae_merged_explanations.csv"
#             else:
#                 print(f" -> Detected VQ-VAE/other configuration. Using default file names.")
#                 vector_map_file = "vector_map.json"
#                 token_map_file = "token_to_index_map.json"
#                 explanations_file = "merged_explanations.csv"

#             all_config_data[config_name] = {
#                 # Load SAE's 'neuron_map' into the 'vector_map' key for compatibility
#                 "vector_map": json.load(open(os.path.join(path, vector_map_file))),
#                 "token_map": json.load(open(os.path.join(path, token_map_file))),
#                 "explanations": pd.read_csv(os.path.join(path, explanations_file))
#             }
#         except FileNotFoundError as e:
#             print(f"Error: Could not find a required file in '{path}'. Details: {e}")
#             print("Please ensure the model has been trained and the output files exist.")
#             return None

#     return all_config_data



# def load_dataset_sentences(dataset_file):
#     """Load sentences from dataset file."""
#     try:
#         with open(dataset_file, 'r') as f:
#             sentences = [line.strip() for line in f.readlines()]
#         return sentences
#     except (FileNotFoundError, TypeError):
#         print(f"Error: Dataset file not found at '{dataset_file}'.")
#         return None


# def perform_stratified_sampling(reference_explanations, dev_json_dataset, num_samples, label_map):
#     """Perform stratified sampling based on prediction correctness."""
#     try:
#         print(f"Loading true labels from: {dev_json_dataset}")
#         with open(dev_json_dataset, 'r') as f:
#             true_labels_data = json.load(f)
#         true_labels_map = {i: item['label'] for i, item in enumerate(true_labels_data)}
#     except FileNotFoundError:
#         print(f"\n[Warning] True labels file not found at '{dev_json_dataset}'.")
#         print("Falling back to simple random sampling.\n")
#         return reference_explanations.sample(n=num_samples, random_state=42)

#     reference_explanations['true_label'] = reference_explanations['sentence_index'].map(true_labels_map).astype(str)
#     is_binary = len(label_map) == 2

#     def get_category(row):
#         pred = str(row['prediction'])
#         true = str(row['true_label'])
#         if is_binary:
#             if pred == '1' and true == '1': return 'TP'
#             if pred == '1' and true == '0': return 'FP'
#             if pred == '0' and true == '0': return 'TN'
#             if pred == '0' and true == '1': return 'FN'
#         else:
#             return 'Correct' if pred == true else 'Incorrect'
#         return 'N/A'

#     reference_explanations['category'] = reference_explanations.apply(get_category, axis=1)

#     group_names = ["TP", "FP", "TN", "FN"] if is_binary else ["Correct", "Incorrect"]
#     print(f"Performing stratified sampling ({', '.join(group_names)})...")

#     # Calculate samples per group to distribute as evenly as possible
#     samples_per_group = num_samples // len(group_names)
#     remainder = num_samples % len(group_names)

#     sampled_indices = []
#     for i, name in enumerate(group_names):
#         group_df = reference_explanations[reference_explanations['category'] == name]

#         # Add remainder to the first few groups
#         num_to_sample = samples_per_group + 1 if i < remainder else samples_per_group

#         # Ensure we don't try to sample more than available
#         num_to_sample = min(len(group_df), num_to_sample)

#         sampled_indices.extend(group_df.sample(n=num_to_sample, random_state=42).index.tolist())

#     # Initial sampled DataFrame
#     sampled_explanations = reference_explanations.loc[sampled_indices]

#     # Fill up to num_samples if any group was smaller than its target
#     if len(sampled_explanations) < num_samples:
#         remaining_count = num_samples - len(sampled_explanations)
#         # Find indices that have not been sampled yet
#         remaining_indices = reference_explanations.index.difference(sampled_explanations.index)

#         if len(remaining_indices) > 0:
#              # Sample from the remaining pool to fill the deficit
#              num_to_fill = min(remaining_count, len(remaining_indices))
#              additional_samples = reference_explanations.loc[remaining_indices].sample(n=num_to_fill, random_state=42)
#              sampled_explanations = pd.concat([sampled_explanations, additional_samples])

#     # Shuffle the final result and reset the index
#     sampled_explanations = sampled_explanations.sample(frac=1, random_state=42).reset_index(drop=True)

#     print(f"Sampled {len(sampled_explanations)} explanations with stratification.")
#     return sampled_explanations

# # =============================================================================
# # LLM EVALUATION
# # =============================================================================

# def get_llm_judge_function(llm_name):
#     """Return the appropriate LLM judge function based on the specified LLM."""
#     if llm_name == "claude":
#         return call_claude_haiku_judge
#     elif llm_name == "openai":
#         return lambda prompt: call_llm_judge(prompt, model_name="gpt-4o-mini")
#     elif llm_name == "gemini_flash":
#         return lambda prompt: call_gemini_judge(prompt, model_name="gemini-2.0-flash-lite-001")
#     elif llm_name == "gemini":
#         return lambda prompt: call_gemini_judge(prompt, model_name="gemini-2.0-flash-001")
#     else:
#         raise ValueError(f"Unsupported LLM: {llm_name}")

# def is_valid_tournament_response(response, config_a_name, config_b_name):
#     """Check if the LLM response for tournament comparison is valid."""
#     if not response or not isinstance(response, dict):
#         return False

#     # Check if winner is present and valid
#     if 'winner' not in response:
#         return False

#     winner = response['winner']
#     if winner not in [config_a_name, config_b_name]:
#         return False

#     # Check both configurations have valid scores and reasons
#     for config_name in [config_a_name, config_b_name]:
#         if config_name not in response:
#             return False

#         config_data = response[config_name]
#         if not isinstance(config_data, dict):
#             return False

#         if 'score' not in config_data or 'reason' not in config_data:
#             return False

#         score_value = config_data['score']
#         # Handle both string and numeric scores
#         if isinstance(score_value, str):
#             try:
#                 score_value = int(score_value)
#             except ValueError:
#                 return False
#         elif not isinstance(score_value, int):
#             return False

#         if score_value < 1 or score_value > 3:
#             return False

#     return True

# def create_tournament_bracket(configurations, random_seed=42):
#     """Create a tournament bracket with random seeding."""
#     random.seed(random_seed)

#     # Shuffle configurations randomly for seeding
#     seeded_configs = configurations.copy()
#     random.shuffle(seeded_configs)

#     # Pad with byes if needed to make it a power of 2
#     num_configs = len(seeded_configs)
#     next_power_of_2 = 2 ** math.ceil(math.log2(num_configs))

#     # Add "bye" placeholders for missing slots
#     while len(seeded_configs) < next_power_of_2:
#         seeded_configs.append("BYE")

#     return seeded_configs

# def run_tournament_round(config_pairs, base_data, all_config_data, all_train_sentences,
#                         dataset_name, args, llm_judge):
#     """Run one round of the tournament with pairwise comparisons."""
#     winners = []
#     round_results = []

#     for pair in config_pairs:
#         config_a_name, config_b_name = pair

#         # Handle bye cases
#         if config_a_name == "BYE":
#             winners.append(config_b_name)
#             continue
#         elif config_b_name == "BYE":
#             winners.append(config_a_name)
#             continue

#         # Get concept representations for both configurations
#         sentence_idx = int(base_data['sentence_index'])
#         position_idx = int(base_data['position_index'])
#         salient_token_key = f"{base_data['token']}_{position_idx}_{sentence_idx}"

#         concept_a_type, concept_a_data = get_concept_representation(
#             salient_token_key, all_config_data[config_a_name], all_train_sentences,
#             args.top_k_words, args.top_k_sentences, args.min_sentence_len, args.max_sentence_len
#         )

#         concept_b_type, concept_b_data = get_concept_representation(
#             salient_token_key, all_config_data[config_b_name], all_train_sentences,
#             args.top_k_words, args.top_k_sentences, args.min_sentence_len, args.max_sentence_len
#         )

#         # Skip if both configurations are empty
#         if not (concept_a_type and concept_a_data) and not (concept_b_type and concept_b_data):
#             # Random winner if both are empty
#             winners.append(random.choice([config_a_name, config_b_name]))
#             continue

#         # Handle empty configurations
#         if not (concept_a_type and concept_a_data):
#             winners.append(config_b_name)
#             continue
#         elif not (concept_b_type and concept_b_data):
#             winners.append(config_a_name)
#             continue

#         # Both have valid concepts, run LLM comparison
#         config_a_info = {"name": config_a_name, "type": concept_a_type, "data": concept_a_data}
#         config_b_info = {"name": config_b_name, "type": concept_b_type, "data": concept_b_data}

#         prompt_data = {
#             "sentence": base_data['sentence'],
#             "prediction": base_data['prediction'],
#             "label_meaning": base_data['label_meaning']
#         }

#         prompt = build_tournament_prompt(prompt_data, config_a_info, config_b_info, dataset_name)
#         llm_result = llm_judge(prompt)

#         if llm_result and is_valid_tournament_response(llm_result, config_a_name, config_b_name):
#             winner = llm_result['winner']
#             winners.append(winner)

#             # Store the detailed comparison result
#             round_results.append({
#                 "config_a": config_a_name,
#                 "config_b": config_b_name,
#                 "winner": winner,
#                 "llm_response": llm_result
#             })
#         else:
#             # Fallback to score-based comparison if LLM fails
#             print(f"\n[Warning] LLM judge failed for {config_a_name} vs {config_b_name}. Using fallback.")
#             # Choose randomly as fallback
#             winner = random.choice([config_a_name, config_b_name])
#             winners.append(winner)

#     return winners, round_results

# def run_tournament(configurations, base_data, all_config_data, all_train_sentences,
#                   dataset_name, args, llm_judge):
#     """Run a complete tournament bracket for one sample."""
#     # Create initial bracket
#     bracket = create_tournament_bracket(configurations, args.tournament_seed)
#     all_round_results = []
#     round_num = 1

#     current_round = bracket

#     while len(current_round) > 1:
#         print(f"  Round {round_num}: {len(current_round)} configurations")

#         # Create pairs for this round
#         pairs = [(current_round[i], current_round[i + 1])
#                 for i in range(0, len(current_round), 2)]

#         # Run the round
#         winners, round_results = run_tournament_round(
#             pairs, base_data, all_config_data, all_train_sentences,
#             dataset_name, args, llm_judge
#         )

#         all_round_results.append({
#             "round": round_num,
#             "pairs": pairs,
#             "results": round_results
#         })

#         current_round = winners
#         round_num += 1

#     final_winner = current_round[0] if current_round else None

#     return {
#         "winner": final_winner,
#         "bracket": bracket,
#         "rounds": all_round_results
#     }

# def evaluate_samples(sampled_explanations, all_config_data, configurations, all_train_sentences, all_dev_sentences,
#                     label_map, args, output_file):
#     """Evaluate concepts using tournament bracket system for all sampled explanations."""
#     llm_judge = get_llm_judge_function(args.llm)

#     with open(output_file, 'w') as f_out:
#         for _, row in tqdm(sampled_explanations.iterrows(), total=len(sampled_explanations), desc="Running Tournaments"):
#             sentence_idx = int(row['sentence_index'])
#             position_idx = int(row['position_index'])
#             prediction_key = str(row['prediction']).strip()

#             # Check if sentence index is valid
#             if sentence_idx >= len(all_dev_sentences):
#                 print(f"\n[Warning] Sentence index {sentence_idx} out of bounds for dev sentences. Skipping sample.")
#                 continue

#             # Filter out configurations that have no concept representation
#             valid_configs = []
#             for config_name in configurations:
#                 salient_token_key = f"{row['token']}_{position_idx}_{sentence_idx}"
#                 concept_type, concept_data = get_concept_representation(
#                     salient_token_key, all_config_data[config_name], all_train_sentences,
#                     args.top_k_words, args.top_k_sentences, args.min_sentence_len, args.max_sentence_len
#                 )
#                 if concept_type and concept_data:
#                     valid_configs.append(config_name)

#             # Skip if fewer than 2 valid configurations
#             if len(valid_configs) < 2:
#                 print(f"\n[Warning] Sample {sentence_idx} has {len(valid_configs)} valid configurations. Skipping tournament.")
#                 continue

#             print(f"\n[Info] Running tournament for sample {sentence_idx} with {len(valid_configs)} configurations: {valid_configs}")

#             base_tournament_data = {
#                 "sentence": all_dev_sentences[sentence_idx],
#                 "prediction": prediction_key,
#                 "label_meaning": label_map.get(prediction_key, "Meaning Not Found"),
#                 "sentence_index": sentence_idx,
#                 "position_index": position_idx,
#                 "token": row['token']
#             }

#             # Run the tournament
#             tournament_result = run_tournament(
#                 valid_configs, base_tournament_data, all_config_data, all_train_sentences,
#                 args.dataset_name, args, llm_judge
#             )

#             # Save tournament results
#             tournament_record = {
#                 "sample_data": {
#                     "sentence": base_tournament_data['sentence'],
#                     "prediction": base_tournament_data['prediction'],
#                     "label_meaning": base_tournament_data['label_meaning'],
#                     "sentence_index": sentence_idx,
#                     "position_index": position_idx,
#                     "token": row['token']
#                 },
#                 "tournament_result": tournament_result,
#                 "ground_truth_label": str(row.get('true_label', 'N/A')),
#                 "category": row.get('category', 'N/A'),
#                 "configurations_entered": valid_configs,
#                 "total_configurations": len(configurations)
#             }

#             f_out.write(json.dumps(tournament_record) + "\n")

#             print(f"  Tournament winner: {tournament_result.get('winner', 'Unknown')}")

# # =============================================================================
# # ARGUMENT PARSING
# # =============================================================================

# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description="Use an LLM to judge the quality of learned concepts.")

#     parser.add_argument('--config-names', nargs='+', required=True, help='A list of names for the configurations.')
#     parser.add_argument('--config-paths', nargs='+', required=True, help='A list of paths to the output directories.')
#     parser.add_argument('--train-dataset-file', type=str, required=True, help='Path to the training dataset text file.')
#     parser.add_argument('--dev-dataset-file', type=str, required=True, help='Path to the development dataset text file.')
#     parser.add_argument('--dev-dataset-json', type=str, required=True, help='Path to the development dataset JSON file.')
#     parser.add_argument('--dataset-name', type=str, required=True, help='Name of the dataset (e.g., agnews).')
#     parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the LLM judgments.')
#     parser.add_argument('--num-samples', type=int, default=50, help='Number of samples to evaluate.')
#     parser.add_argument('--top-k-words', type=int, default=10, help='Number of top frequent words to show.')
#     parser.add_argument('--top-k-sentences', type=int, default=5, help='Number of top representative sentences.')
#     parser.add_argument('--min-sentence-len', type=int, default=5, help='Minimum number of words for a concept sentence.')
#     parser.add_argument('--max-sentence-len', type=int, default=30, help='Maximum number of words for a concept sentence.')
#     parser.add_argument('--llm', type=str, default='claude', choices=['openai', 'claude', 'gemini', 'gemini_flash'], help='Which LLM to use.')
#     parser.add_argument('--tournament-seed', type=int, default=42, help='Random seed for tournament bracket generation.')

#     return parser.parse_args()

# # =============================================================================
# # MAIN FUNCTION
# # =============================================================================

# def main():
#     """Main execution function."""
#     args = parse_args()

#     if len(args.config_names) != len(args.config_paths):
#         raise ValueError("Error: The number of --config-names must match the number of --config-paths.")

#     if args.dataset_name not in DATASET_LABEL_MAPS:
#         raise ValueError(f"Error: Label meanings for '{args.dataset_name}' not defined.")

#     label_map = DATASET_LABEL_MAPS[args.dataset_name]
#     config_paths = dict(zip(args.config_names, args.config_paths))

#     os.makedirs(args.output_dir, exist_ok=True)
#     output_file = os.path.join(args.output_dir, "tournament_results.jsonl")

#     print("Loading data...")
#     all_config_data = load_configuration_data(config_paths)
#     all_train_sentences = load_dataset_sentences(args.train_dataset_file)
#     all_dev_sentences = load_dataset_sentences(args.dev_dataset_file)

#     if not all([all_config_data, all_train_sentences, all_dev_sentences]):
#         print("Aborting due to data loading errors.")
#         return

#     reference_explanations = all_config_data[args.config_names[0]]["explanations"]
#     sampled_explanations = perform_stratified_sampling(
#         reference_explanations, args.dev_dataset_json, args.num_samples, label_map
#     )

#     print(f"\nStarting evaluation on {len(sampled_explanations)} samples...")
#     evaluate_samples(
#         sampled_explanations, all_config_data, list(config_paths.keys()),
#         all_train_sentences, all_dev_sentences, label_map, args, output_file
#     )

#     print(f"\nEvaluation complete. Judgments saved to {output_file}")


# if __name__ == '__main__':
#     main()


