# read codebook from codebook.json
import argparse
import json
import random
import numpy as np
from nltk.corpus import words
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS


def read_dataset_file(file_name):
    inputs = []
    with open(file_name, 'r') as f:
        for line in f:
            inputs.append(line.strip())
    return inputs


# read vector map from vector_map.json
def load_vector_map(file_name):
    with open(file_name, 'r') as json_file:
        return json.load(json_file)


def preprocess_Llama_vectors(vector_map):
    new_vector_map = {}
    for key, value in vector_map.items():
        new_vector_map[key] = []
        for item in value:
            new_item = "[CLS]_" + item.split("_")[1] + "_" + item.split("_")[2] + "_" + item.split("_")[3]
            new_vector_map[key].append(new_item)
    return new_vector_map


def createWordCloud(concept, wordList, output_dir):
    result_tokens = [token.split("_")[0] for token in wordList]
    try:
        text = ' '.join(result_tokens)
        stopwords = set()
        wordcloud = WordCloud(width=1000, height=600, background_color='white', collocations=False,
                            normalize_plurals=False, stopwords=stopwords).generate(text)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.savefig(output_dir + "/" + concept + ".png")
        plt.close(fig)  # Close the figure to free memory
        return fig
    except:
        print("Error in generating word cloud for concept: ", result_tokens)
        plt.close('all')  # Close all figures in case of error




def save_sentences_list_as_figure(concept, sentences_list, output_dir, fontsize=12):
    # Calculate dimensions based on content
    num_sentences = len(sentences_list)
    # Set a fixed narrow width and adjust height to accommodate wrapped text
    fig_width = 8  # Fixed narrow width
    fig_height = max(num_sentences * 1.5, 6)  # Increased multiplier to account for wrapped lines
    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    # Add text with enhanced word wrapping
    for i, sentence in enumerate(sentences_list):
        y_pos = 1.0 - (i + 0.5) / num_sentences
        escaped_sentence = sentence.replace('$', r'\$').replace('%', r'\%').replace('#', r'\#')
        bullet_sentence = f'• {escaped_sentence}'
        # Add text with enhanced wrapping settings
        plt.text(0.05, y_pos, bullet_sentence,
                transform=fig.transFigure,
                ha='left',
                va='center',
                wrap=True,
                fontsize=fontsize,
                usetex=False,
                bbox=dict(facecolor='none', edgecolor='none', pad=0),
                linespacing=1.3)
    plt.axis('off')
    # Save figure with minimal padding and force text wrapping by setting a tight layout
    plt.tight_layout(pad=0.2, h_pad=0.5)
    plt.savefig(output_dir + "/" + concept + ".png",
                bbox_inches='tight',
                dpi=300,
                pad_inches=0.2)
    plt.close(fig)
    return fig



def get_sentences_list(concept, CLS_tokens, train_sentences, k, output_dir):
    sentences_list = []
    for token in CLS_tokens:
        token_idx = token.split("_")[-1]
        sentence = train_sentences[int(token_idx)]
        sentences_list.append(sentence)
    # randomly select k sentences from the sentences_list and save corresponding CLS tokens
    random.seed(0)
    k = min(k, len(sentences_list))
    selected_sentences = random.sample(sentences_list, k)  # Optimized: Simplified sentence selection
    fig = save_sentences_list_as_figure(concept, selected_sentences, output_dir)
    return fig



def visualize_latent_concept(cloudMap, cluster_idx, wordList, train_sentences, k, output_dir):
    cluster = cloudMap[str(cluster_idx)]
    CLS_tokens = []
    other_tokens = []
    num_CLS_token = 0
    for word in cluster:
        if '[CLS]' in word or '<s>' in word or '</s>' in word:
            num_CLS_token += 1
            CLS_tokens.append(word)
        else:
            other_tokens.append(word)
    if len(CLS_tokens) > 0:
      if num_CLS_token > round(len(cluster) / 2):
        # sentence_list = print_sentences_list(CLS_tokens, input_sentence, train_sentences, k)
        figure = get_sentences_list(str(cluster_idx), CLS_tokens, train_sentences, k, output_dir)
      else:
        # createWordCloud(wordList)
        figure = createWordCloud(str(cluster_idx), wordList, output_dir)
    else:
      figure = createWordCloud(str(cluster_idx), wordList, output_dir)
    # return sentence_list
    return figure



def visualize_word_cloud(concept_tokens, train_sentences, output_dir, k=5):
    for cluster_idx, wordList in concept_tokens.items():
        cluster_idx = int(cluster_idx)
        visualize_latent_concept(concept_tokens, cluster_idx, wordList, train_sentences, k, output_dir)



def main():
    parse = argparse.ArgumentParser()

    parse.add_argument('--input_data', type=str)
    parse.add_argument('--vector_map', type=str, default='vector_map-orig.json')
    parse.add_argument('--model_type', type=str, default='bert')
    parse.add_argument('--output_dir', type=str, default='concepts_layer12')
    args = parse.parse_args()

    input_data = args.input_data
    vector_map_file = args.vector_map
    output_dir = args.output_dir
    inputs = read_dataset_file(input_data)
    vector_map = load_vector_map(vector_map_file)
    if args.model_type == 'llama':
        vector_map = preprocess_Llama_vectors(vector_map)

    print("Start generating latent concepts_norm")
    visualize_word_cloud(vector_map, inputs, output_dir)



if __name__ == '__main__':
    main()