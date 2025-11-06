# """Representations Extractor for ``transformers`` toolkit models.

# Module that given a file with input sentences and a ``transformers``
# model, extracts representations from all layers of the model. The script
# supports aggregation over sub-words created due to the tokenization of
# the provided model.

# Can also be invoked as a script as follows:
#     ``python -m neurox.data.extraction.transformers_extractor``
# """

# import argparse
# import csv
# import sys

# import numpy as np
# import torch

# from NeuroX.neurox.data.writer import ActivationsWriter

# from tqdm import tqdm
# from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM



# def get_model_and_tokenizer(model_desc, device="cpu", random_weights=False):
#     """
#     Automatically get the appropriate ``transformers`` model and tokenizer based
#     on the model description
#     """
#     model_desc = model_desc.split(",")
#     model_name = model_desc[0]
#     tokenizer_name = model_desc[0] if len(model_desc) == 1 else model_desc[1]

#     # Conditionally load the correct model architecture
#     is_decoder = any(m_name in model_name.lower() for m_name in ['gpt', 'mistral', 'llama'])

#     if is_decoder:
#         print("Loading model as AutoModelForCausalLM for decoder architecture.")
#         model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
#     else:
#         print("Loading model as AutoModel for encoder architecture.")
#         model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)

#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

#     # Add a padding token if the tokenizer doesn't have one (common for decoders)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
        
#     # --- NEW CODE TO ADD THE SEPARATOR TOKEN ---
#     # 1. Define the separator token.
#     SEPARATOR = "[PROMPT_SEP]"

#     # 2. Add the new token to the tokenizer's vocabulary.
#     tokenizer.add_special_tokens({'additional_special_tokens': [SEPARATOR]})
    
#     # 3. Resize the model's token embeddings to include the new token.
#     model.resize_token_embeddings(len(tokenizer))
#     # --- END NEW CODE ---

#     if random_weights:
#         print("Randomizing weights")
#         model.init_weights()

#     return model, tokenizer




# def aggregate_repr(state, start, end, aggregation):
#     """
#     Function that aggregates activations/embeddings over a span of subword tokens.
#     """
#     if end < start:
#         sys.stderr.write(
#             "WARNING: An empty slice of tokens was encountered. "
#             + "This probably implies a special unicode character or text "
#             + "encoding issue in your original data that was dropped by the "
#             + "transformer model's tokenizer.\n"
#         )
#         return np.zeros((state.shape[0], state.shape[2]))
#     if aggregation == "first":
#         return state[:, start, :]
#     elif aggregation == "last":
#         return state[:, end, :]
#     elif aggregation == "average":
#         return np.average(state[:, start : end + 1, :], axis=1)


# def extract_sentence_representations(
#     sentence,
#     model,
#     tokenizer,
#     device="cpu",
#     include_embeddings=True,
#     aggregation="last",
#     dtype="float32",
#     include_special_tokens=False,
#     tokenization_counts={},
#     input_type="text"
# ):
#     """
#     Get representations for a single sentence. This function remains unchanged.
#     """
#     special_tokens = [
#         x for x in tokenizer.all_special_tokens if x != tokenizer.unk_token
#     ]
#     special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

#     if input_type == "text":
#         original_tokens = sentence.split(" ")

#         # Add letters and spaces around each word since some tokenizers are context sensitive
#         tmp_tokens = []
#         if len(original_tokens) > 0:
#             tmp_tokens.append(f"{original_tokens[0]} a")
#         tmp_tokens += [f"a {x} a" for x in original_tokens[1:-1]]
#         if len(original_tokens) > 1:
#             tmp_tokens.append(f"a {original_tokens[-1]}")

#         sentence = (sentence, )
#         original_tokens = (original_tokens, )
#         tmp_tokens = (tmp_tokens, )
#     elif input_type == "tsv":
#         original_sentence_1, original_sentence_2 = next(csv.reader([sentence], delimiter="\t", quotechar='"'))
#         sentence = (original_sentence_1, original_sentence_2)
#         original_tokens_1 = original_sentence_1.split(" ")
#         original_tokens_2 = original_sentence_2.split(" ")

#         tmp_tokens_1 = []

#         if len(original_tokens_1) > 0:
#             tmp_tokens_1.append(f"{original_tokens_1[0]} a")
#         tmp_tokens_1 += [f"a {x} a" for x in original_tokens_1[1:-1]]
#         if len(original_tokens_1) > 1:
#             tmp_tokens_1.append(f"a {original_tokens_1[-1]}")

#         tmp_tokens_2 = []
#         if len(original_tokens_2) > 0:
#             tmp_tokens_2.append(f"{original_tokens_2[0]} a")
#         tmp_tokens_2 += [f"a {x} a" for x in original_tokens_2[1:-1]]
#         if len(original_tokens_2) > 1:
#             tmp_tokens_2.append(f"a {original_tokens_2[-1]}")

#         original_tokens = (original_tokens_1, original_tokens_2)
#         tmp_tokens = (tmp_tokens_1, tmp_tokens_2)

#     for original_tokens_i, tmp_tokens_i in zip(original_tokens, tmp_tokens):
#         assert len(original_tokens_i) == len(
#             tmp_tokens_i
#         ), f"Original: {original_tokens_i}, Temp: {tmp_tokens_i}"

#     with torch.no_grad():
#         for tmp_tokens_i in tmp_tokens:
#             for token_idx, token in enumerate(tmp_tokens_i):
#                 tok_ids = [
#                     x for x in tokenizer.encode(token) if x not in special_tokens_ids
#                 ]
#                 if token_idx != 0 and token_idx != len(tmp_tokens_i) - 1:
#                     tok_ids = tok_ids[1:-1]
#                 elif token_idx == 0:
#                     tok_ids = tok_ids[:-1]
#                 else:
#                     tok_ids = tok_ids[1:]

#                 if token in tokenization_counts:
#                     assert tokenization_counts[token] == len(
#                         tok_ids
#                     ), "Got different tokenization for already processed word " + token + " " + str(len(tok_ids))
#                 else:
#                     tokenization_counts[token] = len(tok_ids)
#         ids = tokenizer.encode(*sentence, truncation=True)
#         input_ids = torch.tensor([ids]).to(device)
#         all_hidden_states = model(input_ids)[-1]

#         if include_embeddings:
#             all_hidden_states = [
#                 hidden_states[0].cpu().numpy() for hidden_states in all_hidden_states
#             ]
#         else:
#             all_hidden_states = [
#                 hidden_states[0].cpu().numpy()
#                 for hidden_states in all_hidden_states[1:]
#             ]
#         all_hidden_states = np.array(all_hidden_states, dtype=dtype)

#     sentence = "\t".join(sentence)
#     original_tokens = [token for subtokens in original_tokens for token in subtokens]
#     tmp_tokens = [token for subtokens in tmp_tokens for token in subtokens]

#     print('Sentence         : "%s"' % (sentence))
#     print("Original    (%03d): %s" % (len(original_tokens), original_tokens))
#     print(
#         "Tokenized   (%03d): %s"
#         % (
#             len(tokenizer.convert_ids_to_tokens(ids)),
#             tokenizer.convert_ids_to_tokens(ids),
#         )
#     )

#     assert all_hidden_states.shape[1] == len(ids)

#     filtered_ids = ids
#     idx_special_tokens = [t_i for t_i, x in enumerate(ids) if x in special_tokens_ids]
#     special_token_ids = [ids[t_i] for t_i in idx_special_tokens]

#     if not include_special_tokens:
#         idx_without_special_tokens = [
#             t_i for t_i, x in enumerate(ids) if x not in special_tokens_ids
#         ]
#         filtered_ids = [ids[t_i] for t_i in idx_without_special_tokens]
#         all_hidden_states = all_hidden_states[:, idx_without_special_tokens, :]
#         special_token_ids = []

#     assert all_hidden_states.shape[1] == len(filtered_ids)
#     print(
#         "Filtered   (%03d): %s"
#         % (
#             len(tokenizer.convert_ids_to_tokens(filtered_ids)),
#             tokenizer.convert_ids_to_tokens(filtered_ids),
#         )
#     )

#     segmented_tokens = tokenizer.convert_ids_to_tokens(filtered_ids)
#     counter = 0
#     detokenized = []
#     final_hidden_states = np.zeros(
#         (
#             all_hidden_states.shape[0],
#             len(original_tokens) + len(special_token_ids),
#             all_hidden_states.shape[2],
#         ),
#         dtype=dtype,
#     )
#     inputs_truncated = False
#     prev_token_type = "NONE"

#     last_special_token_pointer = 0
#     for token_idx, token in enumerate(tmp_tokens):
#         if include_special_tokens and tokenization_counts[token] != 0:
#             if last_special_token_pointer < len(idx_special_tokens):
#                 while (
#                     last_special_token_pointer < len(idx_special_tokens)
#                     and counter == idx_special_tokens[last_special_token_pointer]
#                 ):
#                     assert prev_token_type != "DROPPED", (
#                         "A token dropped by the tokenizer appeared next "
#                         + "to a special token. Detokenizer cannot resolve "
#                         + f"the ambiguity, please remove '{sentence}' from"
#                         + "the dataset, or try a different tokenizer"
#                     )
#                     prev_token_type = "SPECIAL"
#                     final_hidden_states[:, len(detokenized), :] = all_hidden_states[
#                         :, counter, :
#                     ]
#                     detokenized.append(
#                         segmented_tokens[idx_special_tokens[last_special_token_pointer]]
#                     )
#                     last_special_token_pointer += 1
#                     counter += 1

#         current_word_start_idx = counter
#         current_word_end_idx = counter + tokenization_counts[token]

#         if (
#             tokenization_counts[token] != 0
#             and current_word_start_idx >= all_hidden_states.shape[1]
#         ) or current_word_end_idx > all_hidden_states.shape[1]:
#             final_hidden_states = final_hidden_states[
#                 :,
#                 : len(detokenized)
#                 + len(special_token_ids)
#                 - last_special_token_pointer,
#                 :,
#             ]
#             inputs_truncated = True
#             break

#         if tokenization_counts[token] == 0:
#             assert prev_token_type != "SPECIAL", (
#                 "A token dropped by the tokenizer appeared next "
#                 + "to a special token. Detokenizer cannot resolve "
#                 + f"the ambiguity, please remove '{sentence}' from"
#                 + "the dataset, or try a different tokenizer"
#             )
#             prev_token_type = "DROPPED"
#         else:
#             prev_token_type = "NORMAL"

#         final_hidden_states[:, len(detokenized), :] = aggregate_repr(
#             all_hidden_states,
#             current_word_start_idx,
#             current_word_end_idx - 1,
#             aggregation,
#         )
#         detokenized.append(
#             "".join(segmented_tokens[current_word_start_idx:current_word_end_idx])
#         )
#         counter += tokenization_counts[token]

#     if include_special_tokens:
#         while counter < len(segmented_tokens):
#             if last_special_token_pointer >= len(idx_special_tokens):
#                 break

#             if counter == idx_special_tokens[last_special_token_pointer]:
#                 assert prev_token_type != "DROPPED", (
#                     "A token dropped by the tokenizer appeared next "
#                     + "to a special token. Detokenizer cannot resolve "
#                     + f"the ambiguity, please remove '{sentence}' from"
#                     + "the dataset, or try a different tokenizer"
#                 )
#                 prev_token_type = "SPECIAL"
#                 final_hidden_states[:, len(detokenized), :] = all_hidden_states[
#                     :, counter, :
#                 ]
#                 detokenized.append(
#                     segmented_tokens[idx_special_tokens[last_special_token_pointer]]
#                 )
#                 last_special_token_pointer += 1
#             counter += 1

#     print("Detokenized (%03d): %s" % (len(detokenized), detokenized))
#     print("Counter: %d" % (counter))

#     if inputs_truncated:
#         print("WARNING: Input truncated because of length, skipping check")
#     else:
#         assert counter == len(filtered_ids)
#         assert len(detokenized) == len(original_tokens) + len(special_token_ids)
#     print("===================================================================")
#     return final_hidden_states, detokenized


# def extract_representations(
#     model_desc,
#     input_corpus,
#     output_file,
#     device="cpu",
#     aggregation="last",
#     output_type="json",
#     random_weights=False,
#     ignore_embeddings=False,
#     decompose_layers=False,
#     filter_layers=None,
#     dtype="float32",
#     include_special_tokens=False,
#     input_type="text",
#     is_decoder=False,
#     prompt_path=None
# ):
#     """
#     Extract representations for an entire corpus and save them to disk
#     """
#     print(f"Loading model: {model_desc}")
#     model, tokenizer = get_model_and_tokenizer(
#         model_desc, device=device, random_weights=random_weights
#     )

#     print("Reading input corpus")

#     prompt = ""
#     # Define a unique separator that is highly unlikely to appear in the text
#     # and will be treated as a single token by most tokenizers.
#     SEPARATOR = "[PROMPT_SEP]"

#     if is_decoder:
#         if not prompt_path:
#             raise ValueError("The --is_decoder flag requires a --prompt_path to be specified.")
#         with open(prompt_path, "r") as f:
#             prompt = f.read().strip()
#         print(f"Using decoder model. Prepending prompt: '{prompt}'")

#     def corpus_generator(input_corpus_path, p, sep):
#         with open(input_corpus_path, "r") as fp:
#             for line in fp:
#                 if p: # If there is a prompt, use the separator
#                     yield f"{p} {sep} {line.strip()}"
#                 else: # Otherwise, just yield the line
#                     yield line.strip()

#     print("Preparing output file")
#     writer = ActivationsWriter.get_writer(
#         output_file,
#         filetype=output_type,
#         decompose_layers=decompose_layers,
#         filter_layers=filter_layers,
#         dtype=dtype,
#     )

#     print("Extracting representations from model")
#     tokenization_counts = {}
#     for sentence_idx, sentence_with_prompt in enumerate(corpus_generator(input_corpus, prompt, SEPARATOR)):

#         hidden_states, extracted_words = extract_sentence_representations(
#             sentence_with_prompt, # Pass the full text (prompt + sep + sentence)
#             model,
#             tokenizer,
#             device=device,
#             include_embeddings=(not ignore_embeddings),
#             aggregation=aggregation,
#             dtype=dtype,
#             include_special_tokens=include_special_tokens,
#             tokenization_counts=tokenization_counts,
#             input_type=input_type,
#         )

#         # For decoders, slice out the prompt and separator using the separator as a reliable anchor
#         if is_decoder and prompt:
#             try:
#                 # Find the index of our unique separator in the list of processed words
#                 separator_index = extracted_words.index(SEPARATOR)
                
#                 # The words and embeddings for the actual sentence start one position after the separator
#                 hidden_states = hidden_states[:, separator_index + 1:, :]
#                 extracted_words = extracted_words[separator_index + 1:]

#             except ValueError:
#                 # This fallback will be triggered if the separator is not found,
#                 # which would indicate a problem with the tokenizer.
#                 print(f"Warning: Could not find the prompt separator for line {sentence_idx}. Output may be incorrect.")


#         print("Hidden states: ", hidden_states.shape)
#         print("# Extracted words: ", len(extracted_words))

#         writer.write_activations(sentence_idx, extracted_words, hidden_states)

#     writer.close()


# HDF5_SPECIAL_TOKENS = {".": "__DOT__", "/": "__SLASH__"}


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("model_desc", help="Name of model")
#     parser.add_argument(
#         "input_corpus", help="Text file path with one sentence per line"
#     )
#     parser.add_argument(
#         "output_file",
#         help="Output file path where extracted representations will be stored",
#     )
#     parser.add_argument(
#         "--aggregation",
#         help="first, last or average aggregation for word representation in the case of subword segmentation",
#         default="last",
#     )
#     parser.add_argument(
#         "--dtype",
#         choices=["float16", "float32"],
#         default="float32",
#         help="Output dtype of the extracted representations",
#     )
#     parser.add_argument("--disable_cuda", action="store_true")
#     parser.add_argument("--ignore_embeddings", action="store_true")
#     parser.add_argument(
#         "--random_weights",
#         action="store_true",
#         help="generate representations from randomly initialized model",
#     )
#     parser.add_argument(
#         "--include_special_tokens",
#         action="store_true",
#         help="Include special tokens like [CLS] and [SEP] in the extracted representations",
#     )
#     parser.add_argument(
#         "--input_type",
#         choices=["text", "tsv"],
#         help="Format of the input file, use tsv for multi-sentence inputs",
#         default="text",
#     )

#     ActivationsWriter.add_writer_options(parser)

#     # --- ADDED DECODER ARGUMENTS ---
#     parser.add_argument("--is_decoder", action="store_true",
#                         help="Set this flag for decoder-only models.")
#     parser.add_argument("--prompt_path", type=str, default=None,
#                         help="Path to the prompt file.")
#     # --- END DECODER ARGUMENTS ---

#     args = parser.parse_args()

#     assert args.aggregation in [
#         "average",
#         "first",
#         "last",
#     ], "Invalid aggregation option, please specify first, average or last."

#     assert not (
#         args.filter_layers is not None and args.ignore_embeddings is True
#     ), "--filter_layers and --ignore_embeddings cannot be used at the same time"

#     if not args.disable_cuda and torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")

#     extract_representations(
#         args.model_desc,
#         args.input_corpus,
#         args.output_file,
#         device=device,
#         aggregation=args.aggregation,
#         output_type=args.output_type,
#         random_weights=args.random_weights,
#         ignore_embeddings=args.ignore_embeddings,
#         dtype=args.dtype,
#         decompose_layers=args.decompose_layers,
#         filter_layers=args.filter_layers,
#         include_special_tokens=args.include_special_tokens,
#         input_type=args.input_type,
#         is_decoder=args.is_decoder,
#         prompt_path=args.prompt_path
#     )


# if __name__ == "__main__":
#     main()



# """Representations Extractor for ``transformers`` toolkit models.
# Module that given a file with input sentences and a ``transformers``
# model, extracts representations from all layers of the model. The script
# supports aggregation over sub-words created due to the tokenization of
# the provided model.
# Can also be invoked as a script as follows:
#     ``python -m neurox.data.extraction.transformers_extractor``
# """

# import argparse
# import csv
# import sys
# import os

# import numpy as np
# import torch

# from torch.utils.data import Dataset, DataLoader

# from NeuroX.neurox.data.writer import ActivationsWriter

# from tqdm import tqdm
# from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


# # --- This Dataset class supports the parallel DataLoader ---
# class TextDataset(Dataset):
#     def __init__(self, file_path):
#         with open(file_path, 'r', encoding='utf-8') as f:
#             self.lines = [line.strip() for line in f]

#     def __len__(self):
#         return len(self.lines)

#     def __getitem__(self, idx):
#         return self.lines[idx]


# def get_model_and_tokenizer(model_desc, device="cpu", random_weights=False):
#     # This function is reverted to its original state (no dtype).
#     model_desc = model_desc.split(",")
#     model_name = model_desc[0]
#     tokenizer_name = model_desc[0] if len(model_desc) == 1 else model_desc[1]
#     is_decoder = any(m_name in model_name.lower() for m_name in ['gpt', 'mistral', 'llama'])
#     if is_decoder:
#         print("Loading model as AutoModelForCausalLM for decoder architecture.")
#         model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
#     else:
#         print("Loading model as AutoModel for encoder architecture.")
#         model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     SEPARATOR = "[PROMPT_SEP]"
#     if SEPARATOR not in tokenizer.additional_special_tokens:
#         tokenizer.add_special_tokens({'additional_special_tokens': [SEPARATOR]})
#         model.resize_token_embeddings(len(tokenizer))
#     if random_weights:
#         print("Randomizing weights")
#         model.init_weights()
#     return model, tokenizer


# def aggregate_repr(state, start, end, aggregation):
#     # This function is UNCHANGED
#     if end < start:
#         sys.stderr.write("WARNING: An empty slice of tokens was encountered...\n")
#         return np.zeros((state.shape[0], state.shape[2]))
#     if aggregation == "first":
#         return state[:, start, :]
#     elif aggregation == "last":
#         return state[:, end, :]
#     elif aggregation == "average":
#         return np.average(state[:, start : end + 1, :], axis=1)


# def extract_sentence_representations(
#     sentence, model, tokenizer, device="cpu", include_embeddings=True, aggregation="last",
#     dtype="float32", include_special_tokens=False, tokenization_counts={}, input_type="text"
# ):
#     # This entire core logic function is UNCHANGED from your original.
#     special_tokens = [x for x in tokenizer.all_special_tokens if x != tokenizer.unk_token]
#     special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)
#     if input_type == "text":
#         original_tokens = sentence.split(" ")
#         tmp_tokens = []
#         if len(original_tokens) > 0:
#             tmp_tokens.append(f"{original_tokens[0]} a")
#         tmp_tokens += [f"a {x} a" for x in original_tokens[1:-1]]
#         if len(original_tokens) > 1:
#             tmp_tokens.append(f"a {original_tokens[-1]}")
#         sentence = (sentence,); original_tokens = (original_tokens,); tmp_tokens = (tmp_tokens,)
#     elif input_type == "tsv":
#         original_sentence_1, original_sentence_2 = next(csv.reader([sentence], delimiter="\t", quotechar='"'))
#         sentence = (original_sentence_1, original_sentence_2); original_tokens_1 = original_sentence_1.split(" "); original_tokens_2 = original_sentence_2.split(" ")
#         tmp_tokens_1 = [];
#         if len(original_tokens_1) > 0:
#             tmp_tokens_1.append(f"{original_tokens_1[0]} a")
#         tmp_tokens_1 += [f"a {x} a" for x in original_tokens_1[1:-1]]
#         if len(original_tokens_1) > 1:
#             tmp_tokens_1.append(f"a {original_tokens_1[-1]}")
#         tmp_tokens_2 = []
#         if len(original_tokens_2) > 0:
#             tmp_tokens_2.append(f"{original_tokens_2[0]} a")
#         tmp_tokens_2 += [f"a {x} a" for x in original_tokens_2[1:-1]]
#         if len(original_tokens_2) > 1:
#             tmp_tokens_2.append(f"a {original_tokens_2[-1]}")
#         original_tokens = (original_tokens_1, original_tokens_2); tmp_tokens = (tmp_tokens_1, tmp_tokens_2)
#     for original_tokens_i, tmp_tokens_i in zip(original_tokens, tmp_tokens):
#         assert len(original_tokens_i) == len(tmp_tokens_i), f"Original: {original_tokens_i}, Temp: {tmp_tokens_i}"
#     with torch.no_grad():
#         for tmp_tokens_i in tmp_tokens:
#             for token_idx, token in enumerate(tmp_tokens_i):
#                 tok_ids = [x for x in tokenizer.encode(token) if x not in special_tokens_ids]
#                 if token_idx != 0 and token_idx != len(tmp_tokens_i) - 1:
#                     tok_ids = tok_ids[1:-1]
#                 elif token_idx == 0:
#                     tok_ids = tok_ids[:-1]
#                 else:
#                     tok_ids = tok_ids[1:]
#                 if token in tokenization_counts:
#                     assert tokenization_counts[token] == len(tok_ids), "Got different tokenization for already processed word " + token + " " + str(len(tok_ids))
#                 else:
#                     tokenization_counts[token] = len(tok_ids)
#         ids = tokenizer.encode(*sentence, truncation=True)
#         input_ids = torch.tensor([ids]).to(device)
#         all_hidden_states = model(input_ids)[-1]

#         # REVERTED: The .float() call is no longer needed
#         if include_embeddings:
#             all_hidden_states = [hidden_states[0].cpu().numpy() for hidden_states in all_hidden_states]
#         else:
#             all_hidden_states = [hidden_states[0].cpu().numpy() for hidden_states in all_hidden_states[1:]]

#         all_hidden_states = np.array(all_hidden_states, dtype=dtype)
#     sentence = "\t".join(sentence)
#     original_tokens = [token for subtokens in original_tokens for token in subtokens]
#     tmp_tokens = [token for subtokens in tmp_tokens for token in subtokens]
#     print('Sentence         : "%s"' % (sentence))
#     print("Original    (%03d): %s" % (len(original_tokens), original_tokens))
#     print("Tokenized   (%03d): %s" % (len(tokenizer.convert_ids_to_tokens(ids)), tokenizer.convert_ids_to_tokens(ids),))
#     assert all_hidden_states.shape[1] == len(ids)
#     filtered_ids = ids
#     idx_special_tokens = [t_i for t_i, x in enumerate(ids) if x in special_tokens_ids]
#     special_token_ids = [ids[t_i] for t_i in idx_special_tokens]
#     if not include_special_tokens:
#         idx_without_special_tokens = [t_i for t_i, x in enumerate(ids) if x not in special_tokens_ids]
#         filtered_ids = [ids[t_i] for t_i in idx_without_special_tokens]
#         all_hidden_states = all_hidden_states[:, idx_without_special_tokens, :]
#         special_token_ids = []
#     assert all_hidden_states.shape[1] == len(filtered_ids)
#     print("Filtered   (%03d): %s" % (len(tokenizer.convert_ids_to_tokens(filtered_ids)), tokenizer.convert_ids_to_tokens(filtered_ids),))
#     segmented_tokens = tokenizer.convert_ids_to_tokens(filtered_ids)
#     counter = 0; detokenized = [];
#     final_hidden_states = np.zeros((all_hidden_states.shape[0], len(original_tokens) + len(special_token_ids), all_hidden_states.shape[2],), dtype=dtype,)
#     inputs_truncated = False; prev_token_type = "NONE"; last_special_token_pointer = 0
#     for token_idx, token in enumerate(tmp_tokens):
#         if include_special_tokens and tokenization_counts[token] != 0:
#             if last_special_token_pointer < len(idx_special_tokens):
#                 while (last_special_token_pointer < len(idx_special_tokens) and counter == idx_special_tokens[last_special_token_pointer]):
#                     assert prev_token_type != "DROPPED", ("A token dropped by the tokenizer appeared next to a special token...")
#                     prev_token_type = "SPECIAL"
#                     final_hidden_states[:, len(detokenized), :] = all_hidden_states[:, counter, :]
#                     detokenized.append(segmented_tokens[idx_special_tokens[last_special_token_pointer]])
#                     last_special_token_pointer += 1; counter += 1
#         current_word_start_idx = counter; current_word_end_idx = counter + tokenization_counts[token]
#         if (tokenization_counts[token] != 0 and current_word_start_idx >= all_hidden_states.shape[1]) or current_word_end_idx > all_hidden_states.shape[1]:
#             final_hidden_states = final_hidden_states[:, : len(detokenized) + len(special_token_ids) - last_special_token_pointer, :,]; inputs_truncated = True; break
#         if tokenization_counts[token] == 0:
#             assert prev_token_type != "SPECIAL", ("A token dropped by the tokenizer appeared next to a special token...")
#             prev_token_type = "DROPPED"
#         else:
#             prev_token_type = "NORMAL"
#         final_hidden_states[:, len(detokenized), :] = aggregate_repr(all_hidden_states, current_word_start_idx, current_word_end_idx - 1, aggregation,)
#         detokenized.append("".join(segmented_tokens[current_word_start_idx:current_word_end_idx])); counter += tokenization_counts[token]
#     if include_special_tokens:
#         while counter < len(segmented_tokens):
#             if last_special_token_pointer >= len(idx_special_tokens):
#                 break
#             if counter == idx_special_tokens[last_special_token_pointer]:
#                 assert prev_token_type != "DROPPED", ("A token dropped by the tokenizer appeared next to a special token...")
#                 prev_token_type = "SPECIAL"
#                 final_hidden_states[:, len(detokenized), :] = all_hidden_states[:, counter, :]
#                 detokenized.append(segmented_tokens[idx_special_tokens[last_special_token_pointer]])
#                 last_special_token_pointer += 1
#             counter += 1
#     print("Detokenized (%03d): %s" % (len(detokenized), detokenized))
#     print("Counter: %d" % (counter))
#     if inputs_truncated:
#         print("WARNING: Input truncated because of length, skipping check")
#     else:
#         assert counter == len(filtered_ids)
#         assert len(detokenized) == len(original_tokens) + len(special_token_ids)
#     print("===================================================================")
#     return final_hidden_states, detokenized


# def extract_representations(
#     model_desc, input_corpus, output_file, device="cpu", aggregation="last", output_type="json",
#     random_weights=False, ignore_embeddings=False, decompose_layers=False, filter_layers=None,
#     dtype="float32", include_special_tokens=False, input_type="text", is_decoder=False,
#     prompt_path=None, batch_size=32, num_workers=0
# ):
#     print(f"Loading model: {model_desc}")
#     # REVERTED: Model is loaded without specifying a computation dtype.
#     model, tokenizer = get_model_and_tokenizer(model_desc, device=device, random_weights=random_weights)

#     prompt = ""
#     SEPARATOR = "[PROMPT_SEP]"
#     if is_decoder:
#         if not prompt_path: raise ValueError("The --is_decoder flag requires a --prompt_path to be specified.")
#         with open(prompt_path, "r") as f:
#             prompt = f.read().strip()
#         print(f"Using decoder model. Prepending prompt: '{prompt}'")

#     dataset = TextDataset(input_corpus)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     print("Preparing output file")
#     # Note: The `dtype` argument here only affects the output file, not the computation.
#     writer = ActivationsWriter.get_writer(output_file, filetype=output_type, decompose_layers=decompose_layers, filter_layers=filter_layers, dtype=dtype)
#     print("Extracting representations from model")

#     tokenization_counts = {}
#     sentence_idx = 0
#     for batch_of_sentences in tqdm(data_loader, desc="Processing Batches"):
#         for sentence in batch_of_sentences:
#             sentence_with_prompt = f"{prompt} {SEPARATOR} {sentence}" if prompt else sentence

#             hidden_states, extracted_words = extract_sentence_representations(
#                 sentence_with_prompt, model, tokenizer, device=device,
#                 include_embeddings=(not ignore_embeddings), aggregation=aggregation, dtype=dtype,
#                 include_special_tokens=include_special_tokens, tokenization_counts=tokenization_counts,
#                 input_type=input_type,
#             )

#             if is_decoder and prompt:
#                 try:
#                     separator_index = extracted_words.index(SEPARATOR)
#                     hidden_states = hidden_states[:, separator_index + 1:, :]; extracted_words = extracted_words[separator_index + 1:]
#                 except ValueError:
#                     print(f"Warning: Could not find the prompt separator for line {sentence_idx}. Output may be incorrect.")

#             print("Hidden states: ", hidden_states.shape)
#             print("# Extracted words: ", len(extracted_words))
#             writer.write_activations(sentence_idx, extracted_words, hidden_states)
#             sentence_idx += 1
#     writer.close()


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("model_desc", help="Name of model")
#     parser.add_argument("input_corpus", help="Text file path with one sentence per line")
#     parser.add_argument("output_file", help="Output file path where extracted representations will be stored")
#     parser.add_argument("--aggregation", help="first, last or average aggregation...", default="last")
#     # REVERTED: dtype argument is back to its original state.
#     parser.add_argument("--dtype", choices=["float16", "float32"], default="float32", help="Output dtype of the extracted representations.")
#     parser.add_argument("--disable_cuda", action="store_true")
#     parser.add_argument("--ignore_embeddings", action="store_true")
#     parser.add_argument("--random_weights", action="store_true", help="generate representations from randomly initialized model")
#     parser.add_argument("--include_special_tokens", action="store_true", help="Include special tokens...")
#     parser.add_argument("--input_type", choices=["text", "tsv"], help="Format of the input file...", default="text")
#     ActivationsWriter.add_writer_options(parser)
#     parser.add_argument("--is_decoder", action="store_true", help="Set this flag for decoder-only models.")
#     parser.add_argument("--prompt_path", type=str, default=None, help="Path to the prompt file.")

#     # Arguments for parallel data loading are kept for speed.
#     parser.add_argument("--batch_size", type=int, default=128, help="Number of sentences to load at once by data workers.")
#     parser.add_argument("--num_workers", type=int, default=0, help="Number of parallel CPU workers for data loading. Set to SLURM's --cpus-per-task.")

#     args = parser.parse_args()
#     if not args.disable_cuda and torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")

#     extract_representations(
#         args.model_desc, args.input_corpus, args.output_file, device=device, aggregation=args.aggregation,
#         output_type=args.output_type, random_weights=args.random_weights, ignore_embeddings=args.ignore_embeddings,
#         dtype=args.dtype, decompose_layers=args.decompose_layers, filter_layers=args.filter_layers,
#         include_special_tokens=args.include_special_tokens, input_type=args.input_type, is_decoder=args.is_decoder,
#         prompt_path=args.prompt_path, batch_size=args.batch_size, num_workers=args.num_workers
#     )

# if __name__ == "__main__":
#     main()




# import argparse
# import csv
# import sys
# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from NeuroX.neurox.data.writer import ActivationsWriter
# from tqdm import tqdm
# from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# class TextDataset(Dataset):
#     def __init__(self, file_path):
#         with open(file_path, 'r', encoding='utf-8') as f:
#             self.lines = [line.strip() for line in f if line.strip()]
#     def __len__(self):
#         return len(self.lines)
#     def __getitem__(self, idx):
#         return self.lines[idx]

# def get_model_and_tokenizer(model_desc, device="cpu", random_weights=False):
#     model_desc = model_desc.split(",")
#     model_name = model_desc[0]
#     tokenizer_name = model_desc[0] if len(model_desc) == 1 else model_desc[1]
#     is_decoder = any(m_name in model_name.lower() for m_name in ['gpt', 'mistral', 'llama'])
    
#     if is_decoder:
#         model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
#     else:
#         model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
        
#     if random_weights:
#         model.init_weights()
#     return model, tokenizer

# def aggregate_repr(state, start, end, aggregation):
#     if end < start:
#         return np.zeros((state.shape[0], state.shape[2]))
#     if aggregation == "first":
#         return state[:, start, :]
#     elif aggregation == "last":
#         return state[:, end, :]
#     elif aggregation == "average":
#         return np.average(state[:, start : end + 1, :], axis=1)

# def perform_subword_aggregation(
#     sentence, sentence_activations, tokenizer, aggregation, dtype, include_special_tokens, tokenization_counts
# ):
#     """
#     This function contains the original, correct logic for sub-word aggregation.
#     It operates ONLY on a clean sentence and its corresponding activations.
#     """
#     # --- MODIFIED: Convert special token objects to strings before use ---
#     special_tokens_str = [str(token) for token in tokenizer.all_special_tokens_extended]
#     special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens_str)

#     ids = tokenizer.encode(sentence, truncation=True)

#     if sentence_activations.shape[1] != len(ids):
#         print(f"Warning: Mismatch between sliced activations ({sentence_activations.shape[1]}) and tokenized sentence ({len(ids)}). Skipping.")
#         return None, None
        
#     original_tokens = sentence.split(" ")
#     tmp_tokens = [f"{original_tokens[0]} a"] + [f"a {x} a" for x in original_tokens[1:-1]] + ([f"a {original_tokens[-1]}"] if len(original_tokens) > 1 else [])

#     # for token_idx, token in enumerate(tmp_tokens):
#     #     tok_ids = tokenizer.encode(token, add_special_tokens=False)
#     #     if token not in tokenization_counts:
#     #         tokenization_counts[token] = len(tok_ids)

#     for token_idx, token in enumerate(tmp_tokens):
#         tok_ids = [x for x in tokenizer.encode(token) if x not in special_tokens_ids]
#         # This part correctly slices and removes the sub-tokens for the surrounding "a"s
#         if token_idx != 0 and token_idx != len(tmp_tokens) - 1:
#             tok_ids = tok_ids[1:-1]
#         elif token_idx == 0:
#             tok_ids = tok_ids[:-1]
#         else:
#             tok_ids = tok_ids[1:]
        
#         if token not in tokenization_counts:
#             tokenization_counts[token] = len(tok_ids)

#     filtered_ids = ids
#     idx_special_tokens = [t_i for t_i, x in enumerate(ids) if x in special_tokens_ids]
#     special_token_ids = [ids[t_i] for t_i in idx_special_tokens]
#     if not include_special_tokens:
#         idx_without_special_tokens = [t_i for t_i, x in enumerate(ids) if x not in special_tokens_ids]
#         filtered_ids = [ids[t_i] for t_i in idx_without_special_tokens]
#         sentence_activations = sentence_activations[:, idx_without_special_tokens, :]
#         special_token_ids = []
    
#     segmented_tokens = tokenizer.convert_ids_to_tokens(filtered_ids)
#     counter = 0
#     detokenized = []
#     final_hidden_states = np.zeros((sentence_activations.shape[0], len(original_tokens) + len(special_token_ids), sentence_activations.shape[2]), dtype=dtype)
    
#     for token_idx, token in enumerate(original_tokens):
#         subword_count = tokenization_counts[tmp_tokens[token_idx]]
#         start_idx = counter
#         end_idx = counter + subword_count

#         if end_idx > sentence_activations.shape[1]:
#             print(f"Warning: Truncation detected. Stopping aggregation for sentence.")
#             detokenized = detokenized[:token_idx]
#             final_hidden_states = final_hidden_states[:,:token_idx,:]
#             break

#         final_hidden_states[:, token_idx, :] = aggregate_repr(sentence_activations, start_idx, end_idx - 1, aggregation)
#         detokenized.append("".join(segmented_tokens[start_idx:end_idx]))
#         counter = end_idx

#     if len(detokenized) != len(original_tokens):
#          return None, None
    
#     return final_hidden_states, detokenized


# # def extract_representations(
# #     model_desc, input_corpus, output_file, device="cpu", aggregation="last", output_type="json",
# #     random_weights=False, ignore_embeddings=False, decompose_layers=False, filter_layers=None,
# #     dtype="float32", include_special_tokens=False,
# #     prompt_path=None, batch_size=32, num_workers=0
# # ):
# #     model, tokenizer = get_model_and_tokenizer(model_desc, device=device, random_weights=random_weights)
# #     is_decoder = any(m_name in model_desc.lower() for m_name in ['gpt', 'mistral', 'llama'])
    
# #     prompt_ids = []
# #     if is_decoder and prompt_path:
# #         with open(prompt_path, "r") as f:
# #             prompt = f.read().strip()
# #         # Tokenize prompt WITHOUT special tokens as they will be added with the sentence
# #         prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
# #         print(f"Using decoder model. Prompt token length: {len(prompt_ids)}")

# #     dataset = TextDataset(input_corpus)
# #     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
# #     writer = ActivationsWriter.get_writer(output_file, filetype=output_type, decompose_layers=decompose_layers, filter_layers=filter_layers, dtype=dtype)
# #     tokenization_counts = {}
    
# #     with torch.no_grad():
# #         for sentence_idx_offset, batch_of_sentences in enumerate(tqdm(data_loader, desc="Processing Batches")):
            
# #             # --- STEP 1: JOIN BEFORE INFERENCE ---
# #             if is_decoder and prompt_ids:
# #                 # Tokenize sentences normally (will add BOS/EOS etc.)
# #                 tokenized_sents = tokenizer(batch_of_sentences, padding=True, truncation=True, return_tensors='pt').to(device)
# #                 # Manually create the combined input
# #                 prompt_tensor = torch.tensor(prompt_ids, device=device).unsqueeze(0).expand(len(batch_of_sentences), -1)
# #                 input_ids = torch.cat([prompt_tensor, tokenized_sents['input_ids']], dim=1)
# #                 attention_mask = torch.cat([torch.ones_like(prompt_tensor), tokenized_sents['attention_mask']], dim=1)
                
# #                 # Check for tokenizer truncation on the combined input
# #                 if input_ids.shape[1] > tokenizer.model_max_length:
# #                     input_ids = input_ids[:, :tokenizer.model_max_length]
# #                     attention_mask = attention_mask[:, :tokenizer.model_max_length]

# #             else: # Encoder models
# #                 input_ids = tokenizer(batch_of_sentences, padding=True, truncation=True, return_tensors='pt').to(device)['input_ids']
# #                 attention_mask = tokenizer(batch_of_sentences, padding=True, truncation=True, return_tensors='pt').to(device)['attention_mask']

# #             # --- STEP 2: CONTEXTUALIZED MODEL CALL ---
# #             outputs = model(input_ids, attention_mask=attention_mask)
# #             all_hidden_states_batch = outputs.hidden_states
            
# #             # Process each sentence in the batch individually
# #             for i, sentence in enumerate(batch_of_sentences):
                
# #                 # --- STEP 3: SEPARATE AFTER INFERENCE ---
# #                 # This logic slices the activations to isolate the sentence part
# #                 if is_decoder and prompt_ids:
# #                     # Find the length of the actual sentence tokens (including special tokens)
# #                     sent_ids_len = tokenizer(sentence, truncation=True, return_tensors='pt').to(device)['input_ids'].shape[1]
# #                     prompt_len = len(prompt_ids)
                    
# #                     # Ensure we don't slice past the truncated length
# #                     max_len = all_hidden_states_batch[0].shape[1]
# #                     if prompt_len + sent_ids_len > max_len:
# #                         sent_ids_len = max_len - prompt_len

# #                     start_idx = prompt_len
# #                     end_idx = prompt_len + sent_ids_len
                    
# #                     sentence_hidden_states = [layer[i, start_idx:end_idx, :] for layer in all_hidden_states_batch]

# #                 else: # Encoder models, just remove padding
# #                     actual_len = attention_mask[i].sum()
# #                     sentence_hidden_states = [layer[i, :actual_len, :] for layer in all_hidden_states_batch]

# #                 activations_np = np.array([s.cpu().numpy() for s in sentence_hidden_states])
# #                 if ignore_embeddings:
# #                     activations_np = activations_np[1:]

# #                 # Now, call the clean aggregation function
# #                 final_hidden_states, extracted_words = perform_subword_aggregation(
# #                     sentence, activations_np, tokenizer, aggregation, dtype, include_special_tokens, tokenization_counts
# #                 )
                
# #                 if final_hidden_states is not None:
# #                     writer.write_activations(sentence_idx_offset * batch_size + i, extracted_words, final_hidden_states)

# #     writer.close()





# def extract_representations(
#     model_desc, input_corpus, output_file, device="cpu", aggregation="last", output_type="json",
#     random_weights=False, ignore_embeddings=False, decompose_layers=False, filter_layers=None,
#     dtype="float32", include_special_tokens=False,
#     prompt_path=None, batch_size=32, num_workers=0
# ):
#     model, tokenizer = get_model_and_tokenizer(model_desc, device=device, random_weights=random_weights)
#     is_decoder = any(m_name in model_desc.lower() for m_name in ['gpt', 'mistral', 'llama'])
    
#     prompt_ids = []
#     if is_decoder and prompt_path:
#         with open(prompt_path, "r") as f:
#             prompt = f.read().strip()
#         prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
#         print(f"Using decoder model. Prompt token length: {len(prompt_ids)}")

#     dataset = TextDataset(input_corpus)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     writer = ActivationsWriter.get_writer(output_file, filetype=output_type, decompose_layers=decompose_layers, filter_layers=filter_layers, dtype=dtype)
#     tokenization_counts = {}
    
#     with torch.no_grad():
#         for sentence_idx_offset, batch_of_sentences in enumerate(tqdm(data_loader, desc="Processing Batches")):
            
#             # Tokenize the batch of sentences once
#             tokenized_sents = tokenizer(batch_of_sentences, padding=True, truncation=True, return_tensors='pt').to(device)
            
#             # Join with prompt if necessary
#             if is_decoder and prompt_ids:
#                 prompt_tensor = torch.tensor(prompt_ids, device=device).unsqueeze(0).expand(len(batch_of_sentences), -1)
#                 input_ids = torch.cat([prompt_tensor, tokenized_sents['input_ids']], dim=1)
#                 attention_mask = torch.cat([torch.ones_like(prompt_tensor), tokenized_sents['attention_mask']], dim=1)
                
#                 # Handle potential truncation of the combined input
#                 if hasattr(tokenizer, 'model_max_length') and input_ids.shape[1] > tokenizer.model_max_length:
#                     input_ids = input_ids[:, :tokenizer.model_max_length]
#                     attention_mask = attention_mask[:, :tokenizer.model_max_length]
#             else: # Encoder models
#                 input_ids = tokenized_sents['input_ids']
#                 attention_mask = tokenized_sents['attention_mask']

#             outputs = model(input_ids, attention_mask=attention_mask)
#             all_hidden_states_batch = outputs.hidden_states
            
#             # Process each sentence in the batch individually
#             for i, sentence in enumerate(batch_of_sentences):
                
#                 # This logic slices the activations to isolate the sentence part
#                 if is_decoder and prompt_ids:
#                     # --- MODIFIED: Use the attention mask from the batch for the correct length ---
#                     sent_ids_len = tokenized_sents['attention_mask'][i].sum().item()
#                     prompt_len = len(prompt_ids)
                    
#                     max_len = all_hidden_states_batch[0].shape[1]
#                     if prompt_len + sent_ids_len > max_len:
#                         sent_ids_len = max_len - prompt_len

#                     start_idx = prompt_len
#                     end_idx = prompt_len + sent_ids_len
                    
#                     sentence_hidden_states = [layer[i, start_idx:end_idx, :] for layer in all_hidden_states_batch]
#                 else: # Encoder models, just remove padding
#                     actual_len = attention_mask[i].sum()
#                     sentence_hidden_states = [layer[i, :actual_len, :] for layer in all_hidden_states_batch]

#                 activations_np = np.array([s.cpu().numpy() for s in sentence_hidden_states])
#                 if ignore_embeddings:
#                     activations_np = activations_np[1:]

#                 final_hidden_states, extracted_words = perform_subword_aggregation(
#                     sentence, activations_np, tokenizer, aggregation, dtype, include_special_tokens, tokenization_counts
#                 )
                
#                 if final_hidden_states is not None:
#                     writer.write_activations(sentence_idx_offset * batch_size + i, extracted_words, final_hidden_states)
#     writer.close()

# def main():
#     parser = argparse.ArgumentParser()
#     # Arguments are restored to support prompts correctly
#     parser.add_argument("model_desc")
#     parser.add_argument("input_corpus")
#     parser.add_argument("output_file")
#     parser.add_argument("--aggregation", default="last")
#     parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
#     parser.add_argument("--disable_cuda", action="store_true")
#     parser.add_argument("--ignore_embeddings", action="store_true")
#     parser.add_argument("--random_weights", action="store_true")
#     parser.add_argument("--include_special_tokens", action="store_true")
#     ActivationsWriter.add_writer_options(parser)
#     parser.add_argument("--prompt_path", type=str, default=None) # Prompt feature is restored
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--num_workers", type=int, default=0)
#     args = parser.parse_args()

#     device = torch.device("cpu") if args.disable_cuda or not torch.cuda.is_available() else torch.device("cuda")

#     extract_representations(
#         args.model_desc, args.input_corpus, args.output_file, device=device, aggregation=args.aggregation,
#         output_type=args.output_type, random_weights=args.random_weights, ignore_embeddings=args.ignore_embeddings,
#         dtype=args.dtype, decompose_layers=args.decompose_layers, filter_layers=args.filter_layers,
#         include_special_tokens=args.include_special_tokens,
#         prompt_path=args.prompt_path, 
#         batch_size=args.batch_size, num_workers=args.num_workers
#     )

# if __name__ == "__main__":
#     main()





# import argparse
# import csv
# import sys
# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from NeuroX.neurox.data.writer import ActivationsWriter
# from tqdm import tqdm
# from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# class TextDataset(Dataset):
#     def __init__(self, file_path):
#         with open(file_path, 'r', encoding='utf-8') as f:
#             self.lines = [line.strip() for line in f if line.strip()]
#     def __len__(self):
#         return len(self.lines)
#     def __getitem__(self, idx):
#         return self.lines[idx]

# def get_model_and_tokenizer(model_desc, device="cpu", random_weights=False):
#     model_desc = model_desc.split(",")
#     model_name = model_desc[0]
#     tokenizer_name = model_desc[0] if len(model_desc) == 1 else model_desc[1]
#     is_decoder = any(m_name in model_name.lower() for m_name in ['gpt', 'mistral', 'llama'])
    
#     if is_decoder:
#         model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
#     else:
#         model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
        
#     if random_weights:
#         model.init_weights()
#     return model, tokenizer

# def aggregate_repr(state, start, end, aggregation):
#     if end < start:
#         sys.stderr.write("WARNING: An empty slice of tokens was encountered.\n")
#         return np.zeros((state.shape[0], state.shape[2]))
#     if aggregation == "first":
#         return state[:, start, :]
#     elif aggregation == "last":
#         return state[:, end, :]
#     elif aggregation == "average":
#         return np.average(state[:, start : end + 1, :], axis=1)

# def perform_subword_aggregation(
#     sentence_str, all_hidden_states, tokenizer, aggregation, dtype, include_special_tokens, tokenization_counts
# ):
#     """
#     This is a direct and faithful port of the original, working aggregation logic.
#     It takes pre-computed activations and correctly performs sub-word merging.
#     """
#     special_tokens = [str(x) for x in tokenizer.all_special_tokens if x != tokenizer.unk_token]
#     special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

#     ids = tokenizer.encode(sentence_str, truncation=True)

#     if all_hidden_states.shape[1] != len(ids):
#         print(f"Warning: Mismatch between activations ({all_hidden_states.shape[1]}) and tokenized sentence ({len(ids)}). Skipping.")
#         return None, None


#     original_tokens = sentence_str.split(" ")
#     tmp_tokens = []
#     if len(original_tokens) > 0:
#         tmp_tokens.append(f"{original_tokens[0]} a")
#     tmp_tokens += [f"a {x} a" for x in original_tokens[1:-1]]
#     if len(original_tokens) > 1:
#         tmp_tokens.append(f"a {original_tokens[-1]}")
    
#     original_tokens_tuple = (original_tokens,)
#     tmp_tokens_tuple = (tmp_tokens,)

#     for original_tokens_i, tmp_tokens_i in zip(original_tokens_tuple, tmp_tokens_tuple):
#         for token_idx, token in enumerate(tmp_tokens_i):
#             if token not in tokenization_counts:
#                 tok_ids = [x for x in tokenizer.encode(token) if x not in special_tokens_ids]
#                 if token_idx != 0 and token_idx != len(tmp_tokens_i) - 1:
#                     tok_ids = tok_ids[1:-1]
#                 elif token_idx == 0:
#                     tok_ids = tok_ids[:-1]
#                 else:
#                     tok_ids = tok_ids[1:]
#                 tokenization_counts[token] = len(tok_ids)

#     filtered_ids = ids
#     idx_special_tokens = [t_i for t_i, x in enumerate(ids) if x in special_tokens_ids]
#     special_token_ids = [ids[t_i] for t_i in idx_special_tokens]

#     if not include_special_tokens:
#         idx_without_special_tokens = [t_i for t_i, x in enumerate(ids) if x not in special_tokens_ids]
#         filtered_ids = [ids[t_i] for t_i in idx_without_special_tokens]
#         all_hidden_states = all_hidden_states[:, idx_without_special_tokens, :]
#         special_token_ids = []

#     segmented_tokens = tokenizer.convert_ids_to_tokens(filtered_ids)
#     counter = 0
#     detokenized = []
#     final_hidden_states = np.zeros((all_hidden_states.shape[0], len(original_tokens) + len(special_token_ids), all_hidden_states.shape[2]), dtype=dtype)
#     inputs_truncated = False
#     prev_token_type = "NONE"
#     last_special_token_pointer = 0
    
#     for token_idx, token in enumerate(tmp_tokens):
#         if include_special_tokens and tokenization_counts[token] != 0:
#             if last_special_token_pointer < len(idx_special_tokens):
#                 while (last_special_token_pointer < len(idx_special_tokens) and counter == idx_special_tokens[last_special_token_pointer]):
#                     prev_token_type = "SPECIAL"
#                     final_hidden_states[:, len(detokenized), :] = all_hidden_states[:, counter, :]
#                     detokenized.append(segmented_tokens[idx_special_tokens[last_special_token_pointer]])
#                     last_special_token_pointer += 1
#                     counter += 1

#         current_word_start_idx = counter
#         current_word_end_idx = counter + tokenization_counts[token]

#         if current_word_end_idx > all_hidden_states.shape[1]:
#             inputs_truncated = True
#             break

#         if tokenization_counts[token] == 0:
#             prev_token_type = "DROPPED"
#         else:
#             prev_token_type = "NORMAL"

#         final_hidden_states[:, len(detokenized), :] = aggregate_repr(all_hidden_states, current_word_start_idx, current_word_end_idx - 1, aggregation)
#         detokenized.append("".join(segmented_tokens[current_word_start_idx:current_word_end_idx]))
#         counter += tokenization_counts[token]

#     if not inputs_truncated:
#         assert counter == len(filtered_ids)
#         assert len(detokenized) == len(original_tokens) + len(special_token_ids)


#     return final_hidden_states, detokenized

# def extract_representations(
#     model_desc, input_corpus, output_file, device="cpu", aggregation="last", output_type="json",
#     random_weights=False, ignore_embeddings=False, decompose_layers=False, filter_layers=None,
#     dtype="float32", include_special_tokens=False,
#     prompt_path=None, batch_size=32, num_workers=0
# ):
#     model, tokenizer = get_model_and_tokenizer(model_desc, device=device, random_weights=random_weights)
#     is_decoder = any(m_name in model_desc.lower() for m_name in ['gpt', 'mistral', 'llama'])
    
#     prompt_ids = []
#     if is_decoder and prompt_path:
#         with open(prompt_path, "r") as f:
#             prompt = f.read().strip()
#         prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

#     dataset = TextDataset(input_corpus)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     writer = ActivationsWriter.get_writer(output_file, filetype=output_type, decompose_layers=decompose_layers, filter_layers=filter_layers, dtype=dtype)
#     tokenization_counts = {}
    
#     with torch.no_grad():
#         for sentence_idx_offset, batch_of_sentences in enumerate(tqdm(data_loader, desc="Processing Batches")):
#             tokenized_sents = tokenizer(batch_of_sentences, padding=True, truncation=True, return_tensors='pt').to(device)
            
#             if is_decoder and prompt_ids:
#                 prompt_tensor = torch.tensor(prompt_ids, device=device).unsqueeze(0).expand(len(batch_of_sentences), -1)
#                 input_ids = torch.cat([prompt_tensor, tokenized_sents['input_ids']], dim=1)
#                 attention_mask = torch.cat([torch.ones_like(prompt_tensor), tokenized_sents['attention_mask']], dim=1)
#                 if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length is not None and input_ids.shape[1] > tokenizer.model_max_length:
#                     input_ids = input_ids[:, :tokenizer.model_max_length]
#                     attention_mask = attention_mask[:, :tokenizer.model_max_length]
#             else:
#                 input_ids = tokenized_sents['input_ids']
#                 attention_mask = tokenized_sents['attention_mask']

#             outputs = model(input_ids, attention_mask=attention_mask)
#             all_hidden_states_batch = outputs.hidden_states
            
#             for i, sentence in enumerate(batch_of_sentences):
#                 if is_decoder and prompt_ids:
#                     sent_ids_len = tokenized_sents['attention_mask'][i].sum().item()
#                     prompt_len = len(prompt_ids)
#                     max_len = all_hidden_states_batch[0].shape[1]
#                     if prompt_len + sent_ids_len > max_len: sent_ids_len = max_len - prompt_len
#                     start_idx, end_idx = prompt_len, prompt_len + sent_ids_len
#                     sentence_hidden_states = [layer[i, start_idx:end_idx, :] for layer in all_hidden_states_batch]
#                 else:
#                     actual_len = attention_mask[i].sum()
#                     sentence_hidden_states = [layer[i, :actual_len, :] for layer in all_hidden_states_batch]

#                 activations_np = np.array([s.cpu().numpy() for s in sentence_hidden_states])
#                 if ignore_embeddings:
#                     activations_np = activations_np[1:]

#                 final_hidden_states, extracted_words = perform_subword_aggregation(
#                     sentence, activations_np, tokenizer, aggregation, dtype, include_special_tokens, tokenization_counts
#                 )
                
#                 if final_hidden_states is not None:
#                     writer.write_activations(sentence_idx_offset * batch_size + i, extracted_words, final_hidden_states)
#     writer.close()

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("model_desc")
#     parser.add_argument("input_corpus")
#     parser.add_argument("output_file")
#     parser.add_argument("--aggregation", default="last")
#     parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
#     parser.add_argument("--disable_cuda", action="store_true")
#     parser.add_argument("--ignore_embeddings", action="store_true")
#     parser.add_argument("--random_weights", action="store_true")
#     parser.add_argument("--include_special_tokens", action="store_true")
#     ActivationsWriter.add_writer_options(parser)
#     parser.add_argument("--prompt_path", type=str, default=None)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--num_workers", type=int, default=0)
#     args = parser.parse_args()

#     device = torch.device("cpu") if args.disable_cuda or not torch.cuda.is_available() else torch.device("cuda")
#     extract_representations(
#         args.model_desc, args.input_corpus, args.output_file, device=device, aggregation=args.aggregation,
#         output_type=args.output_type, random_weights=args.random_weights, ignore_embeddings=args.ignore_embeddings,
#         dtype=args.dtype, decompose_layers=args.decompose_layers, filter_layers=args.filter_layers,
#         include_special_tokens=args.include_special_tokens, prompt_path=args.prompt_path, 
#         batch_size=args.batch_size, num_workers=args.num_workers
#     )

# if __name__ == "__main__":
#     main()




# import argparse
# import csv
# import sys
# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from NeuroX.neurox.data.writer import ActivationsWriter
# from tqdm import tqdm
# from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# class TextDataset(Dataset):
#     def __init__(self, file_path):
#         with open(file_path, 'r', encoding='utf-8') as f:
#             self.lines = [line.strip() for line in f if line.strip()]
#     def __len__(self):
#         return len(self.lines)
#     def __getitem__(self, idx):
#         return self.lines[idx]

# def get_model_and_tokenizer(model_desc, device="cpu", random_weights=False):
#     model_desc = model_desc.split(",")
#     model_name = model_desc[0]
#     tokenizer_name = model_desc[0] if len(model_desc) == 1 else model_desc[1]
#     is_decoder = any(m_name in model_name.lower() for m_name in ['gpt', 'mistral', 'llama'])
    
#     if is_decoder:
#         model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
#     else:
#         model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
        
#     if random_weights:
#         model.init_weights()
#     return model, tokenizer

# def aggregate_repr(state, start, end, aggregation):
#     if end < start:
#         sys.stderr.write("WARNING: An empty slice of tokens was encountered.\n")
#         return np.zeros((state.shape[0], state.shape[2]))
#     if aggregation == "first":
#         return state[:, start, :]
#     elif aggregation == "last":
#         return state[:, end, :]
#     elif aggregation == "average":
#         return np.average(state[:, start : end + 1, :], axis=1)

# def perform_subword_aggregation(
#     sentence_str, all_hidden_states, tokenizer, aggregation, dtype, include_special_tokens, tokenization_counts
# ):
#     special_tokens = [str(x) for x in tokenizer.all_special_tokens if x != tokenizer.unk_token]
#     special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

#     ids = tokenizer.encode(sentence_str, truncation=True)

#     if all_hidden_states.shape[1] != len(ids):
#         print(f"Warning: Mismatch between activations ({all_hidden_states.shape[1]}) and tokenized sentence ({len(ids)}). Skipping.")
#         return None, None

#     original_tokens = sentence_str.split(" ")
    
#     # --- ADDED PRINT STATEMENTS ---
#     print('Sentence          : "%s"' % (sentence_str))
#     print("Original    (%03d): %s" % (len(original_tokens), original_tokens))
#     print(
#         "Tokenized   (%03d): %s"
#         % (
#             len(tokenizer.convert_ids_to_tokens(ids)),
#             tokenizer.convert_ids_to_tokens(ids),
#         )
#     )
#     # -----------------------------

#     tmp_tokens = []
#     if len(original_tokens) > 0:
#         tmp_tokens.append(f"{original_tokens[0]} a")
#     tmp_tokens += [f"a {x} a" for x in original_tokens[1:-1]]
#     if len(original_tokens) > 1:
#         tmp_tokens.append(f"a {original_tokens[-1]}")
    
#     for token in tmp_tokens:
#         if token not in tokenization_counts:
#              tok_ids = [
#                 x for x in tokenizer.encode(token, add_special_tokens=False) if x not in special_tokens_ids
#             ]
#              tokenization_counts[token] = len(tok_ids)

#     filtered_ids = ids
#     idx_special_tokens = [t_i for t_i, x in enumerate(ids) if x in special_tokens_ids]
#     special_token_ids = [ids[t_i] for t_i in idx_special_tokens]

#     if not include_special_tokens:
#         idx_without_special_tokens = [t_i for t_i, x in enumerate(ids) if x not in special_tokens_ids]
#         filtered_ids = [ids[t_i] for t_i in idx_without_special_tokens]
#         all_hidden_states = all_hidden_states[:, idx_without_special_tokens, :]
#         special_token_ids = []

#     segmented_tokens = tokenizer.convert_ids_to_tokens(filtered_ids)
#     counter = 0
#     detokenized = []
#     final_hidden_states = np.zeros((all_hidden_states.shape[0], len(original_tokens) + len(special_token_ids), all_hidden_states.shape[2]), dtype=dtype)
#     inputs_truncated = False
    
#     for token_idx, token in enumerate(tmp_tokens):
#         # This part of the loop remains the same
#         if include_special_tokens and tokenization_counts.get(token, 0) != 0:
#             if last_special_token_pointer < len(idx_special_tokens):
#                 while (last_special_token_pointer < len(idx_special_tokens) and counter == idx_special_tokens[last_special_token_pointer]):
#                     final_hidden_states[:, len(detokenized), :] = all_hidden_states[:, counter, :]
#                     detokenized.append(segmented_tokens[idx_special_tokens[last_special_token_pointer]])
#                     last_special_token_pointer += 1
#                     counter += 1

#         current_word_start_idx = counter
#         current_word_end_idx = counter + tokenization_counts.get(token, 0)

#         if current_word_end_idx > all_hidden_states.shape[1]:
#             inputs_truncated = True
#             break

#         final_hidden_states[:, len(detokenized), :] = aggregate_repr(all_hidden_states, current_word_start_idx, current_word_end_idx - 1, aggregation)
#         detokenized.append("".join(segmented_tokens[current_word_start_idx:current_word_end_idx]))
#         counter += tokenization_counts.get(token, 0)

#     # --- ADDED PRINT STATEMENTS ---
#     print("Detokenized (%03d): %s" % (len(detokenized), detokenized))
#     print("===================================================================")
#     # -----------------------------

#     if not inputs_truncated:
#         if len(filtered_ids) != counter:
#              print(f"Warning: Counter ({counter}) does not match filtered_ids length ({len(filtered_ids)}). This may be due to truncation.")

#     # Apply bug fix here to prevent mismatched output sizes
#     return final_hidden_states[:, :len(detokenized), :], detokenized

# # The rest of the script remains unchanged...
# def extract_representations(
#     model_desc, input_corpus, output_file, device="cpu", aggregation="last", output_type="json",
#     random_weights=False, ignore_embeddings=False, decompose_layers=False, filter_layers=None,
#     dtype="float32", include_special_tokens=False,
#     prompt_path=None, batch_size=32, num_workers=0
# ):
#     model, tokenizer = get_model_and_tokenizer(model_desc, device=device, random_weights=random_weights)
#     is_decoder = any(m_name in model_desc.lower() for m_name in ['gpt', 'mistral', 'llama'])
    
#     prompt_text = ""
#     prompt_ids = []
#     if is_decoder and prompt_path:
#         with open(prompt_path, "r") as f:
#             prompt_text = f.read().strip()
#         prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

#     dataset = TextDataset(input_corpus)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     writer = ActivationsWriter.get_writer(output_file, filetype=output_type, decompose_layers=decompose_layers, filter_layers=filter_layers, dtype=dtype)
#     tokenization_counts = {}
    
#     with torch.no_grad():
#         for sentence_idx_offset, batch_of_sentences in enumerate(tqdm(data_loader, desc="Processing Batches")):
#             tokenized_sents = tokenizer(batch_of_sentences, padding=True, truncation=True, return_tensors='pt').to(device)
            
#             if is_decoder and prompt_path:
#                 prompt_tensor = torch.tensor(prompt_ids, device=device).unsqueeze(0).expand(len(batch_of_sentences), -1)
#                 input_ids = torch.cat([prompt_tensor, tokenized_sents['input_ids']], dim=1)
#                 attention_mask = torch.cat([torch.ones_like(prompt_tensor), tokenized_sents['attention_mask']], dim=1)
#                 if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length is not None and input_ids.shape[1] > tokenizer.model_max_length:
#                     input_ids = input_ids[:, :tokenizer.model_max_length]
#                     attention_mask = attention_mask[:, :tokenizer.model_max_length]
#             else:
#                 input_ids = tokenized_sents['input_ids']
#                 attention_mask = tokenized_sents['attention_mask']

#             outputs = model(input_ids, attention_mask=attention_mask)
#             all_hidden_states_batch = outputs.hidden_states
            
#             for i, sentence in enumerate(batch_of_sentences):
#                 full_text = (prompt_text + " " + sentence) if (is_decoder and prompt_path) else sentence
#                 actual_len = attention_mask[i].sum()
#                 sentence_hidden_states = [layer[i, :actual_len, :] for layer in all_hidden_states_batch]

#                 activations_np = np.array([s.cpu().numpy() for s in sentence_hidden_states])
#                 if ignore_embeddings:
#                     activations_np = activations_np[1:]

#                 final_hidden_states, extracted_words = perform_subword_aggregation(
#                     full_text, activations_np, tokenizer, aggregation, dtype, include_special_tokens, tokenization_counts
#                 )
                
#                 if final_hidden_states is not None:
#                     writer.write_activations(sentence_idx_offset * batch_size + i, extracted_words, final_hidden_states)
#     writer.close()

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("model_desc")
#     parser.add_argument("input_corpus")
#     parser.add_argument("output_file")
#     parser.add_argument("--aggregation", default="last")
#     parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
#     parser.add_argument("--disable_cuda", action="store_true")
#     parser.add_argument("--ignore_embeddings", action="store_true")
#     parser.add_argument("--random_weights", action="store_true")
#     parser.add_argument("--include_special_tokens", action="store_true")
#     ActivationsWriter.add_writer_options(parser)
#     parser.add_argument("--prompt_path", type=str, default=None)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--num_workers", type=int, default=0)
#     args = parser.parse_args()

#     device = torch.device("cpu") if args.disable_cuda or not torch.cuda.is_available() else torch.device("cuda")
#     extract_representations(
#         args.model_desc, args.input_corpus, args.output_file, device=device, aggregation=args.aggregation,
#         output_type=args.output_type, random_weights=args.random_weights, ignore_embeddings=args.ignore_embeddings,
#         dtype=args.dtype, decompose_layers=args.decompose_layers, filter_layers=args.filter_layers,
#         include_special_tokens=args.include_special_tokens, prompt_path=args.prompt_path, 
#         batch_size=args.batch_size, num_workers=args.num_workers
#     )

# if __name__ == "__main__":
#     main()



# import argparse
# import csv
# import sys
# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from NeuroX.neurox.data.writer import ActivationsWriter
# from tqdm import tqdm
# from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
# import re # <-- Added for robust splitting

# class TextDataset(Dataset):
#     def __init__(self, file_path):
#         with open(file_path, 'r', encoding='utf-8') as f:
#             self.lines = [line.strip() for line in f if line.strip()]
#     def __len__(self):
#         return len(self.lines)
#     def __getitem__(self, idx):
#         return self.lines[idx]

# def get_model_and_tokenizer(model_desc, device="cpu", random_weights=False):
#     model_desc = model_desc.split(",")
#     model_name = model_desc[0]
#     tokenizer_name = model_desc[0] if len(model_desc) == 1 else model_desc[1]
#     is_decoder = any(m_name in model_name.lower() for m_name in ['gpt', 'mistral', 'llama'])
    
#     if is_decoder:
#         model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
#     else:
#         model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
        
#     if random_weights:
#         model.init_weights()
#     return model, tokenizer

# def aggregate_repr(state, start, end, aggregation):
#     if end < start:
#         sys.stderr.write("WARNING: An empty slice of tokens was encountered.\n")
#         return np.zeros((state.shape[0], state.shape[2]))
#     if aggregation == "first":
#         return state[:, start, :]
#     elif aggregation == "last":
#         return state[:, end, :]
#     elif aggregation == "average":
#         return np.average(state[:, start : end + 1, :], axis=1)

# def perform_subword_aggregation(
#     sentence_str, all_hidden_states, tokenizer, aggregation, dtype, include_special_tokens, tokenization_counts
# ):
#     """
#     This is a direct and faithful port of the original, working aggregation logic.
#     It takes pre-computed activations and correctly performs sub-word merging.
#     """
#     special_tokens = [str(x) for x in tokenizer.all_special_tokens if x != tokenizer.unk_token]
#     special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

#     ids = tokenizer.encode(sentence_str, truncation=True)

#     if all_hidden_states.shape[1] != len(ids):
#         print(f"Warning: Mismatch between activations ({all_hidden_states.shape[1]}) and tokenized sentence ({len(ids)}). Skipping.")
#         return None, None

#     # --- FIX 1: Use robust regex split to handle all whitespace (spaces, newlines) ---
#     original_tokens = [token for token in re.split(r'\s+', sentence_str) if token]
    
#     # --- PRINT STATEMENTS ADDED BACK ---
#     print('Sentence          : "%s"' % (sentence_str))
#     print("Original    (%03d): %s" % (len(original_tokens), original_tokens))
#     print(
#         "Tokenized   (%03d): %s"
#         % (
#             len(tokenizer.convert_ids_to_tokens(ids)),
#             tokenizer.convert_ids_to_tokens(ids),
#         )
#     )
#     # ------------------------------------

#     tmp_tokens = []
#     if len(original_tokens) > 0:
#         tmp_tokens.append(f"{original_tokens[0]} a")
#     tmp_tokens += [f"a {x} a" for x in original_tokens[1:-1]]
#     if len(original_tokens) > 1:
#         tmp_tokens.append(f"a {original_tokens[-1]}")
    
#     # This logic now works correctly because original_tokens is clean
#     for token in tmp_tokens:
#         if token not in tokenization_counts:
#              tok_ids = [
#                 x for x in tokenizer.encode(token, add_special_tokens=False) if x not in special_tokens_ids
#             ]
#              tokenization_counts[token] = len(tok_ids)

#     filtered_ids = ids
#     if not include_special_tokens:
#         idx_without_special_tokens = [t_i for t_i, x in enumerate(ids) if x not in special_tokens_ids]
#         filtered_ids = [ids[t_i] for t_i in idx_without_special_tokens]
#         all_hidden_states = all_hidden_states[:, idx_without_special_tokens, :]

#     segmented_tokens = tokenizer.convert_ids_to_tokens(filtered_ids)
#     counter = 0
#     detokenized = []
#     final_hidden_states = np.zeros((all_hidden_states.shape[0], len(original_tokens), all_hidden_states.shape[2]), dtype=dtype)
#     inputs_truncated = False
    
#     for token_idx, token in enumerate(tmp_tokens):
#         if len(detokenized) >= len(original_tokens):
#             break
            
#         num_subwords = tokenization_counts.get(token, 0)
#         current_word_start_idx = counter
#         current_word_end_idx = counter + num_subwords

#         if current_word_end_idx > all_hidden_states.shape[1]:
#             inputs_truncated = True
#             break

#         final_hidden_states[:, len(detokenized), :] = aggregate_repr(all_hidden_states, current_word_start_idx, current_word_end_idx - 1, aggregation)
        
#         # --- FIX 2: Use tokenizer.decode for proper word reconstruction ---
#         subword_ids_slice = filtered_ids[current_word_start_idx:current_word_end_idx]
#         reconstructed_word = tokenizer.decode(subword_ids_slice)
#         detokenized.append(reconstructed_word)
#         counter += num_subwords

#     if not inputs_truncated:
#         if len(filtered_ids) != counter:
#              print(f"Warning: Counter ({counter}) does not match filtered_ids length ({len(filtered_ids)}). This may be due to truncation.")

#     print("Detokenized (%03d): %s" % (len(detokenized), detokenized))
#     print("===================================================================")
    
#     # Return correctly sized array
#     return final_hidden_states[:, :len(detokenized), :], detokenized

# def extract_representations(
#     model_desc, input_corpus, output_file, device="cpu", aggregation="last", output_type="json",
#     random_weights=False, ignore_embeddings=False, decompose_layers=False, filter_layers=None,
#     dtype="float32", include_special_tokens=False,
#     prompt_path=None, batch_size=32, num_workers=0
# ):
#     model, tokenizer = get_model_and_tokenizer(model_desc, device=device, random_weights=random_weights)
#     is_decoder = any(m_name in model_desc.lower() for m_name in ['gpt', 'mistral', 'llama'])
    
#     prompt_text = ""
#     if prompt_path:
#         with open(prompt_path, "r") as f:
#             prompt_text = f.read() # Read the whole template

#     dataset = TextDataset(input_corpus)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     writer = ActivationsWriter.get_writer(output_file, filetype=output_type, decompose_layers=decompose_layers, filter_layers=filter_layers, dtype=dtype)
#     tokenization_counts = {}
    
#     with torch.no_grad():
#         for sentence_idx_offset, batch_of_sentences in enumerate(tqdm(data_loader, desc="Processing Batches")):
            
#             # --- FIX 3: Use .format() to correctly insert review into prompt ---
#             if '{review_text_goes_here}' in prompt_text:
#                 full_texts = [prompt_text.format(review_text_goes_here=s) for s in batch_of_sentences]
#             else:
#                 full_texts = [prompt_text + " " + s for s in batch_of_sentences]

#             tokenized_batch = tokenizer(full_texts, padding=True, truncation=True, return_tensors='pt').to(device)
            
#             outputs = model(**tokenized_batch)
#             all_hidden_states_batch = outputs.hidden_states
            
#             for i in range(len(full_texts)):
#                 full_text = full_texts[i]
#                 actual_len = tokenized_batch['attention_mask'][i].sum()
#                 sentence_hidden_states = [layer[i, :actual_len, :] for layer in all_hidden_states_batch]

#                 activations_np = np.array([s.cpu().numpy() for s in sentence_hidden_states])
#                 if ignore_embeddings:
#                     activations_np = activations_np[1:]

#                 final_hidden_states, extracted_words = perform_subword_aggregation(
#                     full_text, activations_np, tokenizer, aggregation, dtype, include_special_tokens, tokenization_counts
#                 )
                
#                 if final_hidden_states is not None and extracted_words and final_hidden_states.shape[1] > 0:
#                     writer.write_activations(sentence_idx_offset * batch_size + i, extracted_words, final_hidden_states)
#     writer.close()

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("model_desc")
#     parser.add_argument("input_corpus")
#     parser.add_argument("output_file")
#     parser.add_argument("--aggregation", default="last")
#     parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
#     parser.add_argument("--disable_cuda", action="store_true")
#     parser.add_argument("--ignore_embeddings", action="store_true")
#     parser.add_argument("--random_weights", action="store_true")
#     parser.add_argument("--include_special_tokens", action="store_true")
#     ActivationsWriter.add_writer_options(parser)
#     parser.add_argument("--prompt_path", type=str, default=None)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--num_workers", type=int, default=0)
#     args = parser.parse_args()

#     device = torch.device("cpu") if args.disable_cuda or not torch.cuda.is_available() else torch.device("cuda")
#     extract_representations(
#         args.model_desc, args.input_corpus, args.output_file, device=device, aggregation=args.aggregation,
#         output_type=args.output_type, random_weights=args.random_weights, ignore_embeddings=args.ignore_embeddings,
#         dtype=args.dtype, decompose_layers=args.decompose_layers, filter_layers=args.filter_layers,
#         include_special_tokens=args.include_special_tokens, prompt_path=args.prompt_path, 
#         batch_size=args.batch_size, num_workers=args.num_workers
#     )

# if __name__ == "__main__":
#     main()



# import argparse
# import csv
# import sys
# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from NeuroX.neurox.data.writer import ActivationsWriter
# from tqdm import tqdm
# from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# class TextDataset(Dataset):
#     def __init__(self, file_path):
#         with open(file_path, 'r', encoding='utf-8') as f:
#             self.lines = [line.strip() for line in f if line.strip()]
#     def __len__(self):
#         return len(self.lines)
#     def __getitem__(self, idx):
#         return self.lines[idx]

# def get_model_and_tokenizer(model_desc, device="cpu", random_weights=False):
#     model_desc = model_desc.split(",")
#     model_name = model_desc[0]
#     tokenizer_name = model_desc[0] if len(model_desc) == 1 else model_desc[1]
#     is_decoder = any(m_name in model_name.lower() for m_name in ['gpt', 'mistral', 'llama'])
    
#     if is_decoder:
#         model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
#     else:
#         model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
        
#     if random_weights:
#         model.init_weights()
#     return model, tokenizer

# def aggregate_repr(state, start, end, aggregation):
#     if end < start:
#         sys.stderr.write("WARNING: An empty slice of tokens was encountered.\n")
#         return np.zeros((state.shape[0], state.shape[2]))
#     if aggregation == "first":
#         return state[:, start, :]
#     elif aggregation == "last":
#         return state[:, end, :]
#     elif aggregation == "average":
#         return np.average(state[:, start : end + 1, :], axis=1)

# def perform_subword_aggregation(
#     sentence_str, all_hidden_states, tokenizer, aggregation, dtype, include_special_tokens, tokenization_counts
# ):
#     """
#     This is a direct and faithful port of the original, working aggregation logic.
#     It takes pre-computed activations and correctly performs sub-word merging.
#     """
#     special_tokens = [str(x) for x in tokenizer.all_special_tokens if x != tokenizer.unk_token]
#     special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

#     ids = tokenizer.encode(sentence_str, truncation=True)

#     if all_hidden_states.shape[1] != len(ids):
#         print(f"Warning: Mismatch between activations ({all_hidden_states.shape[1]}) and tokenized sentence ({len(ids)}). Skipping.")
#         return None, None


#     original_tokens = [token for token in sentence_str.strip().split(" ") if token]
    
#     print('Sentence          : "%s"' % (sentence_str))
#     print("Original    (%03d): %s" % (len(original_tokens), original_tokens))
#     print(
#         "Tokenized   (%03d): %s"
#         % (
#             len(tokenizer.convert_ids_to_tokens(ids)),
#             tokenizer.convert_ids_to_tokens(ids),
#         )
#     )

#     tmp_tokens = []
#     if len(original_tokens) > 0:
#         tmp_tokens.append(f"{original_tokens[0]} a")
#     tmp_tokens += [f"a {x} a" for x in original_tokens[1:-1]]
#     if len(original_tokens) > 1:
#         tmp_tokens.append(f"a {original_tokens[-1]}")
    
#     # This is the original subword counting logic
#     original_tokens_tuple = (original_tokens,)
#     tmp_tokens_tuple = (tmp_tokens,)

#     for original_tokens_i, tmp_tokens_i in zip(original_tokens_tuple, tmp_tokens_tuple):
#         for token_idx, token in enumerate(tmp_tokens_i):
#             if token not in tokenization_counts:
#                 tok_ids = [x for x in tokenizer.encode(token) if x not in special_tokens_ids]
#                 if token_idx != 0 and token_idx != len(tmp_tokens_i) - 1:
#                     tok_ids = tok_ids[1:-1]
#                 elif token_idx == 0:
#                     tok_ids = tok_ids[:-1]
#                 else:
#                     tok_ids = tok_ids[1:]
#                 tokenization_counts[token] = len(tok_ids)

#     filtered_ids = ids
#     if not include_special_tokens:
#         idx_without_special_tokens = [t_i for t_i, x in enumerate(ids) if x not in special_tokens_ids]
#         filtered_ids = [ids[t_i] for t_i in idx_without_special_tokens]
#         all_hidden_states = all_hidden_states[:, idx_without_special_tokens, :]

#     segmented_tokens = tokenizer.convert_ids_to_tokens(filtered_ids)
#     counter = 0
#     detokenized = []
#     final_hidden_states = np.zeros((all_hidden_states.shape[0], len(original_tokens), all_hidden_states.shape[2]), dtype=dtype)
#     inputs_truncated = False
    
#     for token_idx, token in enumerate(tmp_tokens):
#         if len(detokenized) >= len(original_tokens):
#             break

#         num_subwords = tokenization_counts.get(token, 0)
#         current_word_start_idx = counter
#         current_word_end_idx = counter + num_subwords

#         if current_word_end_idx > all_hidden_states.shape[1]:
#             inputs_truncated = True
#             break

#         final_hidden_states[:, len(detokenized), :] = aggregate_repr(all_hidden_states, current_word_start_idx, current_word_end_idx - 1, aggregation)
        
#         # --- REVERTED to original "".join() method ---
#         detokenized.append("".join(segmented_tokens[current_word_start_idx:current_word_end_idx]))
#         counter += num_subwords

#     print("Detokenized (%03d): %s" % (len(detokenized), detokenized))
#     print("===================================================================")
    
#     return final_hidden_states[:, :len(detokenized), :], detokenized

# def extract_representations(
#     model_desc, input_corpus, output_file, device="cpu", aggregation="last", output_type="json",
#     random_weights=False, ignore_embeddings=False, decompose_layers=False, filter_layers=None,
#     dtype="float32", include_special_tokens=False,
#     prompt_path=None, batch_size=32, num_workers=0
# ):
#     model, tokenizer = get_model_and_tokenizer(model_desc, device=device, random_weights=random_weights)
#     is_decoder = any(m_name in model_desc.lower() for m_name in ['gpt', 'mistral', 'llama'])
    
#     prompt_text = ""
#     if prompt_path:
#         with open(prompt_path, "r") as f:
#             prompt_text = f.read()

#     dataset = TextDataset(input_corpus)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     writer = ActivationsWriter.get_writer(output_file, filetype=output_type, decompose_layers=decompose_layers, filter_layers=filter_layers, dtype=dtype)
#     tokenization_counts = {}
    
#     with torch.no_grad():
#         for sentence_idx_offset, batch_of_sentences in enumerate(tqdm(data_loader, desc="Processing Batches")):
            
#             if '{review_text_goes_here}' in prompt_text:
#                 full_texts = [prompt_text.format(review_text_goes_here=s) for s in batch_of_sentences]
#             else:
#                 full_texts = [prompt_text + " " + s for s in batch_of_sentences]

#             tokenized_batch = tokenizer(full_texts, padding=True, truncation=True, return_tensors='pt').to(device)
            
#             outputs = model(**tokenized_batch)
#             all_hidden_states_batch = outputs.hidden_states
            
#             for i in range(len(full_texts)):
#                 full_text = full_texts[i]
#                 actual_len = tokenized_batch['attention_mask'][i].sum()
#                 sentence_hidden_states = [layer[i, :actual_len, :] for layer in all_hidden_states_batch]

#                 activations_np = np.array([s.cpu().numpy() for s in sentence_hidden_states])
#                 if ignore_embeddings:
#                     activations_np = activations_np[1:]

#                 final_hidden_states, extracted_words = perform_subword_aggregation(
#                     full_text, activations_np, tokenizer, aggregation, dtype, include_special_tokens, tokenization_counts
#                 )
                
#                 if final_hidden_states is not None and extracted_words and final_hidden_states.shape[1] > 0:
#                     writer.write_activations(sentence_idx_offset * batch_size + i, extracted_words, final_hidden_states)
#     writer.close()

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("model_desc")
#     parser.add_argument("input_corpus")
#     parser.add_argument("output_file")
#     parser.add_argument("--aggregation", default="last")
#     parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
#     parser.add_argument("--disable_cuda", action="store_true")
#     parser.add_argument("--ignore_embeddings", action="store_true")
#     parser.add_argument("--random_weights", action="store_true")
#     parser.add_argument("--include_special_tokens", action="store_true")
#     ActivationsWriter.add_writer_options(parser)
#     parser.add_argument("--prompt_path", type=str, default=None)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--num_workers", type=int, default=0)
#     args = parser.parse_args()

#     device = torch.device("cpu") if args.disable_cuda or not torch.cuda.is_available() else torch.device("cuda")
#     extract_representations(
#         args.model_desc, args.input_corpus, args.output_file, device=device, aggregation=args.aggregation,
#         output_type=args.output_type, random_weights=args.random_weights, ignore_embeddings=args.ignore_embeddings,
#         dtype=args.dtype, decompose_layers=args.decompose_layers, filter_layers=args.filter_layers,
#         include_special_tokens=args.include_special_tokens, prompt_path=args.prompt_path, 
#         batch_size=args.batch_size, num_workers=args.num_workers
#     )

# if __name__ == "__main__":
#     main()




"""Representations Extractor for ``transformers`` toolkit models.

Module that given a file with input sentences and a ``transformers``
model, extracts representations from all layers of the model. The script
supports aggregation over sub-words created due to the tokenization of
the provided model.

Can also be invoked as a script as follows:
    ``python -m neurox.data.extraction.transformers_extractor``
"""

import argparse
import csv
import sys

import numpy as np
import torch

from NeuroX.neurox.data.writer import ActivationsWriter

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def get_model_and_tokenizer(model_desc, device="cpu", random_weights=False):
    """
    Automatically get the appropriate ``transformers`` model and tokenizer based
    on the model description

    Parameters
    ----------
    model_desc : str
        Model description; can either be a model name like ``bert-base-uncased``,
        a comma separated list indicating <model>,<tokenizer> (since 1.0.8),
        or a path to a trained model

    device : str, optional
        Device to load the model on, cpu or gpu. Default is cpu.

    random_weights : bool, optional
        Whether the weights of the model should be randomized. Useful for analyses
        where one needs an untrained model.

    Returns
    -------
    model : transformers model
        An instance of one of the transformers.modeling classes
    tokenizer : transformers tokenizer
        An instance of one of the transformers.tokenization classes
    """
    model_desc = model_desc.split(",")
    if len(model_desc) == 1:
        model_name = model_desc[0]
        tokenizer_name = model_desc[0]
    else:
        model_name = model_desc[0]
        tokenizer_name = model_desc[1]
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if random_weights:
        print("Randomizing weights")
        model.init_weights()

    return model, tokenizer


def aggregate_repr(state, start, end, aggregation):
    """
    Function that aggregates activations/embeddings over a span of subword tokens.
    This function will usually be called once per word. For example, if we had the sentence::

        This is an example

    which is tokenized by BPE into::

        this is an ex @@am @@ple

    The function should be called 4 times::

        aggregate_repr(state, 0, 0, aggregation)
        aggregate_repr(state, 1, 1, aggregation)
        aggregate_repr(state, 2, 2, aggregation)
        aggregate_repr(state, 3, 5, aggregation)

    Returns a zero vector if end is less than start, i.e. the request is to
    aggregate over an empty slice.

    Parameters
    ----------
    state : numpy.ndarray
        Matrix of size [ NUM_LAYERS x NUM_SUBWORD_TOKENS_IN_SENT x LAYER_DIM]
    start : int
        Index of the first subword of the word being processed
    end : int
        Index of the last subword of the word being processed
    aggregation : {'first', 'last', 'average'}
        Aggregation method for combining subword activations

    Returns
    -------
    word_vector : numpy.ndarray
        Matrix of size [NUM_LAYERS x LAYER_DIM]
    """
    if end < start:
        sys.stderr.write(
            "WARNING: An empty slice of tokens was encountered. "
            + "This probably implies a special unicode character or text "
            + "encoding issue in your original data that was dropped by the "
            + "transformer model's tokenizer.\n"
        )
        return np.zeros((state.shape[0], state.shape[2]))
    if aggregation == "first":
        return state[:, start, :]
    elif aggregation == "last":
        return state[:, end, :]
    elif aggregation == "average":
        return np.average(state[:, start : end + 1, :], axis=1)


def extract_sentence_representations(
    sentence,
    model,
    tokenizer,
    device="cpu",
    include_embeddings=True,
    aggregation="last",
    dtype="float32",
    include_special_tokens=False,
    tokenization_counts={},
    input_type="text"
):
    """
    Get representations for a single sentence

    The extractor runs a detokenization procedure to combine subwords
    automatically. For instance, a sentence "Hello, how are you?" may be
    tokenized by the model as "Hell @@o , how are you @@?". This extractor
    automatically detokenizes the subtokens back into the original token.


    Parameters
    ----------
    sentence : str
        Sentence for which the extraction needs to be done. The returned output
        will have representations for exactly the same number of elements as
        tokens in this sentence (counted by `sentence.split(' ')`).

    model : transformers model
        An instance of one of the transformers.modeling classes

    tokenizer : transformers tokenizer
        An instance of one of the transformers.tokenization classes

    device : str, optional
        Specifies the device (CPU/GPU) on which the extraction should be
        performed. Defaults to 'cpu'

    include_embeddings : bool, optional
        Whether the embedding layer should be included in the final output, or
        just regular layers. Defaults to True

    aggregation : {'first', 'last', 'average'}, optional
        Aggregation method for combining subword activations. Defaults to 'last'

    dtype : str, optional
        Data type in which the activations will be stored. Supports all numpy
        based tensor types. Common values are 'float32' and 'float16'. Defaults
        to 'float16'

    include_special_tokens : bool, optional
        Whether or not to special tokens in the extracted representations.
        Special tokens are tokens not present in the original sentence, but are
        added by the tokenizer, such as [CLS], [SEP] etc.

    tokenization_counts : dict, optional
        Tokenization counts to use across a dataset for efficiency

    Returns
    -------
    final_hidden_states : numpy.ndarray
        Numpy Matrix of size [``NUM_LAYERs`` x ``NUM_TOKENS`` x ``NUM_NEURONS``].

    detokenizer : list
        List of detokenized words. This will have the same number of elements as
        tokens in the original sentence, plus special tokens if requested. Each element
        preserves tokenization artifacts (such as `##`, `@@` etc) to enable further
        automatic processing.
    """

    special_tokens = [
        x for x in tokenizer.all_special_tokens if x != tokenizer.unk_token
    ]
    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    print("special_tokens:", special_tokens)
    print("special_tokens_ids:", special_tokens_ids)

    if input_type == "text":
        original_tokens = sentence.split(" ")

        # Add letters and spaces around each word since some tokenizers are context sensitive
        tmp_tokens = []
        if len(original_tokens) > 0:
            tmp_tokens.append(f"{original_tokens[0]} a")
        tmp_tokens += [f"a {x} a" for x in original_tokens[1:-1]]
        if len(original_tokens) > 1:
            tmp_tokens.append(f"a {original_tokens[-1]}")

        sentence = (sentence, )
        original_tokens = (original_tokens, )
        tmp_tokens = (tmp_tokens, )
    elif input_type == "tsv":
        original_sentence_1, original_sentence_2 = next(csv.reader([sentence], delimiter="\t", quotechar='"'))
        sentence = (original_sentence_1, original_sentence_2)
        original_tokens_1 = original_sentence_1.split(" ")
        original_tokens_2 = original_sentence_2.split(" ")

        # Add letters and spaces around each word since some tokenizers are context sensitive
        tmp_tokens_1 = []

        if len(original_tokens_1) > 0:
            tmp_tokens_1.append(f"{original_tokens_1[0]} a")
        tmp_tokens_1 += [f"a {x} a" for x in original_tokens_1[1:-1]]
        if len(original_tokens_1) > 1:
            tmp_tokens_1.append(f"a {original_tokens_1[-1]}")

        # TODO: modularize the token context anchors
        # TODO: Make sure all tokenizers support multi-sentence
        tmp_tokens_2 = []
        if len(original_tokens_2) > 0:
            tmp_tokens_2.append(f"{original_tokens_2[0]} a")
        tmp_tokens_2 += [f"a {x} a" for x in original_tokens_2[1:-1]]
        if len(original_tokens_2) > 1:
            tmp_tokens_2.append(f"a {original_tokens_2[-1]}")

        original_tokens = (original_tokens_1, original_tokens_2)
        tmp_tokens = (tmp_tokens_1, tmp_tokens_2)

    for original_tokens_i, tmp_tokens_i in zip(original_tokens, tmp_tokens):
        assert len(original_tokens_i) == len(
            tmp_tokens_i
        ), f"Original: {original_tokens_i}, Temp: {tmp_tokens_i}"

    with torch.no_grad():
        # Get tokenization counts if not already available
        for tmp_tokens_i in tmp_tokens:
            for token_idx, token in enumerate(tmp_tokens_i):
                tok_ids = [
                    x for x in tokenizer.encode(token) if x not in special_tokens_ids
                ]
                # Ignore the added letter tokens
                if token_idx != 0 and token_idx != len(tmp_tokens_i) - 1:
                    # Word appearing in the middle of the sentence
                    tok_ids = tok_ids[1:-1]
                elif token_idx == 0:
                    # Word appearing at the beginning
                    tok_ids = tok_ids[:-1]
                else:
                    # Word appearing at the end
                    tok_ids = tok_ids[1:]

                if token in tokenization_counts:
                    assert tokenization_counts[token] == len(
                        tok_ids
                    ), "Got different tokenization for already processed word " + token + " " + str(len(tok_ids))
                else:
                    tokenization_counts[token] = len(tok_ids)
        ids = tokenizer.encode(*sentence, truncation=True)
        print("ids:", ids)
        print("ids tokens:", tokenizer.convert_ids_to_tokens(ids))

        input_ids = torch.tensor([ids]).to(device)
        # Hugging Face format: tuple of torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
        # Tuple has 13 elements for base model: embedding outputs + hidden states at each layer
        all_hidden_states = model(input_ids)[-1]

        if include_embeddings:
            all_hidden_states = [
                hidden_states[0].cpu().numpy() for hidden_states in all_hidden_states
            ]
        else:
            all_hidden_states = [
                hidden_states[0].cpu().numpy()
                for hidden_states in all_hidden_states[1:]
            ]
        all_hidden_states = np.array(all_hidden_states, dtype=dtype)

    sentence = "\t".join(sentence)
    original_tokens = [token for subtokens in original_tokens for token in subtokens]
    tmp_tokens = [token for subtokens in tmp_tokens for token in subtokens]

    print('Sentence         : "%s"' % (sentence))
    print("Original    (%03d): %s" % (len(original_tokens), original_tokens))
    print(
        "Tokenized   (%03d): %s"
        % (
            len(tokenizer.convert_ids_to_tokens(ids)),
            tokenizer.convert_ids_to_tokens(ids),
        )
    )

    assert all_hidden_states.shape[1] == len(ids)

    # Handle special tokens
    # filtered_ids will contain all ids if we are extracting with
    #  special tokens, and only normal word/subword ids if we are
    #  extracting without special tokens
    # all_hidden_states will also be filtered at this step to match
    #  the ids in filtered ids
    filtered_ids = ids
    idx_special_tokens = [t_i for t_i, x in enumerate(ids) if x in special_tokens_ids]
    special_token_ids = [ids[t_i] for t_i in idx_special_tokens]

    if not include_special_tokens:
        idx_without_special_tokens = [
            t_i for t_i, x in enumerate(ids) if x not in special_tokens_ids
        ]
        filtered_ids = [ids[t_i] for t_i in idx_without_special_tokens]
        all_hidden_states = all_hidden_states[:, idx_without_special_tokens, :]
        special_token_ids = []

    assert all_hidden_states.shape[1] == len(filtered_ids)
    print(
        "Filtered   (%03d): %s"
        % (
            len(tokenizer.convert_ids_to_tokens(filtered_ids)),
            tokenizer.convert_ids_to_tokens(filtered_ids),
        )
    )

    # Get actual tokens for filtered ids in order to do subword
    #  aggregation
    segmented_tokens = tokenizer.convert_ids_to_tokens(filtered_ids)

    # Perform subword aggregation/detokenization
    #  After aggregation, we should have |original_tokens| embeddings,
    #  one for each word. If special tokens are included, then we will
    #  have |original_tokens| + |special_tokens|
    counter = 0
    detokenized = []
    final_hidden_states = np.zeros(
        (
            all_hidden_states.shape[0],
            len(original_tokens) + len(special_token_ids),
            all_hidden_states.shape[2],
        ),
        dtype=dtype,
    )
    inputs_truncated = False

    # Keep track of what the previous token was. This is used to detect
    #  special tokens followed/preceeded by dropped tokens, which is an
    #  ambiguous situation for the detokenizer
    prev_token_type = "NONE"

    last_special_token_pointer = 0
    for token_idx, token in enumerate(tmp_tokens):
        # Handle special tokens
        if include_special_tokens and tokenization_counts[token] != 0:
            if last_special_token_pointer < len(idx_special_tokens):
                while (
                    last_special_token_pointer < len(idx_special_tokens)
                    and counter == idx_special_tokens[last_special_token_pointer]
                ):
                    assert prev_token_type != "DROPPED", (
                        "A token dropped by the tokenizer appeared next "
                        + "to a special token. Detokenizer cannot resolve "
                        + f"the ambiguity, please remove '{sentence}' from"
                        + "the dataset, or try a different tokenizer"
                    )
                    prev_token_type = "SPECIAL"
                    final_hidden_states[:, len(detokenized), :] = all_hidden_states[
                        :, counter, :
                    ]
                    detokenized.append(
                        segmented_tokens[idx_special_tokens[last_special_token_pointer]]
                    )
                    last_special_token_pointer += 1
                    counter += 1

        current_word_start_idx = counter
        current_word_end_idx = counter + tokenization_counts[token]

        # Check for truncated hidden states in the case where the
        # original word was actually tokenized
        if (
            tokenization_counts[token] != 0
            and current_word_start_idx >= all_hidden_states.shape[1]
        ) or current_word_end_idx > all_hidden_states.shape[1]:
            final_hidden_states = final_hidden_states[
                :,
                : len(detokenized)
                + len(special_token_ids)
                - last_special_token_pointer,
                :,
            ]
            inputs_truncated = True
            break

        if tokenization_counts[token] == 0:
            assert prev_token_type != "SPECIAL", (
                "A token dropped by the tokenizer appeared next "
                + "to a special token. Detokenizer cannot resolve "
                + f"the ambiguity, please remove '{sentence}' from"
                + "the dataset, or try a different tokenizer"
            )
            prev_token_type = "DROPPED"
        else:
            prev_token_type = "NORMAL"

        final_hidden_states[:, len(detokenized), :] = aggregate_repr(
            all_hidden_states,
            current_word_start_idx,
            current_word_end_idx - 1,
            aggregation,
        )
        detokenized.append(
            "".join(segmented_tokens[current_word_start_idx:current_word_end_idx])
        )
        counter += tokenization_counts[token]

    if include_special_tokens:
        while counter < len(segmented_tokens):
            if last_special_token_pointer >= len(idx_special_tokens):
                break

            if counter == idx_special_tokens[last_special_token_pointer]:
                assert prev_token_type != "DROPPED", (
                    "A token dropped by the tokenizer appeared next "
                    + "to a special token. Detokenizer cannot resolve "
                    + f"the ambiguity, please remove '{sentence}' from"
                    + "the dataset, or try a different tokenizer"
                )
                prev_token_type = "SPECIAL"
                final_hidden_states[:, len(detokenized), :] = all_hidden_states[
                    :, counter, :
                ]
                detokenized.append(
                    segmented_tokens[idx_special_tokens[last_special_token_pointer]]
                )
                last_special_token_pointer += 1
            counter += 1

    print("Detokenized (%03d): %s" % (len(detokenized), detokenized))
    print("Counter: %d" % (counter))
    print(1)

    if inputs_truncated:
        print("WARNING: Input truncated because of length, skipping check")
    else:
        print()
        print('detokenized_length:', len(detokenized))
        print('detokenized:', detokenized)
        print('original_tokens_length:', len(original_tokens))
        print('original_tokens:', original_tokens)
        print('special_token_ids_length:', len(special_token_ids))
        print('special_token_ids:', special_token_ids)

        # convert special_token_ids to token
        print(tokenizer.convert_ids_to_tokens(special_token_ids))

        assert counter == len(filtered_ids)
        assert len(detokenized) == len(original_tokens) + len(special_token_ids)
    print("===================================================================")
    return final_hidden_states, detokenized


def extract_representations(
    model_desc,
    input_corpus,
    output_file,
    device="cpu",
    aggregation="last",
    output_type="json",
    random_weights=False,
    ignore_embeddings=False,
    decompose_layers=False,
    filter_layers=None,
    dtype="float32",
    include_special_tokens=False,
    input_type="text"
):
    """
    Extract representations for an entire corpus and save them to disk

    Parameters
    ----------
    model_desc : str
        Model description; can either be a model name like ``bert-base-uncased``,
        a comma separated list indicating <model>,<tokenizer> (since 1.0.8),
        or a path to a trained model

    input_corpus : str
        Path to the input corpus, where each sentence is on its separate line

    output_file : str
        Path to output file. Supports all filetypes supported by
        ``data.writer.ActivationsWriter``.

    device : str, optional
        Specifies the device (CPU/GPU) on which the extraction should be
        performed. Defaults to 'cpu'

    aggregation : {'first', 'last', 'average'}, optional
        Aggregation method for combining subword activations. Defaults to 'last'

    output_type : str, optional
        Explicit definition of output file type if it cannot be derived from the
        ``output_file`` path

    random_weights : bool, optional
        Whether the weights of the model should be randomized. Useful for analyses
        where one needs an untrained model. Defaults to False.

    ignore_embeddings : bool, optional
        Whether the embedding layer should be excluded in the final output, or
        kept with the regular layers. Defaults to False

    decompose_layers : bool, optional
        Whether each layer should have it's own output file, or all layers be saved
        in a single file. Defaults to False, i.e. single file

    filter_layers : str
        Comma separated list of layer indices to save. The format is the same as
        the one accepted by ``data.writer.ActivationsWriter``.

    dtype : str, optional
        Data type in which the activations will be stored. Supports all numpy
        based tensor types. Common values are 'float32' and 'float16'. Defaults
        to 'float16'

    include_special_tokens : bool, optional
        Whether or not to special tokens in the extracted representations.
        Special tokens are tokens not present in the original sentence, but are
        added by the tokenizer, such as [CLS], [SEP] etc.
    """
    print(f"Loading model: {model_desc}")
    model, tokenizer = get_model_and_tokenizer(
        model_desc, device=device, random_weights=random_weights
    )

    print("Reading input corpus")

    def corpus_generator(input_corpus_path):
        with open(input_corpus_path, "r") as fp:
            for line in fp:
                yield line.strip()
            return

    print("Preparing output file")
    writer = ActivationsWriter.get_writer(
        output_file,
        filetype=output_type,
        decompose_layers=decompose_layers,
        filter_layers=filter_layers,
        dtype=dtype,
    )

    print("Extracting representations from model")
    tokenization_counts = {}  # Cache for tokenizer rules
    for sentence_idx, sentence in enumerate(corpus_generator(input_corpus)):
        hidden_states, extracted_words = extract_sentence_representations(
            sentence,
            model,
            tokenizer,
            device=device,
            include_embeddings=(not ignore_embeddings),
            aggregation=aggregation,
            dtype=dtype,
            include_special_tokens=include_special_tokens,
            tokenization_counts=tokenization_counts,
            input_type=input_type,
        )

        print("Hidden states: ", hidden_states.shape)
        print("# Extracted words: ", len(extracted_words))

        writer.write_activations(sentence_idx, extracted_words, hidden_states)

    writer.close()


HDF5_SPECIAL_TOKENS = {".": "__DOT__", "/": "__SLASH__"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_desc", help="Name of model")
    parser.add_argument(
        "input_corpus", help="Text file path with one sentence per line"
    )
    parser.add_argument(
        "output_file",
        help="Output file path where extracted representations will be stored",
    )
    parser.add_argument(
        "--aggregation",
        help="first, last or average aggregation for word representation in the case of subword segmentation",
        default="last",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float32",
        help="Output dtype of the extracted representations",
    )
    parser.add_argument("--disable_cuda", action="store_true")
    parser.add_argument("--ignore_embeddings", action="store_true")
    parser.add_argument(
        "--random_weights",
        action="store_true",
        help="generate representations from randomly initialized model",
    )
    parser.add_argument(
        "--include_special_tokens",
        action="store_true",
        help="Include special tokens like [CLS] and [SEP] in the extracted representations",
    )
    parser.add_argument(
        "--input_type",
        choices=["text", "tsv"],
        help="Format of the input file, use tsv for multi-sentence inputs",
        default="text",
    )

    ActivationsWriter.add_writer_options(parser)

    args = parser.parse_args()

    assert args.aggregation in [
        "average",
        "first",
        "last",
    ], "Invalid aggregation option, please specify first, average or last."

    assert not (
        args.filter_layers is not None and args.ignore_embeddings is True
    ), "--filter_layers and --ignore_embeddings cannot be used at the same time"

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    extract_representations(
        args.model_desc,
        args.input_corpus,
        args.output_file,
        device=device,
        aggregation=args.aggregation,
        output_type=args.output_type,
        random_weights=args.random_weights,
        ignore_embeddings=args.ignore_embeddings,
        dtype=args.dtype,
        decompose_layers=args.decompose_layers,
        filter_layers=args.filter_layers,
        include_special_tokens=args.include_special_tokens,
        input_type=args.input_type
    )


if __name__ == "__main__":
    main()
