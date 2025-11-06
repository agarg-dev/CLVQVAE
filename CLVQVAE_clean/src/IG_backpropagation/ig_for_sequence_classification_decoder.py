import torch
import argparse
import pandas as pd
from captum.attr import LayerIntegratedGradients
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import torch.nn as nn

class ModelWrapper(nn.Module):
    """
    A simple wrapper for a Causal LM to make it compatible with Captum's
    Integrated Gradients, which requires the model to be a `nn.Module`.
    """
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):
        """
        Performs a forward pass and returns the logits for the last token.
        This is necessary for explaining the model's single-token classification output.
        """
        logits = self.model(input_ids, attention_mask=attention_mask).logits
        return logits[:, -1, :]

class ModelIGExplainer:
    def __init__(self, model_path, device=None):
        # Determine if the model is a Qwen model to set appropriate flags
        is_qwen = 'qwen' in model_path.lower()

        # Qwen models often require trusting remote code for both model and tokenizer
        model_kwargs = {'trust_remote_code': True} if is_qwen else {}
        tokenizer_kwargs = {'trust_remote_code': True} if is_qwen else {}

        # Use AutoModelForCausalLM for broader compatibility (works for Mistral, Qwen, etc.)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.init_explainer()




    def init_explainer(self, layer=0):
        self.wrapped_model = ModelWrapper(self.model)
        if layer == 0:
            target_layer = self.model.model.embed_tokens
        else:
            target_layer = self.model.model.layers[layer - 1]

        self.interpreter = LayerIntegratedGradients(
            forward_func=self.wrapped_model,
            layer=target_layer
        )

    def interpret(self, text, valid_labels, n_steps=30):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        target_token_ids = [self.tokenizer.convert_tokens_to_ids(lbl) for lbl in valid_labels]
        target_id_to_label_map = {tid: lbl for tid, lbl in zip(target_token_ids, valid_labels)}

        predicted_class_str = None
        ig_target_id = None

        with torch.no_grad():
            last_token_logits = self.wrapped_model(input_ids, attention_mask=inputs['attention_mask'])
            top_token_id = torch.argmax(last_token_logits, dim=-1).item()
            top_token_str = self.tokenizer.decode(top_token_id)

            # --- CONDITIONAL LOGIC ---
            # Condition: If the top prediction is a space or empty...
            if not top_token_str.strip():
                print(" First token was a space. Generating a second token.")
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=2,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                generated_ids = generated[0][input_len:]
                
                # Check if a second token was actually generated
                if len(generated_ids) > 1:
                    second_token_id = generated_ids[1].item()
                    second_token_decoded = self.tokenizer.decode(second_token_id)
                    print(f" Second token is: '{second_token_decoded.strip()}'") 
                    
                    if second_token_id in target_id_to_label_map:
                        ig_target_id = second_token_id
                        predicted_class_str = target_id_to_label_map[second_token_id]
                else:
                    generated_decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    print(f" Generation stopped early. Generated: '{generated_decoded}'")

            # Condition: If the top prediction was NOT a space, check if it's a valid label.
            elif top_token_id in target_id_to_label_map:
                ig_target_id = top_token_id
                predicted_class_str = target_id_to_label_map[top_token_id]
        
        # Fallback: If neither of the above conditions found a valid label.
        if ig_target_id is None:
            top_pred_decoded = self.tokenizer.decode(top_token_id, skip_special_tokens=True).strip()
            print(f"⚠️ Could not find a valid label. Top prediction was '{top_pred_decoded}'.")
            
            target_logits = last_token_logits[0, target_token_ids]
            most_probable_target_idx = torch.argmax(target_logits).item()
            ig_target_id = target_token_ids[most_probable_target_idx]
            predicted_class_str = target_id_to_label_map[ig_target_id]
            print(f"⚠️ Explaining class '{predicted_class_str}' instead.")

        # --- The Explanation Step ---
        #if model is mistral, use bos_token_id else use pad_token_id
        if 'mistral' in self.model.config._name_or_path.lower():
            baseline_ids = torch.full_like(input_ids, self.tokenizer.bos_token_id)
        else:
            baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)

        attributions, _ = self.interpreter.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(inputs["attention_mask"],),
            target=ig_target_id,
            n_steps=n_steps,
            return_convergence_delta=True,
            internal_batch_size=5
        )

        return self._process_attributions(attributions, input_ids), predicted_class_str

    def _process_attributions(self, attributions, input_ids):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        return list(zip(tokens, attributions.tolist()))

    def aggregate_repr(self, state, start_idx, end_idx, aggregation, token):
        if end_idx < start_idx:
            sys.stderr.write("WARNING: empty token slides\n")
            return (token, 0.0)

        attributions = [state[idx][1] for idx in range(start_idx, end_idx + 1)]

        if aggregation == "first":
            return (token, attributions[0])
        elif aggregation == "last":
            return (token, attributions[-1])
        elif aggregation == "average":
            return (token, sum(attributions) / len(attributions))
        elif aggregation == "max":
            return (token, max(attributions))
        else:
            raise ValueError(f"不支持的聚合方法: {aggregation}")



    def detokenize_explanation(self, sentence, tokenized_explanation, aggregation="max"):
        assert aggregation in ["max", "average", "first", "last"]

        original_tokens = sentence.split()
        special_tokens = self.tokenizer.all_special_tokens
        special_tokens_ids = self.tokenizer.convert_tokens_to_ids(special_tokens)

        tokenization_map = {}
        for token in original_tokens:
            token_ids = [
                tid for tid in self.tokenizer.encode(token, add_special_tokens=False)
                if tid not in special_tokens_ids
            ]
            # --- CHANGE 1: INITIALIZE WITH A LIST TO STORE POSITIONS ---
            if token not in tokenization_map:
                tokenization_map[token] = {
                    'count': len(token_ids),
                    'positions': [] # Instead of 'start' and 'end'
                }

        full_ids = self.tokenizer.encode(sentence)
        full_tokens = self.tokenizer.convert_ids_to_tokens(full_ids)

        ptr = 0
        word_ptr = 0
        while ptr < len(full_tokens) and word_ptr < len(original_tokens):
            current_token = original_tokens[word_ptr]
            if full_tokens[ptr] in special_tokens:
                ptr += 1
                continue

            start = ptr
            expected_count = tokenization_map[current_token]['count']
            end = start + expected_count - 1
            if end >= len(full_tokens):
                end = len(full_tokens) - 1
                sys.stderr.write(f"WARNING: not matching: {current_token}\n")

            # --- CHANGE 2: APPEND POSITIONS INSTEAD OF OVERWRITING ---
            tokenization_map[current_token]['positions'].append({'start': start, 'end': end})
            
            ptr = end + 1
            word_ptr += 1

        detokenized = []
        # --- CHANGE 3: KEEP TRACK OF WHICH POSITION TO USE FOR EACH WORD ---
        map_indices = {token: 0 for token in original_tokens}

        for token in original_tokens:
            # Check if we have any positions recorded for this token
            if not tokenization_map.get(token) or not tokenization_map[token]['positions']:
                detokenized.append((token, 0.0))
                continue

            current_idx = map_indices[token]
            
            # Make sure we don't go out of bounds (for safety)
            if current_idx >= len(tokenization_map[token]['positions']):
                detokenized.append((token, 0.0))
                continue

            pos = tokenization_map[token]['positions'][current_idx]
            start = pos['start']
            end = pos['end']
            
            merged = self.aggregate_repr(tokenized_explanation, start, end, aggregation, token)
            detokenized.append(merged)

            # Increment the index for the next time we see this token
            map_indices[token] += 1
            
        return detokenized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mistral Integrated Gradients Explainer")
    parser.add_argument("input_file", help="Path to input text file")
    parser.add_argument("model_path", help="Path to Mistral model")
    parser.add_argument("layer", type=int, help="Layer index (0 for embeddings)")
    parser.add_argument("save_file", help="Path to save output results")
    parser.add_argument("dataset_name", choices=['eraser-movie', 'jigsaw', 'agnews'], help="The name of the dataset which determines the output labels.")
    args = parser.parse_args()

    TARGET_LABELS = {
        'eraser-movie': ['0', '1'],
        'jigsaw': ['0', '1'],
        'agnews': ['0', '1', '2', '3']
    }
    
    valid_labels = TARGET_LABELS[args.dataset_name]
    print(f"✅ Configured for dataset '{args.dataset_name}' with labels: {valid_labels}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    explainer = ModelIGExplainer(
        model_path=args.model_path,
        device=device
    )
    explainer.init_explainer(layer=args.layer)

    df = pd.DataFrame(columns=["sentence_id", "predicted_class", "saliencies"])

    senten_idx = []
    prediction = []
    all_saliencies = []
    with open(args.input_file) as fp:
        for sentence_idx, line in enumerate(fp):
            sentence = line.strip()
            if not sentence:
                continue

            result = explainer.interpret(sentence, valid_labels=valid_labels, n_steps=100)

            if result is None:
                continue

            raw_attributions, label = result
            word_attributions = explainer.detokenize_explanation(sentence, raw_attributions, aggregation="max")
            saliencies = [(w, abs(s)) for w, s in word_attributions]
            print(f"Saliencies: {saliencies}")

            senten_idx.append(sentence_idx)
            prediction.append(label)
            all_saliencies.append(saliencies)

    df["sentence_id"] = senten_idx
    df["predicted_class"] = prediction
    df["saliencies"] = all_saliencies

    df.to_csv(args.save_file, index=False)
    print(f"results saving: {args.save_file}")