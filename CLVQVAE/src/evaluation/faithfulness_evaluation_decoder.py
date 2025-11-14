import argparse
import json
import os
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import defaultdict

# --- UTILITIES ---
def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset(file_name):
    """Load dataset from a text file."""
    with open(file_name, 'r') as f:
        data = f.readlines()
    return [line.strip() for line in data]

def load_codebook_vectors(file_path):
    """Load the codebook vectors from the PyTorch file."""
    return torch.load(file_path)

def load_ground_truth_labels(file_path):
    """Load ground truth labels and convert strings to numeric indices if needed."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    labels = [item.get('label') for item in data if 'label' in item and item.get('label') is not None]
    if labels and isinstance(labels[0], str):
        unique_labels = sorted(set(labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        labels = [label_map[label] for label in labels]
    return labels

def load_merged_explanation(file_path):
    """Load the token to index mapping from the CSV file, ensuring correct types."""
    df = pd.read_csv(file_path, sep=",", engine="python", on_bad_lines="skip")
    df = df.dropna(subset=["sentence_index", "position_index", "vector_idx", "token"])
    df["sentence_index"] = df["sentence_index"].astype(int)
    df["position_index"] = df["position_index"].astype(int)
    df["vector_idx"] = df["vector_idx"].astype(int)
    return df

# --- EMBEDDING LOADING LOGIC ---
def load_last_token_embeddings_from_json(json_path, sentence_ids_to_load):
    """
    Correctly loads pre-extracted embeddings for the LAST token of specified sentences.
    """
    print(f"Loading pre-extracted embeddings for LAST tokens from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    embedding_map, last_token_indices = {}, {}
    for token_info, embedding in data:
        parts = token_info.split("|||")
        sent_idx, word_idx = int(parts[2]), int(parts[3])
        embedding_map[(sent_idx, word_idx)] = embedding
        if sent_idx not in last_token_indices or word_idx > last_token_indices[sent_idx]:
            last_token_indices[sent_idx] = word_idx

    final_embeddings, final_sentence_ids = [], []
    for sent_idx in sentence_ids_to_load:
        if sent_idx in last_token_indices:
            last_idx = last_token_indices[sent_idx]
            final_embeddings.append(embedding_map[(sent_idx, last_idx)])
            final_sentence_ids.append(sent_idx)
        else:
            print(f"Warning: No embeddings found for sentence {sent_idx} in JSON file.")

    print(f"Successfully loaded last token embeddings for {len(final_embeddings)} sentences.")
    return np.array(final_embeddings), final_sentence_ids

def load_mean_token_embeddings_from_json(json_path, sentence_ids_to_load):
    """
    Loads pre-extracted embeddings and calculates the MEAN for specified sentences.
    """
    print(f"Loading pre-extracted embeddings for MEAN pooling from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    sentence_embedding_dict = defaultdict(list)
    for token_info, embedding in data:
        sent_idx = int(token_info.split("|||")[2])
        sentence_embedding_dict[sent_idx].append(np.array(embedding))

    final_embeddings, final_sentence_ids = [], []
    for sent_idx in sentence_ids_to_load:
        if sent_idx in sentence_embedding_dict:
            mean_embedding = np.stack(sentence_embedding_dict[sent_idx]).mean(axis=0)
            final_embeddings.append(mean_embedding)
            final_sentence_ids.append(sent_idx)
        else:
            print(f"Warning: No embeddings found for sentence {sent_idx} in JSON file.")
            
    print(f"Extracted mean-pooled token embeddings for {len(final_embeddings)} sentences.")
    return np.array(final_embeddings), final_sentence_ids

# --- DATA ALIGNMENT ---
def filter_data_by_sentence_ids(embeddings, labels, explanation_df, sentence_ids):
    """
    Ensures that embeddings, labels, and explanations are all aligned to the same set of sentence_ids.
    """
    print("Filtering and aligning data...")
    sid_to_idx_map = {sid: i for i, sid in enumerate(sentence_ids)}
    
    valid_sids = set(sentence_ids) & set(explanation_df['sentence_index'])
    valid_sids = {sid for sid in valid_sids if sid < len(labels)}
    
    final_indices = sorted([sid_to_idx_map[sid] for sid in valid_sids])
    final_sids = sorted(list(valid_sids))

    filtered_embeddings = embeddings[final_indices]
    filtered_labels = np.array([labels[sid] for sid in final_sids])
    filtered_explanation_df = explanation_df[explanation_df['sentence_index'].isin(final_sids)]
    
    assert len(filtered_embeddings) == len(filtered_labels) == len(final_sids)
    print(f"Data aligned. Final sample size: {len(final_sids)}")
    
    return filtered_embeddings, filtered_labels, filtered_explanation_df, final_sids

# --- PERTURBATION LOGIC ---
def project_orthogonal(vector, direction):
    """Project vector onto the subspace orthogonal to direction."""
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-10: return vector
    direction_unit = direction / direction_norm
    projection = np.dot(vector, direction_unit) * direction_unit
    return vector - projection

def perturb_salient_embeddings_orthogonal(embeddings, sentences, sentence_ids, merged_explanation_df, codebook_vectors):
    """Remove the concept direction from embeddings using orthogonal projection."""
    perturbed_embeddings = embeddings.copy()
    perturbed_indices = []
    explanation_map = {idx: group for idx, group in merged_explanation_df.groupby('sentence_index')}

    print("\n--- Verifying Perturbation Logic (showing first 3 examples) ---")
    for i, sentence_id in enumerate(sentence_ids):
        if sentence_id not in explanation_map:
            continue
            
        salient_token_df = explanation_map[sentence_id]
        codebook_idx = salient_token_df.iloc[0]['vector_idx']
        salient_codebook_vector = codebook_vectors[codebook_idx]

        if isinstance(salient_codebook_vector, torch.Tensor):
            salient_codebook_vector = salient_codebook_vector.cpu().numpy()
        elif isinstance(salient_codebook_vector, list):
            salient_codebook_vector = np.array(salient_codebook_vector)

        # This block will print detailed info for the first 3 valid sentences
        if i < 3:
            # Extract info for printing from available sources
            salient_token = salient_token_df.iloc[0]['token']
            salient_position = salient_token_df.iloc[0]['position_index']
            current_sentence_text = sentences[sentence_id]
            
            # Print the formatted block
            print("\n" + "="*80)
            print(f"Processing Item (Batch Index: {i}, Original Sentence ID: {sentence_id})")
            print(f"  Sentence: \"{current_sentence_text}\"")
            print(f"  > Salient Token (from explanation file): '{salient_token}' at position {salient_position}")
            print(f"  > Action: Perturbing the pre-loaded sentence embedding using the vector for '{salient_token}'")
            print(f"  > Codebook Vector Index: {codebook_idx}")
            print(f"    - Vector Shape: {salient_codebook_vector.shape}")
            print(f"    - Vector Preview: {salient_codebook_vector[:5]}")
            if i == 2: # After the 3rd example, print a footer and stop
                print("="*80)
                print("--- End of Verification Block ---\n")

        perturbed_embeddings[i] = project_orthogonal(perturbed_embeddings[i], salient_codebook_vector)
        perturbed_indices.append(i)
    
    print(f"Applied orthogonal projection to {len(perturbed_indices)} embeddings")
    return perturbed_embeddings, perturbed_indices

def perturb_salient_embeddings_orthogonal_random(embeddings, perturbed_indices, seed=42):
    """Remove a random concept direction from embeddings."""
    random.seed(seed)
    np.random.seed(seed)
    perturbed_embeddings = embeddings.copy()
    embedding_dim = embeddings.shape[1]
    for idx in perturbed_indices: 
        random_vector = np.random.randn(embedding_dim)
        perturbed_embeddings[idx] = project_orthogonal(perturbed_embeddings[idx], random_vector)
    
    print(f"Applied random concept direction removal to {len(perturbed_indices)} embeddings")
    return perturbed_embeddings


class SimpleClassifier(nn.Module):
    """A simple feed-forward neural network classifier."""
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super(SimpleClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.layer2(x)
        return x



def train_and_evaluate_model(train_embeddings, train_labels, other_embeddings_list, other_labels_list, 
                           embedding_names, n_folds=5, batch_size=32, epochs=10, lr=0.001, 
                           device="cuda", patience=3, min_delta=0.001, seed=42):
    """
    Train a model on train_embeddings and evaluate on multiple embedding types.
    
    Args:
        train_embeddings: The embeddings to train the model on
        train_labels: Labels for training
        other_embeddings_list: List of other embedding sets to evaluate
        other_labels_list: List of label sets corresponding to other_embeddings_list
        embedding_names: Names for reporting (first name is for train_embeddings)
        n_folds: Number of cross-validation folds
        
    Returns:
        Dictionary of results for each embedding type
    """
    # Convert numpy arrays to PyTorch tensors for training data
    train_embeddings_tensor = torch.FloatTensor(train_embeddings)
    train_labels_tensor = torch.LongTensor(train_labels)
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # We'll store results for each embedding type
    all_results = {name: {"fold_accuracies": [], "predictions": [], "true_labels": []} 
                   for name in embedding_names}
    
    fold_num = 0
    
    print(f"\nTraining with {n_folds}-fold cross-validation...")
    
    for train_idx, val_idx in cv.split(train_embeddings, train_labels):
        fold_num += 1
        
        # Split data according to current fold
        X_train, X_val = train_embeddings_tensor[train_idx], train_embeddings_tensor[val_idx]
        y_train, y_val = train_labels_tensor[train_idx], train_labels_tensor[val_idx]
        
        # Create data loaders for training data
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        other_val_data = []
        for other_embeddings, other_labels in zip(other_embeddings_list, other_labels_list):
            other_embeddings_tensor = torch.FloatTensor(other_embeddings)
            other_labels_tensor = torch.LongTensor(other_labels)
            
            # Use the same validation indices
            X_other_val = other_embeddings_tensor[val_idx]
            y_other_val = other_labels_tensor[val_idx]
            
            other_val_dataset = TensorDataset(X_other_val, y_other_val)
            other_val_loader = DataLoader(other_val_dataset, batch_size=batch_size)
            
            other_val_data.append(other_val_loader)
        
        # Initialize model
        input_dim = train_embeddings.shape[1]
        num_classes = len(np.unique(train_labels))
        model = SimpleClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        epochs_no_improve = 0
        last_epoch = 0
        
        for epoch in range(epochs):
            last_epoch = epoch + 1
            
            # Training phase
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase on original embeddings
            model.eval()
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    
                    val_preds.extend(predicted.cpu().numpy())
                    val_true.extend(targets.cpu().numpy())
            
            val_acc = accuracy_score(val_true, val_preds)
            
            # Save best model
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            # Early stopping check
            if epochs_no_improve >= patience:
                break
        
        # Load the best model for evaluation
        model.load_state_dict(best_model_state)
        model.eval()
        
        # Evaluate on validation set for each embedding type
        # First, the original train embeddings (validation portion)
        with torch.no_grad():
            val_preds = []
            val_true = []
            
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(targets.cpu().numpy())
            
            val_acc = accuracy_score(val_true, val_preds)
            all_results[embedding_names[0]]["fold_accuracies"].append(val_acc)
            all_results[embedding_names[0]]["predictions"].extend(val_preds)
            all_results[embedding_names[0]]["true_labels"].extend(val_true)
        
        # Now evaluate on other embedding types
        for i, other_val_loader in enumerate(other_val_data):
            other_preds = []
            other_true = []
            
            with torch.no_grad():
                for inputs, targets in other_val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    
                    other_preds.extend(predicted.cpu().numpy())
                    other_true.extend(targets.cpu().numpy())
            
            other_acc = accuracy_score(other_true, other_preds)
            all_results[embedding_names[i+1]]["fold_accuracies"].append(other_acc)
            all_results[embedding_names[i+1]]["predictions"].extend(other_preds)
            all_results[embedding_names[i+1]]["true_labels"].extend(other_true)
        
        # Print results for this fold
        fold_results = ", ".join([f"{name}: {all_results[name]['fold_accuracies'][-1]:.4f}" 
                                for name in embedding_names])
        print(f"Fold {fold_num}/{n_folds}: {fold_results} (after {last_epoch} epochs)")
    
    # Calculate cross-validation metrics for each embedding type
    for name in embedding_names:
        fold_accs = all_results[name]["fold_accuracies"]
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        
        all_results[name]["mean_accuracy"] = mean_acc
        all_results[name]["std_accuracy"] = std_acc
        
        print(f"\n{name} Results:")
        print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return all_results


# --- MAIN EXECUTION BLOCK ---
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate classifiers on perturbed embeddings")
    
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--merged-explanation-file', type=str, required=True)
    parser.add_argument('--codebook-vectors', type=str, required=True)
    parser.add_argument('--ground-truth-file', type=str, required=True)
    parser.add_argument('--eval-embedding', type=str, required=True, help='Path to the JSON file with pre-extracted embeddings.')
    parser.add_argument('--output-dir', type=str, default='classifier_results/')
    
    parser.add_argument('--model-name', type=str, required=True, help="Name of the model (for logging purposes).")
    parser.add_argument('--layer-idx', type=int, default=-1, help="Layer index used for embeddings (for logging purposes).")
    parser.add_argument('--ablation-method', type=str, default="last_token", choices=['last_token', 'mean'], help='Method to represent sentence embedding.')
    
    parser.add_argument('--batch-size', type=int, default=32)

    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Maximum number of epochs for training classifiers.')

    parser.add_argument('--patience',
                        type=int,
                        default=5,
                        help='Early stopping patience (default: 3)')
                        
    parser.add_argument('--min-delta',
                        type=float,
                        default=0.001,
                        help='Minimum improvement for early stopping (default: 0.001)')
    
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    parser.add_argument('--n-folds',
                        type=int,
                        default=20,
                        help='Number of folds for cross-validation (default: 5)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sentences = get_dataset(args.dataset_path)

    # 1. Load all necessary data files
    merged_explanation_df = load_merged_explanation(args.merged_explanation_file)
    codebook_vectors = load_codebook_vectors(args.codebook_vectors)
    ground_truth_labels = load_ground_truth_labels(args.ground_truth_file)
    
    sentence_ids_with_explanations = sorted(list(set(merged_explanation_df['sentence_index'])))
    print(f"Found explanations for {len(sentence_ids_with_explanations)} unique sentences.")

    # 2. Load pre-extracted embeddings
    if args.ablation_method == "last_token":
        original_embeddings, loaded_sids = load_last_token_embeddings_from_json(args.eval_embedding, sentence_ids_with_explanations)
    elif args.ablation_method == "mean":
        original_embeddings, loaded_sids = load_mean_token_embeddings_from_json(args.eval_embedding, sentence_ids_with_explanations)
    else:
        raise ValueError(f"Unknown ablation method: {args.ablation_method}")

    # 3. Filter all data to ensure perfect alignment
    original_embeddings, labels, merged_explanation_df, sentence_ids = filter_data_by_sentence_ids(
        original_embeddings, ground_truth_labels, merged_explanation_df, loaded_sids
    )

    # 4. Create perturbed embeddings
    print("\nCreating perturbed embeddings using orthogonal projection...")

    perturbed_embeddings, perturbed_indices = perturb_salient_embeddings_orthogonal(
        original_embeddings, sentences, sentence_ids, merged_explanation_df, codebook_vectors
    )
    
    print("\nCreating perturbed embeddings with random orthogonal projection...")
    random_perturbed_embeddings = perturb_salient_embeddings_orthogonal_random(
        original_embeddings, perturbed_indices, seed=args.seed
    )
    
    # 5. Train and evaluate the classifier
    embedding_names = ["Normal", "Perturbed", "Random Perturbed"]
    print("\nTraining model on original embeddings and evaluating on all embedding types:")
    results = train_and_evaluate_model(
        original_embeddings, labels, 
        [perturbed_embeddings, random_perturbed_embeddings],
        [labels, labels],
        embedding_names,
        n_folds=args.n_folds, epochs=args.epochs, patience=args.patience, 
        min_delta=args.min_delta, device=device, seed=args.seed, batch_size=args.batch_size
    )
    
    # 6. Log and save results
    print("\nComparison of Cross-Validation Accuracy:")
    for name in embedding_names:
        print(f"{name} embeddings: {results[name]['mean_accuracy']:.4f} ± {results[name]['std_accuracy']:.4f}")
    
    specific_effect = results['Normal']['mean_accuracy'] - results['Perturbed']['mean_accuracy']
    random_effect = results['Normal']['mean_accuracy'] - results['Random Perturbed']['mean_accuracy']
    
    print(f"Specific perturbation effect: {specific_effect:.4f}")
    print(f"Random perturbation effect: {random_effect:.4f}")

    summary = {
        "model_name": args.model_name,
        "layer_idx": args.layer_idx,
        "total_sentences": len(original_embeddings),
        "perturbed_sentences": len(perturbed_indices),
        "ablation_method": args.ablation_method,
        "perturbation_method": "orthogonal_projection",
        "random_seed": args.seed,
        "n_folds": args.n_folds
    }
    
    for name in embedding_names:
        summary[name.lower().replace(" ", "_") + "_accuracy"] = {
            "mean": float(results[name]['mean_accuracy']),
            "std": float(results[name]['std_accuracy']),
        }
    
    summary["specific_perturbation_effect"] = float(specific_effect)
    summary["random_perturbation_effect"] = float(random_effect)
    
    output_filename = f"summary_{os.path.basename(args.model_name)}_layer{args.layer_idx}.json"
    with open(os.path.join(args.output_dir, output_filename), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nResults saved to {os.path.join(args.output_dir, output_filename)}")

if __name__ == '__main__':
    main()
