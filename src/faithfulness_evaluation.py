import argparse
import json
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


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
    data = [line.strip() for line in data]
    return data


def load_merged_explanation(file_path):
    """Load the token to index mapping from the CSV file."""
    df = pd.read_csv(file_path, sep=",", engine="python", on_bad_lines="skip")
    return df


def load_codebook_vectors(file_path):
    """Load the codebook vectors from the PyTorch file."""
    codebook_dict = torch.load(file_path)
    return codebook_dict


def load_ground_truth_labels(json_file_path):
    """Load ground truth labels from the JSON file."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    # Create a mapping from sentence index to ground truth label
    labels = [item['label'] for item in data]
    return labels


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


class LogisticRegressionClassifier(nn.Module):
    """A simple logistic regression classifier."""
    def __init__(self, input_dim, num_classes=2):
        super(LogisticRegressionClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)


def extract_cls_embeddings(model, tokenizer, sentences, device, layer_idx=11):
    """Extract CLS token embeddings from a specific layer in one go."""
    # Tokenize all sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    
    # Forward pass with output_hidden_states=True to get all layer outputs
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get hidden states from specified layer
    hidden_states = outputs.hidden_states
    layer_output = hidden_states[layer_idx]

    # Extract CLS token embeddings (first token in each sequence)
    cls_embeddings = layer_output[:, 0, :].cpu().numpy()
    
    return cls_embeddings


def project_orthogonal(vector, direction):
    """
    Project vector onto the subspace orthogonal to direction.
    This effectively removes the component of vector in the direction.
    """
    # Ensure direction is not a zero vector
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-10:
        return vector  # Return original vector if direction is effectively zero
    
    # Normalize the direction vector
    direction_unit = direction / direction_norm
    
    # Calculate the projection of vector onto direction
    projection = np.dot(vector, direction_unit) * direction_unit
    
    # Subtract the projection to get the orthogonal component
    orthogonal = vector - projection
    
    return orthogonal


def perturb_salient_embeddings_orthogonal(cls_embeddings, sentence_ids, merged_explanation_df, codebook_vectors):
    """Remove the concept direction from CLS token embeddings using orthogonal projection."""
    perturbed_cls_embeddings = cls_embeddings.copy()
    perturbed_indices = []
    
    for i, sentence_id in enumerate(sentence_ids):
        # Find the salient token for this sentence in the merged_explanation_df
        salient_token_df = merged_explanation_df[merged_explanation_df['sentence_index'] == sentence_id]
        codebook_idx = salient_token_df.iloc[0]['vector_idx']
        salient_codebook_vector = codebook_vectors[codebook_idx]

        if isinstance(salient_codebook_vector, torch.Tensor):
            salient_codebook_vector = salient_codebook_vector.cpu().numpy()
        
        # Remove the concept direction using orthogonal projection
        perturbed_cls_embeddings[i] = project_orthogonal(perturbed_cls_embeddings[i], salient_codebook_vector)
        perturbed_indices.append(i)
    
    print(f"Applied orthogonal projection to remove concept directions from {len(perturbed_indices)} embeddings")
    return perturbed_cls_embeddings, perturbed_indices


def perturb_salient_embeddings_orthogonal_random(cls_embeddings, perturbed_indices, seed=42):
    """
    Remove a random concept direction from CLS token embeddings.
    For each perturbed embedding, select a random codebook vector and project orthogonally to it.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    perturbed_cls_embeddings = cls_embeddings.copy()
    embedding_dim = cls_embeddings.shape[1]

    for idx in perturbed_indices: 
        # Select a random vector
        random_vector = np.random.randn(embedding_dim)
        
        # Remove the random concept direction using orthogonal projection
        perturbed_cls_embeddings[idx] = project_orthogonal(perturbed_cls_embeddings[idx], random_vector)
    
    print(f"Applied random concept direction removal to {len(perturbed_indices)} embeddings")
    return perturbed_cls_embeddings


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
        model = SimpleClassifier(input_dim=input_dim).to(device)
        # model = LogisticRegressionClassifier(input_dim=input_dim).to(device)
        
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate classifiers on perturbed embeddings")
    
    parser.add_argument('--dataset-path',
                        type=str,
                        required=True,
                        help='Path to the dataset file with sentences.')
    
    parser.add_argument('--merged-explanation-file',
                        type=str,
                        required=True,
                        help='Path to the token_to_index_map.json file.')
    
    parser.add_argument('--codebook-vectors',
                        type=str,
                        required=True,
                        help='Path to the codebook vectors file.')
    
    parser.add_argument('--model-name',
                        type=str,
                        required=True,
                        help='The name or path of the RoBERTa model.')
    
    parser.add_argument('--output-dir',
                        type=str,
                        default='cls_classifier_results/',
                        help='Directory to save the results.')
    
    parser.add_argument('--layer-idx',
                        type=int,
                        default=8,
                        help='Layer index to extract CLS token from (default: 11)')

    parser.add_argument('--ground-truth-file',
                        type=str,
                        required=True,
                        help='Path to the ground truth JSON file with labels.')
    
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
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset, token map, and codebook vectors
    sentences = get_dataset(args.dataset_path)
    merged_explanation_df = load_merged_explanation(args.merged_explanation_file)
    codebook_vectors = load_codebook_vectors(args.codebook_vectors)
    ground_truth_labels = load_ground_truth_labels(args.ground_truth_file)
    
    print(f"Loaded {len(sentences)} sentences and {len(ground_truth_labels)} ground truth labels")
    
    # Get all sentence IDs from token map
    sentence_ids = set(merged_explanation_df['sentence_index'])
    sentence_ids = sorted(list(sentence_ids), key=lambda x: int(x))
    print(f"Found {len(sentence_ids)} unique sentence IDs in token map")

    max_sentences = len(sentences)
    # Create sentence ID mapping
    sentence_id_mapping = {i: sentence_ids[i] for i in range(len(sentences))}
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()
    
    print(f"Loaded model: {args.model_name}")
    
    # Extract CLS token embeddings
    print("\nExtracting CLS token embeddings...")
    cls_embeddings = extract_cls_embeddings(
        model, tokenizer, sentences[:max_sentences], device, layer_idx=args.layer_idx
    )

    # Create perturbed CLS token embeddings using orthogonal projection
    print("\nCreating perturbed CLS token embeddings using orthogonal projection...")
    perturbed_cls_embeddings, perturbed_indices = perturb_salient_embeddings_orthogonal(
        cls_embeddings, sentence_ids, merged_explanation_df, codebook_vectors
    )
    
    # Create perturbed CLS token embeddings with random vectors
    print("\nCreating perturbed CLS token embeddings with random orthogonal projection...")
    random_perturbed_cls_embeddings = perturb_salient_embeddings_orthogonal_random(
        cls_embeddings, perturbed_indices, seed=args.seed
    )
    
    # Prepare labels
    labels = np.array(ground_truth_labels[:max_sentences])
    
    # Define names for the embedding types
    embedding_names = ["Normal CLS", "Perturbed CLS", "Random Perturbed CLS"]
    
    # Train on original embeddings and evaluate on all types
    print("\nTraining model on original CLS embeddings and evaluating on all embedding types:")
    results = train_and_evaluate_model(
        cls_embeddings, labels, 
        [perturbed_cls_embeddings, random_perturbed_cls_embeddings],
        [labels, labels],
        embedding_names,
        n_folds=args.n_folds, 
        epochs=args.epochs, 
        patience=args.patience, 
        min_delta=args.min_delta, 
        device=device, 
        seed=args.seed
    )
    
    # Compare results
    print("\nComparison of Cross-Validation Accuracy:")
    for name in embedding_names:
        print(f"{name} embeddings: {results[name]['mean_accuracy']:.4f} ± {results[name]['std_accuracy']:.4f}")
    
    print(f"Specific perturbation effect: {results['Normal CLS']['mean_accuracy'] - results['Perturbed CLS']['mean_accuracy']:.4f}")
    print(f"Random perturbation effect: {results['Normal CLS']['mean_accuracy'] - results['Random Perturbed CLS']['mean_accuracy']:.4f}")
    print(f"Specific vs Random difference: {results['Random Perturbed CLS']['mean_accuracy'] - results['Perturbed CLS']['mean_accuracy']:.4f}")
    
    # Save summary results
    summary = {
        "layer_idx": args.layer_idx,
        "total_sentences": len(cls_embeddings),
        "perturbed_sentences": len(perturbed_indices),
        "perturbation_method": "orthogonal_projection",
        "random_seed": args.seed,
        "n_folds": args.n_folds
    }
    
    # Add results for each embedding type
    for name in embedding_names:
        summary[name.lower().replace(" ", "_") + "_accuracy"] = {
            "mean": float(results[name]['mean_accuracy']),
            "std": float(results[name]['std_accuracy']),
            "fold_accuracies": [float(acc) for acc in results[name]['fold_accuracies']]
        }
    
    summary["specific_perturbation_effect"] = float(results['Normal CLS']['mean_accuracy'] - results['Perturbed CLS']['mean_accuracy'])
    summary["random_perturbation_effect"] = float(results['Normal CLS']['mean_accuracy'] - results['Random Perturbed CLS']['mean_accuracy'])
    summary["specific_vs_random_difference"] = float(results['Random Perturbed CLS']['mean_accuracy'] - results['Perturbed CLS']['mean_accuracy'])
    
    with open(os.path.join(args.output_dir, "single_model_comparison_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()