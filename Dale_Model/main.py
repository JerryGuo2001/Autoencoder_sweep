import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from itertools import product
from tqdm import tqdm
import pandas as pd
from collections import Counter
import torch.nn.functional as F
import pickle
import os
import re
from scipy import stats
import itertools
from scipy.stats import ranksums
import json
import pandas as pd
import networkx as nx
from datetime import datetime
from itertools import combinations
from torchsummary import summary  # TODO: This should be a param / file write
from sys import path
path.append('/dfs6/pub/jianleg/Autoencoder_sweep/Dale_model/''')
from graphtask import *
from graphmeta import mappingN, Gedges
from graphtrain import Data, TrainTorch, get_graph_dataset
from graphplots import plot_graphtask
from learner import AE, get_hidden_activations
from utils import calc_dist


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Using CPU")


### THE GRAPH TASK
G = nx.from_numpy_array(Gedges)
edges = np.array(list(G.edges))
if 0: plot_graphtask(G, mappingN, Gedges, font_size=10)  # TODO: I don't think there's a show call


def find_connected_pairs(adj_matrix):
    """
    Find all unique connected pairs in an adjacency matrix.
    Only includes each edge once (no duplicates with swapped indices).

    Parameters:
    adj_matrix (numpy.ndarray): Square adjacency matrix where 1 indicates connection

    Returns:
    list: List of tuples containing pairs of connected indices
    """
    # Convert to numpy array if not already
    adj_matrix = np.array(adj_matrix)

    # Get indices of connected pairs (where value is 1)
    connected_indices = np.where(adj_matrix == 1)

    # Create list of unique pairs, only keeping pairs where first index < second index
    # This ensures we don't get duplicates like (0,1) and (1,0)
    unique_pairs = [
        (row, col) for row, col in zip(connected_indices[0], connected_indices[1])
        if row < col
    ]

    return sorted(unique_pairs)

# get the 16 unique pairs
connected_pairs = find_connected_pairs(Gedges)


def randomize_pair_order(pair):
    """Randomly return either (a,b) or (b,a) for a pair"""
    return tuple(np.random.permutation(pair))


def find_overlapping_pairs(pair1, pair2):
    """Return True if pairs share any nodes"""
    return len(set(pair1) & set(pair2)) > 0


def check_valid_group(group):
    """Check if a group has any overlapping pairs"""
    for pair1, pair2 in combinations(group, 2):
        if find_overlapping_pairs(pair1, pair2):
            return False
    return True


def find_valid_groups(pairs, num_groups=4, max_attempts=10000):
    """Find a valid grouping of pairs with no overlaps within groups."""
    pairs = [tuple(sorted(pair)) for pair in pairs]  # Ensure canonical order for grouping
    for _ in range(max_attempts):
        groups = []
        remaining_pairs = pairs.copy()
        np.random.shuffle(remaining_pairs)

        valid_grouping = True
        for _ in range(num_groups):
            if not remaining_pairs:
                valid_grouping = False
                break

            target_size = len(remaining_pairs) // (num_groups - len(groups))
            current_group = []
            candidates = remaining_pairs.copy()
            np.random.shuffle(candidates)

            for pair in candidates:
                if len(current_group) >= target_size:
                    break

                if not any(find_overlapping_pairs(pair, existing_pair)
                           for existing_pair in current_group):
                    current_group.append(pair)
                    remaining_pairs.remove(pair)

            if not current_group:
                valid_grouping = False
                break

            groups.append(current_group)

        if valid_grouping and len(groups) == num_groups and \
                all(check_valid_group(group) for group in groups) and \
                sum(len(group) for group in groups) == len(pairs):
            return groups

    return None


def generate_trial_sequences(connected_pairs, total_trials=704):
    """
    Generate both intermixed and blocked sequences of trials.
    Randomly shuffles the order of numbers within each pair.

    Parameters:
    connected_pairs: list of tuples representing connected node pairs
    total_trials: total number of trials (default 704)

    Returns:
    dict containing both sequences with randomized pair orders
    """
    repetitions = total_trials // len(connected_pairs)  # Should be 44
    connected_pairs = [tuple(sorted(pair)) for pair in connected_pairs]  # Ensure canonical order

    # Generate intermixed sequence with randomized pair orders
    intermixed = []
    for pair in connected_pairs:
        intermixed.extend([randomize_pair_order(pair) for _ in range(repetitions)])
    np.random.shuffle(intermixed)

    # Keep trying until we find valid blocked sequence
    while True:
        grouped_pairs = find_valid_groups(connected_pairs)
        if grouped_pairs is None:
            continue

        # Generate blocked sequence with randomized pair orders
        blocked = []
        for group in grouped_pairs:
            block = []
            for pair in group:
                block.extend([randomize_pair_order(pair) for _ in range(repetitions)])
            np.random.shuffle(block)
            blocked.extend(block)

        # Validate the sequence
        try:
            validate_sequences({'intermixed': intermixed, 'blocked': blocked},
                               connected_pairs)
            return {
                'intermixed': intermixed,
                'blocked': blocked
            }
        except AssertionError:
            continue


def validate_sequences(sequences, connected_pairs, total_trials=704):
    """Validate that both sequences meet all requirements"""
    intermixed = sequences['intermixed']
    blocked = sequences['blocked']

    # Convert pairs to canonical form (sorted) for validation
    connected_pairs = [tuple(sorted(pair)) for pair in connected_pairs]
    canonical_intermixed = [tuple(sorted(pair)) for pair in intermixed]
    canonical_blocked = [tuple(sorted(pair)) for pair in blocked]

    # Basic checks
    assert len(intermixed) == total_trials, f"Intermixed sequence wrong length: {len(intermixed)}"
    assert len(blocked) == total_trials, f"Blocked sequence wrong length: {len(blocked)}"

    # Check intermixed counts
    for pair in connected_pairs:
        count = canonical_intermixed.count(pair)
        expected = total_trials // len(connected_pairs)
        assert count == expected, \
            f"Wrong count for pair {pair} in intermixed: got {count}, expected {expected}"

    # Check blocked structure
    block_size = total_trials // 4  # Should be 176
    for i in range(4):
        block = canonical_blocked[i * block_size:(i + 1) * block_size]
        unique_pairs = set(block)

        # Check for overlapping pairs within block
        for pair1, pair2 in combinations(unique_pairs, 2):
            assert not find_overlapping_pairs(pair1, pair2), \
                f"Overlapping pairs found in block {i}: {pair1}, {pair2}"

        # Check counts within block
        for pair in unique_pairs:
            count = block.count(pair)
            expected = total_trials // len(connected_pairs)
            assert count == expected, \
                f"Wrong count for pair {pair} in block {i}: got {count}, expected {expected}"


# Example usage
np.random.seed(42)  # For reproducibility
sequences = generate_trial_sequences(connected_pairs)

# Print first few trials of each sequence to show randomized pair orders
print("\nFirst 20 trials of intermixed sequence:")
print(sequences['intermixed'][:20])
print("\nFirst 20 trials of blocked sequence:")
print(sequences['blocked'][:20])

with open('graphwalk_sequences.pkl', 'wb') as f:
    pickle.dump(sequences, f)

##### visualize sequences

def visualize_sequences(sequences, connected_pairs, figsize=(20, 8)):
    """
    Create two separate visualizations for intermixed and blocked sequences.
    Plots pairs in the same row regardless of number order.

    Parameters:
    sequences: dict with 'intermixed' and 'blocked' sequences
    connected_pairs: list of original pairs for ordering
    figsize: tuple for figure size
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Create canonical (sorted) pairs for y-axis mapping
    canonical_pairs = [tuple(sorted(pair)) for pair in connected_pairs]
    unique_canonical_pairs = list(dict.fromkeys(canonical_pairs))  # Remove duplicates while preserving order

    # Create y-axis mapping for each canonical pair
    pair_to_y = {pair: i for i, pair in enumerate(unique_canonical_pairs)}

    # Plot intermixed sequence
    for x, pair in enumerate(sequences['intermixed']):
        canonical_pair = tuple(sorted(pair))
        y = pair_to_y[canonical_pair]
        ax1.scatter(x, y, color='blue', s=20, alpha=0.5)

    ax1.set_xlabel('Trials', fontsize=16)
    ax1.set_ylabel('Item pairs', fontsize=16)
    ax1.set_title('Intermixed Sequence', fontsize=18)
    ax1.set_yticks(range(len(unique_canonical_pairs)))
    ax1.set_yticklabels([f"{pair[0]},{pair[1]}" for pair in unique_canonical_pairs], fontsize=14)
    ax1.tick_params(axis='x', labelsize=14)

    # Plot blocked sequence
    block_size = len(sequences['blocked']) // 4
    colors = ['red', 'blue', 'green', 'purple']  # Different color for each block

    for x, pair in enumerate(sequences['blocked']):
        canonical_pair = tuple(sorted(pair))
        y = pair_to_y[canonical_pair]
        block_num = x // block_size
        ax2.scatter(x, y, color=colors[block_num], s=20, alpha=0.5)

    ax2.set_xlabel('Trials', fontsize=16)
    ax2.set_ylabel('Item pairs', fontsize=16)
    ax2.set_title('Blocked Sequence', fontsize=18)
    ax2.set_yticks(range(len(unique_canonical_pairs)))
    ax2.set_yticklabels([f"{pair[0]},{pair[1]}" for pair in unique_canonical_pairs], fontsize=14)
    ax2.tick_params(axis='x', labelsize=14)

    # Add block separators for blocked sequence
    for i in range(1, 4):
        ax2.axvline(x=i * block_size, color='gray', linestyle='--', alpha=0.5)

    # Legend for blocks
    block_elements = [plt.Line2D([0], [0], marker='o', color=color,
                                 label=f'Block {i + 1}', markersize=10)
                      for i, color in enumerate(colors)]
    ax2.legend(handles=block_elements,
               title='Blocks',
               loc='upper right',
               title_fontsize=14,
               fontsize=12)

    plt.tight_layout()
    return fig


# Example usage:
fig = visualize_sequences(sequences, connected_pairs)
# plt.show()
fig.savefig('graphwalk_sequence_visualization.pdf', format='pdf', dpi=300, bbox_inches='tight')

######################
# model and training #
#####################

# Total number of unique stimuli
NUM_STIMULI = 12

# Find the maximum integer to determine the one-hot vector size
vector_size = NUM_STIMULI  # One-hot vectors are indexed from 0 to max_int

# Create one-hot dictionary with 12 entries
one_hot_dict = {num: np.zeros(NUM_STIMULI) for num in range(NUM_STIMULI)}
for num in range(NUM_STIMULI):
    one_hot_dict[num][num] = 1

# Print the resulting one-hot dictionary
for key, value in one_hot_dict.items():
    print(f"{key}: {value}")

unique_integers = set(one_hot_dict.keys())

# Define the function to get one-hot encodings
def get_one_hot(stimulus):
    return one_hot_dict.get(stimulus, None)


# Function to extract the bottleneck representation for a stimulus
def get_bottleneck_representation_for_stimulus(stimulus, model):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        stimulus_input = torch.tensor(get_one_hot(stimulus), dtype=torch.float32, device=device).unsqueeze(0)
        bottleneck_rep = model.bottleneck(model.encoder2(model.encoder1(stimulus_input)))
        bottleneck_rep = bottleneck_rep.squeeze(0)

    return bottleneck_rep


def load_model(model_path):
    # Extract parameters from filename
    match = re.search(r'hs_(\d+)_bs_(\d+)', model_path)

    if match:
        hidden_size1 = int(match.group(1))
        bottleneck_size = int(match.group(2))
    else:
        # Raise an error if values cannot be extracted
        raise ValueError(f"Could not extract hidden size and bottleneck size from filename: {model_path}")

    input_size = 12
    hidden_size2 = hidden_size1 // 2

    model = StimulusPredictionNetwork(
        input_size=input_size,
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        bottleneck_size=bottleneck_size
    ).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Example prediction function
def predict_stimulus(input_stimuli):
    # Convert input to one-hot encoding
    input_encoding = torch.tensor(get_one_hot(input_stimuli)).float().unsqueeze(0)  # Add batch dimension

    # Set model to evaluation mode
    model.eval()

    # Predict
    with torch.no_grad():
        prediction = model(input_encoding)

    # Convert back to stimulus number
    predicted_stimulus = torch.argmax(prediction) + 1

    return predicted_stimulus


class StimulusPredictionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, bottleneck_size=18, output_size=None,
                 l1_lambda=1e-2, l2_lambda=1e-2, elastic_beta=0.5):
        super().__init__()

        # Regularization parameters
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.elastic_beta = elastic_beta  # Mixing parameter between L1 and L2

        # If output_size is not provided, use input_size
        output_size = output_size or input_size

        # Encoding layers with Batch Normalization
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.bn_bottleneck = nn.BatchNorm1d(bottleneck_size)

        # Encoding layers
        self.encoder1 = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            self.bn1,
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size2),
            self.bn2,
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_size2, bottleneck_size),
            self.bn_bottleneck,
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Decoding layers
        self.decoder2 = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Output layer with sigmoid (for reconstruction)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size1, output_size),
        )

    def forward(self, x):
        # Encoding path
        x = self.encoder1(x)
        x = self.encoder2(x)

        # Bottleneck
        bottleneck_output = self.bottleneck(x)

        # Decoding path
        x = self.decoder2(bottleneck_output)
        x = self.decoder1(x)

        # Final output
        x = self.output_layer(x)

        return x, bottleneck_output  # Return both the output and bottleneck

    def compute_elastic_net_regularization(self):
        """
        Compute Elastic Net regularization (combined L1 and L2)

        Returns:
            torch.Tensor: Regularization loss
        """
        l1_reg = torch.tensor(0., requires_grad=True)
        l2_reg = torch.tensor(0., requires_grad=True)

        # Iterate through all parameters
        for param in self.parameters():
            if param.requires_grad:
                # L1 regularization (Lasso)
                l1_reg = l1_reg + torch.norm(param, 1)
                # L2 regularization (Ridge)
                l2_reg = l2_reg + torch.norm(param, 2) ** 2

        # Apply scaling factors for L1 and L2 regularization
        l1_reg_scaled = self.l1_lambda * self.elastic_beta * l1_reg
        l2_reg_scaled = self.l2_lambda * (1 - self.elastic_beta) * l2_reg

        # Combine L1 and L2 regularization terms
        elastic_reg = l1_reg_scaled + l2_reg_scaled

        return elastic_reg

    def pretrain_autoencoder(self, dataloader, criterion, optimizer, num_epochs=100):
        best_loss = float('inf')
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0

            for batch_inputs, _ in dataloader:
                batch_inputs = batch_inputs.to(device)  # Move inputs to the device

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass - reconstruct input
                reconstructions, bottleneck_output = self(batch_inputs)

                # Compute reconstruction loss
                reconstruction_loss = criterion(reconstructions, batch_inputs)

                # Compute elastic regularization
                elastic_reg_loss = self.compute_elastic_net_regularization()

                # Combine reconstruction loss with elastic regularization
                loss = reconstruction_loss + elastic_reg_loss

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            if avg_loss < best_loss:
                best_loss = avg_loss

        return best_loss


def train_stimulus_prediction_network(sequence_data, seed, hidden_size1, bottleneck_size, weight_decay, learning_rate, model_num,
                                      data_type, l1_lambda=1e-2, l2_lambda=1e-2, elastic_beta=0.5, num_epochs=25):
    # Hyperparameters
    input_size = vector_size
    hidden_size1 = hidden_size1
    hidden_size2 = hidden_size1 // 2
    bottleneck_size = bottleneck_size
    output_size = vector_size
    learning_rate = learning_rate
    weight_decay = weight_decay

    # Initialize the network with regularization parameters
    torch.manual_seed(seed)
    model = StimulusPredictionNetwork(
        input_size,
        hidden_size1,
        hidden_size2,
        bottleneck_size,
        output_size,
        l1_lambda=l1_lambda,
        l2_lambda=l2_lambda,
        elastic_beta=elastic_beta
    ).to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # Better weight initialization
    model.apply(init_weights)

    # Prepare data for both pre-training and main training
    inputs = []
    for input_stim in unique_integers:
        input_encoding = get_one_hot(input_stim)
        if input_encoding is None:
            continue
        inputs.append(torch.tensor(input_encoding).float())

    # Convert to tensor
    inputs = torch.stack(inputs).to(device)

    # Create DataLoader for pre-training
    pretrain_dataset = TensorDataset(inputs.to(device), inputs.to(device))  # Same input as target for autoencoder
    pretrain_dataloader = DataLoader(
        pretrain_dataset,
        batch_size=32,
        shuffle=True
    )

    # Pre-training optimizer and loss
    pretrain_criterion = nn.MSELoss()
    pretrain_optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Perform pre-training as an autoencoder
    pretrain_loss = model.pretrain_autoencoder(
        pretrain_dataloader,
        pretrain_criterion,
        pretrain_optimizer
    )

    # Save the model's weights
    model_filename = (
        f"pre_trained_model_{data_type}_"
        f"hs_{hidden_size1}_"
        f"bs_{bottleneck_size}_"
        f"beta_{elastic_beta}_"
        f"{model_num}_weights.pth"
    )
    torch.save(model.state_dict(), os.path.join(model_dir+"_pretrained", model_filename))

    # Metrics tracking DataFrame
    metrics_df = pd.DataFrame(columns=[
        'epoch', 'model_type', 'seed', 'hidden_size1', 'bottleneck_size',
        'stim_A', 'stim_C', 'C_activation', 'cosine_sim',
        'representation_entropy', 'representation_sparsity'
    ])

    # Main training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        min_lr=1e-5
    )

    # Prepare data for main training
    inputs = []
    targets = []

    for (input_stim, output_stim) in sequence_data:
        input_encoding = get_one_hot(input_stim)
        target_encoding = get_one_hot(output_stim)

        if target_encoding is None:
            continue

        inputs.append(torch.tensor(input_encoding).float())
        targets.append(torch.tensor(target_encoding).float())

    # Convert to tensors
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    # Create DataLoader
    dataset = TensorDataset(inputs.to(device), targets.to(device))
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    # Tracking variables
    # Tracking metrics every 5 epochs
    def track_metrics(sequence_data, model, epoch, model_type, seed, hidden_size1, bottleneck_size):
        """
        Track various metrics, including cosine similarity, for given pairs of stimuli.
        """
        device = next(model.parameters()).device
        pair_metrics = []

        for pair in connected_pairs:
            stim_A, stim_C = pair

            # Get bottleneck representations for both stimuli
            stim_A_rep = get_bottleneck_representation_for_stimulus(stim_A, model)
            stim_C_rep = get_bottleneck_representation_for_stimulus(stim_C, model)

            # Ensure float32 precision
            stim_A_rep = stim_A_rep.float()  # This ensures float32
            stim_C_rep = stim_C_rep.float()  # This ensures float32

            # Calculate cosine similarity
            cosine_sim = F.cosine_similarity(stim_A_rep.unsqueeze(0), stim_C_rep.unsqueeze(0), dim=1).item()

            # Get one-hot encoding and move to correct device with float32
            stim_A_encoding = get_one_hot(stim_A)
            input_encoding = torch.tensor(stim_A_encoding, device=device, dtype=torch.float32).unsqueeze(0)

            # Predict
            with torch.no_grad():
                prediction = model(input_encoding)
                softmax_output = F.softmax(prediction[0], dim=1)

            # Get the activation for stim_C
            C_activation = softmax_output[0, stim_C - 1].item()

            # Calculate entropy of bottleneck representation
            bottleneck_rep = stim_A_rep.detach()
            # Normalize the representation
            bottleneck_prob = F.softmax(bottleneck_rep, dim=0)
            # Calculate entropy (ensure float32)
            entropy = -torch.sum(bottleneck_prob * torch.log2(bottleneck_prob + 1e-10).float()).item()

            # Calculate sparsity
            sparsity = torch.sum(torch.abs(bottleneck_rep)).item()

            pair_metrics.append({
                'epoch': epoch,
                'model_type': model_type,  # Fixed data_type to model_type
                'seed': seed,
                'hidden_size1': hidden_size1,
                'bottleneck_size': bottleneck_size,
                'stim_A': stim_A,
                'stim_C': stim_C,
                'C_activation': C_activation,
                'cosine_sim': cosine_sim,
                'representation_entropy': entropy,
                'representation_sparsity': sparsity
            })

        return pair_metrics

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_inputs, batch_targets in dataloader:
            # Move the batch to the device (GPU)
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions, bottleneck_output = model(batch_inputs)

            # Compute prediction loss
            prediction_loss = criterion(predictions, batch_targets)

            # Compute elastic net regularization loss
            elastic_reg_loss = model.compute_elastic_net_regularization()

            # Combine losses
            total_loss = prediction_loss + elastic_reg_loss

            # Backpropagation and optimization
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        # Update learning rate
        scheduler.step(epoch_loss)

        # Track metrics every 5 epochs
        if epoch % 5 == 0:
            epoch_metrics = track_metrics(sequence_data, model, epoch, data_type, seed, hidden_size1, bottleneck_size)
            metrics_df = pd.concat([
                metrics_df,
                pd.DataFrame(epoch_metrics)
            ], ignore_index=True)

    # Save metrics DataFrame
    metrics_filename = f'metrics_{data_type}_hs_{hidden_size1}_bs_{bottleneck_size}_beta_{elastic_beta}_{model_num}.csv'
    metrics_df.to_csv(os.path.join(model_dir, metrics_filename), index=False)
    print(f"Metrics saved to {metrics_filename}")

    return model


model_dir = './trained_models_graphwalk6'  # Replace with the directory where your models are saved

# Train the network multiple times for each scheduler and memory capacities
def train_and_save_multiple_models(num_models=1):
    # Elastic beta sweep values
    elastic_beta_sweep = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    param_grid = [
        {'learning_rate': 0.001, 'weight_decay': 0.001, 'hidden_size1': 324, 'bottleneck_size': 18},
        {'learning_rate': 0.001, 'weight_decay': 0.001, 'hidden_size1': 288, 'bottleneck_size': 18},
        {'learning_rate': 0.001, 'weight_decay': 0.001, 'hidden_size1': 252, 'bottleneck_size': 18},
        {'learning_rate': 0.001, 'weight_decay': 0.001, 'hidden_size1': 216, 'bottleneck_size': 18},
        {'learning_rate': 0.001, 'weight_decay': 0.001, 'hidden_size1': 180, 'bottleneck_size': 18},
        {'learning_rate': 0.001, 'weight_decay': 0.001, 'hidden_size1': 144, 'bottleneck_size': 18},
        {'learning_rate': 0.001, 'weight_decay': 0.001, 'hidden_size1': 108, 'bottleneck_size': 18},
        {'learning_rate': 0.001, 'weight_decay': 0.001, 'hidden_size1': 72, 'bottleneck_size': 18},
        {'learning_rate': 0.001, 'weight_decay': 0.001, 'hidden_size1': 36, 'bottleneck_size': 18},
        {'learning_rate': 0.001, 'weight_decay': 0.001, 'hidden_size1': 12, 'bottleneck_size': 18},
        {'learning_rate': 0.001, 'weight_decay': 0.001, 'hidden_size1': 6, 'bottleneck_size': 18}
    ]

    # Loop over each combination of parameters
    counter = 0
    for data in [sequences['intermixed'], sequences['blocked']]:
        if counter == 0:
            data_type = "intermixed"
        elif counter == 1:
            data_type = "blocked"

        for params in param_grid:
            for elastic_beta in elastic_beta_sweep:
                for model_num in range(num_models):
                    print(f"Training model {model_num + 1}/{num_models} with elastic_beta={elastic_beta}...")

                    # Compute L1 and L2 lambda values
                    l1_lambda = 1e-2
                    l2_lambda = 1e-2

                    model = train_stimulus_prediction_network(
                        sequence_data = data,
                        seed=model_num,
                        hidden_size1=params['hidden_size1'],
                        bottleneck_size=params['bottleneck_size'],
                        weight_decay=params['weight_decay'],
                        learning_rate=params['learning_rate'],
                        model_num=model_num,
                        data_type=data_type,
                        l1_lambda=l1_lambda,
                        l2_lambda=l2_lambda,
                        elastic_beta=elastic_beta
                    )

                    # Save the model's weights
                    model_filename = (
                        f"model_{data_type}_"
                        f"hs_{params['hidden_size1']}_"
                        f"bs_{params['bottleneck_size']}_"
                        f"beta_{elastic_beta}_"
                        f"{model_num}_weights.pth"
                    )
                    torch.save(model.state_dict(), os.path.join(model_dir, model_filename))
                    print(f"Model {model_num + 1} weights saved to {model_filename}")

        counter += 1


# Train and save models
train_and_save_multiple_models(10)


# Relative distance task

def calculate_pairwise_cosine_distances(tensor_list):
    """
    Calculate pairwise cosine distances between a list of tensors.

    Args:
        tensor_list (list of torch.Tensor): List of 1D tensors to compare

    Returns:
        torch.Tensor: A square matrix of pairwise cosine distances
    """
    # Stack the tensors into a matrix
    matrix = torch.stack(tensor_list)

    # Normalize the vectors
    normalized = F.normalize(matrix, p=2, dim=1)

    # Calculate cosine similarity matrix
    cosine_sim = torch.mm(normalized, normalized.t())

    # Convert similarity to distance (1 - similarity)
    cosine_dist = 1 - cosine_sim

    return cosine_dist


path_lens = nx.floyd_warshall_numpy(G)


def create_relative_distance_trials(n_items, path_lens, Gedges, niter=17, verbose=False):
    """
    Creates trials for the relative distance task where for each reference item (i2),
    two comparison items (i1, i3) are selected with different path lengths to i2.

    Parameters:
    -----------
    n_items : int
        Number of items in the graph (assumed to be 12)
    path_lens : np.ndarray
        Matrix of path lengths between all pairs of items
    Gedges : np.ndarray
        Adjacency matrix representing graph edges
    niter : int
        Number of iterations per reference item
    verbose : bool
        Whether to print debug information

    Returns:
    --------
    list of tuples
        Each tuple contains (i2, i1, i3) where:
        - i2 is the reference item
        - i1 and i3 are comparison items
        - correct_choice is 0 if i1 is closer, 1 if i3 is closer
    """
    trials = []

    # Cycle through each possible i2 value
    for i2 in range(1, 13):  # 1 to 12
        counter = 1
        forced_trials = [(i, j, k) for (i, j, k) in force_distance_3_results if i == i2]

        while counter <= niter:  # repetitions for each i2
            valid_pair = False
            while not valid_pair:
                # first use up all the distance 3 possibilities
                if counter <= len(forced_trials):
                    _, i1, i3 = forced_trials[counter - 1]
                else:
                    # Get available numbers (excluding i2)
                    available = [x for x in range(1, 13) if x != i2]
                    # Randomly select i1 and i3
                    i1, i3 = np.random.choice(available, size=2, replace=False)

                # Calculate path lengths and their difference
                d12 = path_lens[i1 - 1, i2 - 1]  # Adjust indices for 0-based indexing
                d32 = path_lens[i3 - 1, i2 - 1]
                dist_diff = np.abs(d32 - d12)

                # Check if conditions are met
                if (Gedges[i2 - 1][i1 - 1] == 0 and  # i1 and i3 are not connected to i2
                        Gedges[i2 - 1][i3 - 1] == 0 and
                        dist_diff != 0):  # Absolute difference in path lengths is nonzero

                    if verbose:
                        print('PL', d12, d32, dist_diff)

                    # Determine correct choice (0 = i1, 1 = i3)
                    correct_choice = int(np.argmin([d12, d32]))

                    # Add trial to list
                    trials.append((i2, i1, i3, correct_choice, dist_diff))
                    valid_pair = True

            counter += 1

    return trials


def evaluate_relative_distance(trials, model_dists, trained_model_instance, pretrained_model_instance, verbose=False):
    """
    Evaluates model performance on relative distance trials.
    """
    choice_accs_dist = {1: [], 2: [], 3: [], 4: []}  # Hardcoded at max PL 4
    integration_dist = {1: [], 2: [], 3: [], 4: []}  # Hardcoded at max PL 4

    for i2, i1, i3, correct_choice, dist_diff in trials:
        # Get model distances
        m12 = model_dists[i1 - 1, i2 - 1]
        m32 = model_dists[i3 - 1, i2 - 1]

        if verbose:
            print('MD', m12, m32)

        # Use torch.tensor and keep on MPS instead of numpy
        choices = torch.tensor([m12, m32])
        model_choice = int(torch.argmin(choices))

        # Assess accuracy
        choice_acc = int(correct_choice == model_choice)
        choice_accs_dist[dist_diff].append(choice_acc)

        # If correct, calculate integration measures
        if choice_acc == 1:
            stim_B = i2
            stim_C = i3 if correct_choice == 1 else i1
            stim_A = i1 if correct_choice == 1 else i3

            # Get trained model representations (keep on MPS)
            stim_B_rep = get_bottleneck_representation_for_stimulus(stim_B - 1, trained_model_instance)
            stim_C_rep = get_bottleneck_representation_for_stimulus(stim_C - 1, trained_model_instance)
            stim_A_rep = get_bottleneck_representation_for_stimulus(stim_A - 1, trained_model_instance)

            # Calculate trained model similarities (stays on MPS)
            cosine_sim_BC = F.cosine_similarity(stim_B_rep.unsqueeze(0), stim_C_rep.unsqueeze(0), dim=1).item()
            cosine_sim_BA = F.cosine_similarity(stim_B_rep.unsqueeze(0), stim_A_rep.unsqueeze(0), dim=1).item()

            # Get pretrained model representations (keep on MPS)
            stim_B_rep = get_bottleneck_representation_for_stimulus(stim_B - 1, pretrained_model_instance)
            stim_C_rep = get_bottleneck_representation_for_stimulus(stim_C - 1, pretrained_model_instance)
            stim_A_rep = get_bottleneck_representation_for_stimulus(stim_A - 1, pretrained_model_instance)

            # Calculate pretrained model similarities (stays on MPS)
            pretrained_cosine_sim_BC = F.cosine_similarity(stim_B_rep.unsqueeze(0), stim_C_rep.unsqueeze(0),
                                                           dim=1).item()
            pretrained_cosine_sim_BA = F.cosine_similarity(stim_B_rep.unsqueeze(0), stim_A_rep.unsqueeze(0),
                                                           dim=1).item()

            # Calculate integration changes
            integration_change_BC = cosine_sim_BC - pretrained_cosine_sim_BC
            integration_change_BA = cosine_sim_BA - pretrained_cosine_sim_BA

            # Calculate difference between BC and BA integration changes
            integration_difference = integration_change_BC - integration_change_BA

            integration_dist[dist_diff].append(integration_difference)

    return choice_accs_dist, integration_dist


# Iterate over each model file (assuming models are named 'model_*_weights.pth')
model_dir = './trained_models_graphwalk'  # Replace with the directory where your models are saved
model_files = [f for f in os.listdir(model_dir) if f.endswith('_weights.pth')]
with open(os.path.join(model_dir,"graphwalk_sequences.pkl"), 'rb') as file:
    sequences = pickle.load(file)

# Create a list to store all the extracted information
model_info = []
results = {'name':[], 'path':[], 'task':[], 'L2':[], '1':[], '2':[], '3':[], '4':[],
            'scores':[], 'beta':[], 'integration':[],
           'integration_1':[], 'integration_2':[], 'integration_3':[], 'integration_4':[]}

# Initialize the result to store pairs
force_distance_3_results = []

# Iterate through each row
for i, row in enumerate(path_lens):
    # Check all pairs in the row
    for j in range(len(row)):
        for k in range(j+1, len(row)):  # Avoid checking the same pair twice
            if abs(row[j] - row[k]) == 3 and i!=j and i!=k and row[j]!=1 and row[k]!=1:
                force_distance_3_results.append((i+1, j+1, k+1))  # Store the indices of the pair

# Create the trials once before the model loop
trials = create_relative_distance_trials(12, path_lens=path_lens, Gedges=Gedges, verbose=False)

# Loop over each model file and calculate cosine similarity for each model
for model_file in tqdm(model_files):
    # Extract information from the filename
    match = re.search(r'model_(\w+)_hs_(\d+)_bs_(\d+)_beta_(\d+(?:\.\d+)?)_(\d+)_weights\.pth', model_file)
    # match = re.search(r'model_(\w+)_lr_[\d.e-]+_wd_[\d.e-]+_hs_(\d+)_bs_(\d+)_(\d+)_weights\.pth', model_file)

    if match:
        model_type = match.group(1)

        hs = int(match.group(2))
        bs = int(match.group(3))
        beta = match.group(4)
        model_number = int(match.group(5))

        model_path = os.path.join(model_dir, model_file)

        # Load the model weights
        model = load_model(model_path)

        # Load corresponding pretrained model
        pretrained_file = f"pre_trained_{model_file}"
        pretrained_path = os.path.join(model_dir+'_pretrained', pretrained_file)
        pretrained_model = load_model(pretrained_path)

        # Get bottleneck representations for all stimuli
        tensor_list = []
        for stim in list(range(12)):
            tensor_list.append(get_bottleneck_representation_for_stimulus(stim, model))

        # Get distnace matrix
        distances = calculate_pairwise_cosine_distances(tensor_list)

        # Use the pre-generated trials for evaluation
        choice_accs_dist, integration_dist = evaluate_relative_distance(trials,
                                                                        model_dists=distances,
                                                                        trained_model_instance=model,
                                                                        pretrained_model_instance=pretrained_model,
                                                                        verbose=False)

        dist_pct = {}
        for dist, vals in choice_accs_dist.items():
            acc = (np.sum(vals) / len(vals)) * 100
            dist_pct[dist] = acc
            if 0: print(f'{dist}: {acc:.2f}% {len(vals)}')


        mean_integration = {}
        for dist, vals in integration_dist.items():
            acc = np.mean(vals)
            mean_integration[dist] = acc
            if 0: print(f'{dist}: {acc:.2f}% {len(vals)}')

        flat_list = [item for sublist in integration_dist.values() for item in sublist]
        mean_value = np.mean(flat_list)

        # Pack up into dictionary
        # TODO: this should be an interable
        results['name'].append(model_file)
        results['path'].append(model_dir)
        results['task'].append(model_type)
        results['L2'].append(hs)
        results['1'].append(dist_pct[1])
        results['2'].append(dist_pct[2])
        results['3'].append(dist_pct[3])
        results['4'].append(dist_pct[4])
        results['scores'].append(dist_pct)
        results['beta'].append(beta)
        results['integration'].append(mean_value)
        results['integration_1'].append(mean_integration[1])
        results['integration_2'].append(mean_integration[2])
        results['integration_3'].append(mean_integration[3])
        results['integration_4'].append(mean_integration[4])

r_frame = pd.DataFrame(results)

r_frame.to_csv('dalesgraphwalkae.csv', index=False)