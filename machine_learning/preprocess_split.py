"""
This script preprocesses and splits a dataset of paired `.svg` and `.pcap` files into training, validation, 
and test sets. It ensures that the files are correctly paired, shuffles them, and saves the splits to disk. 
Additionally, it preprocesses the data for each split using the `SVGPCAPDataset` class.
Modules:
    - os: Provides functions for interacting with the operating system.
    - random: Used for shuffling the dataset in a reproducible manner.
    - gnn: Contains the `SVGPCAPDataset` class for handling dataset preprocessing.
Functions:
    - preprocess_split(split_files, split_name): Preprocesses and saves the data for a given split.
Variables:
    - svg_dir (str): Directory containing `.svg` files.
    - pcap_dir (str): Directory containing `.pcap` files.
    - vectorizer_path (str): Path to the vectorizer file.
    - output_dir (str): Directory where split file lists will be saved.
    - preprocessed_dir (str): Directory where preprocessed data will be saved.
    - svg_files (list): Sorted list of `.svg` files in `svg_dir`.
    - pcap_files (list): Sorted list of `.pcap` files in `pcap_dir`.
    - combined (list): List of paired `.svg` and `.pcap` files.
    - train_ratio (float): Proportion of the dataset to use for training.
    - val_ratio (float): Proportion of the dataset to use for validation.
    - test_ratio (float): Proportion of the dataset to use for testing.
    - train_files (list): List of paired files for the training set.
    - val_files (list): List of paired files for the validation set.
    - test_files (list): List of paired files for the test set.
    - dataset (SVGPCAPDataset): Instance of the dataset class for preprocessing.
Steps:
    1. Ensures the output and preprocessed directories exist.
    2. Reads and validates `.svg` and `.pcap` file pairs.
    3. Shuffles the dataset and splits it into training, validation, and test sets.
    4. Saves the file lists for each split to the output directory.
    5. Initializes the `SVGPCAPDataset` class for preprocessing.
    6. Preprocesses and saves the data for each split.
Output:
    - Three text files (`train_files.txt`, `val_files.txt`, `test_files.txt`) containing the file pairs for each split.
    - Preprocessed data saved in the `preprocessed_dir` for each split.
Usage:
    Run this script to preprocess and split the dataset into training, validation, and test sets. 
    Ensure that the input directories (`svg_dir` and `pcap_dir`) and the labels CSV file are correctly set.
"""

import os
import random
from gnn import SVGPCAPDataset

# Set paths
svg_dir = 'data/powerset/svg'
pcap_dir = 'data/powerset/pcap'
vectorizer_path = 'powerset/vectorizer.pkl'
output_dir = 'data/powerset_splits'
preprocessed_dir = 'data/powerset_preprocessed'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)
os.makedirs(preprocessed_dir, exist_ok=True)

# Get list of preprocessed files
svg_files = sorted([f for f in os.listdir(svg_dir) if f.endswith('.svg')])
pcap_files = sorted([f for f in os.listdir(pcap_dir) if f.endswith('.pcap')])

# Ensure that .svg and .pcap files are paired correctly
assert len(svg_files) == len(pcap_files), "Mismatch between .svg and .pcap files"
for svg_file, pcap_file in zip(svg_files, pcap_files):
    assert svg_file.split('.')[0] == pcap_file.split('.')[0], f"Mismatched files: {svg_file} and {pcap_file}"

# Shuffle the files (while maintaining .svg and .pcap pairing)
combined = list(zip(svg_files, pcap_files))
random.seed(42)  # For reproducibility
random.shuffle(combined)

# Split into training, validation, and test sets
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

dataset_size = len(combined)
train_size = int(train_ratio * dataset_size)
val_size = int(val_ratio * dataset_size)
test_size = dataset_size - train_size - val_size

train_files = combined[:train_size]
val_files = combined[train_size:train_size + val_size]
test_files = combined[train_size + val_size:]

# Save the splits
with open(os.path.join(output_dir, 'train_files.txt'), 'w') as f:
    for svg_file, pcap_file in train_files:
        f.write(f"{svg_file}.pt\t{pcap_file}.pt\n")

with open(os.path.join(output_dir, 'val_files.txt'), 'w') as f:
    for svg_file, pcap_file in val_files:
        f.write(f"{svg_file}.pt\t{pcap_file}.pt\n")

with open(os.path.join(output_dir, 'test_files.txt'), 'w') as f:
    for svg_file, pcap_file in test_files:
        f.write(f"{svg_file}.pt\t{pcap_file}.pt\n")

print(f"Training set size: {len(train_files)}")
print(f"Validation set size: {len(val_files)}")
print(f"Test set size: {len(test_files)}")
print(f"Splits saved to {output_dir}")

# Initialize the dataset
dataset = SVGPCAPDataset(
    labels_csv='data/powerset/labels.csv',
    svg_dir=svg_dir,
    pcap_dir=pcap_dir,
    preprocessed_dir=preprocessed_dir,
)

# Preprocess and save the data for each split
def preprocess_split(split_files, split_name):
    svg_files, pcap_files = zip(*split_files)
    split_output_dir = os.path.join(preprocessed_dir, split_name)
    os.makedirs(split_output_dir, exist_ok=True)
    dataset.preprocess_data(
        svg_files=svg_files,
        pcap_files=pcap_files,
        svg_dir=svg_dir,
        pcap_dir=pcap_dir,
        output_dir=split_output_dir
    )
preprocess_split(train_files, 'train')
preprocess_split(val_files, 'val')
preprocess_split(test_files, 'test')

print("Preprocessing completed for all splits.")