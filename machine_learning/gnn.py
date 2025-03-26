"""
This script implements a Graph Neural Network (GNN) model for combining graph-based features from SVG files 
and network traffic features from PCAP files for classification tasks. The script includes the following components:
1. **Imports**:
    - Necessary libraries for graph processing, machine learning, and data manipulation.
    - PyTorch Geometric for graph-based operations.
    - Scapy for PCAP file parsing.
    - Dask for scalable data processing.
2. **Custom Functions**:
    - `custom_collate_fn`: A custom collate function for preparing batches of data for the DataLoader.
    - `is_connected`: Checks if a graph is connected.
    - `pad_node_features`: Pads node features to a fixed size.
3. **Dataset Class**:
    - `SVGPCAPDataset`: A PyTorch Dataset class for handling data that combines SVG and PCAP files. 
      It includes methods for:
        - Data augmentation and preprocessing.
4. **Model Definition**:
    - `GNNModel`: A Graph Neural Network model that combines graph-based features and PCAP features for classification tasks.
    - `WeightedFocalLoss`: A custom implementation of the Weighted Focal Loss function for addressing class imbalance.
5. **Training and Validation**:
    - `train_model`: A function to train the GNN model using the preprocessed datasets.
    - `validate_model`: A function to validate the model's performance on a validation dataset.
6. **Main Execution**:
    - The script trains the model and plots the training and validation losses.
### Key Features:
- **Graph Processing**: Uses PyTorch Geometric for graph-based operations such as graph convolution and pooling.
- **PCAP Feature Extraction**: Extracts features from PCAP files using Scapy and applies data augmentation.
- **Custom Loss Function**: Implements a Weighted Focal Loss to handle class imbalance.
- **Data Augmentation**: Includes methods for augmenting both SVG and PCAP features.
- **Threshold Tuning**: Tunes thresholds for process-level predictions to maximize F1 score.
### Usage:
- Ensure the required libraries are installed, including PyTorch, PyTorch Geometric, Scapy, and Dask.
- Prepare the dataset by preprocessing SVG and PCAP files.
- Run the script to train the model and evaluate its performance.
### Dependencies:
- PyTorch
- PyTorch Geometric
- Scapy
- Dask
- Matplotlib
- scikit-learn
### Notes:
- The script assumes the existence of specific file paths and preprocessed data.
"""

import os
import re
import math
import time
import hashlib
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import is_undirected, to_undirected, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import to_networkx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from xml.etree import ElementTree as ET
from scapy.all import rdpcap
import networkx as nx
from collections import Counter
import pickle
from functools import partial
from multiprocessing import Pool
import matplotlib.pyplot as plt
import dask.array as da
from dask_ml.decomposition import TruncatedSVD

def custom_collate_fn(batch):
    """
    Custom collate function that handles:
    - Filtering invalid graphs
    - Padding variable-length PCAP sequences
    - Creating masks for RNN
    - Batching graphs with PyG
    """
    # Filter out invalid graphs
    valid_indices = [i for i, item in enumerate(batch) if item[0].num_nodes > 0]
    batch = [batch[i] for i in valid_indices]
    
    svg_graphs = [item[0] for item in batch]
    pcap_features = [item[1] for item in batch]
    origin_ips = torch.tensor([item[2] for item in batch], dtype=torch.long)
    target_process_indices = [item[3] for item in batch]
    target_process_values = [item[4] for item in batch]
    
    # PCAP Feature Processing
    # ======================
    
    # 1. Verify feature dimensions
    for pcap in pcap_features:
        assert pcap.dim() == 2, f"Expected 2D tensor, got {pcap.dim()}D"
        assert pcap.size(1) == 106, f"Expected 106 features, got {pcap.size(1)}"
    
    # 2. Get sequence lengths before padding
    lengths = torch.tensor([pcap.size(0) for pcap in pcap_features], dtype=torch.long)
    
    # 3. Pad sequences to max length
    padded_pcap = torch.nn.utils.rnn.pad_sequence(
        pcap_features,
        batch_first=True,
        padding_value=0.0
    )  # [batch_size, max_len, 106]
    
    # 4. Create mask (1 = real data, 0 = padding)
    max_len = padded_pcap.size(1)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    mask = mask.float()  # [batch_size, max_len]
    
    # Graph Processing
    # ================
    graph_batch = Batch.from_data_list(svg_graphs)
    
    return (
        graph_batch,      # PyG Batch object
        padded_pcap,      # Padded PCAP features [batch_size, max_len, 106]
        mask,             # Sequence mask [batch_size, max_len]
        lengths,          # Original lengths [batch_size]
        origin_ips,       # Origin IP indices [batch_size]
        target_process_indices,  # List of target indices
        target_process_values    # List of target values
    )

def is_connected(edge_index, num_nodes=None):
    """
    Check if a graph is connected.
    
    Args:
        edge_index (torch.Tensor): The edge indices of the graph.
        num_nodes (int, optional): The number of nodes in the graph.
    
    Returns:
        bool: True if the graph is connected, False otherwise.
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if num_nodes == 0:
        return True  # An empty graph is trivially connected
    
    # Convert to undirected graph
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    
    # Compute degrees
    deg = degree(edge_index[0], num_nodes=num_nodes)
    
    # Check if all nodes have degree >= 1
    if torch.any(deg == 0):
        return False
    
    # Perform BFS to check connectivity
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    queue = [0]
    visited[0] = True
    
    while queue:
        node = queue.pop(0)
        neighbors = edge_index[1][edge_index[0] == node]
        for neighbor in neighbors.tolist():
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    
    return torch.all(visited).item()

def pad_node_features(graph, max_features):
    """
    Pad node features to a fixed size.
    Args:
        graph (Data): The input graph.
        max_features (int): The maximum number of features in the dataset.
    Returns:
        Data: The graph with padded node features.
    """
    if graph.num_nodes == 0:
        # Ensure empty graphs have the correct feature dimension
        graph.x = torch.zeros((0, max_features), dtype=torch.float32)
        return graph

    num_nodes = graph.num_nodes
    num_features = graph.x.size(1) if graph.num_nodes > 0 else 0

    if num_features < max_features:
        # Create a padding tensor
        padding = torch.zeros((num_nodes, max_features - num_features), dtype=graph.x.dtype)
        graph.x = torch.cat([graph.x, padding], dim=1)

    return graph

# Command line argument for verbose output
parser = argparse.ArgumentParser(description="Action Level Detection Transformer")
parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
parser.add_argument("--dataset", "-d", type=str, help="Directory containing the preprocessed dataset")
parser.add_argument("--split", "-s", type=str, help="Directory containing the preprocessed split")
args, unknown = parser.parse_known_args()

# Assign the dataset directory to the variable
dataset = args.dataset
split = args.split

# Clear CUDA cache
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

train_losses = []
val_losses = []

# Dataset Class
class SVGPCAPDataset(Dataset):
    """
    SVGPCAPDataset is a PyTorch Dataset class designed to handle data that combines SVG (Scalable Vector Graphics) files 
    and PCAP (Packet Capture) files. This dataset is particularly useful for tasks involving graph-based data from SVG files 
    and network traffic data from PCAP files.
    The class provides functionality for:
    - Loading and preprocessing SVG and PCAP data.
    - Extracting graph structures and features from SVG files.
    - Extracting network traffic features from PCAP files.
    - Applying data augmentation techniques to both SVG and PCAP data.
    - Handling preprocessed data for faster loading and training.
        data (list): A list to store dataset information, including file paths and labels.
        drop_prob (float): Probability of dropping nodes or packets for augmentation.
        dup_prob (float): Probability of duplicating nodes or packets for augmentation.
        vectorizer (TfidfVectorizer): Vectorizer for text data in SVG files.
        max_features (int): Maximum number of features across all graphs.
    Methods:
        __init__(file, svg_dir, pcap_dir, process_to_index=None, augment=False, noise_scale=0.1, drop_prob=0.1, 
                 dup_prob=0.1, time_noise_scale=0.1, preprocessed_dir=None, preprocessed=False):
            Initializes the dataset with the given parameters and loads data.
        __len__():
            Returns the number of data samples in the dataset.
        __getitem__(idx):
        get_ip_index(ip):
            Retrieves the index of a given IP address.
        collect_all_node_texts(svg_dir):
            Collects all unique text elements from SVG files in the specified directory.
        validate_graph(graph):
            Validates the graph structure and ensures it is connected.
        add_graph_features(graph):
            Adds node degrees and centrality measures as features to the graph.
        extract_unique_processes(svg_dir):
            Extracts unique text elements from SVG files and maps them to unique indices.
        extract_svg_graph(svg_path):
            Extracts a graph from an SVG file and processes it into a PyTorch Geometric Data object.
        extract_pcap_features(pcap_path):
            Extracts enhanced features from a PCAP file, such as unique IPs, ports, payload sizes, and time intervals.
        parse_connection(path_element):
            Parses an SVG path element to extract source and target nodes of a connection.
        parallel_extract_svg_graph(svg_files):
            Extracts graphs from a list of SVG files in parallel using multiprocessing.
        add_positional_encoding(graph):
            Adds random positional encoding to node features in the graph.
        preprocess_data(svg_files, pcap_files, svg_dir, pcap_dir, output_dir):
        get_file_hash(file_path):
        load_pcap_features(pcap_path):
            Extracts basic features from a PCAP file, such as unique IPs, ports, and average payload size.
        augment_svg_features(svg_features):
            Augments SVG features by adding random Gaussian noise.
        augment_pcap_features(pcap_features):
            Augments PCAP features by simulating network conditions, such as scaling and adding noise.
        augment_pcap_file(pcap_path):
            Augments a PCAP file by applying random transformations to packets and extracts features.
    Usage:
        This dataset can be used in PyTorch DataLoader for training machine learning models that require both graph-based 
        and network traffic data. It supports data augmentation and preprocessing for efficient training.
    """

    def __init__(self, file, svg_dir, pcap_dir, process_to_index=None, augment=False, noise_scale=0.1, drop_prob=0.1, dup_prob=0.1, time_noise_scale=0.1, preprocessed_dir=None, preprocessed=False):
        """
        Initialize the dataset with the given parameters.
        Args:
            labels_csv (str): Path to the CSV file containing labels.
            svg_dir (str): Directory containing SVG files.
            pcap_dir (str): Directory containing PCAP files.
            process_to_index (dict, optional): Mapping of process names to indices. Defaults to None.
            augment (bool, optional): Whether to apply data augmentation. Defaults to False.
            noise_scale (float, optional): Scale of noise to add for augmentation. Defaults to 0.1.
            drop_prob (float, optional): Probability of dropping nodes for augmentation. Defaults to 0.1.
            dup_prob (float, optional): Probability of duplicating nodes for augmentation. Defaults to 0.1.
            time_noise_scale (float, optional): Scale of time noise to add for augmentation. Defaults to 0.1.
            preprocessed_dir (str, optional): Directory to save/load preprocessed data. Defaults to None.
        Attributes:
            data (list): List to store dataset information.
            svg_dir (str): Directory containing SVG files.
            pcap_dir (str): Directory containing PCAP files.
            process_to_index (dict): Mapping of process names to indices.
            ip_to_index (dict): Mapping of IP addresses to indices.
            augment (bool): Whether to apply data augmentation.
            noise_scale (float): Scale of noise to add for augmentation.
            drop_prob (float): Probability of dropping nodes for augmentation.
            dup_prob (float): Probability of duplicating nodes for augmentation.
            time_noise_scale (float): Scale of time noise to add for augmentation.
            preprocessed_dir (str): Directory to save/load preprocessed data.
            vectorizer (TfidfVectorizer): Vectorizer for text data.
        """

        self.data = []
        self.svg_dir = svg_dir
        self.pcap_dir = pcap_dir
        self.process_to_index = process_to_index if process_to_index is not None else SVGPCAPDataset.extract_unique_processes(svg_dir)
        self.ip_to_index = {}
        self.augment = augment
        self.noise_scale = noise_scale
        self.drop_prob = drop_prob
        self.dup_prob = dup_prob
        self.time_noise_scale = time_noise_scale
        self.preprocessed_dir = preprocessed_dir

        if preprocessed:
            # Load the split
            with open(file, 'r') as f:
                self.files = [line.strip().split('\t') for line in f.readlines()]

            # Calculate the maximum number of features across all graphs
            self.max_features = 0
            for svg_file, _ in self.files:
                graph = torch.load(os.path.join(self.svg_dir, svg_file), weights_only=False)
                if graph.num_nodes > 0:
                    self.max_features = max(self.max_features, graph.x.size(1))
        else:
            # Ensure preprocessed_dir exists if provided
            if self.preprocessed_dir is not None:
                os.makedirs(self.preprocessed_dir, exist_ok=True)

            # Initialize vectorizer
            if preprocessed_dir and os.path.exists(os.path.join(preprocessed_dir, 'vectorizer.pkl')):
                with open(os.path.join(preprocessed_dir, 'vectorizer.pkl'), 'rb') as f:
                    self.vectorizer = pickle.load(f)
            else:
                all_texts = self.collect_all_node_texts(svg_dir)
                self.vectorizer = TfidfVectorizer().fit(all_texts)

            # Load data
            with open(file, 'r') as f:
                for line in f.readlines()[1:]:  # Skip header
                    svg_file, pcap_file, origin_ip, target_processes = line.strip().split(',')

                    # Split target processes into a list
                    target_processes_list = target_processes.split(';')

                    target_process_vector = set()
                    for process in target_processes_list:
                        if process in self.process_to_index:
                            target_process_vector.add(self.process_to_index[process])

                    # Map IP address to an integer
                    if origin_ip not in self.ip_to_index:
                        self.ip_to_index[origin_ip] = len(self.ip_to_index)

                    self.data.append((svg_file, pcap_file, origin_ip, target_process_vector))
                
    def __len__(self):
        return len(self.files) # self.data

    def get_ip_index(self, ip):
        return self.ip_to_index.get(ip, -1)  # Return -1 if IP is not found
    
    def collect_all_node_texts(self, svg_dir):
        """
        Collect all unique node texts from SVG files in the specified directory.
        This method parses each SVG file in the given directory, extracts text elements,
        and collects all unique texts to fit the TF-IDF vectorizer.
        Args:
            svg_dir (str): The directory containing SVG files to be processed.
        Returns:
            list: A list of unique text strings extracted from the SVG files.
        Raises:
            Exception: If there is an error parsing an SVG file, it will be caught and
                   printed if verbose mode is enabled.
        """

        # Initialize a set to store all unique texts
        all_texts = set()
        
        # Iterate over all SVG files in the directory
        for svg_file in os.listdir(svg_dir):
            if svg_file.endswith(".svg"):
                svg_path = os.path.join(svg_dir, svg_file)
                try:
                    # Parse the SVG file
                    tree = ET.parse(svg_path)
                    root = tree.getroot()
                    
                    # Extract text elements and add to the set
                    for elem in root.findall(".//{http://www.w3.org/2000/svg}text"):
                        all_texts.add(elem.text.strip())
                except Exception as e:
                    # Print error message if verbose mode is enabled
                    if args.verbose:
                        print(f"Error parsing {svg_path}: {e}")
        
        # Return the list of unique texts
        return list(all_texts)
    
    def validate_graph(self, graph):
        """
        Validate the graph structure and ensure it's connected.
        This method performs the following validations and modifications:
        1. If the graph has no nodes, it returns the graph as is.
        2. Ensures the graph is undirected. If not, converts the edge index to an undirected format.
        3. Checks if the graph is connected. If not, adds a dummy connection to handle disconnected graphs.
        Args:
            graph (torch_geometric.data.Data): The input graph to validate.
        Returns:
            torch_geometric.data.Data: The validated graph with necessary modifications.
        """
        if graph.num_nodes == 0:
            return graph
        
        # Ensure the graph is undirected
        if not is_undirected(graph.edge_index):
            graph.edge_index = to_undirected(graph.edge_index)
        
        # Check if the graph is connected
        if not is_connected(graph.edge_index, num_nodes=graph.num_nodes):
            # Handle disconnected graphs by adding a dummy connection
            if graph.num_nodes > 1:
                dummy_edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
                graph.edge_index = torch.cat([graph.edge_index, dummy_edge], dim=1)
        
        # Ensure the graph has at least one edge
        if graph.edge_index.shape[1] == 0:
            # Add a self-loop to the first node
            dummy_edge = torch.tensor([[0], [0]], dtype=torch.long)
            graph.edge_index = torch.cat([graph.edge_index, dummy_edge], dim=1)
        
        return graph
    
    def add_graph_features(self, graph):
        """
        Add node degrees and centrality measures as features to the graph.
        Parameters:
        graph (torch_geometric.data.Data): The input graph.
        Returns:
        torch_geometric.data.Data: The graph with added features.
        The function adds the following features to the graph:
        - Node degrees: The degree of each node is added as a feature.
        - Centrality measures: The degree centrality of each node is added as a feature if the graph has 1000 or fewer nodes.
        """

        if graph.num_nodes == 0:
            return graph
        
        # Add node degrees
        deg = degree(graph.edge_index[0], num_nodes=graph.num_nodes).float().unsqueeze(1)
        graph.x = torch.cat([graph.x, deg], dim=1)
        
        # Add centrality measures (if the graph is small enough)
        if graph.num_nodes <= 1000:
            nx_graph = to_networkx(graph, to_undirected=True)
            centrality = nx.degree_centrality(nx_graph)
            centrality = torch.tensor([centrality[i] for i in range(graph.num_nodes)], dtype=torch.float).unsqueeze(1)
            graph.x = torch.cat([graph.x, centrality], dim=1)
        
        return graph

    def extract_unique_processes(svg_dir):
        """
        Extracts unique text elements from SVG files in a given directory.
        This function scans through all SVG files in the specified directory, 
        extracts text elements, and returns a dictionary mapping each unique 
        text element to a unique index.
        Args:
            svg_dir (str): The directory containing SVG files to be processed.
        Returns:
            dict: A dictionary where keys are unique text elements found in the 
                  SVG files and values are unique indices assigned to each text element.
        Raises:
            Exception: If there is an error parsing an SVG file, an exception is caught 
                       and an error message is printed if verbose mode is enabled.
        """

        # Initialize a set to store unique processes
        unique_processes = set()
        
        # Iterate over all SVG files in the directory
        for svg_file in os.listdir(svg_dir):
            if svg_file.endswith(".svg"):
                start_time = time.time()
                svg_path = os.path.join(svg_dir, svg_file)
                try:
                    # Parse the SVG file
                    tree = ET.parse(svg_path)
                    root = tree.getroot()
                    
                    # Extract text elements and add to the set
                    for elem in root.findall(".//{http://www.w3.org/2000/svg}text"):
                        unique_processes.add(elem.text)
                except Exception as e:
                    # Print error message if verbose mode is enabled
                    if args.verbose:
                        print(f"Error parsing {svg_path}: {e}")
            
                # Print elapsed time for extracting processes if verbose mode is enabled
                elapsed_time = time.time() - start_time
                if args.verbose:
                    print(f"Elapsed time for extracting processes for file {svg_file}: {elapsed_time:.2f} seconds")
        
        # Return a dictionary mapping each unique process to a unique index
        return {process: idx for idx, process in enumerate(unique_processes)}
    
    def extract_svg_graph(self, svg_path):
        """
        Extracts a graph from an SVG file.
        Args:
            svg_path (str): The path to the SVG file.
        Returns:
            Data: A PyTorch Geometric Data object representing the graph. The Data object contains:
                - x (torch.Tensor): Node features.
                - edge_index (torch.Tensor): Edge indices.
                - num_nodes (int): Number of nodes in the graph.
        The function performs the following steps:
            1. Parses the SVG file to extract nodes and edges.
            2. Converts node texts to numerical features using TF-IDF and applies dimensionality reduction.
            3. Converts the extracted data to PyTorch tensors.
            4. Creates a PyTorch Geometric Data object and applies additional transformations.
        If an error occurs during parsing, an empty graph is returned.
        Raises:
            Exception: If there is an error parsing the SVG file.
        """

        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # Extract nodes with their IDs and text content
            nodes = {}
            for elem in root.findall(".//{http://www.w3.org/2000/svg}g"):
                if 'class' in elem.attrib and elem.attrib['class'] == 'node':
                    node_id = elem.attrib['id']
                    text_elem = elem.find(".//{http://www.w3.org/2000/svg}text")
                    if text_elem is not None:
                        nodes[node_id] = text_elem.text.strip()
            
            # If no nodes are found, return an empty graph
            if not nodes:
                return Data(
                    x=torch.empty((0, 1), dtype=torch.float),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    num_nodes=0
                )
            
            # Create node ID to index mapping
            node_id_to_idx = {node_id: idx for idx, node_id in enumerate(nodes.keys())}
            
            # Extract edges from path elements
            edges = []
            for path in root.findall(".//{http://www.w3.org/2000/svg}path"):
                # Extract source and target from the path's class attribute
                if 'class' in path.attrib and path.attrib['class'] == 'edge':
                    # Parse the edge title to get source and target
                    title = path.find(".//{http://www.w3.org/2000/svg}title")
                    if title is not None:
                        src_tgt = title.text.split('->')
                        if len(src_tgt) == 2:
                            src = src_tgt[0].strip()
                            tgt = src_tgt[1].strip()
                            if src in node_id_to_idx and tgt in node_id_to_idx:
                                edges.append([node_id_to_idx[src], node_id_to_idx[tgt]])
            
            # Convert node texts to numerical features using TF-IDF
            node_texts = list(nodes.values())
            if node_texts:
                node_features = self.vectorizer.transform(node_texts).toarray()
                # Apply dimensionality reduction

                #svd = TruncatedSVD(n_components=512, algorithm='randomized')
                #node_features = svd.fit_transform(node_features)
                # Convert the feature matrix to a Dask array
                node_features_dask = da.from_array(node_features, chunks=(1000, 1000))

                # Perform SVD using Dask
                svd = TruncatedSVD(n_components=1000)
                node_features = svd.fit_transform(node_features_dask).compute()
                print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum()}")
            else:
                node_features = np.zeros((len(nodes), 1))
            
            # Convert to PyTorch tensors
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
            
            # Create the graph and apply transformations
            graph = Data(x=torch.tensor(node_features, dtype=torch.float), edge_index=edge_index, num_nodes=len(nodes))
            graph = self.validate_graph(graph)  # Ensure the graph is valid
            graph = self.add_graph_features(graph)  # Add graph-based features
            graph = self.add_positional_encoding(graph)  # Add positional encoding
            
            return graph
        
        except Exception as e:
            if args.verbose:
                print(f"Error parsing {svg_path}: {e}")
            return Data(
                x=torch.empty((0, 1), dtype=torch.float),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                num_nodes=0
            )

    def parse_connection(self, path_element):
        """
        Args:
            path_element (xml.etree.ElementTree.Element): The SVG path element to parse.
        Returns:
            tuple: A tuple containing the source node and target node as strings. 
               If the path element does not represent a connection, returns (None, None).
        Raises:
            Exception: If an error occurs during parsing and verbose mode is enabled, 
                   the error message will be printed.
        """

        try:
            # Extract connection information from the path's class attribute
            if 'class' in path_element.attrib and path_element.attrib['class'] == 'edge':
                title = path_element.find(".//{http://www.w3.org/2000/svg}title")
                if title is not None:
                    src_tgt = title.text.split('->')
                    if len(src_tgt) == 2:
                        return src_tgt[0].strip(), src_tgt[1].strip()
            return None, None
        except Exception as e:
            if args.verbose:
                print(f"Error parsing connection: {e}")
            return None, None

    def parallel_extract_svg_graph(self, svg_files):
        """
        Extract graphs from a list of SVG files in parallel.
        This method utilizes multiple processes to extract graphs from SVG files,
        leveraging all available CPU cores to speed up the extraction process.
        Args:
            svg_files (list of str): A list of file paths to the SVG files.
        Returns:
            list: A list of extracted graphs from the SVG files.
        """

        with Pool(processes=os.cpu_count()) as pool:
            graphs = pool.map(self.extract_svg_graph, svg_files)
        return graphs
    
    def add_positional_encoding(self, graph):
        """
        Add positional encoding to node features.
        This method generates a random 16-dimensional positional encoding for each node
        in the graph and concatenates it to the existing node features.
        Args:
            graph (torch_geometric.data.Data): The input graph with node features.
        Returns:
            torch_geometric.data.Data: The graph with updated node features including positional encoding.
        """

        if graph.num_nodes == 0:
            return graph
        
        # Generate random positional encoding
        pos_enc = torch.randn(graph.num_nodes, 16)  # 16-dimensional encoding
        graph.x = torch.cat([graph.x, pos_enc], dim=1)
        return graph

    def preprocess_data(self, svg_files, pcap_files, svg_dir, pcap_dir, output_dir):
        """
        Preprocesses SVG and PCAP files, extracts features, and saves them in the specified output directory.
        Args:
            svg_files (list): List of SVG file names to process.
            pcap_files (list): List of PCAP file names to process.
            svg_dir (str): Directory containing the SVG files.
            pcap_dir (str): Directory containing the PCAP files.
            output_dir (str): Directory where the processed data and vectorizer will be saved.
        Workflow:
            1. Creates the output directory if it does not exist.
            2. Processes each SVG file:
                - Extracts graph data from the SVG file.
                - Saves the graph data as a PyTorch tensor in the output directory.
            3. Processes each PCAP file:
                - Extracts features from the PCAP file.
                - Saves the features as a PyTorch tensor in the output directory.
            4. Saves the vectorizer object as a pickle file in the output directory.
            5. Processes all PCAP files in the PCAP directory:
                - Extracts features from each PCAP file.
                - Saves the features as PyTorch tensors in a subdirectory named 'pcap' within the output directory.
            6. Saves the vectorizer object again as a pickle file in the output directory.
        Note:
            - The method assumes the existence of `self.extract_svg_graph` and `self.extract_pcap_features` methods 
              for extracting graph data and features, respectively.
            - The `self.vectorizer` object is expected to be defined and initialized prior to calling this method.
        Raises:
            OSError: If there are issues creating directories or saving files.
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process SVG files
        for svg_file in svg_files:
            svg_path = os.path.join(svg_dir, svg_file)
            graph = self.extract_svg_graph(svg_path)
            torch.save(graph, os.path.join(f"{output_dir}/svg", f"{svg_file}.pt"))
        
        # Save vectorizer
        with open(os.path.join(output_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Process PCAP files
        self.save_pcap_sequences(pcap_files, pcap_dir, f"{output_dir}/pcap")
        
        # Save vectorizer
        with open(os.path.join(output_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
    def get_file_hash(self, file_path):
        """
        Computes the MD5 hash of a file.
        Args:
            file_path (str): The path to the file for which the hash is to be computed.
        Returns:
            str: The hexadecimal MD5 hash of the file's contents.
        Raises:
            FileNotFoundError: If the specified file does not exist.
            IOError: If there is an error reading the file.
        """
        
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def __getitem__(self, idx):
        """
        Retrieves and processes data for a given index.
        Args:
            idx (int): Index of the data to retrieve.
        Returns:
            tuple: A tuple containing the following elements:
            - svg_graph (torch.Tensor): The processed SVG graph with padded node features.
            - pcap_features (torch.Tensor): The processed PCAP features as a tensor.
            - origin_ip (int): A placeholder for the origin IP index (replace with actual value if available).
            - target_process_indices (torch.Tensor): A tensor containing indices of target processes.
            - target_process_values (torch.Tensor): A tensor containing values corresponding to the target processes.
        Notes:
            - The SVG graph is loaded from a file and its node features are padded to a fixed size.
            - The PCAP features are loaded from a file and converted to a tensor if necessary.
            - Dummy labels are used for `origin_ip`, `target_process_indices`, and `target_process_values`.
              Replace these with actual labels if available.
        """

        svg_file, pcap_file = self.files[idx]

        # Load SVG graph
        svg_graph = torch.load(os.path.join(self.svg_dir, svg_file), weights_only=False)

        # Pad node features to a fixed size
        svg_graph = pad_node_features(svg_graph, self.max_features)

        # Load PCAP features and convert to tensor
        pcap_features = torch.load(os.path.join(self.pcap_dir, pcap_file), weights_only=False)
        if isinstance(pcap_features, list):  # If features are loaded as a list
            pcap_features = torch.tensor(pcap_features, dtype=torch.float32)

        # Dummy labels (replace with actual labels if available)
        origin_ip = 0  # Replace with actual IP index
        target_process_indices = torch.tensor([0], dtype=torch.long)  # Replace with actual process indices
        target_process_values = torch.tensor([1.0], dtype=torch.float32)  # Replace with actual process values

        return (
            svg_graph,
            pcap_features,
            origin_ip,
            target_process_indices,
            target_process_values
        )

    def extract_payload(self, packet):
        if packet.payload:
            return bytes(packet.payload)
        return b''

    def encode_payload(self, payload):
        # Byte-level encoding (truncate/pad to fixed size)
        max_payload_size = 100  # Adjust as needed
        payload_encoded = list(payload[:max_payload_size])  # Truncate to max size
        payload_encoded += [0] * (max_payload_size - len(payload_encoded))  # Pad to max size
        return payload_encoded

    def extract_packet_features(self, pcap_path):
        packets = rdpcap(pcap_path)
        packet_features = []

        for packet in packets:
            features = {}

            # Extract headers
            if packet.haslayer('IP'):
                features['src_ip'] = packet['IP'].src
                features['dst_ip'] = packet['IP'].dst
                features['protocol'] = packet['IP'].proto

            if packet.haslayer('TCP'):
                features['src_port'] = packet['TCP'].sport
                features['dst_port'] = packet['TCP'].dport
                features['tcp_flags'] = packet['TCP'].flags  # FlagValue object

            if packet.haslayer('UDP'):
                features['src_port'] = packet['UDP'].sport
                features['dst_port'] = packet['UDP'].dport

            # Extract and analyze payload
            if packet.haslayer('Raw'):
                payload = packet['Raw'].load
                features['payload_analysis'] = self.analyze_payload(payload)

            # Temporal features
            features['timestamp'] = float(packet.time)
            packet_features.append(features)

        return packet_features

    def preprocess_pcap_file(self, pcap_path):
        packet_features = self.extract_packet_features(pcap_path)
        sequences = []

        for packet in packet_features:
            # Encode headers
            src_ip = int(packet.get('src_ip', '0.0.0.0').replace('.', ''))
            dst_ip = int(packet.get('dst_ip', '0.0.0.0').replace('.', ''))
            src_port = packet.get('src_port', 0)
            dst_port = packet.get('dst_port', 0)
            protocol = packet.get('protocol', 0)
            flags = packet.get('flags', 0)

            # Encode payload
            payload_encoded = self.encode_payload(packet.get('payload', b''))

            # Combine features into a single vector
            feature_vector = [src_ip, dst_ip, src_port, dst_port, protocol, flags] + payload_encoded
            sequences.append(torch.tensor(feature_vector, dtype=torch.float32))

        return torch.stack(sequences)  # Convert list of tensors to a single tensor

    def save_pcap_sequences(self, pcap_files, pcap_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for pcap_file in pcap_files:
            pcap_path = os.path.join(pcap_dir, pcap_file)
            sequences = self.preprocess_pcap_file(pcap_path)
            torch.save(sequences, os.path.join(output_dir, f"{pcap_file}.pt"))

    def analyze_payload(self, payload):
        features = {}

        # If payload is empty, return empty features
        if not payload:
            return features

        # Try to decode payload as text
        try:
            payload_text = payload.decode('utf-8', errors='ignore')
            features['is_text'] = True
        except UnicodeDecodeError:
            payload_text = None
            features['is_text'] = False

        # Analyze text-based payloads
        if payload_text:
            # HTTP analysis
            if 'HTTP' in payload_text:
                features['protocol'] = 'HTTP'
                http_headers = {}
                for line in payload_text.split('\r\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        http_headers[key.strip()] = value.strip()
                features['http_headers'] = http_headers

                # Extract URLs
                urls = re.findall(r'https?://[^\s"]+', payload_text)
                features['urls'] = urls

            # DNS analysis
            if payload.startswith(b'\x00\x01'):  # DNS query magic number
                features['protocol'] = 'DNS'
                try:
                    dns_query = payload[12:].split(b'\x00', 1)[0].decode('utf-8', errors='ignore')
                    features['dns_query'] = dns_query
                except Exception as e:
                    features['dns_query'] = None

            # FTP analysis
            if payload_text.startswith('USER') or payload_text.startswith('PASS'):
                features['protocol'] = 'FTP'
                features['ftp_command'] = payload_text.split('\r\n')[0]

            # SMTP analysis
            if payload_text.startswith('EHLO') or payload_text.startswith('MAIL FROM'):
                features['protocol'] = 'SMTP'
                features['smtp_command'] = payload_text.split('\r\n')[0]

        # Analyze binary payloads
        else:
            features['is_binary'] = True

            # Detect TLS/SSL handshake
            if payload.startswith(b'\x16\x03'):  # TLS handshake magic number
                features['protocol'] = 'TLS'
                features['tls_handshake'] = True

            # Detect file transfers (e.g., PDF, ZIP, etc.)
            if payload.startswith(b'%PDF'):
                features['file_type'] = 'PDF'
            elif payload.startswith(b'PK'):  # ZIP file magic number
                features['file_type'] = 'ZIP'
            elif payload.startswith(b'\x89PNG'):  # PNG file magic number
                features['file_type'] = 'PNG'

            # Detect encrypted data (e.g., AES, RSA)
            if len(payload) > 16 and all(32 <= byte <= 126 for byte in payload[:16]):  # ASCII range
                features['encryption'] = 'Possible AES/RSA'

        # Extract payload size
        features['payload_size'] = len(payload)

        # Extract entropy of payload (useful for detecting encrypted/compressed data)
        if payload:
            entropy = self.calculate_entropy(payload)
            features['entropy'] = entropy

        return features

    def calculate_entropy(self, data):
        if not data:
            return 0.0

        # Count frequency of each byte
        byte_counts = Counter(data)
        total_bytes = len(data)

        # Calculate entropy
        entropy = 0.0
        for count in byte_counts.values():
            probability = count / total_bytes
            entropy -= probability * math.log2(probability)

        return entropy
    
    def augment_svg_features(self, svg_features):
        """
        Augments the given SVG features by adding random noise.
        This method perturbs the input SVG features by adding Gaussian noise
        to each feature. The noise is generated with a mean of 0 and a standard
        deviation defined by the `self.noise_scale` attribute.
        Args:
            svg_features (numpy.ndarray): A 1D array of SVG features to be augmented.
        Returns:
            numpy.ndarray: A 1D array of SVG features with added noise.
        """

        # Perturb SVG features with noise
        noise = np.random.normal(0, self.noise_scale, size=len(svg_features))
        return svg_features + noise

    def augment_pcap_features(self, pcap_features):
        """
        Augments the given PCAP (Packet Capture) features by simulating network conditions.
        This method applies random scaling and noise to specific features of the input
        PCAP data to simulate variations in network conditions. The modifications include:
        - Scaling the number of source IPs and destination IPs by a random factor.
        - Adding Gaussian noise to the mean packet size.
        Args:
            pcap_features (list or np.ndarray): A list or array containing PCAP features.
                Expected indices:
                - Index 0: Number of source IPs.
                - Index 1: Number of destination IPs.
                - Index 4: Mean packet size.
        Returns:
            list or np.ndarray: The augmented PCAP features with simulated variations.
        """

        # Simulate network conditions
        pcap_features[0] *= np.random.uniform(0.8, 1.2)  # Randomly scale the number of source IPs
        pcap_features[1] *= np.random.uniform(0.8, 1.2)  # Randomly scale the number of destination IPs
        pcap_features[4] += np.random.normal(0, self.noise_scale)  # Add noise to mean packet size
        return pcap_features

    def augment_pcap_file(self, pcap_path):
        """
        Augments a PCAP file by applying random transformations to the packets and extracts features.
        This method performs the following operations on the packets in the given PCAP file:
        - Randomly drops packets based on a predefined probability (`self.drop_prob`).
        - Randomly duplicates packets based on a predefined probability (`self.dup_prob`).
        - Adds Gaussian noise to packet timestamps with a scale defined by `self.time_noise_scale`.
        - Randomly reorders the packets.
        After augmentation, the method extracts the following features from the packets:
        - Total number of packets (`num_packets`).
        - Mean packet size (`mean_packet_size`).
        - Standard deviation of packet sizes (`std_packet_size`).
        - Number of unique protocols in the packets (`unique_protocols`).
        Args:
            pcap_path (str): Path to the PCAP file to be augmented.
        Returns:
            list: A list of extracted feature values in the following order:
            [num_packets, mean_packet_size, std_packet_size, unique_protocols].
        """

        packets = rdpcap(pcap_path)
        augmented_packets = []

        for packet in packets:
            # Randomly drop packets
            if np.random.rand() < self.drop_prob:
                continue
            
            # Randomly duplicate packets
            if np.random.rand() < self.dup_prob:
                augmented_packets.append(packet)
            
            # Add noise to packet timestamps
            packet.time += np.random.normal(0, self.time_noise_scale)
            
            augmented_packets.append(packet)
        
        # Reorder packets randomly
        np.random.shuffle(augmented_packets)
        
        # Extract features from augmented packets
        num_packets = len(augmented_packets)
        packet_sizes = [len(pkt) for pkt in augmented_packets]
        protocols = [pkt.payload.name for pkt in augmented_packets]
        intervals = np.diff([pkt.time for pkt in augmented_packets]) if len(augmented_packets) > 1 else [0]

        features = {
            'num_packets': num_packets,
            'mean_packet_size': np.mean(packet_sizes) if packet_sizes else 0,
            'std_packet_size': np.std(packet_sizes) if packet_sizes else 0,
            'unique_protocols': len(set(protocols)),
        }
        return list(features.values())

# Model Definition
class GNNModel(nn.Module):
    def __init__(self, svg_dim, pcap_dim, hidden_dim, num_processes, num_ips):
        super().__init__()
        # Graph components (unchanged)
        self.conv1 = GCNConv(svg_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        
        # Replace CNN with LSTM for PCAP processing
        self.pcap_rnn = nn.LSTM(
            input_size=pcap_dim,  # Number of features per timestep
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.rnn_dropout = nn.Dropout(0.3)
        
        # Output heads (modified for RNN output)
        combined_dim = hidden_dim * 3  # Graph + bidirectional RNN
        self.process_head = nn.Linear(combined_dim, num_processes)
        self.origin_head = nn.Linear(combined_dim, num_ips)

    def forward(self, graph_batch, pcap_features, lengths=None):
        # Graph processing (unchanged)
        graph_batch.x = self.pad_features(graph_batch.x, target_dim=self.conv1.in_channels)
        x = self.conv1(graph_batch.x, graph_batch.edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, graph_batch.edge_index)
        graph_embed = global_mean_pool(x, graph_batch.batch)  # [batch_size, hidden_dim]

        # RNN processing with lengths
        pcap_features = pcap_features.float()
        
        if lengths is not None:
            # Sort sequences by length (descending) for pack_padded_sequence
            lengths = lengths.cpu()  # Move to CPU if needed
            lengths, sort_idx = torch.sort(lengths, descending=True)
            pcap_features = pcap_features[sort_idx]
            
            # Pack the sequences
            packed_input = nn.utils.rnn.pack_padded_sequence(
                pcap_features, 
                lengths, 
                batch_first=True
            )
            packed_output, (h_n, c_n) = self.pcap_rnn(packed_input)
            
            # Get the last hidden states (bidirectional concatenated)
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch_size, hidden_dim*2]
            
            # Undo the sorting
            _, unsort_idx = torch.sort(sort_idx)
            last_hidden = last_hidden[unsort_idx]
        else:
            # Process without lengths (all sequences same length)
            rnn_output, (h_n, c_n) = self.pcap_rnn(pcap_features)
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        last_hidden = self.rnn_dropout(last_hidden)
        
        # Combine features
        combined = torch.cat([graph_embed, last_hidden], dim=1)
        
        return self.origin_head(combined), self.process_head(combined)

    def pad_features(self, x, target_dim):
        if x.shape[1] < target_dim:
            pad_size = target_dim - x.shape[1]
            x = torch.cat([x, torch.zeros((x.shape[0], pad_size), device=x.device)], dim=1)
        return x

class WeightedFocalLoss(nn.Module):
    """
    A custom implementation of the Weighted Focal Loss function, which is commonly used 
    for addressing class imbalance in classification tasks. This loss function applies 
    a modulating factor to the standard binary cross-entropy loss to focus learning on 
    hard-to-classify examples.
    Attributes:
        alpha (float): A scaling factor to balance the importance of positive/negative examples.
        gamma (float): The focusing parameter that reduces the relative loss for well-classified examples.
        weights (torch.Tensor, optional): A tensor of weights for each class to handle class imbalance.
    Methods:
        forward(inputs, targets):
            Computes the Weighted Focal Loss for the given inputs and targets.
    Args:
        alpha (float, optional): A scaling factor for the loss. Default is 1.
        gamma (float, optional): The focusing parameter to adjust the rate at which easy examples are down-weighted. Default is 2.
        weights (torch.Tensor, optional): A tensor of class weights. If provided, it is used to weight the loss for each class. Default is None.
    Forward Args:
        inputs (torch.Tensor): The predicted logits from the model (before applying sigmoid).
        targets (torch.Tensor): The ground truth binary labels (0 or 1).
    Returns:
        torch.Tensor: The computed Weighted Focal Loss value.
    """

    def __init__(self, alpha=1, gamma=2, weights=None):
        """
        Initializes the WeightedFocalLoss class.
        Parameters:
            alpha (float, optional): A scaling factor for the positive class. Default is 1.
            gamma (float, optional): A focusing parameter to adjust the rate at which easy examples are down-weighted. Default is 2.
            weights (torch.Tensor or None, optional): Class-specific weights to handle class imbalance. Default is None.
        """

        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights

    def forward(self, inputs, targets):
        """
        Computes the forward pass for the loss function, which is a variant of the 
        Focal Loss used for addressing class imbalance in binary classification tasks.
        Args:
            inputs (torch.Tensor): The predicted logits from the model. Shape: (N, *) 
                where N is the batch size and * represents additional dimensions.
            targets (torch.Tensor): The ground truth binary labels. Shape: (N, *), 
                matching the shape of `inputs`.
        Returns:
            torch.Tensor: The computed focal loss, averaged over the batch.
        Notes:
            - The loss is based on the Binary Cross-Entropy (BCE) with logits, 
              modified by a weighting factor and a focusing parameter.
            - If `self.weights` is provided, it is used to apply class-specific 
              weights to the loss.
            - `self.alpha` controls the balance between positive and negative samples.
            - `self.gamma` adjusts the focusing parameter to reduce the impact of 
              easy-to-classify examples.
        """

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        if self.weights is not None:
            weights = self.weights[targets.long()]
            focal_loss = self.alpha * weights * (1 - pt) ** self.gamma * BCE_loss
        else:
            focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# Training and Validation Loop
def train_model():
    """
    Trains a Graph Neural Network (GNN) model using preprocessed SVG and PCAP datasets.
    The function performs the following steps:
    1. Loads a pre-trained vectorizer from a pickle file.
    2. Initializes training, validation, and test datasets using the `SVGPCAPDataset` class.
    3. Creates DataLoaders for batching and shuffling the datasets.
    4. Initializes the GNN model, optimizer, and loss functions.
    5. Executes a training loop for a fixed number of epochs:
        - Processes batches of data, including graphs and PCAP features.
        - Computes the process loss using binary cross-entropy with logits.
        - Computes the origin loss using cross-entropy.
        - Combines the losses and performs backpropagation.
        - Updates the model parameters using the optimizer.
        - Tracks and prints the training loss for each epoch.
    6. Validates the model after each epoch using a separate validation function.
    Note:
    - The model is trained on a GPU if available, otherwise on a CPU.
    - The number of processes and other hyperparameters are hardcoded in the function.
    - The function assumes the existence of specific file paths and preprocessed data.
    Dependencies:
    - PyTorch for model training and tensor operations.
    - Custom classes and functions such as `SVGPCAPDataset`, `GNNModel`, `custom_collate_fn`, 
      and `validate_model`.
    Raises:
    - FileNotFoundError: If the vectorizer file or dataset files are not found.
    - RuntimeError: If there are issues with tensor operations or GPU availability.
    Returns:
    None
    """

    # Load the vectorizer
    with open(f"{dataset}/train/vectorizer.pkl", 'rb') as f:
        vectorizer = pickle.load(f)

    # Create datasets
    train_dataset = SVGPCAPDataset(f"{split}/train_files.txt", f"{dataset}/train/svg", f"{dataset}/train/pcap", preprocessed=True)
    val_dataset = SVGPCAPDataset(f"{split}/val_files.txt", f"{dataset}/val/svg", f"{dataset}/val/pcap", preprocessed=True)
    test_dataset = SVGPCAPDataset(f"{split}/test_files.txt", f"{dataset}/test/svg", f"{dataset}/test/pcap", preprocessed=True)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    # Initialize model, optimizer, and loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(svg_dim=train_dataset.max_features, pcap_dim=106, hidden_dim=128, num_processes=10, num_ips=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion_process = WeightedFocalLoss()
    criterion_origin = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(50):
        model.train()
        total_loss = 0

        for batch in train_loader:
            graph_batch, pcap_features, masks, lengths, origin_indices, proc_indices, proc_values = batch
            graph_batch = graph_batch.to(device)  # Move graph_batch to the correct device
            pcap_features = pcap_features.to(device)  # Move pcap_features to the correct device
            masks = masks.to(device)  # Move masks to the correct device
            lengths = lengths.to(device)  # Move lengths to the correct device

            origin_output, process_output = model(graph_batch, pcap_features, lengths)
            
            # Process loss
            process_target = torch.sparse_coo_tensor(
                torch.stack([torch.arange(len(proc_indices)).repeat_interleave(
                    torch.tensor([len(indices) for indices in proc_indices])),
                    torch.cat(proc_indices)
                ]),
                torch.cat(proc_values),
                (len(proc_indices), 10)  # Replace 10 with the actual number of processes
            ).to(device)
            
            loss_process = F.binary_cross_entropy_with_logits(
                process_output, 
                process_target.to_dense()
            )
            
            origin_indices = origin_indices.to(device)
            # Origin loss
            loss_origin = F.cross_entropy(origin_output, origin_indices)
            
            # Combine the losses
            total_batch_loss = loss_process + loss_origin
            
            # Backpropagate the total loss
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()

        train_losses.append(total_loss / len(train_loader))
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
        #torch.cuda.empty_cache()
        val_loss = validate_model(model, val_loader, device, val_dataset, 10, criterion_process, criterion_origin)  # Replace 10 with the actual number of processes
        val_losses.append(val_loss)

# Validation Function
def validate_model(model, dataloader, device, dataset, num_processes, criterion_process, criterion_origin):
    """
    Validates the performance of a given model on a dataset using a dataloader.
    Args:
        model (torch.nn.Module): The model to be validated.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the validation data.
        device (torch.device): The device (CPU or GPU) to run the validation on.
        dataset (Dataset): The dataset being used for validation.
        num_processes (int): The number of processes in the dataset.
        criterion_process (torch.nn.Module): Loss function for process-level predictions.
        criterion_origin (torch.nn.Module): Loss function for origin-level predictions.
    Returns:
        float: The average validation loss.
    This function performs the following steps:
    - Sets the model to evaluation mode.
    - Iterates through the dataloader to compute predictions and losses.
    - Computes the combined loss for process-level and origin-level predictions.
    - Stores predictions and labels for further evaluation.
    - Tunes thresholds for process-level predictions to maximize F1 score.
    - Computes and prints various metrics, including precision, recall, F1 score, and ROC-AUC.
    - Computes and prints accuracy for origin-level predictions.
    """

    model.eval()
    all_origin_labels = []
    all_origin_preds = []
    all_process_labels = []
    all_process_probs = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            # Unpack the batch (include masks)
            graph_batch, pcap_features, masks, lengths, origin_ips, target_process_indices, target_process_values = batch
            
            # Move data to the correct device
            graph_batch = graph_batch.to(device)
            pcap_features = pcap_features.to(device)
            masks = masks.to(device)  # Move masks to the correct device
            lengths = lengths.to(device)  # Move lengths to the correct device
            
            # Convert origin_ips (indices) to tensor
            origin_indices = origin_ips.to(device)
            
            # Create sparse tensor from indices and values
            sparse_indices = []
            for i, indices in enumerate(target_process_indices):
                for idx in indices:
                    sparse_indices.append([i, idx])  # [batch_index, process_index]
            
            sparse_indices = torch.tensor(sparse_indices, dtype=torch.long).t()  # Transpose for sparse tensor format
            sparse_values = torch.tensor([v for values in target_process_values for v in values], dtype=torch.float32)
            
            # Create sparse tensor
            sparse_size = (len(graph_batch), num_processes)  # [batch_size, num_processes]
            target_process_sparse = torch.sparse_coo_tensor(sparse_indices, sparse_values, sparse_size).to(device)
            
            # Convert sparse tensor to dense tensor (if needed)
            process_labels = target_process_sparse.to_dense()

            # Get both outputs from the model
            origin_output, process_output = model(graph_batch, pcap_features, lengths)

            # Compute process loss
            loss_process = criterion_process(process_output, process_labels)
            
            # Compute origin loss
            loss_origin = criterion_origin(origin_output, origin_indices)
            
            # Combine the losses
            total_batch_loss = loss_process + loss_origin
            total_loss += total_batch_loss.item()

            # Store origin predictions and labels
            origin_preds = torch.argmax(origin_output, dim=1)
            all_origin_labels.extend(origin_indices.cpu().numpy())
            all_origin_preds.extend(origin_preds.cpu().numpy())

            # Store process probabilities and labels for threshold tuning
            process_probs = torch.sigmoid(process_output).cpu().numpy()
            all_process_labels.extend(process_labels.cpu().numpy().flatten())
            all_process_probs.extend(process_probs.flatten())

    # Compute average validation loss
    avg_val_loss = total_loss / len(dataloader)

    # Threshold Tuning
    thresholds = np.linspace(0.1, 0.9, 9)
    best_threshold = 0.5
    best_f1 = 0

    # Store metrics for each threshold
    threshold_metrics = []

    for threshold in thresholds:
        process_preds = (np.array(all_process_probs) > threshold).astype(int)
        
        # Compute per-class metrics
        precision_per_class = precision_score(all_process_labels, process_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_process_labels, process_preds, average=None, zero_division=0)
        f1_per_class = f1_score(all_process_labels, process_preds, average=None, zero_division=0)
        
        # Compute macro/micro averages
        precision_macro = precision_score(all_process_labels, process_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_process_labels, process_preds, average='macro', zero_division=0)
        f1_macro = f1_score(all_process_labels, process_preds, average='macro', zero_division=0)
        
        precision_micro = precision_score(all_process_labels, process_preds, average='micro', zero_division=0)
        recall_micro = recall_score(all_process_labels, process_preds, average='micro', zero_division=0)
        f1_micro = f1_score(all_process_labels, process_preds, average='micro', zero_division=0)
        
        # Store metrics
        threshold_metrics.append({
            'threshold': threshold,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
        })
        
        # Update best threshold based on micro F1
        if f1_micro > best_f1:
            best_f1 = f1_micro
            best_threshold = threshold

    # Print best threshold and metrics
    print(f"Best Threshold: {best_threshold}, Best Micro F1 Score: {best_f1:.4f}")

    # Compute metrics using the best threshold
    best_metrics = next(m for m in threshold_metrics if m['threshold'] == best_threshold)
    print("\nMetrics at Best Threshold:")
    print(f"Per-Class Precision: {best_metrics['precision_per_class']}")
    print(f"Per-Class Recall: {best_metrics['recall_per_class']}")
    print(f"Per-Class F1: {best_metrics['f1_per_class']}")
    print(f"Macro Precision: {best_metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {best_metrics['recall_macro']:.4f}")
    print(f"Macro F1: {best_metrics['f1_macro']:.4f}")
    print(f"Micro Precision: {best_metrics['precision_micro']:.4f}")
    print(f"Micro Recall: {best_metrics['recall_micro']:.4f}")
    print(f"Micro F1: {best_metrics['f1_micro']:.4f}")

    # Compute ROC-AUC
    try:
        avg_process_auc = roc_auc_score(all_process_labels, all_process_probs, average='macro')
    except ValueError:
        avg_process_auc = 0.5  # Default value if ROC-AUC cannot be computed
    print(f"ROC-AUC (Macro): {avg_process_auc:.4f}")

    # Metrics for origin IP
    origin_accuracy = accuracy_score(all_origin_labels, all_origin_preds)
    print(f"\nOrigin IP Accuracy: {origin_accuracy:.4f}")

    return avg_val_loss

if __name__ == "__main__":
    train_model()

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()