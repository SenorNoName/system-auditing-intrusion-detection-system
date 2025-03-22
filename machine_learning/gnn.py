import os
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
import pickle
from functools import partial
from multiprocessing import Pool
import matplotlib.pyplot as plt
import dask.array as da
from dask_ml.decomposition import TruncatedSVD

def custom_collate_fn(batch):
    """
    Custom collate function to prepare batches for a DataLoader.
    Args:
        batch (list of tuples): A list where each element is a tuple containing:
            - svg_graphs (Data): Graph data object.
            - pcap_features (Tensor): Tensor of pcap features.
            - origin_ips (int): Origin IP address as an integer.
            - target_process_indices (list): List of target process indices.
            - target_process_values (list): List of target process values.
    Returns:
        tuple: A tuple containing:
            - graph_batch (Batch): Batched graph data object.
            - pcap_features (Tensor): Batched tensor of pcap features.
            - origin_ips (Tensor): Batched tensor of origin IP addresses.
            - target_process_indices (list): List of target process indices.
            - target_process_values (list): List of target process values.
    """

    # Filter out invalid graphs
    valid_indices = [i for i, item in enumerate(batch) if item[0].num_nodes > 0]
    batch = [batch[i] for i in valid_indices]
    
    svg_graphs = [item[0] for item in batch]
    pcap_features = torch.stack([item[1] for item in batch])
    origin_ips = torch.tensor([item[2] for item in batch], dtype=torch.long)
    target_process_indices = [item[3] for item in batch]
    target_process_values = [item[4] for item in batch]
    
    # Batch the graphs using PyTorch Geometric's utility
    graph_batch = Batch.from_data_list(svg_graphs)
    
    return (
        graph_batch,
        pcap_features,
        origin_ips,
        target_process_indices,
        target_process_values
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
args = parser.parse_args()

# Clear CUDA cache
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

train_losses = []
val_losses = []

# Dataset Class
class SVGPCAPDataset(Dataset):
    def __init__(self, split_file, svg_dir, pcap_dir, process_to_index=None, augment=False, noise_scale=0.1, drop_prob=0.1, dup_prob=0.1, time_noise_scale=0.1, preprocessed_dir=None):
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

        # Load the split
        with open(split_file, 'r') as f:
            self.files = [line.strip().split('\t') for line in f.readlines()]

        # Calculate the maximum number of features across all graphs
        self.max_features = 0
        for svg_file, _ in self.files:
            graph = torch.load(os.path.join(self.svg_dir, svg_file), weights_only=False)
            if graph.num_nodes > 0:
                self.max_features = max(self.max_features, graph.x.size(1))
        '''

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
        with open(labels_csv, 'r') as f:
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

        '''
                
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
        
    def extract_pcap_features(self, pcap_path):
        """
        Extract enhanced features from a PCAP file.
        Parameters:
        pcap_path (str): The file path to the PCAP file.
        Returns:
        list: A list of 10 features extracted from the PCAP file:
            - Number of unique source IP addresses
            - Number of unique destination IP addresses
            - Number of unique source ports
            - Number of unique destination ports
            - Mean payload size
            - Standard deviation of payload sizes
            - Number of unique protocols
            - Mean time interval between packets
            - Standard deviation of time intervals between packets
            - Total number of packets
        """

        packets = rdpcap(pcap_path)
        if len(packets) == 0:
            return [0] * 10  # Return zero features for empty PCAPs
        
        src_ips = set()
        dst_ips = set()
        src_ports = set()
        dst_ports = set()
        payload_sizes = []
        protocols = set()
        timestamps = []
        
        for pkt in packets:
            if 'IP' in pkt:
                src_ips.add(pkt['IP'].src)
                dst_ips.add(pkt['IP'].dst)
            if 'TCP' in pkt:
                src_ports.add(pkt['TCP'].sport)
                dst_ports.add(pkt['TCP'].dport)
                payload_sizes.append(len(pkt['TCP'].payload))
            if 'UDP' in pkt:
                src_ports.add(pkt['UDP'].sport)
                dst_ports.add(pkt['UDP'].dport)
                payload_sizes.append(len(pkt['UDP'].payload))
            protocols.add(pkt.payload.name if hasattr(pkt, 'payload') else 'unknown')
            timestamps.append(float(pkt.time))  # Convert Scapy timestamp to float
        
        # Calculate temporal features
        time_intervals = np.diff(timestamps) if len(timestamps) > 1 else np.array([0.0])
        
        features = [
            len(src_ips), len(dst_ips), len(src_ports), len(dst_ports),
            float(np.mean(payload_sizes)) if payload_sizes else 0.0,  # Convert to float
            float(np.std(payload_sizes)) if payload_sizes else 0.0,   # Convert to float
            len(protocols),
            float(np.mean(time_intervals)) if len(time_intervals) > 0 else 0.0,  # Convert to float
            float(np.std(time_intervals)) if len(time_intervals) > 0 else 0.0,   # Convert to float
            len(packets),
        ]
        return features

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
        os.makedirs(output_dir, exist_ok=True)
        
        # Process SVG files
        for svg_file in svg_files:
            svg_path = os.path.join(svg_dir, svg_file)
            graph = self.extract_svg_graph(svg_path)
            torch.save(graph, os.path.join(output_dir, f"{svg_file}.pt"))
        
        # Process PCAP files
        for pcap_file in pcap_files:
            pcap_path = os.path.join(pcap_dir, pcap_file)
            features = self.extract_pcap_features(pcap_path)
            torch.save(features, os.path.join(output_dir, f"{pcap_file}.pt"))
        
        # Save vectorizer
        with open(os.path.join(output_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Process PCAP files
        pcap_output_dir = os.path.join(output_dir, 'pcap')
        os.makedirs(pcap_output_dir, exist_ok=True)
        for pcap_file in os.listdir(pcap_dir):
            if pcap_file.endswith(".pcap"):
                pcap_path = os.path.join(pcap_dir, pcap_file)
                features = self.extract_pcap_features(pcap_path)
                torch.save(features, os.path.join(pcap_output_dir, f"{pcap_file}.pt"))
        
        # Save vectorizer
        with open(os.path.join(output_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        

    def get_file_hash(self, file_path):
        """Generate a hash for a file to use as a cache key."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def __getitem__(self, idx):
        '''
        svg_file, pcap_file, origin_ip, target_process_vector = self.data[idx]
        target_process_indices = sorted(target_process_vector)
        
        # Use cached data if available and preprocessed_dir is provided
        if self.preprocessed_dir is not None:
            cache_key = f"{self.get_file_hash(os.path.join(self.svg_dir, svg_file))}_{self.get_file_hash(os.path.join(self.pcap_dir, pcap_file))}"
            cache_path = os.path.join(self.preprocessed_dir, f"{cache_key}.pt")
            
            if os.path.exists(cache_path):
                svg_graph, pcap_features = torch.load(cache_path)
            else:
                # Extract SVG graph
                svg_graph = self.extract_svg_graph(os.path.join(self.svg_dir, svg_file))
                
                # Load PCAP features
                pcap_features = self.load_pcap_features(os.path.join(self.pcap_dir, pcap_file))
                if self.augment:
                    pcap_features = self.augment_pcap_features(pcap_features)
                
                # Convert to tensors
                pcap_features = torch.tensor(pcap_features, dtype=torch.float32)
                
                # Cache the data
                torch.save((svg_graph, pcap_features), cache_path)
        else:
            # Extract SVG graph
            svg_graph = self.extract_svg_graph(os.path.join(self.svg_dir, svg_file))
            
            # Load PCAP features
            pcap_features = self.load_pcap_features(os.path.join(self.pcap_dir, pcap_file))
            if self.augment:
                pcap_features = self.augment_pcap_features(pcap_features)
            
            # Convert to tensors
            pcap_features = torch.tensor(pcap_features, dtype=torch.float32)
        
        target_process_indices = torch.tensor(sorted(target_process_vector), dtype=torch.long)
        target_process_values = torch.ones_like(target_process_indices, dtype=torch.float32)
        
        return (
            svg_graph,
            pcap_features,
            self.ip_to_index[origin_ip],  # Directly return index instead of IP string
            target_process_indices,
            target_process_values
        )
        '''

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

    def load_pcap_features(self, pcap_path):
        packets = rdpcap(pcap_path)
        if len(packets) == 0:
            return [0, 0, 0, 0, 0]
        
        src_ips = set()
        dst_ips = set()
        src_ports = set()
        dst_ports = set()
        payload_sizes = []
        
        for pkt in packets:
            if 'IP' in pkt:
                src_ips.add(pkt['IP'].src)
                dst_ips.add(pkt['IP'].dst)
            if 'TCP' in pkt:
                src_ports.add(pkt['TCP'].sport)
                dst_ports.add(pkt['TCP'].dport)
                payload_sizes.append(len(pkt['TCP'].payload))
        
        features = [
            len(src_ips), len(dst_ips), len(src_ports), len(dst_ports),
            np.mean(payload_sizes) if payload_sizes else 0,
        ]
        return features
    
    def augment_svg_features(self, svg_features):
        """
        Apply augmentation to SVG features.
        :param svg_features: List or array of SVG features.
        :return: Augmented SVG features.
        """
        # Perturb SVG features with noise
        noise = np.random.normal(0, self.noise_scale, size=len(svg_features))
        return svg_features + noise

    def augment_pcap_features(self, pcap_features):
        """
        Apply augmentation to PCAP features.
        :param pcap_features: List or array of PCAP features.
        :return: Augmented PCAP features.
        """
        # Simulate network conditions
        pcap_features[0] *= np.random.uniform(0.8, 1.2)  # Randomly scale the number of source IPs
        pcap_features[1] *= np.random.uniform(0.8, 1.2)  # Randomly scale the number of destination IPs
        pcap_features[4] += np.random.normal(0, self.noise_scale)  # Add noise to mean packet size
        return pcap_features

    def augment_pcap_file(self, pcap_path):
        """
        Apply network variations to a PCAP file.
        :param pcap_path: Path to the input PCAP file.
        :return: Augmented PCAP features.
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
        self.conv1 = GCNConv(svg_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)  # Add dropout
        self.pcap_cnn = nn.Sequential(
            nn.Conv1d(pcap_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1)
        )
        self.process_head = nn.Linear(hidden_dim * 2, num_processes)
        self.origin_head = nn.Linear(hidden_dim * 2, num_ips)

    def forward(self, graph_batch, pcap_features):
        graph_batch.x = self.pad_features(graph_batch.x, target_dim=self.conv1.in_channels)

        x = self.conv1(graph_batch.x, graph_batch.edge_index).relu()
        x = self.dropout(x)  # Apply dropout
        x = self.conv2(x, graph_batch.edge_index)
        graph_embed = global_mean_pool(x, graph_batch.batch)
        
        pcap_features = pcap_features.unsqueeze(2)
        pcap_embed = self.pcap_cnn(pcap_features).mean(dim=2)
        
        combined = torch.cat([graph_embed, pcap_embed], dim=1)
        return self.origin_head(combined), self.process_head(combined)

    def pad_features(self, x, target_dim=1015):
        """
        Pads the feature matrix to the target dimension if necessary.
        Args:
            x (torch.Tensor): Input feature matrix.
            target_dim (int): Desired feature size.
        Returns:
            torch.Tensor: Padded feature matrix.
        """
        if x.shape[1] < target_dim:
            pad_size = target_dim - x.shape[1]
            x = torch.cat([x, torch.zeros((x.shape[0], pad_size), device=x.device)], dim=1)
        return x

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weights=None):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights

    def forward(self, inputs, targets):
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
    '''
    process_to_index = SVGPCAPDataset.extract_unique_processes('data/svg')
    num_processes = len(process_to_index)  # Number of unique processes
    print(f"Number of unique processes: {num_processes}")
    
    # Create datasets with and without augmentation
    start_time = time.time()
    train_dataset = SVGPCAPDataset('data/labels.csv', 'data/svg', 'data/pcap', process_to_index=process_to_index, augment=True, noise_scale=0.1, preprocessed_dir=None)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time for loading training dataset: {elapsed_time:.2f} seconds")

    start_time = time.time()
    val_dataset = SVGPCAPDataset('data/labels.csv', 'data/svg', 'data/pcap', process_to_index=process_to_index, augment=False)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time for loading validation dataset: {elapsed_time:.2f} seconds")

    # Use functools.partial to pass num_processes to custom_collate_fn
    collate_fn = partial(custom_collate_fn)
    
    # Use the partial function in DataLoader
    start_time = time.time()
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=6, 
        pin_memory=True,
        collate_fn=collate_fn  # Use the partial function
    )
    print(f"Training dataloader size: {len(train_dataloader)}")
    elapsed_time = time.time() - start_time
    print(f"Elapsed time for training dataloader: {elapsed_time:.2f} seconds")

    start_time = time.time()
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=6, 
        pin_memory=True,
        collate_fn=collate_fn  # Use the partial function
    )
    print(f"Validation dataloader size: {len(val_dataloader)}")
    elapsed_time = time.time() - start_time
    print(f"Elapsed time for validation dataloader: {elapsed_time:.2f} seconds")
    
    # In train_model()
    num_processes = len(train_dataset.process_to_index)
    num_ips = len(train_dataset.ip_to_index)
    svg_dim = len(train_dataset.vectorizer.get_feature_names_out())
    model = GNNModel(
        svg_dim=svg_dim,
        pcap_dim=5,  # Update based on enhanced PCAP features
        hidden_dim=128,
        num_processes=len(train_dataset.process_to_index),
        num_ips=len(train_dataset.ip_to_index)
    )

    # Initialize model, optimizer, and loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(svg_dim=3, pcap_dim=5, hidden_dim=128, num_processes=10, num_ips=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion_process = WeightedFocalLoss()
    criterion_origin = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            graph_batch, pcap_features, origin_indices, proc_indices, proc_values = batch
            
            origin_output, process_output = model(graph_batch, pcap_features)
            
            # Process loss
            process_target = torch.sparse_coo_tensor(
                torch.stack([torch.arange(len(proc_indices)).repeat_interleave(
                    torch.tensor([len(indices) for indices in proc_indices])),
                    torch.cat(proc_indices)
                ]),
                torch.cat(proc_values),
                (len(proc_indices), num_processes)
            ).to(device)
            
            loss_process = F.binary_cross_entropy_with_logits(
                process_output, 
                process_target.to_dense()
            )
            
            # Origin loss
            loss_origin = F.cross_entropy(origin_output, origin_indices)
            
            # Combine the losses
            total_batch_loss = loss_process + loss_origin
            
            # Backpropagate the total loss
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()

        train_losses.append(total_loss / len(train_dataloader))
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
        torch.cuda.empty_cache()
        validate_model(model, val_dataloader, device, val_dataset, num_processes)
    '''
    # Load the vectorizer
    with open('data/powerset_preprocessed/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Create datasets
    #train_dataset = SVGPCAPDataset('data/powerset_splits/train_files.txt', 'data/powerset_preprocessed/train/svg', 'data/powerset_preprocessed/train/pcap')
    train_dataset = SVGPCAPDataset('data/powerset_splits/train_files.txt', 'data/powerset_preprocessed/train/svg', 'data/powerset_preprocessed/train/pcap')
    #train_dataset.files = train_dataset.files[:100]  # Use only 100 samples
    val_dataset = SVGPCAPDataset('data/powerset_splits/val_files.txt', 'data/powerset_preprocessed/val/svg', 'data/powerset_preprocessed/val/pcap')
    #val_dataset.files = val_dataset.files[:30]
    test_dataset = SVGPCAPDataset('data/powerset_splits/test_files.txt', 'data/powerset_preprocessed/test/svg', 'data/powerset_preprocessed/test/pcap')

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    # Initialize model, optimizer, and loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(svg_dim=train_dataset.max_features, pcap_dim=10, hidden_dim=128, num_processes=10, num_ips=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion_process = WeightedFocalLoss()
    criterion_origin = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(20):
        model.train()
        total_loss = 0

        for batch in train_loader:
            graph_batch, pcap_features, origin_indices, proc_indices, proc_values = batch
            #print(f"TRAINING graph_batch.x shape: {graph_batch.x.shape}")
            #origin_output, process_output = model(graph_batch, pcap_features)
            graph_batch = graph_batch.to(device)  # Move graph_batch to the correct device
            pcap_features = pcap_features.to(device)  # Move pcap_features to the correct device

            origin_output, process_output = model(graph_batch, pcap_features)

            
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
    model.eval()
    all_origin_labels = []
    all_origin_preds = []
    all_process_labels = []
    all_process_probs = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            # Unpack the batch
            graph_batch, pcap_features, origin_ips, target_process_indices, target_process_values = batch
            #print(f"VALIDATION graph_batch.x shape: {graph_batch.x.shape}")
            # Move data to the correct device
            graph_batch = graph_batch.to(device)
            pcap_features = pcap_features.to(device)
            
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
            origin_output, process_output = model(graph_batch, pcap_features)

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