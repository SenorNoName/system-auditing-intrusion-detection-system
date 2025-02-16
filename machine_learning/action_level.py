import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from xml.etree import ElementTree as ET
from scapy.all import rdpcap
import matplotlib.pyplot as plt
import argparse
import time
from torch.utils.data._utils.collate import default_collate
from functools import partial

def custom_collate_fn(batch, num_processes):
    svg_features = default_collate([item[0] for item in batch])
    pcap_features = default_collate([item[1] for item in batch])
    origin_ips = [item[2] for item in batch]  # List of IP addresses (no need to collate)
    
    # Process indices and values for sparse tensor
    target_process_indices = [item[3] for item in batch]
    target_process_values = [item[4] for item in batch]
    
    return svg_features, pcap_features, origin_ips, target_process_indices, target_process_values

parser = argparse.ArgumentParser(description="Action Level Detection Transformer")
parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

train_losses = []
val_losses = []

# Dataset Class
class SVGPCAPDataset(Dataset):
    def __init__(self, labels_csv, svg_dir, pcap_dir, process_to_index=None, augment=False, noise_scale=0.1, drop_prob=0.1, dup_prob=0.1, time_noise_scale=0.01):
        self.data = []
        self.svg_dir = svg_dir
        self.pcap_dir = pcap_dir
        self.process_to_index = process_to_index if process_to_index is not None else self.extract_unique_processes(svg_dir)
        self.ip_to_index = {}  # Mapping for IP addresses
        self.augment = augment
        self.noise_scale = noise_scale
        self.drop_prob = drop_prob
        self.dup_prob = dup_prob
        self.time_noise_scale = time_noise_scale

        with open(labels_csv, 'r') as f:
            for line in f.readlines()[1:]:  # Skip header
                svg_file, pcap_file, origin_ip, target_processes = line.strip().split(',')

                # Split target processes into a list
                target_processes_list = target_processes.split(';')

                target_process_vector = set()
                for process in target_processes_list:
                    if process in self.process_to_index:
                        target_process_vector.add(self.process_to_index[process])

                # Convert to multi-hot encoding
                #target_process_vector = [0] * len(self.process_to_index)
                #for process in target_processes_list:
                    #if process in self.process_to_index:
                        #target_process_vector[self.process_to_index[process]] = 1

                # Initialize a sparse matrix
                #target_process_vector = lil_matrix((1, len(self.process_to_index)), dtype=np.int8)
                #for process in target_processes_list:
                    #if process in self.process_to_index:
                        #target_process_vector[0, self.process_to_index[process]] = 1

                # Map IP address to an integer
                if origin_ip not in self.ip_to_index:
                    self.ip_to_index[origin_ip] = len(self.ip_to_index)

                self.data.append((svg_file, pcap_file, origin_ip, target_process_vector))
        '''
        # Store the number of samples
        with open(labels_csv, 'r') as f:
            self.num_samples = sum(1 for _ in f) - 1  # Subtract header
        '''

    def __len__(self):
        return len(self.data)

    def get_ip_index(self, ip):
        return self.ip_to_index.get(ip, -1)  # Return -1 if IP is not found

    def extract_unique_processes(svg_dir):
        unique_processes = set()
        for svg_file in os.listdir(svg_dir):
            if svg_file.endswith(".svg"):
                start_time = time.time()
                svg_path = os.path.join(svg_dir, svg_file)
                try:
                    tree = ET.parse(svg_path)
                    root = tree.getroot()
                    for elem in root.findall(".//{http://www.w3.org/2000/svg}text"):
                        unique_processes.add(elem.text)
                except Exception as e:
                    if args.verbose: print(f"Error parsing {svg_path}: {e}")
                elapsed_time = time.time() - start_time
                if args.verbose: print(f"Elapsed time for extracting processes for file {svg_file}: {elapsed_time:.2f} seconds")
        return {process: idx for idx, process in enumerate(unique_processes)}

    def __getitem__(self, idx):
        svg_file, pcap_file, origin_ip, target_process_vector = self.data[idx]

        # Extract SVG features
        svg_path = os.path.join(self.svg_dir, svg_file)
        svg_features = self.extract_svg_features(svg_path)
        if self.augment:
            svg_features = self.augment_svg_features(svg_features)  # Apply augmentation

        # Load PCAP features
        pcap_path = os.path.join(self.pcap_dir, pcap_file)
        pcap_features = self.load_pcap_features(pcap_path)
        if self.augment:
            pcap_features = self.augment_pcap_features(pcap_features)  # Apply augmentation

        svg_features = torch.tensor(svg_features, dtype=torch.float32)
        pcap_features = torch.tensor(pcap_features, dtype=torch.float32)
        
        # Convert the set to a list of indices
        target_process_indices = list(target_process_vector)
        target_process_values = [1.0] * len(target_process_indices)  # All values are 1 for multi-hot encoding

        return svg_features, pcap_features, origin_ip, target_process_indices, target_process_values

    def extract_svg_features(self, svg_path):
        start_time = time.time()
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            features = {
                "num_paths": len(root.findall(".//{http://www.w3.org/2000/svg}path")),
                "num_rects": len(root.findall(".//{http://www.w3.org/2000/svg}rect")),
                "num_circles": len(root.findall(".//{http://www.w3.org/2000/svg}circle")),
                "num_groups": len(root.findall(".//{http://www.w3.org/2000/svg}g")),
            }
            elapsed_time = time.time() - start_time
            if args.verbose: print(f"Elapsed time for extracting SVG features: {elapsed_time:.2f} seconds")
            return list(features.values())
        except Exception as e:
            print(f"Error extracting SVG features from {svg_path}: {e}")
            return [0, 0, 0, 0]

    def load_pcap_features(self, pcap_path):
        start_time = time.time()
        try:
            packets = rdpcap(pcap_path)
            num_packets = len(packets)
            packet_sizes = [len(pkt) for pkt in packets]
            protocols = [pkt.payload.name for pkt in packets]
            intervals = np.diff([pkt.time for pkt in packets]) if len(packets) > 1 else [0]

            features = {
                'num_packets': num_packets,
                'mean_packet_size': np.mean(packet_sizes) if packet_sizes else 0,
                'std_packet_size': np.std(packet_sizes) if packet_sizes else 0,
                'unique_protocols': len(set(protocols)),
            }
            elapsed_time = time.time() - start_time
            if args.verbose: print(f"Elapsed time for loading PCAP features: {elapsed_time:.2f} seconds")
            return list(features.values())
        except Exception as e:
            print(f"Error loading PCAP file: {e}")
            return [0, 0, 0, 0]
    
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
        # Perturb PCAP features with noise
        noise = np.random.normal(0, self.noise_scale, size=len(pcap_features))
        return pcap_features + noise

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
class SVGPCAPPredictionModel(nn.Module):
    def __init__(self, svg_dim=4, pcap_dim=4, transformer_dim=256, nhead=8, num_layers=4, num_processes=0, num_ips=0):
        super(SVGPCAPPredictionModel, self).__init__()
        self.svg_embed = nn.Linear(svg_dim, transformer_dim)
        self.pcap_embed = nn.Linear(pcap_dim, transformer_dim)

        transformer_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=nhead, dim_feedforward=512)
        self.svg_transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.pcap_transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        self.cross_attention = nn.MultiheadAttention(embed_dim=transformer_dim, num_heads=nhead, batch_first=True)
        
        # Output layers
        self.fc_out = nn.Sequential(
            nn.Linear(transformer_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_processes)  # Multi-hot encoded processes
        )
        self.fc_origin = nn.Sequential(
            nn.Linear(transformer_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_ips)  # Number of unique IPs
        )

    def forward(self, svg_features, pcap_features):
        svg_features = self.svg_embed(svg_features).unsqueeze(1)
        pcap_features = self.pcap_embed(pcap_features).unsqueeze(1)

        svg_encoded = self.svg_transformer(svg_features)
        pcap_encoded = self.pcap_transformer(pcap_features)

        attention_output, _ = self.cross_attention(query=svg_encoded, key=pcap_encoded, value=pcap_encoded)

        combined_features = torch.cat([svg_encoded.squeeze(1), attention_output.squeeze(1)], dim=-1)
        
        # Predict target processes
        process_output = self.fc_out(combined_features)
        
        # Predict origin IPs
        origin_output = self.fc_origin(combined_features)

        return origin_output, process_output  # Return two outputs

# Training and Validation Loop

def train_model():
    process_to_index = SVGPCAPDataset.extract_unique_processes('data/svg')
    num_processes = len(process_to_index)  # Number of unique processes
    print(f"Number of unique processes: {num_processes}")
    
    # Create datasets with and without augmentation
    start_time = time.time()
    train_dataset = SVGPCAPDataset('data/labels.csv', 'data/svg', 'data/pcap', process_to_index=process_to_index, augment=True, noise_scale=0.1)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time for loading training dataset: {elapsed_time:.2f} seconds")

    start_time = time.time()
    val_dataset = SVGPCAPDataset('data/labels.csv', 'data/svg', 'data/pcap', process_to_index=process_to_index, augment=False)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time for loading validation dataset: {elapsed_time:.2f} seconds")
    
    # Use functools.partial to pass num_processes to custom_collate_fn
    collate_fn = partial(custom_collate_fn, num_processes=num_processes)
    
    # Use the partial function in DataLoader
    start_time = time.time()
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=64, 
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
        batch_size=64, 
        shuffle=False, 
        num_workers=6, 
        pin_memory=True,
        collate_fn=collate_fn  # Use the partial function
    )
    print(f"Validation dataloader size: {len(val_dataloader)}")
    elapsed_time = time.time() - start_time
    print(f"Elapsed time for validation dataloader: {elapsed_time:.2f} seconds")
    
    num_processes = len(train_dataset.process_to_index)
    num_ips = len(train_dataset.ip_to_index)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SVGPCAPPredictionModel(num_processes=num_processes, num_ips=num_ips).to(device)
    
    # Define loss functions and optimizer
    criterion_process = nn.BCEWithLogitsLoss()
    criterion_origin = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0

        for svg_batch, pcap_batch, origin_labels, target_process_indices, target_process_values in train_dataloader:
            svg_batch = svg_batch.to(device)
            pcap_batch = pcap_batch.to(device)
            
            # Convert origin_labels (IP addresses) to indices
            origin_indices = [train_dataset.get_ip_index(ip) for ip in origin_labels]
            origin_indices = torch.tensor(origin_indices, dtype=torch.long, device=device)
            
            # Create sparse tensor from indices and values
            sparse_indices = []
            for i, indices in enumerate(target_process_indices):
                for idx in indices:
                    sparse_indices.append([i, idx])  # [batch_index, process_index]
            
            sparse_indices = torch.tensor(sparse_indices, dtype=torch.long).t()  # Transpose for sparse tensor format
            sparse_values = torch.tensor([v for values in target_process_values for v in values], dtype=torch.float32)
            
            # Create sparse tensor
            sparse_size = (len(svg_batch), num_processes)  # [batch_size, num_processes]
            target_process_sparse = torch.sparse_coo_tensor(sparse_indices, sparse_values, sparse_size).to(device)
            
            # Convert sparse tensor to dense tensor (if needed)
            target_process_dense = target_process_sparse.to_dense()

            optimizer.zero_grad()
            
            # Get both outputs from the model
            origin_output, process_output = model(svg_batch, pcap_batch)

            # Compute loss for target processes
            loss_process = criterion_process(process_output, target_process_dense)
            
            # Compute loss for origin IPs
            loss_origin = criterion_origin(origin_output, origin_indices)
            
            # Combine the losses
            total_batch_loss = loss_process + loss_origin
            
            # Backpropagate the total loss
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()

            # Free up memory
            del svg_batch, pcap_batch, origin_indices, target_process_sparse, target_process_dense, origin_output, process_output
            torch.cuda.empty_cache()

        train_losses.append(total_loss / len(train_dataloader))
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
        torch.cuda.empty_cache()
        validate_model(model, val_dataloader, device, val_dataset, num_processes)

# Validation Function
def validate_model(model, dataloader, device, dataset, num_processes):
    model.eval()
    all_origin_labels = []
    all_origin_preds = []
    process_f1_scores = []
    process_auc_scores = []

    with torch.no_grad():
        for svg_batch, pcap_batch, origin_labels, target_process_indices, target_process_values in dataloader:
            svg_batch = svg_batch.to(device)
            pcap_batch = pcap_batch.to(device)
            
            # Convert origin_labels (IP addresses) to indices
            origin_indices = [dataset.get_ip_index(ip) for ip in origin_labels]
            origin_indices = torch.tensor(origin_indices, dtype=torch.long, device=device)
            
            # Create sparse tensor from indices and values
            sparse_indices = []
            for i, indices in enumerate(target_process_indices):
                for idx in indices:
                    sparse_indices.append([i, idx])  # [batch_index, process_index]
            
            sparse_indices = torch.tensor(sparse_indices, dtype=torch.long).t()  # Transpose for sparse tensor format
            sparse_values = torch.tensor([v for values in target_process_values for v in values], dtype=torch.float32)
            
            # Create sparse tensor
            sparse_size = (len(svg_batch), num_processes)  # [batch_size, num_processes]
            target_process_sparse = torch.sparse_coo_tensor(sparse_indices, sparse_values, sparse_size).to(device)
            
            # Convert sparse tensor to dense tensor (if needed)
            process_labels = target_process_sparse.to_dense()

            # Get both outputs from the model
            origin_output, process_output = model(svg_batch, pcap_batch)

            origin_preds = torch.argmax(origin_output, dim=1)
            process_preds = torch.sigmoid(process_output) > 0.5

            # Compute metrics for this batch
            process_labels_flat = process_labels.cpu().numpy().flatten()
            process_preds_flat = process_preds.cpu().numpy().flatten()
            
            # Compute F1 score for this batch
            batch_f1 = f1_score(process_labels_flat, process_preds_flat, average='micro')
            process_f1_scores.append(batch_f1)
            
            # Compute ROC-AUC for this batch
            try:
                batch_auc = roc_auc_score(process_labels_flat, process_preds_flat)
                process_auc_scores.append(batch_auc)
            except ValueError:
                # Handle cases where ROC-AUC cannot be computed (e.g., only one class present)
                pass

            all_origin_labels.extend(origin_indices.cpu().numpy())
            all_origin_preds.extend(origin_preds.cpu().numpy())

    # Aggregate metrics across all batches
    avg_process_f1 = np.mean(process_f1_scores)
    avg_process_auc = np.mean(process_auc_scores)

    # Metrics for origin IP
    origin_accuracy = accuracy_score(all_origin_labels, all_origin_preds)
    print(f"Origin IP Accuracy: {origin_accuracy:.4f}")

    # Metrics for target processes (multi-label)
    print(f"Target Process F1 Score: {avg_process_f1:.4f}, ROC-AUC: {avg_process_auc:.4f}")

if __name__ == "__main__":
    train_model()

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()