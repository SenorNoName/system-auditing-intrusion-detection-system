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

parser = argparse.ArgumentParser(description="Action Level Detection Transformer")
parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

train_losses = []
val_losses = []

# Dataset Class
class SVGPCAPDataset(Dataset):
    def __init__(self, labels_csv, svg_dir, pcap_dir, augment=False, noise_scale=0.1, drop_prob=0.1, dup_prob=0.1, time_noise_scale=0.01):
        """
        Initialize the dataset.
        :param labels_csv: Path to the CSV file containing labels.
        :param svg_dir: Directory containing SVG files.
        :param pcap_dir: Directory containing PCAP files.
        :param augment: Whether to apply on-the-fly augmentation.
        :param noise_scale: Scale of the noise for feature perturbation.
        :param drop_prob: Probability of dropping a packet.
        :param dup_prob: Probability of duplicating a packet.
        :param time_noise_scale: Scale of timestamp noise.
        """
        self.data = []
        self.svg_dir = svg_dir
        self.pcap_dir = pcap_dir
        self.process_to_index = self.extract_unique_processes(svg_dir)
        self.ip_to_index = {}  # Mapping for IP addresses
        self.augment = augment  # Whether to apply augmentation
        self.noise_scale = noise_scale  # Scale of the noise for feature perturbation
        self.drop_prob = drop_prob  # Probability of dropping a packet
        self.dup_prob = dup_prob  # Probability of duplicating a packet
        self.time_noise_scale = time_noise_scale  # Scale of timestamp noise

        with open(labels_csv, 'r') as f:
            for line in f.readlines()[1:]:  # Skip header
                svg_file, pcap_file, origin_ip, target_processes = line.strip().split(',')

                if args.verbose: print(f"Processing {svg_file} and {pcap_file}")

                # Split target processes into a list
                target_processes_list = target_processes.split(';')

                # Convert to multi-hot encoding
                target_process_vector = [0] * len(self.process_to_index)
                for process in target_processes_list:
                    if process in self.process_to_index:
                        target_process_vector[self.process_to_index[process]] = 1

                # Map IP address to an integer
                if origin_ip not in self.ip_to_index:
                    self.ip_to_index[origin_ip] = len(self.ip_to_index)

                self.data.append((svg_file, pcap_file, origin_ip, target_process_vector))

    def __len__(self):
        return len(self.data)

    def get_ip_index(self, ip):
        return self.ip_to_index.get(ip, -1)  # Return -1 if IP is not found

    def extract_unique_processes(self, svg_dir):
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
                    print(f"Error parsing {svg_path}: {e}")
                elapsed_time = time.time() - start_time
                print(f"Elapsed time for extracting processes for file {svg_file}: {elapsed_time:.2f} seconds")
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
        target_process_vector = torch.tensor(target_process_vector, dtype=torch.float32)

        return svg_features, pcap_features, origin_ip, target_process_vector

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
    # Create datasets with and without augmentation
    start_time = time.time()
    train_dataset = SVGPCAPDataset('data/labels.csv', 'data/svg', 'data/pcap', augment=True, noise_scale=0.1)
    elapsed_time = time.time() - start_time
    if args.verbose: print(f"Elapsed time for loading training dataset: {elapsed_time:.2f} seconds")

    start_time = time.time()
    val_dataset = SVGPCAPDataset('data/labels.csv', 'data/svg', 'data/pcap', augment=False)
    elapsed_time = time.time() - start_time
    if args.verbose: print(f"Elapsed time for loading training dataset: {elapsed_time:.2f} seconds")
    
    start_time = time.time()
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    if args.verbose: print(f"Training dataloader size: {len(train_dataloader)}")
    elapsed_time = time.time() - start_time
    if args.verbose: print(f"Elapsed time for training dataloader: {elapsed_time:.2f} seconds")

    start_time = time.time()
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    if args.verbose: print(f"Validation dataloader size: {len(val_dataloader)}")
    elapsed_time = time.time() - start_time
    if args.verbose: print(f"Elapsed time for validation dataloader: {elapsed_time:.2f} seconds")
    
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

        for svg_batch, pcap_batch, origin_labels, target_process_vector in train_dataloader:
            svg_batch = svg_batch.to(device)
            pcap_batch = pcap_batch.to(device)
            
            # Convert origin_labels (IP addresses) to indices
            origin_indices = [train_dataset.get_ip_index(ip) for ip in origin_labels]
            origin_indices = torch.tensor(origin_indices, dtype=torch.long, device=device)
            
            target_process_vector = target_process_vector.to(device)

            optimizer.zero_grad()
            
            # Get both outputs from the model
            origin_output, process_output = model(svg_batch, pcap_batch)

            # Compute loss for target processes
            loss_process = criterion_process(process_output, target_process_vector)
            
            # Compute loss for origin IPs
            loss_origin = criterion_origin(origin_output, origin_indices)
            
            # Combine the losses
            total_batch_loss = loss_process + loss_origin
            
            # Backpropagate the total loss
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()

        train_losses.append(total_loss / len(train_dataloader))
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
        validate_model(model, val_dataloader, device, val_dataset)

# Validation Function
def validate_model(model, dataloader, device, dataset):
    val_loss = 0

    model.eval()
    all_origin_labels = []
    all_origin_preds = []
    all_process_labels = []
    all_process_preds = []

    with torch.no_grad():
        for svg_batch, pcap_batch, origin_labels, process_labels in dataloader:
            svg_batch = svg_batch.to(device)
            pcap_batch = pcap_batch.to(device)
            
            # Convert origin_labels (IP addresses) to indices
            origin_indices = [dataset.get_ip_index(ip) for ip in origin_labels]
            origin_indices = torch.tensor(origin_indices, dtype=torch.long, device=device)
            
            process_labels = torch.tensor(process_labels, dtype=torch.float32, device=device)

            # Get both outputs from the model
            origin_output, process_output = model(svg_batch, pcap_batch)

            origin_preds = torch.argmax(origin_output, dim=1)
            process_preds = torch.sigmoid(process_output) > 0.5

            all_origin_labels.extend(origin_indices.cpu().numpy())
            all_origin_preds.extend(origin_preds.cpu().numpy())
            all_process_labels.extend(process_labels.cpu().numpy())
            all_process_preds.extend(process_preds.cpu().numpy())
        val_losses.append(val_loss / len(dataloader))

    # Metrics for origin IP
    origin_accuracy = accuracy_score(all_origin_labels, all_origin_preds)
    print(f"Origin IP Accuracy: {origin_accuracy:.4f}")

    # Metrics for target processes (multi-label)
    process_labels_flat = np.array(all_process_labels).flatten()
    process_preds_flat = np.array(all_process_preds).flatten()
    process_f1 = f1_score(process_labels_flat, process_preds_flat, average='micro')
    process_auc = roc_auc_score(process_labels_flat, process_preds_flat)

    print(f"Target Process F1 Score: {process_f1:.4f}, ROC-AUC: {process_auc:.4f}")

train_model()

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()