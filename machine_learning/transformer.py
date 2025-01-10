import torch
import torch.nn as nn
from torchvision import models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
from xml.etree import ElementTree as ET
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scapy.config import conf
conf.use_ipv6 = False
conf.use_pcap = True
from scapy.all import rdpcap

# Dataset Class
class SVGPCAPDataset(torch.utils.data.Dataset):
    def __init__(self, labels_csv, svg_dir, pcap_dir):
        self.data = []
        self.svg_dir = svg_dir
        self.pcap_dir = pcap_dir

        # Read the CSV file
        with open(labels_csv, 'r') as f:
            for line in f.readlines()[1:]:  # Skip header
                svg_file, pcap_file, label = line.strip().split(',')
                self.data.append((svg_file, pcap_file, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        svg_file, pcap_file, label = self.data[idx]

        # Extract SVG features
        svg_path = os.path.join(self.svg_dir, svg_file).replace('\\', '/')
        svg_features = self.extract_svg_features(svg_path)

        # Load PCAP features
        pcap_path = os.path.join(self.pcap_dir, pcap_file)
        pcap_features = self.load_pcap_features(pcap_path)

        # Ensure all features are tensors
        svg_features = torch.tensor(svg_features, dtype=torch.float32)
        pcap_features = torch.tensor(pcap_features, dtype=torch.float32)

        return svg_features, pcap_features, label

    def extract_svg_features(self, svg_path):
        """
        Parse the SVG file and extract features such as counts of elements.
        """
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()

            features = {
                "num_paths": len(root.findall(".//{http://www.w3.org/2000/svg}path")),
                "num_rects": len(root.findall(".//{http://www.w3.org/2000/svg}rect")),
                "num_circles": len(root.findall(".//{http://www.w3.org/2000/svg}circle")),
                "num_groups": len(root.findall(".//{http://www.w3.org/2000/svg}g")),
            }

            return list(features.values())
        except Exception as e:
            print(f"Error extracting SVG features from {svg_path}: {e}")
            return [0, 0, 0, 0]  # Default features

    def load_pcap_features(self, pcap_path):
        """
        Extract features from a PCAP file using Scapy.
        """
        try:
            packets = rdpcap(pcap_path)
            num_packets = len(packets)
            packet_sizes = [len(pkt) for pkt in packets]
            protocols = [pkt.payload.name for pkt in packets]
            intervals = np.diff([pkt.time for pkt in packets]) if len(packets) > 1 else [0]

            # Feature Engineering
            features = {
                'num_packets': num_packets,
                'mean_packet_size': np.mean(packet_sizes) if packet_sizes else 0,
                'std_packet_size': np.std(packet_sizes) if packet_sizes else 0,
                'unique_protocols': len(set(protocols)),
            }
            return list(features.values())
        except Exception as e:
            print(f"Error loading PCAP file: {e}")
            return [0, 0, 0, 0]  # Default features

# Transformer Model
class SVGPCAPMaliciousDetectionModel(nn.Module):
    def __init__(self, svg_dim=4, pcap_dim=4, transformer_dim=256, nhead=8, num_layers=4):
        super(SVGPCAPMaliciousDetectionModel, self).__init__()

        # Fully Connected Embeddings for SVG and PCAP features
        self.svg_embed = nn.Linear(svg_dim, transformer_dim)
        self.pcap_embed = nn.Linear(pcap_dim, transformer_dim)

        # Transformer Encoders
        transformer_layer = TransformerEncoderLayer(d_model=transformer_dim, nhead=nhead, dim_feedforward=512)
        self.svg_transformer = TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.pcap_transformer = TransformerEncoder(transformer_layer, num_layers=num_layers)

        # Cross-Attention Mechanism
        self.cross_attention = nn.MultiheadAttention(embed_dim=transformer_dim, num_heads=nhead, batch_first=True)

        # Fully Connected Output Layer
        self.fc_out = nn.Sequential(
            nn.Linear(transformer_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Binary classification
            nn.Sigmoid()  # Probability output
        )

    def forward(self, svg_features, pcap_features):
        # Embedding Layers
        svg_features = self.svg_embed(svg_features).unsqueeze(1)
        pcap_features = self.pcap_embed(pcap_features).unsqueeze(1)

        # Transformer Encoders
        svg_encoded = self.svg_transformer(svg_features)
        pcap_encoded = self.pcap_transformer(pcap_features)

        # Cross-Attention
        attention_output, _ = self.cross_attention(query=svg_encoded, key=pcap_encoded, value=pcap_encoded)

        # Combine and Classify
        combined_features = torch.cat([svg_encoded.squeeze(1), attention_output.squeeze(1)], dim=-1)
        output = self.fc_out(combined_features)

        return output

# Load Dataset and Dataloader
dataset = SVGPCAPDataset(labels_csv='data/labels.csv', 
                         svg_dir='data/svg', 
                         pcap_dir='data/pcap')

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, Loss, Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SVGPCAPMaliciousDetectionModel().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for svg_batch, pcap_batch, labels in dataloader:
        svg_batch = svg_batch.to(device)
        pcap_batch = pcap_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(svg_batch, pcap_batch).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/10], Loss: {total_loss:.4f}")

# Validation
model.eval()
all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for svg_batch, pcap_batch, labels in dataloader:
        svg_batch = svg_batch.to(device)
        pcap_batch = pcap_batch.to(device)
        labels = labels.to(device)

        outputs = model(svg_batch, pcap_batch).squeeze()
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(outputs.cpu().numpy())
        predictions = (outputs > 0.5).long()
        all_predictions.extend(predictions.cpu().numpy())

# Metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, zero_division=1)
recall = recall_score(all_labels, all_predictions, zero_division=1)
f1 = f1_score(all_labels, all_predictions, zero_division=1)
roc_auc = roc_auc_score(all_labels, all_probabilities)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUROC: {roc_auc:.4f}")