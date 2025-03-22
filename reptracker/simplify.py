"""
This script simplifies .dot graph files by performing various operations such as 
removing duplicate edges, grouping similar nodes, collapsing nodes with single 
connections, and removing unconnected nodes. The simplified graph is saved to 
an output file.
Functions:
    simplify_dot(input_file, output_file):
        Simplifies a .dot graph file by:
        - Removing duplicate edges and consolidating them with labels indicating the count.
        - Grouping similar nodes into subgraphs based on their type.
        - Removing unnecessary labels from nodes.
        - Collapsing nodes with single incoming and outgoing connections into a single edge.
        - Removing unconnected nodes from the graph.
        - Simplifying edge labels by removing unnecessary information.
Command Line Arguments:
    input_directory (str): Directory containing the input .dot files to be simplified.
    output_directory (str): Directory where the simplified .dot files will be saved.
Usage:
    python simplify.py <input_directory> <output_directory>
Example:
    python simplify.py ./input_dot_files ./output_dot_files
Dependencies:
    - pydot: For parsing and manipulating .dot graph files.
    - argparse: For parsing command-line arguments.
    - os: For file and directory operations.
    - collections.defaultdict: For managing edge and node group counts.
Notes:
    - The script ensures that the output directory exists before saving the simplified files.
    - Only files with a ".dot" extension in the input directory are processed.
"""

import pydot
from collections import defaultdict
import os
import argparse

def simplify_dot(input_file, output_file):
    # Load the graph from the input .dot file
    graphs = pydot.graph_from_dot_file(input_file)
    graph = graphs[0]

    # Step 1: Remove duplicate edges
    edge_counts = defaultdict(int)
    edges_to_remove = []

    for edge in graph.get_edges():
        edge_key = (edge.get_source(), edge.get_destination())
        edge_counts[edge_key] += 1

    for edge in graph.get_edges():
        edge_key = (edge.get_source(), edge.get_destination())
        if edge_counts[edge_key] > 1:
            edges_to_remove.append(edge)

    for edge in edges_to_remove:
        graph.del_edge(edge.get_source(), edge.get_destination())

    # Add consolidated edges
    for (src, dst), count in edge_counts.items():
        if count > 1:
            new_edge = pydot.Edge(src, dst, label=f"{count} connections")
            graph.add_edge(new_edge)

    # Step 2: Group similar nodes
    node_groups = defaultdict(list)
    for node in graph.get_nodes():
        label = node.get_label()
        if label:
            node_type = label.split()[0]  # Extract the first word as the node type
            node_groups[node_type].append(node.get_name())

    # Create subgraphs for grouped nodes
    for node_type, nodes in node_groups.items():
        if len(nodes) > 1:
            subgraph = pydot.Cluster(node_type, label=node_type)
            for node_name in nodes:
                node = graph.get_node(node_name)[0]
                subgraph.add_node(node)
            graph.add_subgraph(subgraph)

    # Step 3: Remove unnecessary labels
    for node in graph.get_nodes():
        label = node.get_label()
        if label and "[0.0]" in label:
            new_label = label.replace(" [0.0]", "")
            node.set_label(new_label)

    # Step 4: Collapse nodes with single connections
    nodes_to_remove = []
    edges_to_add = []

    for node in graph.get_nodes():
        node_name = node.get_name()
        # Find incoming and outgoing edges manually
        in_edges = [edge for edge in graph.get_edges() if edge.get_destination() == node_name]
        out_edges = [edge for edge in graph.get_edges() if edge.get_source() == node_name]

        if len(in_edges) == 1 and len(out_edges) == 1:
            in_edge = in_edges[0]
            out_edge = out_edges[0]
            new_edge = pydot.Edge(in_edge.get_source(), out_edge.get_destination(), label="collapsed")
            edges_to_add.append(new_edge)
            nodes_to_remove.append(node_name)

    for node_name in nodes_to_remove:
        graph.del_node(node_name)

    for edge in edges_to_add:
        graph.add_edge(edge)

    # Step 5: Remove unconnected nodes
    nodes_to_remove = []
    for node in graph.get_nodes():
        node_name = node.get_name()
        # Find incoming and outgoing edges manually
        in_edges = [edge for edge in graph.get_edges() if edge.get_destination() == node_name]
        out_edges = [edge for edge in graph.get_edges() if edge.get_source() == node_name]

        if not in_edges and not out_edges:
            nodes_to_remove.append(node_name)

    for node_name in nodes_to_remove:
        graph.del_node(node_name)

    # Step 6: Simplify edge labels
    for edge in graph.get_edges():
        label = edge.get_label()
        if label and "0.0" in label:
            edge.set_label("")

    # Save the simplified graph to the output file
    graph.write(output_file)

    print(f"Simplified graph saved to {output_file}")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Simplify .dot graph files.")
parser.add_argument("input_directory", type=str, help="Directory containing .dot files")
parser.add_argument("output_directory", type=str, help="Directory to save simplified .dot files")
args = parser.parse_args()

# Directory containing .dot files
input_directory = args.input_directory
output_directory = args.output_directory

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Process each .dot file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".dot"):
        input_file = os.path.join(input_directory, filename)
        output_file = os.path.join(output_directory, filename)
        simplify_dot(input_file, output_file)