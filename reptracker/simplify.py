"""
This script simplifies .dot graph files by performing various transformations
to make the graphs more concise and easier to interpret. The script processes
all .dot files in a specified input directory and saves the simplified versions
to an output directory.
Functions:
    simplify_dot(input_file, output_file):
        Simplifies a .dot graph file by:
        1. Removing duplicate edges and consolidating them with labels indicating
           the number of connections.
        2. Grouping similar nodes into subgraphs based on their types.
        3. Removing unnecessary labels from nodes.
        4. Collapsing nodes with single incoming and outgoing connections into
           a single edge.
        5. Removing unconnected nodes from the graph.
        6. Simplifying edge labels by removing redundant information.
    Parameters:
        input_file (str): Path to the input .dot file.
        output_file (str): Path to save the simplified .dot file.
Directories:
    input_directory: The directory containing the input .dot files to be processed.
    output_directory: The directory where the simplified .dot files will be saved.
Usage:
    - Place the .dot files to be simplified in the `powerset_dot` directory.
    - Run the script to process all .dot files in the directory.
    - Simplified .dot files will be saved in the `clean_powerset` directory.
Dependencies:
    - pydot: A Python interface to Graphviz's Dot language.
    - collections.defaultdict: For counting and grouping elements.
    - os: For file and directory operations.
Example:
    To simplify all .dot files in the `powerset_dot` directory and save the
    results in the `clean_powerset` directory, simply run the script.
Notes:
    - Ensure that Graphviz is installed and properly configured for pydot to work.
    - The script assumes that node labels follow a specific format for grouping
      and collapsing operations.
"""

import pydot
from collections import defaultdict
import os

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

# Directory containing .dot files
input_directory = "powerset_dot"
output_directory = "clean_powerset"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Process each .dot file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".dot"):
        input_file = os.path.join(input_directory, filename)
        output_file = os.path.join(output_directory, filename)
        simplify_dot(input_file, output_file)