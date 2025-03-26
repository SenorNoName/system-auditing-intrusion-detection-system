#!/bin/bash

# This script processes .dot files in a specified source directory and converts them to .svg files
# in a specified destination directory using the Graphviz 'dot' command. It includes error handling
# for missing arguments, existing SVG files, and timeout scenarios.

# Usage:
#   ./graph.sh <source> <destination>
# Arguments:
#   <source>      - The directory containing .dot files to be processed.
#   <destination> - The directory where the generated .svg files will be saved.
# Behavior:
#   - If the number of arguments is not exactly 2, the script exits with an error message.
#   - For each .dot file in the source directory:
#       - If an SVG file with the same base name already exists in the destination directory, it skips processing.
#       - Converts the .dot file to an .svg file using the 'dot' command with a timeout of 60 seconds.
#       - If the conversion times out, it logs a timeout message, removes the incomplete SVG file, and skips to the next file.

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source> <destination>"
    exit 1
fi

# Assign arguments to variables
source=$1
destination=$2

# Loop through each .dot file in the graphs directory
for dotfile in "$source"/*.dot
do
    # Extract the base name of the file (without extension)
    base_name=$(basename "$dotfile" .dot)

    # Check if the SVG file already exists
    if [ -f "${destination}/${base_name}.svg" ]; then
        echo "SVG file for $base_name already exists, skipping"
        continue
    fi

    timeout 60 dot -Tsvg "${dotfile}" > "${destination}/${base_name}.svg"
    if [ $? -eq 124 ]; then
        echo "Graph $base_name timed out"
        rm "${destination}/${base_name}.svg"
        timeout_files+=("$base_name")
        continue
    else
        echo "Graph $base_name processed successfully"
    fi
done