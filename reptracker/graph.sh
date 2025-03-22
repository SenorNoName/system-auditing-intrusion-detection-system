#!/bin/bash

# This script processes .dot files in the "ransom_dot" directory, generates cleaned versions,
# and converts them into SVG files using the `dot` command. It handles specific cases such as
# skipping certain files, handling timeouts, and removing empty SVG files.

# Functionality:
# - Skips processing files with names containing "keylogger", "exfiltration", or "cryptomining".
# - Skips generating SVG files if they already exist in the "ransom/svg" directory.
# - Cleans .dot files by removing labels with non-alphabetic characters.
# - Converts cleaned .dot files to SVG format using the `dot` command with a 60-second timeout.
# - Tracks files that time out during SVG generation and deletes their incomplete SVG files.
# - Tracks files that generate empty SVG files and deletes them.
# - Outputs lists of files that timed out or generated empty SVG files.

# Variables:
# - timeout_files: Array to store the names of files that timed out during SVG generation.
# - empty_files: Array to store the names of files that generated empty SVG files.

# Input:
# - .dot files located in the "ransom_dot" directory.

# Output:
# - Cleaned .dot files saved in the "clean" directory.
# - Generated SVG files saved in the "ransom/svg" directory.
# - Console output indicating the status of each file and lists of problematic files.

# Dependencies:
# - `dot` command from Graphviz must be installed.
# - `timeout` command must be available.

# Usage:
# - Place this script in the root directory containing the "ransom_dot" folder.
# - Ensure the "clean" and "ransom/svg" directories exist before running the script.
# - Run the script in a bash shell.

# Initialize lists to keep track of problematic files
timeout_files=()
empty_files=()

# Loop through each .dot file in the graphs directory
for dotfile in ransom_dot/*.dot
do
    # Extract the base name of the file (without extension)
    base_name=$(basename "$dotfile" .dot)
    if [[ "$base_name" == *"keylogger"* || "$base_name" == *"exfiltration"* || "$base_name" == *"cryptomining"* ]]; then
        echo "Skipping $base_name"
        continue
    fi
    
    # Check if the SVG file already exists
    if [ -f "ransom/svg/${base_name}.svg" ]; then
        echo "SVG file for $base_name already exists, skipping"
        continue
    fi
    
    grep -v 'label="[^a-zA-Z]*"' "$dotfile" > "clean/${dotfile}"
    # Generate SVG files from .dot files with a timeout
    timeout 60 dot -Tsvg "clean/${dotfile}" > "ransom/svg/${base_name}.svg"
    if [ $? -eq 124 ]; then
        echo "Graph $base_name timed out"
        rm "ransom/svg/${base_name}.svg"
        timeout_files+=("$base_name")
        continue
    fi
    
    # Check if the generated SVG file is empty and delete it if it is
    if [ ! -s "ransom/svg/${base_name}.svg" ]; then
        echo "Graph $base_name generated an empty file and will be deleted"
        rm "ransom/svg/${base_name}.svg"
        empty_files+=("$base_name")
    else
        echo "Graph $base_name done"
    fi
done

# Print the list of files that timed out
if [ ${#timeout_files[@]} -ne 0 ]; then
    echo "Files that timed out:"
    for file in "${timeout_files[@]}"; do
        echo "$file"
    done
fi

# Print the list of files that generated empty files
if [ ${#empty_files[@]} -ne 0 ]; then
    echo "Files that generated empty files:"
    for file in "${empty_files[@]}"; do
        echo "$file"
    done
fi