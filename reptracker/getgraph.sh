#!/bin/bash

# This script is used to compile and execute the GetGraph class of the reptracker project.
# 
# Steps performed by the script:
# 1. Compiles the reptracker project using Maven. If the compilation fails, the script exits with an error message.
# 2. Executes the GetGraph class using Maven's exec plugin with three parameters:
#    - First parameter (ipaddr): IP address of the victim.
#    - Second parameter (source): Source directory containing sysdig .txt files.
#    - Third parameter (dot_dest): Destination directory for .dot files.
#    - Fourth parameter (simplify_dest): Destination directory for simplified .svg files.
#    - Fifth parameter (svg_dest): Destination directory for .svg files.
# 3. If both steps succeed, a success message is displayed.
#
# Prerequisites:
# - Maven must be installed and available in the system's PATH.
# - The script must be executed from the root directory of the reptracker project or a directory where Maven can locate the project files.
#
# Usage:
# Run the script in a bash shell with the required parameters:
# ./getgraph.sh <ipaddr> <source> <dot_dest> <simplify_dest> <svg_dest>

# Check if the correct number of arguments is provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <ipaddr> <source> <dot_dest> <simplify_dest> <svg_dest>"
    exit 1
fi

# Assign arguments to variables
ipaddr=$1
source=$2
dot_dest=$3
simplify_dest=$4
svg_dest=$5

# Run Maven compile
echo "Compiling reptracker using Maven..."
mvn compile
if [ $? -ne 0 ]; then
    echo "Maven compilation failed. Exiting."
    exit 1
fi

# Run the GetGraph class using Maven
echo "Running GetGraph using Maven..."
mvn exec:java -Dexec.mainClass="pagerank.GetGraph" -Dexec.args="$ipaddr $source $dot_dest"
if [ $? -ne 0 ]; then
    echo "Execution failed."
    exit 1
fi

echo "GetGraph execution completed successfully."

echo "Simplifying DOT files..."
python3 simplify.py $dot_dest $simplify_dest
echo "Simplification completed successfully."

echo "Converting DOT graphs into SVG files..."
./graph.sh $simplify_dest $svg_dest
echo "SVG conversion completed successfully."