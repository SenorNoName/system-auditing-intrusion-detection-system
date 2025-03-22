#!/bin/bash

# This script is used to compile and execute the GetGraph class of the reptracker project.
# 
# Steps performed by the script:
# 1. Compiles the reptracker project using Maven. If the compilation fails, the script exits with an error message.
# 2. Executes the GetGraph class using Maven's exec plugin with three parameters:
#    - First parameter (ipaddr): IP address of the victim.
#    - Second parameter (source): Source directory containing sysdig .txt files.
#    - Third parameter (destination): Destination directory for .dot files.
# 3. If both steps succeed, a success message is displayed.
#
# Prerequisites:
# - Maven must be installed and available in the system's PATH.
# - The script must be executed from the root directory of the reptracker project or a directory where Maven can locate the project files.
#
# Usage:
# Run the script in a bash shell with the required parameters:
# ./getgraph.sh <ipaddr> <source> <destination>

# Run Maven compile
echo "Compiling reptracker using Maven..."
mvn compile
if [ $? -ne 0 ]; then
    echo "Maven compilation failed. Exiting."
    exit 1
fi

# Run the GetGraph class using Maven
echo "Running GetGraph using Maven..."
mvn exec:java -Dexec.mainClass="pagerank.GetGraph" -Dexec.args="192.168.0.28 powerset/sysdig powerset_dot"
if [ $? -ne 0 ]; then
    echo "Execution failed."
    exit 1
fi

echo "GetGraph execution completed successfully."
