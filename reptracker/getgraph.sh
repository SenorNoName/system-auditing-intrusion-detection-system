#!/bin/bash

# Run Maven compile
echo "Compiling reptracker using Maven..."
mvn compile
if [ $? -ne 0 ]; then
    echo "Maven compilation failed. Exiting."
    exit 1
fi

# Run the GetGraph class using Maven
echo "Running GetGraph using Maven..."
mvn exec:java -Dexec.mainClass="pagerank.GetGraph"
if [ $? -ne 0 ]; then
    echo "Execution failed."
    exit 1
fi

echo "GetGraph execution completed successfully."
