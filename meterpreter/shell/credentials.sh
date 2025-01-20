#!/bin/bash

# Use $HISTFILE to get the zsh history file path
HISTORY_FILE="$HISTFILE"

# Remote server details (mocked for testing)
REMOTE_USER="user"
REMOTE_HOST="remote-linux"
REMOTE_PATH="/tmp/"

# Log file for debugging
DEBUG_LOG="/tmp/zsh_history_debug.log"
> "$DEBUG_LOG"

# Ensure the zsh history file exists
if [ ! -f "$HISTORY_FILE" ]; then
    echo "Zsh history file $HISTORY_FILE does not exist. Exiting." | tee -a "$DEBUG_LOG"
    exit 1
fi

# Simulate transferring the zsh history file to a remote server
echo "Transferring $HISTORY_FILE to $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH..." | tee -a "$DEBUG_LOG"
scp "$HISTORY_FILE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" 2>>"$DEBUG_LOG"

# Completion message
if [ $? -eq 0 ]; then
    echo "Zsh history file transferred successfully." | tee -a "$DEBUG_LOG"
else
    echo "Transfer failed." | tee -a "$DEBUG_LOG"
fi
