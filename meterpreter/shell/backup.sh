#!/bin/bash

# Define the source directory to compress (ensure this path is correct)
SOURCE_DIR="/home/kali/Documents"

# Define the backup file path
BACKUP_FILE="/tmp/data_backup.tar.gz"

# Remote server details
REMOTE_USER="user"
REMOTE_HOST="remote-linux"
REMOTE_PATH="/tmp/"

# Ensure the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory $SOURCE_DIR does not exist. Exiting."
    exit 1
fi

# Compress the specified directory only
echo "Compressing files from $SOURCE_DIR into $BACKUP_FILE..."
tar -czf "$BACKUP_FILE" -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")"

# Transfer the compressed file to the remote server
echo "Transferring $BACKUP_FILE to $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH..."
scp "$BACKUP_FILE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

# Completion message
if [ $? -eq 0 ]; then
    echo "Backup and transfer completed successfully."
else
    echo "Backup or transfer failed."
fi
