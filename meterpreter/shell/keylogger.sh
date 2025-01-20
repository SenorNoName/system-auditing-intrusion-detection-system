#!/bin/bash

# Set the log file where the keyboard input will be saved
LOG_FILE="/tmp/keyboard_log.txt"
> "$LOG_FILE"
# Find the keyboard input device by listing /dev/input/ and looking for event devices
DEVICE=$(ls /dev/input/event* | grep -i 'event' | head -n 1)

# Check if the device is found
if [ -z "$DEVICE" ]; then
    echo "No input devices found."
    exit 1
fi

# Run evtest for 5 seconds, then log only key presses to the file
echo "Logging keyboard activity from device $DEVICE for 5 seconds..."
echo "kali" | sudo -S timeout 5 evtest "$DEVICE" | grep -oP "EV_KEY.*" | while read line; do
    echo "$(date) - $line" >> "$LOG_FILE"
done

# Display the contents of the log file after 5 seconds
echo "Log file contents:"
cat "$LOG_FILE"
