#!/bin/bash

# attacker.sh
#
# This script is designed to continuously run the `attacker.sh` script in an infinite loop,
# monitor its output, and handle specific conditions such as IP address changes or network issues.
#
# Functionality:
# 1. Initializes variables:
#    - `ITERATION`: Tracks the current iteration number, starting at 1.
#    - `IP`: Tracks the current IP address, initialized to "0.0.0.0".
#
# 2. Runs the `attacker.sh` script in a loop, passing the current `ITERATION` as an argument.
#    - Captures and processes the output of `attacker.sh` in real-time.
#
# 3. Monitors the output for specific patterns:
#    - Detects and updates the `ITERATION` variable when the output contains "iteration X".
#    - Detects and updates the `IP` variable when the output contains "IP Address changed to X.X.X.X".
#    - Detects network issues such as "Connection timed out" or "Connection refused".
#
# 4. Handles network issues:
#    - If a valid IP address is detected, removes it from the `eth0` network interface.
#    - Restarts the `attacker.sh` script after a 10-second delay.
#
# 5. Uses `sudo` for commands requiring elevated privileges, such as modifying network settings
#    or killing the `attacker.sh` process.
#
# Notes:
# - The script uses `pkill` to terminate the `attacker.sh` process when restarting.
# - The `sleep` command introduces a delay before restarting the script.
# - The script assumes that `attacker.sh` is located in the same directory and is executable.
#
# Prerequisites:
# - Ensure `attacker.sh` exists and has executable permissions.
# - The user running this script must have `sudo` privileges.
#
# Warning:
# - This script modifies network settings and terminates processes. Use with caution.
# - Ensure proper error handling and logging mechanisms are in place for production use.

# Path to the attacker script
ATTACKER_SCRIPT="./attacker.sh"

# Initialize variables
ITERATION=1
IP="0.0.0.0"

# Infinite loop to keep running the script
while true; do
    echo "Starting attacker.sh with ITERATION=$ITERATION"

    # Run attacker.sh with ITERATION as an argument and capture output in real-time
    while IFS= read -r line; do
        echo "$line"

        # Check for "Iteration X" pattern and extract the number
        if [[ $line =~ iteration\ ([0-9]+) ]]; then
            ITERATION=${BASH_REMATCH[1]}
            echo "Detected Iteration: $ITERATION"
        fi

        # Check for "IP Address changed to X.X.X.X" and extract the IP
        if [[ $line =~ IP\ Address\ changed\ to\ ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+) ]]; then
            IP=${BASH_REMATCH[1]}
            echo "Detected IP Change: $IP"
        fi

        # Check for "Connection timed out" OR "Connection refused"
        if echo "$line" | grep -Eq "Connection timed out|Connection refused"; then
            echo "Network issue detected: $line"

            # Delete the IP address from eth0 before restarting
            if [[ $IP != "0.0.0.0" ]]; then
                echo "Removing IP $IP from eth0..."
                sudo ip addr del $IP/24 dev eth0
            else
                echo "No valid IP to remove."
            fi

            # Kill attacker.sh process
            echo "Restarting attacker.sh after 10 seconds..."
            sudo pkill -f "$ATTACKER_SCRIPT"
            sleep 5
            break
        fi
    done < <(sudo $ATTACKER_SCRIPT $ITERATION 2>&1)
done
