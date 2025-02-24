#!/bin/bash

# Path to the attacker script
ATTACKER_SCRIPT="./attacker.sh"

# Initialize variables
ITERATION=2526
IP="0.0.0.0"

# Infinite loop to keep running the script
while true; do
    echo "Starting attacker.sh with ITERATION=$ITERATION"

    # Run attacker.sh with ITERATION as an argument and capture output in real-time
    while IFS= read -r line; do
        echo "$line"  # Print output to terminal

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
            sleep 5  # Wait before restarting
            break  # Exit the while loop and restart the script
        fi
    done < <(sudo $ATTACKER_SCRIPT $ITERATION 2>&1)
done
