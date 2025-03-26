#!/bin/bash

# attacker.sh
#
# This script automates the execution of various attack simulations by generating
# combinations of predefined scripts, assigning random IP addresses, and executing
# them in a controlled loop. It also handles IP address cleanup and ensures proper
# resource management between iterations.
#
# Features:
# - Generates the powerset of predefined attack scripts to simulate different combinations.
# - Dynamically assigns and cleans up IP addresses for each iteration.
# - Updates a specific configuration file (backup.sh) with a new IP address when required.
# - Ensures reliable execution by retrying failed operations and cleaning up resources.
#
# Components:
# 1. **scripts**: Array containing the names of attack scripts located in the "custom/" directory.
# 2. **powerset()**: Function to generate all possible subsets of the `scripts` array.
# 3. **cleanup()**: Function to remove assigned IP addresses and routes after each iteration.
# 4. **Main Loop**:
#    - Iterates 500 times (from 500 to 1000).
#    - Randomly selects a subset of attack scripts to execute.
#    - Dynamically assigns a new IP address using `ipgen.sh`.
#    - Updates the `REMOTE_HOST` variable in `backup.sh` if "exfiltration" is selected.
#    - Executes `sender.sh` in a retry loop until it succeeds.
#    - Passes the selected scripts, iteration number, and IP address to `meterpreter.sh`.
#    - Cleans up IP addresses and routes after each iteration.
#    - Adds a delay between iterations to free up resources.
#
# Prerequisites:
# - Ensure `ipgen.sh`, `sender.sh`, and `meterpreter.sh` are executable and in the same directory.
# - The `backup.sh` file must exist at `/home/kali/Documents/backup.sh`.
# - Sufficient permissions to modify network configurations (requires `sudo`).
#
# Usage:
# Run the script in a bash shell with appropriate permissions:
#   ./attacker.sh
#
# Notes:
# - The script assumes a specific network configuration (e.g., `eth0` interface).
# - Modify the `scripts` array to include the names of all attack scripts in the "custom/" directory.
# - Ensure the `backup.sh` file's format matches the expected structure for `sed` to work correctly.
#
# Warning:
# This script is intended for educational or testing purposes in a controlled environment.
# Do not use this script for malicious purposes or on unauthorized systems.

# Array of scripts in the "custom/" directory
scripts=("cryptomining" "exfiltration" "ransomware" "keylogger")  # Add all your script names here

# Function to generate the powerset of an array
powerset() {
    local elements=("$@")
    local n=${#elements[@]}
    local subsets=()

    for ((i = 0; i < (1 << n); i++)); do
        subset=()
        for ((j = 0; j < n; j++)); do
            if (( (i & (1 << j)) > 0 )); then
                subset+=("${elements[j]}")
            fi
        done
        # Join elements with commas and add to subsets
        subsets+=("$(IFS=_; echo "${subset[*]}")")
    done

    printf '%s\n' "${subsets[@]}"
}

# Generate the powerset of the scripts
mapfile -t powerset_scripts < <(powerset "${scripts[@]}")

# Enable IP forwarding
echo 1 | sudo tee /proc/sys/net/ipv4/ip_forward

# Function to clean up IP and routes
cleanup() {
    local ip="$1"
    echo "Cleaning up IP and routes for $ip"
    sudo ip route del 192.168.0.0/24 dev eth0 src "$ip" 2>/dev/null
    sudo ip addr del "$ip"/24 dev eth0 2>/dev/null
}

# Loop for 1000 iterations
for ((i = 0; i <= 1000; i++)); do
    # Randomly choose an element from the powerset
    random_index=$((RANDOM % ${#powerset_scripts[@]}))
    selected_scripts=(${powerset_scripts[$random_index]})

    echo "Starting iteration $i with scripts: ${selected_scripts}"

    ip=$(./ipgen.sh)

    # If "exfiltration" is in the selected scripts, update the backup.sh file
    if [[ " ${selected_scripts[*]} " =~ " exfiltration " ]]; then
        # Path to the script you want to modify
        path="/home/kali/Documents/backup.sh"

        # Use sed to replace the IP address on line 8
        sed -i "8s/REMOTE_HOST=\"[^\"]*\"/REMOTE_HOST=\"$ip\"/" "$path"

        echo "Updated the REMOTE_HOST IP in $path to $ip"
    fi

    echo "Running scripts with IP $ip"

    # Assign new IP
    sudo ip addr add "$ip"/24 dev eth0
    sudo ip route add 192.168.0.0/24 dev eth0 src "$ip"
    echo "IP Address changed to ${ip}"

    # Loop until sender.sh succeeds
    while true; do
        sudo ./sender.sh "$ip"
        if [ $? -eq 0 ]; then
            echo "Sender script succeeded!"
            break
        else
            echo "Sender script failed. Retrying..."
            sleep 2
        fi
    done  

    # Pass the selected scripts to meterpreter.sh
    sudo ./meterpreter.sh "${selected_scripts}" "$i" "$ip"

    # Cleanup IP and NAT
    cleanup "$ip"

    # Ensure cleanup is successful before proceeding
    if [ $? -ne 0 ]; then
        echo "Failed to clean up IP and route. Exiting..."
        exit 1
    fi

    sleep 2

    echo "All scripts completed for iteration $i"
done
echo "All data collected!"
