#!/bin/bash

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
for ((i = 500; i <= 1000; i++)); do
    # Randomly choose an element from the powerset
    random_index=$((RANDOM % ${#powerset_scripts[@]}))
    selected_scripts=(${powerset_scripts[$random_index]})
    #scr=$(echo "${selected_scripts}" | tr -d '{}' | tr ' ' ',')

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

    # Add a delay between iterations to allow resources to free up
    sleep 2

    echo "All scripts completed for iteration $i"
done
echo "All data collected!"
