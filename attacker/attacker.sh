#!/bin/bash

# Array of scripts in the "custom/" directory
scripts=("ransomware")

# Enable IP forwarding
echo 1 | sudo tee /proc/sys/net/ipv4/ip_forward

# Function to clean up IP and routes
cleanup() {
    local ip="$1"
    echo "Cleaning up IP and routes for $ip"
    sudo ip route del 192.168.0.0/24 dev eth0 src "$ip" 2>/dev/null
    sudo ip addr del "$ip"/24 dev eth0 2>/dev/null
}

# Loop through each meterpreter script
for script in "${scripts[@]}"; do
    for ((i = $1; i <= 5000; i++)); do
        ip=$(./ipgen.sh)

	# Path to the script you want to modify
	#path="/home/kali/Documents/backup.sh"

	# Use sed to replace the IP address on line 8
	#sed -i "8s/REMOTE_HOST=\"[^\"]*\"/REMOTE_HOST=\"$ip\"/" "$path"

	#echo "Updated the REMOTE_HOST IP in $path to $ip"

        echo "Starting iteration $i with IP $ip for script $script"

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

        # Loop until meterpreter.sh succeeds
        while true; do
            sudo ./meterpreter.sh "$script" "$i" "$ip"
            if [ $? -eq 0 ]; then
                echo "Meterpreter script succeeded!"
                break
            else
                echo "Meterpreter script failed. Retrying..."
                sleep 2
            fi
        done  

        # Cleanup IP and NAT
        cleanup "$ip"

        # Ensure cleanup is successful before proceeding
        if [ $? -ne 0 ]; then
            echo "Failed to clean up IP and route. Exiting..."
            exit 1
        fi

        # Add a delay between iterations to allow resources to free up
        sleep 2
    done
    echo "All iterations completed for script ${script}"
done
echo "All data collected!"
