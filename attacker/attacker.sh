#!/bin/bash

# Array of scripts in the "custom/" directory
scripts=("keylogger" "ransomware" "cryptomining" "kill" "exfiltration")

# Loop through each meterpreter script
for script in "${scripts[@]}"; do
    # Load the child script
    #echo "[*] Loading new script ${script}.rb"
    #source "/usr/share/metasploit-framework/scripts/meterpreter/custom/${script}.rb"

    # Iterate 2000 times
    for ((i = 1; i <= 2; i++)); do

        # Loop until sender.sh returns success (exit code 0)
        while true; do
            sudo ./sender.sh
            if [ $? -eq 0 ]; then
                echo "Sender script succeeded!"
                break
            else
                echo "Sender script failed. Retrying..."
                sleep 5
            fi
        done  

        # Loop until meterpreter.sh returns success (exit code 0)
        while true; do
            sudo ./meterpreter.sh $script $i
            if [ $? -eq 0 ]; then
                echo "Meterpreter script succeeded!"
                break
            else
                echo "Meterpreter script failed. Retrying..."
		sleep 5
            fi
        done  

    done
    echo "All iterations completed for script ${script}"
done
echo "All data collected!"
