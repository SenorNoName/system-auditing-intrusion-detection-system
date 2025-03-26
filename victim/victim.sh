#!/bin/bash

# victim.sh - A script to receive a file over a network, execute it, and monitor its execution.
#
# This script continuously executes the `receiver.sh` script in a loop until it succeeds.
# If `receiver.sh` exits with a success code (0), the loop terminates, and a success message is displayed.
# If `receiver.sh` fails (non-zero exit code), an error message is displayed, and the script retries after a 2-second delay.
#
# USAGE:
# - Ensure `receiver.sh` is in the same directory as this script.
# - Run this script with appropriate permissions (e.g., using `sudo` if required).
# - The script will handle retries automatically until `receiver.sh` succeeds.
#
# NOTES:
# - The script uses `sudo` to execute `receiver.sh`. Ensure the user has the necessary permissions.
# - Modify the sleep duration (currently 2 seconds) if needed to adjust the retry interval.

# Loop until receiver.sh returns success (exit code 0)
while true; do
    sudo ./receiver.sh
    if [ $? -eq 0 ]; then
        echo "Receiver script succeeded!"
    else
        echo "Receiver script failed. Retrying..."
	sleep 2
    fi
done
