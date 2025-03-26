#!/bin/bash

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
