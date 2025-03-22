#!/bin/bash
CD_CMD="cd ../../usr/share/metasploit-framework/scripts/meterpreter/custom"
# Change to the specified directory and open a new terminal there
gnome-terminal --tab --title="custom" -- bash -c "$CD_CMD; exec bash"
