#!/bin/bash
# Scan for open ports on a target machine and save results to /tmp/scan.txt
target="192.168.0.222"
output_file="/tmp/scan.txt"
log_file="/tmp/port_scan.log"

# Clear the output and log files before starting the scan
> "$output_file"
> "$log_file"

# Logging function to record script progress and errors
log_message() {
    echo "$(date) - $1" >> "$log_file"
}

# Starting the port scan
log_message "Starting port scan on target: $target"

for port in {1..65535}; do
    log_message "Scanning port: $port"
    # Scan port and log any errors or success
    if bash -c "echo >/dev/tcp/$target/$port" 2>/dev/null; then
        echo "Port $port is open" >> "$output_file"
        log_message "Port $port is open"
    else
        log_message "Error scanning port: $port"
    fi
done

log_message "Port scan completed. Results saved to $output_file"
