# /usr/share/metasploit-framework/scripts/meterpreter/sync/parent_script.rb
require 'msf/core'

# Array of scripts in the "custom/" directory
scripts = [
  "download", 
  "keylogger", 
  "ransomware", 
  "credentials", 
  "persistence", 
  "port_scanner", 
  "cryptomining", 
  "kill", 
  "exfiltration"
]

# Define the victim's path for the dumpcap binary
dumpcap_path = "/usr/bin/dumpcap"

# Loop through each meterpreter script
scripts.each do |script|

# Load the child script
print_status("Loading new script #{script}.rb")
load "/usr/share/metasploit-framework/scripts/meterpreter/custom/#{script}.rb"

# Iterate 500 times
500.times do |i|
print_status("Collecting data for #{script}.rb, iteration: #{i + 1}/500")

# Define the path where the pcap file will be saved on the victim machine
pcap_file_path = "/home/kali/meterpreter/pcap/#{script}_#{i + 1}.pcap"
sysdig_file_path = "/home/kali/meterpreter/sysdig/#{script}_#{i + 1}.txt"

# Command to run dumpcap for 10 seconds and save the capture to the file
dumpcap_command = "sudo nohup #{dumpcap_path} -i eth0 -w #{pcap_file_path} > /dev/null 2>&1 &"
sysdig_command = "sudo sysdig -p'%evt.num %evt.rawtime.s.%evt.rawtime.ns %evt.cpu %proc.name (%proc.pid) %evt.dir %evt.type cwd=%proc.cwd %evt.args latency=%evt.latency exepath=%proc.exepath' \"proc.name!=tmux and (evt.type=read or evt.type=readv or evt.type=write or evt.type=writev or evt.type=fcntl or evt.type=accept or evt.type=connect)\" > #{sysdig_file_path} &"

# Run the command on the victim machine via Meterpreter
print_status("Starting wireshark packet capture and sysdig audit logging ...")
dumpcap_result = cmd_exec(dumpcap_command)
sysdig_result = cmd_exec(sysdig_command)

print_status("Running Meterpreter script #{script}.rb ...")

# Call the script's method
begin
  run_script(client)
  print_status("#{script}.rb executed successfully.")
rescue => e
  print_error("Error while running #{script}.rb: #{e.message}")
end

# Check if dumpcap was successful
if dumpcap_result.include?("error")
  print_error("Failed to execute dumpcap: #{dumpcap_result}")
else
  print_good("Packet capture completed successfully. Saved as #{pcap_file_path}")
end

# Check if sysdig was successful
if sysdig_result.include?("error")
  print_error("Failed to execute sysdig: #{sysdig_result}")
else
  print_good("Sysdig logging completed successfully. Saved as #{sysdig_file_path}")
end

# Terminate processes and set proper file permissions
cmd_exec("pkill -f sysdig")
cmd_exec("pkill -f dumpcap")
cmd_exec("sudo chmod 0644 #{pcap_file_path}")

print_good("Iteration #{i + 1}/500 for #{script}.rb completed.")

end

print_good("All iterations for script #{script} completed.")

end

print_good("Data collection for all scripts complete!")
