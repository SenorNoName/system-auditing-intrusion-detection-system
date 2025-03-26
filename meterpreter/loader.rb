# This script is a custom Metasploit post-exploitation module designed to handle
# the execution of multiple Meterpreter scripts in a loop, while also performing
# packet capture and system call auditing on the target machine.
#
# Class: Metasploit3
# Inherits from: Msf::Post
#
# Description:
# - The module allows users to specify a list of scripts (via the SCRIPT_NAME option)
#   and the number of iterations (via the ITERATION_COUNT option) to execute them.
# - It also integrates with tools like `dumpcap` for packet capture and `sysdig` for
#   system call auditing, saving the output to specified file paths.
#
# Options:
# - SCRIPT_NAME (String): The name of the script(s) to run, separated by underscores.
#   Example: "script1_script2_script3".
# - ITERATION_COUNT (Integer): The number of iterations to run the specified scripts.
#
# Key Features:
# - Executes multiple Meterpreter scripts in sequence.
# - Captures network packets using `dumpcap` and saves them as `.pcap` files.
# - Logs system call events using `sysdig` and saves them as `.txt` files.
# - Handles errors gracefully and provides detailed status updates.
# - Ensures proper cleanup by terminating background processes and setting file permissions.
#
# Commands:
# - `dumpcap`: Used for capturing network packets.
# - `sysdig`: Used for auditing system calls based on specific filters.
# - `pkill`: Used to terminate `dumpcap` and `sysdig` processes after execution.
# - `chmod`: Used to set appropriate permissions for the generated files.
# - `sed`: Updates environment variables to mark the process as complete.
#
# Usage:
# - Configure the SCRIPT_NAME and ITERATION_COUNT options in the Metasploit datastore.
# - Run the module to execute the specified scripts and perform packet capture and auditing.
#
# Note:
# - Ensure that the required tools (`dumpcap` and `sysdig`) are installed and accessible
#   on the target machine.
# - The module is designed for Linux-based platforms and requires a Meterpreter session.
#
# Author:
# - Ian Johnson

require 'msf/core'

class Metasploit3 < Msf::Post

  def initialize(info = {})
    super(update_info(info,
      'Name'           => 'Loader Script',
      'Description'    => 'Handles running custom scripts with iterations',
      'Author'         => 'Ian Johnson',
      'Platform'       => ['linux'],
      'SessionTypes'   => ['meterpreter']
    ))

    # Register the options for the script
    register_options(
      [
        OptString.new('SCRIPT_NAME', [true, 'The name of the script to run', 'default_script']),
        OptInt.new('ITERATION_COUNT', [true, 'The number of iterations to run', 1])
      ], self.class)
  end

  def run
    # Fetch the script name and iteration count from datastore
    combined = datastore['SCRIPT_NAME']
    scripts = combined.split("_")
    iteration_count = datastore['ITERATION_COUNT']

    print_status("Loader script started with SCRIPT_NAME=#{combined} and ITERATION_COUNT=#{iteration_count}")

    # Validate arguments
    if scripts.nil? || iteration_count.nil?
      print_error("Missing SCRIPT_NAME or ITERATION_COUNT.")
      return
    end

    # Define the victim's path for the dumpcap binary
    dumpcap_path = "/usr/bin/dumpcap"
    pcap_file_path = "/home/kali/meterpreter/pcap/#{combined}_#{iteration_count}.pcap"
    sysdig_file_path = "/home/kali/meterpreter/sysdig/#{combined}_#{iteration_count}.txt"

    # Define the commands to run dumpcap and sysdig
    dumpcap_command = "sudo nohup #{dumpcap_path} -i eth0 -w #{pcap_file_path} > /dev/null 2>&1 &"
    sysdig_command = "sudo sysdig -p'%evt.num %evt.rawtime.s.%evt.rawtime.ns %evt.cpu %proc.name (%proc.pid) %evt.dir %evt.type cwd=%proc.cwd %evt.args latency=%evt.latency exepath=%proc.exepath' \"proc.name!=tmux and (evt.type=read or evt.type=readv or evt.type=write or evt.type=writev or evt.type=fcntl or evt.type=accept or evt.type=connect)\" > #{sysdig_file_path} &"

    # Run the commands
    print_status("Starting wireshark packet capture and sysdig audit logging ...")
    dumpcap_result = cmd_exec(dumpcap_command)
    sysdig_result = cmd_exec(sysdig_command)

    # Loop through each script in the list
    scripts.each do |script|
        begin
                # Attempt to run the specified script
                print_status("Running Meterpreter script #{script}.rb ...")
		session.run_cmd("run custom/#{script}.rb")
                print_status("#{script}.rb executed successfully.")
        rescue => e
                # If an error occurs, print the error message
                print_error("Error while running #{script}.rb: #{e.message}")
        end
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

    print_good("Iteration #{iteration_count}/500 for #{combined}.rb completed.")

   cmd_exec("sudo sed -i 's/^COMPLETE=.*/COMPLETE=1/' /etc/environment")
  end

end
