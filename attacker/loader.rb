
require 'msf/core'

class Metasploit3 < Msf::Post

  def initialize(info = {})
    super(update_info(info,
      'Name'           => 'Loader Script',
      'Description'    => 'Handles running custom scripts with iterations',
      'Author'         => 'Your Name',
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
    #print_error("Scripts array after split: #{scripts.inspect}")
    iteration_count = datastore['ITERATION_COUNT']

    # Add debugging output to verify values
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

    # Running the Meterpreter script
    # print_status("Running Meterpreter script #{script}.rb ...")

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
   #cmd_exec("source /etc/environment")
  end

end
