def run_script(client)
# Port scanner Meterpreter script (scans all ports and only prints open ones)
# Save this as custom/port_scanner.rb and run it via "run custom/port_scanner.rb" in Meterpreter

# Define the victim's IP and the range of ports to scan (1-65535)
victim_ip = '192.168.0.222'
ports = (1..65535)

# Start the port scanning loop
ports.each do |port|
  begin
    # Create a raw socket and attempt to connect
    socket = Socket.new(Socket::AF_INET, Socket::SOCK_STREAM, 0)
    sockaddr = Socket.sockaddr_in(port, victim_ip)

    # Set the socket to non-blocking
    socket.connect_nonblock(sockaddr)
  rescue Errno::EINPROGRESS, Errno::EALREADY
    # If the connection is still pending, do nothing (wait)
    next
  rescue Errno::ECONNREFUSED, Errno::ETIMEDOUT, Errno::EHOSTUNREACH
    # If the connection is refused or times out, skip this port
    next
  rescue => e
    # Catch any other unexpected errors and skip the port
    next
  else
    # If the connection was successful, print the open port
    print_good("Port #{port} is OPEN!")
  ensure
    # Close the socket after checking
    socket.close if socket
  end
end

print_good("Port scanning complete!")
end
