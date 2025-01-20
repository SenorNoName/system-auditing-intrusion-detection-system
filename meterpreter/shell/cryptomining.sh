#!/bin/bash
# Simulate CPU-intensive cryptomining activity for 5 seconds

end=$((SECONDS+5))  # Set the end time to 5 seconds from now
while [ $SECONDS -lt $end ]; do
  echo "Simulating cryptomining..." > /dev/null
  md5sum < /dev/urandom &
done

# End of simulation
echo "Cryptomining simulation finished."
