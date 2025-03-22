#!/bin/bash

# Generate a random runtime between 200ms (0.2s) and 500ms (0.5s)
runtime_ms=$(( RANDOM % 301 + 200 ))  # Random number between 200 and 500 milliseconds
start_time=$(date +%s%N)  # Start time in nanoseconds

pids=()  # Array to store process IDs
hash_algorithms=("md5sum" "sha1sum" "sha256sum")  # Possible hashing algorithms

while true; do
  current_time=$(date +%s%N)  # Get current time in nanoseconds
  elapsed_time_ms=$(( (current_time - start_time) / 1000000 ))  # Convert to milliseconds
  remaining_time_ms=$(( runtime_ms - elapsed_time_ms ))  # Calculate remaining time

  if [ "$remaining_time_ms" -le 0 ]; then
    break  # Stop when the runtime limit is reached
  fi

  # Randomly select a hashing algorithm
  algo=${hash_algorithms[$RANDOM % ${#hash_algorithms[@]}]}
  echo "Simulating cryptomining with $algo..." > /dev/null
  cat /dev/urandom | $algo &  # Start hashing process
  pids+=($!)  # Store PID

  # Introduce a random sleep delay between 5ms and 10ms, but not longer than remaining time
  max_sleep_ms=$(( remaining_time_ms < 10 ? remaining_time_ms : 10 ))  # Avoid overshooting time
  sleep_ms=$(( RANDOM % (max_sleep_ms - 4) + 5 ))  # Ensure at least 5ms delay
  sleep $(awk -v t="$sleep_ms" 'BEGIN { printf "%.3f", t / 1000 }')  # Convert ms to seconds
done

# End of simulation
echo "Cryptomining simulation finished."

# Kill all related processes
for pid in "${pids[@]}"; do
  kill $pid 2>/dev/null
done
