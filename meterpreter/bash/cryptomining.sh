#!/bin/bash

# cryptomining.sh
#
# This script simulates cryptomining by performing hashing operations using 
# various algorithms for a randomly determined runtime between 200ms and 500ms.
#
# Functionality:
# - Randomly selects a hashing algorithm from a predefined list.
# - Continuously generates random data and hashes it using the selected algorithm.
# - Introduces a random delay between hashing operations to simulate realistic behavior.
# - Automatically terminates all hashing processes once the runtime limit is reached.
#
# Variables:
# - runtime_ms: Randomly generated runtime in milliseconds (200ms to 500ms).
# - start_time: Timestamp in nanoseconds when the script starts execution.
# - pids: Array to store process IDs of background hashing operations.
# - hash_algorithms: List of hashing algorithms to randomly select from.
#
# Workflow:
# 1. Calculate the remaining runtime and ensure the script stops when the limit is reached.
# 2. Randomly select a hashing algorithm and simulate hashing using `/dev/urandom`.
# 3. Introduce a random sleep delay between 5ms and 10ms, ensuring it does not exceed the remaining runtime.
# 4. Terminate all background hashing processes after the simulation ends.
#
# Note:
# - The script uses `/dev/urandom` to generate random data for hashing.
# - Background processes are managed and terminated to prevent resource leaks.
# - Output from hashing operations is redirected to `/dev/null` to avoid clutter.

# Generate a random runtime between 200ms (0.2s) and 500ms (0.5s)
runtime_ms=$(( RANDOM % 301 + 200 ))
start_time=$(date +%s%N)

pids=()
hash_algorithms=("md5sum" "sha1sum" "sha256sum" "sha512sum" "b2sum")

while true; do
  current_time=$(date +%s%N)
  elapsed_time_ms=$(( (current_time - start_time) / 1000000 ))
  remaining_time_ms=$(( runtime_ms - elapsed_time_ms ))  # Calculate remaining time

  if [ "$remaining_time_ms" -le 0 ]; then
    break  # Stop when the runtime limit is reached
  fi

  # Randomly select a hashing algorithm
  algo=${hash_algorithms[$RANDOM % ${#hash_algorithms[@]}]}
  echo "Simulating cryptomining with $algo..." > /dev/null
  cat /dev/urandom | $algo &  # Start hashing process
  pids+=($!)

  # Introduce a random sleep delay between 5ms and 10ms, but not longer than remaining time
  max_sleep_ms=$(( remaining_time_ms < 10 ? remaining_time_ms : 10 ))
  sleep_ms=$(( RANDOM % (max_sleep_ms - 4) + 5 ))  # Ensure at least 5ms delay
  sleep $(awk -v t="$sleep_ms" 'BEGIN { printf "%.3f", t / 1000 }')
done

echo "Cryptomining simulation finished."

# Kill all related processes
for pid in "${pids[@]}"; do
  kill $pid 2>/dev/null
done
