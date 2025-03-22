#!/bin/bash

# Function to check if the IP is in a reserved range
is_reserved_ip() {
    local ip=$1
    local first_octet=$(echo $ip | cut -d '.' -f 1)
    local second_octet=$(echo $ip | cut -d '.' -f 2)
    local third_octet=$(echo $ip | cut -d '.' -f 3)
    local fourth_octet=$(echo $ip | cut -d '.' -f 4)
    
    # Check if the IP is in a reserved range
    if [[ $fourth_octet -eq 28 ]] || \
       [[ $first_octet -eq 127 ]] || \
       [[ $first_octet -eq 169 && $second_octet -eq 254 ]] || \
       [[ $first_octet -eq 172 && $second_octet -ge 16 && $second_octet -le 31 ]] || \
       [[ $first_octet -eq 192 && $second_octet -eq 168 ]] || \
       [[ $first_octet -ge 224 && $first_octet -le 239 ]]; then
        return 1  # Reserved IP
    fi
    return 0  # Valid IP
}

# Function to generate a random valid IP address
generate_ip() {
    while :; do
        # Generate the first three octets as before
        octet1=192
        octet2=168
        octet3=0

        # Generate a random fourth octet excluding 0 and 255
        octet4=$((RANDOM % 254 + 1))

        # Combine the octets to form an IP address
        ip="$octet1.$octet2.$octet3.$octet4"

        # Check if the generated IP is reserved
        if ! is_reserved_ip "$ip"; then
            echo $ip
            return
        fi
    done
}

# Call the function to generate and print a valid IP
generate_ip
