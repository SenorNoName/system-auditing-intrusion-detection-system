"""
This script processes SVG files to extract information about processes and IP addresses,
and writes the extracted data into a CSV file. It is designed to analyze SVG files for
specific patterns related to different attack types.
Modules:
    - os: Provides functions for interacting with the operating system.
    - csv: Provides functionality to read and write CSV files.
    - xml.etree.ElementTree: Provides tools for parsing and creating XML data.
Functions:
    - process_svg(svg_file, all_processes):
        Processes an SVG file to extract all instances of processes from the given list.
        Args:
            svg_file (str): Path to the SVG file to be processed.
            all_processes (list): List of process names to search for in the SVG file.
        Returns:
            str: A semicolon-separated string of matched process instances.
    - get_ip_address(svg_file):
        Extracts the IP address from an SVG file based on specific text patterns.
        Args:
            svg_file (str): Path to the SVG file to be processed.
        Returns:
            str: The extracted IP address, or an empty string if no IP address is found.
    - write_csv(all_processes):
        Writes the extracted data from SVG files into a CSV file named "labels.csv".
        If the file does not exist, it creates it and writes the header row.
        Args:
            all_processes (list): List of all process names to search for in the SVG files.
        Returns:
            None
Global Variables:
    - attack_processes (dict): A dictionary mapping attack types to their associated processes.
    - all_processes (list): A flattened list of all processes from the `attack_processes` dictionary.
Usage:
    The script scans the "svg" directory for SVG files, processes each file to extract
    relevant data, and appends the results to "labels.csv". Each row in the CSV file
    contains the SVG file name, corresponding PCAP file name, origin IP address, and
    detected processes.
"""

import os
import csv
import xml.etree.ElementTree as ET

# Define the processes/files for each attack type
attack_processes = {
    "ransomware": ['openssl', '.enc'],
    "exfiltration": ['scp', 'tar', '.tar.gz', '.txt.', '.pcap', '.docx', '.pptx', '.xlsx', '.pdf', '.exe', '.png', '.jpg', '.cpp', '.java', '.mp3'],
    "keylogger": ['evtest', 'grep'],
    "cryptomining": ['md5sum', 'sha1sum', 'sha256sum']
}

# Processes an SVG file to extract all instances of processes from the given list.
def process_svg(svg_file, all_processes):
    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()
        instances = []
        for element in root.iter():
            text = element.text
            if text:
                for process in all_processes:
                    if process in text: instances.append(text)
        return ";".join(instances)
    except ET.ParseError:
        return ""

# Extracts the IP address from an SVG file based on specific text patterns.
def get_ip_address(svg_file):
    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()
        for element in root.iter():
            text = element.text
            if text and "->" in text:
                start = text.find("->") + 2
                end = text.find(":", start)
                if end != -1:
                    return text[start:end].strip()
        return ""
    except ET.ParseError:
        return ""

# Writes the extracted data from SVG files into a CSV file.
def write_csv(all_processes):
    file_exists = os.path.isfile("labels.csv")
    with open("labels.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["svg_file", "pcap_file", "origin_ip", "target_processes"])
        for filename in os.listdir("svg"):
            if filename.endswith(".svg"):
                svg_path = os.path.join("svg", filename)
                pcap_file = filename.replace(".svg", ".pcap")
                instances = process_svg(svg_path, all_processes)
                ip_addr = get_ip_address(svg_path)
                writer.writerow([filename, pcap_file, ip_addr, instances])

all_processes = [process for processes in attack_processes.values() for process in processes]
write_csv(all_processes)