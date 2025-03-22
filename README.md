# Combining System Auditing and Network Traffic Analysis to Detect Malicious Network Activity

## Author  
**Ian Johnson**  
*Ira A. Fulton Schools of Engineering*  
*Barrett, The Honors College at Arizona State University*  
*CSE 492: Honors Directed Study*  
*Advisor: Dr. Xusheng Xiao*

---

## Description  
Cybercrime continues to run rampant worldwide, as security experts race with skilled exploiters and attackers to keep the transfer of sensitive data over our networks reliable and as safe as possible. According to the 2023 Unit 42 Network Threat Trends Research Report from Palo Alto Networks:  

> "While 48.94% of network communication generated during sandbox analysis (including both malicious and benign files) uses encrypted SSL for its traffic, 12.91% of what is caused by malware is SSL traffic."

This alarming ratio of malicious Secure Socket Layer (SSL) traffic underscores the challenges posed by encrypted network traffic, which is not readily inspectable. Traditional methods of detecting malicious traffic rely on inspecting packets for suspicious patterns, but encrypted malware easily operates undetected. Over the years, progressively advanced encryption techniques have enabled malicious traffic to blend in with benign traffic.

The goal of this project is to investigate how network traffic and host-based information can be used in tandem to detect such attacks. Relying solely on network traffic analysis is insufficient in today’s large-scale networks. Instead, this analysis must be combined with host information, particularly system audit logs.  

System audit logs document specific events, timestamps, and the entities involved. For example, a log might record that a process `p` accessed a file `f`, denoted as `f(f→p)`. Logs can capture various events, such as admin processes, system-wide changes, login attempts, and data access. Since every device in a network contains an audit log, these pieces of evidence can be analyzed collectively to detect unusual system behavior. The insights from host-based analysis can then narrow the focus of network traffic analysis to identify potential attacks more effectively.

---

## Methodology  
1. **Infrastructure Phase**  
   - Set up a virtual environment to simulate a network.  
   - Identify a reliable dataset simulating malicious network activity.  
   - Use the `sysdig` tool to collect system audit logs.  
   - Parse collected logs into Java objects using Dr. Xiao’s lab-provided code.

2. **Analysis Detection Phase**  
   - Develop an algorithm to detect anomalies in the parsed system auditing events.

3. **Integration Phase**  
   - Pair detected anomalies with corresponding network events.  
   - Map system processes to IP addresses based on network event pairings (host-based analysis).

---

## The Committee  
- **Director**: Dr. Xusheng Xiao  
- **Second Committee Member**: Dr. Jaejong Baek  

The committee will meet biweekly on Tuesdays at 3:30 PM via Zoom, starting September 10, 2024. Meetings will focus on:  
- Sharing research and thesis progress.  
- Addressing challenges.  
- Receiving feedback and advice.

---

## Goals & Milestones  

1. **Fall 2024 Semester**  
   - **Infrastructure Phase**: 4–5 weeks  
   - **Analysis Detection Phase**: 8–10 weeks  
   - **Integration Phase**: 4–5 weeks  
   - Target completion: End of Fall 2024 semester.  
   - Potential extension: Winter break into early Spring 2025 semester.

2. **Spring 2025 Semester**  
   - **Thesis Drafting**  
     - Begin drafting: January 2025.  
     - Submit first draft: Early March 2025.  
   - **Thesis Defense**  
     - Schedule defense: Mid-March 2025.  
     - Finalize and submit thesis: April 2025.  

After completing the defense and final revisions, the thesis will be submitted to Barrett, marking the conclusion of this research journey.
