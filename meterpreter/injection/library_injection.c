#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/ptrace.h>
#include <sys/wait.h>

void inject_library(pid_t pid, const char *library_path) {
    // Attach to the target process
    if (ptrace(PTRACE_ATTACH, pid, NULL, NULL) < 0) {
        perror("ptrace(ATTACH)");
        exit(1);
    }
    wait(NULL); // Wait for the process to stop

    // Call dlopen in the target process
    void *dlopen_addr = dlsym(RTLD_DEFAULT, "dlopen");
    if (!dlopen_addr) {
        perror("dlsym");
        exit(1);
    }

    // Allocate memory in the target process for the library path
    void *remote_mem = mmap(NULL, strlen(library_path) + 1, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (remote_mem == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    // Write the library path to the target process
    if (ptrace(PTRACE_POKETEXT, pid, remote_mem, *(long *)library_path) < 0) {
        perror("ptrace(POKETEXT)");
        exit(1);
    }

    // Call dlopen in the target process
    struct user_regs_struct regs;
    if (ptrace(PTRACE_GETREGS, pid, NULL, &regs) < 0) {
        perror("ptrace(GETREGS)");
        exit(1);
    }
    regs.eip = (long)dlopen_addr; // Set EIP to dlopen address
    regs.eax = (long)remote_mem;  // Set EAX to the library path
    regs.ebx = RTLD_LAZY;         // Set EBX to dlopen flags
    if (ptrace(PTRACE_SETREGS, pid, NULL, &regs) < 0) {
        perror("ptrace(SETREGS)");
        exit(1);
    }

    // Detach from the target process
    if (ptrace(PTRACE_DETACH, pid, NULL, NULL) < 0) {
        perror("ptrace(DETACH)");
        exit(1);
    }

    printf("Library %s injected into process %d\n", library_path, pid);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <pid> <library_path>\n", argv[0]);
        return 1;
    }

    pid_t pid = atoi(argv[1]); // Target process ID
    const char *library_path = argv[2]; // Path to the shared library

    inject_library(pid, library_path);

    return 0;
}
