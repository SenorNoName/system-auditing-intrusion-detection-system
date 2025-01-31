#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/wait.h>

// Function to hook (e.g., printf)
void hooked_function() {
    printf("Function hooked!\n");
}

void inject_function(pid_t pid, void *target_function, void *hook_function) {
    // Attach to the target process
    if (ptrace(PTRACE_ATTACH, pid, NULL, NULL) < 0) {
        perror("ptrace(ATTACH)");
        exit(1);
    }
    wait(NULL); // Wait for the process to stop

    // Overwrite the target function's entry point with a jump to the hook function
    unsigned char jump_code[] = {
        0xE9, 0x00, 0x00, 0x00, 0x00 // JMP <offset>
    };
    long offset = (long)hook_function - ((long)target_function + 5);
    memcpy(&jump_code[1], &offset, 4);

    // Write the jump code to the target function
    if (ptrace(PTRACE_POKETEXT, pid, target_function, *(long *)jump_code) < 0) {
        perror("ptrace(POKETEXT)");
        exit(1);
    }

    // Detach from the target process
    if (ptrace(PTRACE_DETACH, pid, NULL, NULL) < 0) {
        perror("ptrace(DETACH)");
        exit(1);
    }

    printf("Function hooked in process %d\n", pid);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <pid>\n", argv[0]);
        return 1;
    }

    pid_t pid = atoi(argv[1]); // Target process ID
    void *target_function = (void *)0x080484b6; // Address of the target function (e.g., printf)
    void *hook_function = (void *)hooked_function; // Address of the hook function

    inject_function(pid, target_function, hook_function);

    return 0;
}
