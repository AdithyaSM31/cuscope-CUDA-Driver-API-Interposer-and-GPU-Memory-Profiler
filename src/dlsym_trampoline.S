/* cuscope's exported dlsym (interception path 2), x86-64 System V.
 *
 * glibc resolves RTLD_NEXT, and the local scope of RTLD_DEFAULT, from the
 * return address of dlsym's caller. A C++ override that *calls* the real dlsym
 * makes libcuscope the caller and changes every such lookup in the process.
 * So this trampoline asks cuscope_dlsym_decide (intercept.cpp) what to do,
 * passing the caller's return address, and either returns cuscope's answer or
 * restores the arguments and *jumps* to the real dlsym: glibc then sees the
 * original caller, exactly as if cuscope were not there.
 *
 * cuscope_dlsym_decide returns a two-pointer struct in rax:rdx:
 *   rdx == 0  return rax
 *   rdx != 0  tail-jump to rdx (the real dlsym) with the original arguments
 */
    .text
    .globl  dlsym
    .type   dlsym, @function
dlsym:
    .cfi_startproc
    endbr64
    pushq   %rdi                    /* handle */
    .cfi_adjust_cfa_offset 8
    pushq   %rsi                    /* symbol */
    .cfi_adjust_cfa_offset 8
    subq    $8, %rsp                /* 16-byte alignment for the call */
    .cfi_adjust_cfa_offset 8
    movq    24(%rsp), %rdx          /* the caller's return address */
    call    cuscope_dlsym_decide
    addq    $8, %rsp
    .cfi_adjust_cfa_offset -8
    popq    %rsi
    .cfi_adjust_cfa_offset -8
    popq    %rdi
    .cfi_adjust_cfa_offset -8
    testq   %rdx, %rdx
    jnz     1f
    ret
1:
    jmp     *%rdx
    .cfi_endproc
    .size   dlsym, .-dlsym

    .section .note.GNU-stack,"",@progbits
