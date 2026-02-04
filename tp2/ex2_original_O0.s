	.file	"ex2_original.c"
	.intel_syntax noprefix
	.text
	.section .rdata,"dr"
.LC4:
	.ascii "x = %f, y = %f, time = %f s\12\0"
	.text
	.globl	main
	.def	main;	.scl	2;	.type	32;	.endef
	.seh_proc	main
main:
	push	rbp
	.seh_pushreg	rbp
	mov	rbp, rsp
	.seh_setframe	rbp, 0
	sub	rsp, 80
	.seh_stackalloc	80
	.seh_endprologue
	call	__main
	movsd	xmm0, QWORD PTR .LC0[rip]
	movsd	QWORD PTR -32[rbp], xmm0
	movsd	xmm0, QWORD PTR .LC1[rip]
	movsd	QWORD PTR -40[rbp], xmm0
	pxor	xmm0, xmm0
	movsd	QWORD PTR -8[rbp], xmm0
	pxor	xmm0, xmm0
	movsd	QWORD PTR -16[rbp], xmm0
	call	clock
	mov	DWORD PTR -44[rbp], eax
	mov	DWORD PTR -20[rbp], 0
	jmp	.L2
.L3:
	movsd	xmm0, QWORD PTR -32[rbp]
	mulsd	xmm0, QWORD PTR -40[rbp]
	movsd	xmm1, QWORD PTR -8[rbp]
	addsd	xmm0, xmm1
	movsd	QWORD PTR -8[rbp], xmm0
	movsd	xmm0, QWORD PTR -32[rbp]
	mulsd	xmm0, QWORD PTR -40[rbp]
	movsd	xmm1, QWORD PTR -16[rbp]
	addsd	xmm0, xmm1
	movsd	QWORD PTR -16[rbp], xmm0
	add	DWORD PTR -20[rbp], 1
.L2:
	cmp	DWORD PTR -20[rbp], 99999999
	jle	.L3
	call	clock
	mov	DWORD PTR -48[rbp], eax
	mov	eax, DWORD PTR -48[rbp]
	sub	eax, DWORD PTR -44[rbp]
	pxor	xmm0, xmm0
	cvtsi2sd	xmm0, eax
	movsd	xmm1, QWORD PTR .LC3[rip]
	divsd	xmm0, xmm1
	movapd	xmm2, xmm0
	movq	rcx, xmm0
	movsd	xmm1, QWORD PTR -16[rbp]
	mov	rdx, QWORD PTR -16[rbp]
	movsd	xmm0, QWORD PTR -8[rbp]
	mov	rax, QWORD PTR -8[rbp]
	movapd	xmm3, xmm2
	mov	r9, rcx
	movapd	xmm2, xmm1
	mov	r8, rdx
	movapd	xmm1, xmm0
	mov	rdx, rax
	lea	rax, .LC4[rip]
	mov	rcx, rax
	call	printf
	mov	eax, 0
	add	rsp, 80
	pop	rbp
	ret
	.seh_endproc
	.section .rdata,"dr"
	.align 8
.LC0:
	.long	-1717986918
	.long	1072798105
	.align 8
.LC1:
	.long	858993459
	.long	1072902963
	.align 8
.LC3:
	.long	0
	.long	1083129856
	.def	__main;	.scl	2;	.type	32;	.endef
	.ident	"GCC: (Rev2, Built by MSYS2 project) 14.2.0"
	.def	clock;	.scl	2;	.type	32;	.endef
	.def	printf;	.scl	2;	.type	32;	.endef
