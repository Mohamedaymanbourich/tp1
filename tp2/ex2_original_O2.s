	.file	"ex2_original.c"
	.intel_syntax noprefix
	.text
	.section .rdata,"dr"
.LC3:
	.ascii "x = %f, y = %f, time = %f s\12\0"
	.section	.text.startup,"x"
	.p2align 4
	.globl	main
	.def	main;	.scl	2;	.type	32;	.endef
	.seh_proc	main
main:
	push	rbp
	.seh_pushreg	rbp
	push	rbx
	.seh_pushreg	rbx
	sub	rsp, 56
	.seh_stackalloc	56
	lea	rbp, 48[rsp]
	.seh_setframe	rbp, 48
	.seh_endprologue
	call	__main
	call	clock
	movsd	xmm0, QWORD PTR .LC1[rip]
	pxor	xmm1, xmm1
	mov	ebx, eax
	mov	eax, 100000000
	.p2align 4
	.p2align 4
	.p2align 3
.L2:
	addsd	xmm1, xmm0
	addsd	xmm1, xmm0
	sub	eax, 2
	jne	.L2
	movsd	QWORD PTR -8[rbp], xmm1
	call	clock
	movsd	xmm1, QWORD PTR -8[rbp]
	pxor	xmm0, xmm0
	lea	rcx, .LC3[rip]
	sub	eax, ebx
	cvtsi2sd	xmm0, eax
	movapd	xmm2, xmm1
	divsd	xmm0, QWORD PTR .LC2[rip]
	movq	r8, xmm1
	movq	r9, xmm0
	movapd	xmm3, xmm0
	movq	rdx, xmm1
	call	printf
	xor	eax, eax
	add	rsp, 56
	pop	rbx
	pop	rbp
	ret
	.seh_endproc
	.section .rdata,"dr"
	.align 8
.LC1:
	.long	1374389535
	.long	1073028792
	.align 8
.LC2:
	.long	0
	.long	1083129856
	.def	__main;	.scl	2;	.type	32;	.endef
	.ident	"GCC: (Rev2, Built by MSYS2 project) 14.2.0"
	.def	clock;	.scl	2;	.type	32;	.endef
	.def	printf;	.scl	2;	.type	32;	.endef
