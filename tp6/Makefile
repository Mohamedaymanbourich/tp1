# TP6 - MPI Derived Types
# Makefile
#
# Usage:
#   make all       — compile both exercises
#   make ex1       — compile exercise 1 only
#   make ex2       — compile exercise 2 only
#   make clean     — remove binaries
#   make run_ex1   — run exercise 1
#   make run_ex2   — run exercise 2
#
# Prerequisites: mpicc must be in PATH

MPICC  = mpicc
CFLAGS = -O2 -Wall -std=c99
LDFLAGS = -lm

.PHONY: all clean ex1 ex2 run_ex1 run_ex2

all: ex1 ex2

ex1: ex1_transpose.c
	$(MPICC) $(CFLAGS) -o ex1_transpose ex1_transpose.c

ex2: distrib_grad.c
	$(MPICC) $(CFLAGS) -o distrib_grad distrib_grad.c $(LDFLAGS)

run_ex1: ex1
	mpirun -np 2 ./ex1_transpose

run_ex2: ex2
	mpirun -np 4 ./distrib_grad 1000

clean:
	rm -f ex1_transpose distrib_grad
	rm -f benchmark_results.csv speedup_efficiency.png
