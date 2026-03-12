# TP7 - MPI Communicators (2D Cartesian Topology)
# Makefile
#
# Usage:
#   make all       — compile both exercises
#   make ex1       — compile exercise 1 (Game of Life)
#   make ex2       — compile exercise 2 (Poisson solver)
#   make clean     — remove binaries
#   make run_ex1   — run exercise 1 with 4 processes
#   make run_ex2   — run exercise 2 with 4 processes
#
# Prerequisites: mpicc must be in PATH

MPICC   = mpicc
CFLAGS  = -O2 -Wall -std=c99
LDFLAGS = -lm

.PHONY: all clean ex1 ex2 run_ex1 run_ex2

all: ex1 ex2

ex1: game_of_life.c
	$(MPICC) $(CFLAGS) -o game_of_life game_of_life.c

ex2: poisson.c compute.c
	$(MPICC) $(CFLAGS) -o poisson poisson.c compute.c $(LDFLAGS)

run_ex1: ex1
	mpirun -np 4 ./game_of_life

run_ex2: ex2
	mpirun -np 4 ./poisson

clean:
	rm -f game_of_life poisson
