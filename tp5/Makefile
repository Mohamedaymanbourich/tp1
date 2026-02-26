# TP5 - MPI (P2P & Collectives)
# Top-level Makefile
#
# Usage:
#   make all       — compile all exercises
#   make ex1       — compile exercise 1 only
#   make clean     — remove all binaries
#
# Prerequisites: mpicc must be in PATH

MPICC = mpicc
CFLAGS = -O2 -Wall
LDFLAGS = -lm

.PHONY: all clean ex1 ex2 ex3 ex4 ex5

all: ex1 ex2 ex3 ex4 ex5

ex1:
	$(MPICC) $(CFLAGS) -o ex1/ex1 ex1/ex1.c

ex2:
	$(MPICC) $(CFLAGS) -o ex2/ex2 ex2/ex2.c

ex3:
	$(MPICC) $(CFLAGS) -o ex3/ex3 ex3/ex3.c

ex4:
	$(MPICC) $(CFLAGS) -o ex4/ex4 ex4/ex4.c $(LDFLAGS)

ex5:
	$(MPICC) $(CFLAGS) -o ex5/ex5 ex5/ex5.c $(LDFLAGS)

clean:
	rm -f ex1/ex1 ex2/ex2 ex3/ex3 ex4/ex4 ex5/ex5
	rm -f ex4/timings.csv ex5/timings.csv
	rm -f ex4/*.png ex5/*.png
