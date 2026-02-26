/*
 * TP5 - Exercise 2: Sharing Data (MPI)
 *
 * Rank 0 reads integers from stdin and broadcasts them to all processes.
 * Each process prints its rank and the received value.
 * The loop stops when a negative integer is entered.
 *
 * Compile: mpicc -o ex2 ex2.c
 * Run:     mpirun -np 4 ./ex2
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int value;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    do {
        /* Rank 0 reads the value from terminal */
        if (rank == 0) {
            printf("Enter an integer (negative to quit): ");
            fflush(stdout);
            if (scanf("%d", &value) != 1) {
                value = -1; /* treat read failure as quit */
            }
        }

        /* Broadcast the value from rank 0 to all processes */
        MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);

        /* Each process prints its rank and the received value */
        if (value >= 0) {
            printf("Process %d got %d\n", rank, value);
            fflush(stdout);
        }
    } while (value >= 0);

    MPI_Finalize();
    return 0;
}
