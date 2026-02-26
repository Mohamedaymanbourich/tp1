/*
 * TP5 - Exercise 3: Sending in a Ring (Broadcast by Ring)
 *
 * Process 0 reads a value from the user.
 * It sends it to process 1, which adds its rank and sends to process 2, etc.
 * Each process receives the data, adds its rank, prints the result,
 * and forwards it to the next process.
 *
 * Compile: mpicc -o ex3 ex3.c
 * Run:     mpirun -np 4 ./ex3
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int value;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        /* Process 0 reads the initial value */
        printf("Enter a value: ");
        fflush(stdout);
        if (scanf("%d", &value) != 1) {
            fprintf(stderr, "Invalid input.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Process 0 adds its rank (0) — value stays the same */
        printf("Process %d: value = %d\n", rank, value);
        fflush(stdout);

        /* Send to process 1 if there are more processes */
        if (size > 1) {
            MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        }
    } else {
        /* Receive from the previous process */
        MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);

        /* Add own rank */
        value += rank;

        /* Print the result */
        printf("Process %d: value = %d\n", rank, value);
        fflush(stdout);

        /* Forward to the next process (if not the last) */
        if (rank < size - 1) {
            MPI_Send(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
