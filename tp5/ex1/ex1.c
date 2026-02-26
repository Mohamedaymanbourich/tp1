/*
 * TP5 - Exercise 1: Hello World (MPI)
 *
 * 1. Print "Hello World" from every process.
 * 2. Each process prints its rank and total number of processes.
 * 3. Only rank 0 prints a message.
 * 4. Answer: If MPI_Finalize is omitted, the MPI runtime may not flush
 *    buffered output, may leak resources, and the job can hang or be
 *    aborted by mpirun.
 *
 * Compile: mpicc -o ex1 ex1.c
 * Run:     mpirun -np 4 ./ex1
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Question 1: Every process says Hello World */
    printf("Hello World from process %d\n", rank);

    /* Question 2: Each process prints rank and total */
    printf("I am process %d among %d\n", rank, size);

    /* Question 3: Only rank 0 prints */
    if (rank == 0) {
        printf("[Rank 0 only] There are %d processes running.\n", size);
    }

    /*
     * Question 4: What happens if MPI_Finalize() is omitted?
     *
     * - The MPI standard requires MPI_Finalize to be called before exit.
     * - Omitting it may cause:
     *   * Buffered output not being flushed (missing prints).
     *   * Resources (shared memory, sockets) not being released.
     *   * mpirun detecting abnormal termination and aborting all processes.
     *   * Potential hangs if other processes are blocked in communication.
     */
    MPI_Finalize();
    return 0;
}
