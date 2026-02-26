/*
 * TP5 - Exercise 4: Matrix-Vector Product (MPI)
 *
 * Parallel matrix-vector multiplication  y = A * b  using MPI.
 * - Root generates A and b, computes serial result for verification.
 * - Rows of A are distributed via MPI_Scatterv (handles N % P != 0).
 * - b is broadcast to all processes.
 * - Each process computes its local portion of y.
 * - Results are gathered via MPI_Gatherv.
 * - Speedup / efficiency are reported and appended to timings.csv.
 *
 * Compile: mpicc -O2 -o ex4 ex4.c -lm
 * Run:     mpirun -np 4 ./ex4 1000
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/* Serial matrix-vector multiplication */
void matrixVectorMult(double *A, double *b, double *x, int rows, int N) {
    for (int i = 0; i < rows; ++i) {
        x[i] = 0.0;
        for (int j = 0; j < N; ++j) {
            x[i] += A[i * N + j] * b[j];
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 2) {
        if (rank == 0) printf("Usage: %s <matrix_size>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        if (rank == 0) printf("Matrix size must be positive.\n");
        MPI_Finalize();
        return 1;
    }

    /* ---- Compute row distribution (handles N % P != 0) ---- */
    int base_rows = N / world_size;
    int rem = N % world_size;
    int local_rows = base_rows + (rank < rem ? 1 : 0);

    /* sendcounts / displs for Scatterv (in number of doubles) */
    int *sendcounts = (int *)malloc(world_size * sizeof(int));
    int *displs_A   = (int *)malloc(world_size * sizeof(int));
    /* recvcounts / rdispls for Gatherv (in number of doubles) */
    int *recvcounts = (int *)malloc(world_size * sizeof(int));
    int *rdispls    = (int *)malloc(world_size * sizeof(int));

    for (int r = 0, offset = 0; r < world_size; ++r) {
        int rows_r = base_rows + (r < rem ? 1 : 0);
        sendcounts[r] = rows_r * N;
        displs_A[r]   = offset * N;
        recvcounts[r]  = rows_r;
        rdispls[r]     = offset;
        offset += rows_r;
    }

    /* ---- Allocations ---- */
    double *A          = NULL;  /* full matrix — root only */
    double *b          = (double *)malloc(N * sizeof(double));
    double *x_serial   = NULL;  /* serial result — root only */
    double *x_parallel = NULL;  /* gathered parallel result — root only */

    double t_serial = 0.0;

    if (rank == 0) {
        A          = (double *)malloc((size_t)N * N * sizeof(double));
        x_serial   = (double *)malloc(N * sizeof(double));
        x_parallel = (double *)malloc(N * sizeof(double));

        if (!A || !b || !x_serial || !x_parallel) {
            fprintf(stderr, "Memory allocation failed on root.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* ---- Initialize A and b (same pattern from the assignment) ---- */
        srand48(42);

        /* Zero out A first */
        for (int i = 0; i < N * N; ++i) A[i] = 0.0;

        /* Fill A[0][:100] with random values */
        int limit = (N < 100) ? N : 100;
        for (int j = 0; j < limit; ++j)
            A[0 * N + j] = drand48();

        /* Copy A[0][:100] into A[1][100:200] if possible */
        if (N > 1 && N > 100) {
            int copy_len = (N - 100 < 100) ? (N - 100) : 100;
            for (int j = 0; j < copy_len; ++j)
                A[1 * N + (100 + j)] = A[0 * N + j];
        }

        /* Set diagonal */
        for (int i = 0; i < N; ++i)
            A[i * N + i] = drand48();

        /* Fill vector b */
        for (int i = 0; i < N; ++i)
            b[i] = drand48();

        /* ---- Serial computation for timing and verification ---- */
        double ts0 = MPI_Wtime();
        matrixVectorMult(A, b, x_serial, N, N);
        double ts1 = MPI_Wtime();
        t_serial = ts1 - ts0;
    }

    /* ---- Parallel computation ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double tp0 = MPI_Wtime();

    /* Broadcast b to all processes */
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Allocate local rows of A */
    double *A_local = NULL;
    if (local_rows > 0) {
        A_local = (double *)malloc((size_t)local_rows * N * sizeof(double));
        if (!A_local) {
            fprintf(stderr, "Rank %d: failed to allocate A_local\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Scatter rows of A */
    MPI_Scatterv(A, sendcounts, displs_A, MPI_DOUBLE,
                 A_local, local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    /* Local matrix-vector multiply */
    double *x_local = NULL;
    if (local_rows > 0) {
        x_local = (double *)malloc(local_rows * sizeof(double));
        if (!x_local) {
            fprintf(stderr, "Rank %d: failed to allocate x_local\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        matrixVectorMult(A_local, b, x_local, local_rows, N);
    }

    /* Gather results on root */
    MPI_Gatherv(x_local, local_rows, MPI_DOUBLE,
                x_parallel, recvcounts, rdispls, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double tp1 = MPI_Wtime();
    double t_parallel = tp1 - tp0;

    /* Get maximum parallel time across all processes */
    double max_parallel_time = 0.0;
    MPI_Reduce(&t_parallel, &max_parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ---- Root: verify and report ---- */
    if (rank == 0) {
        /* Compare parallel result with serial */
        double max_error = 0.0;
        for (int i = 0; i < N; ++i) {
            double diff = fabs(x_parallel[i] - x_serial[i]);
            if (diff > max_error) max_error = diff;
        }

        double speedup    = t_serial / max_parallel_time;
        double efficiency = speedup / world_size;

        printf("N=%d P=%d serial=%.6e parallel=%.6e speedup=%.4f efficiency=%.4f max_error=%e\n",
               N, world_size, t_serial, max_parallel_time, speedup, efficiency, max_error);

        /* Append to CSV for plotting */
        FILE *f = fopen("timings.csv", "a");
        if (f) {
            fprintf(f, "%d,%d,%.12e,%.12e,%.6f,%.6f,%e\n",
                    N, world_size, t_serial, max_parallel_time, speedup, efficiency, max_error);
            fclose(f);
        }
    }

    /* ---- Cleanup ---- */
    if (A_local) free(A_local);
    if (x_local) free(x_local);
    if (A) free(A);
    free(b);
    if (x_serial)   free(x_serial);
    if (x_parallel) free(x_parallel);
    free(sendcounts);
    free(displs_A);
    free(recvcounts);
    free(rdispls);

    MPI_Finalize();
    return 0;
}
