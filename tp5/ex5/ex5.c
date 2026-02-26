/*
 * TP5 - Exercise 5: Pi Calculation (MPI)
 *
 * Approximation:  pi = (4/N) * sum_{i=0}^{N-1} 1/(1 + x_i^2)
 *                 where x_i = (i + 0.5) / N
 *
 * - Iterations are split across processes (handles N % P != 0).
 * - Each process computes its local partial sum.
 * - MPI_Reduce sums partial results on root.
 * - Root also computes the serial version for speedup measurement.
 * - Results are appended to timings.csv for plotting.
 *
 * Compile: mpicc -O2 -o ex5 ex5.c -lm
 * Run:     mpirun -np 4 ./ex5 10000000
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    long long N = 1000000; /* default number of intervals */
    if (argc > 1) {
        N = atoll(argv[1]);
    }

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- Parallel computation ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double tp0 = MPI_Wtime();

    /* Distribute iterations: handles N % size != 0 */
    long long base_iters = N / size;
    long long rem = N % size;
    long long local_iters = base_iters + (rank < rem ? 1 : 0);
    long long start_i = rank * base_iters + (rank < rem ? rank : rem);
    long long end_i = start_i + local_iters;

    double local_sum = 0.0;
    for (long long i = start_i; i < end_i; i++) {
        double x = (i + 0.5) / (double)N;
        local_sum += 1.0 / (1.0 + x * x);
    }

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double tp1 = MPI_Wtime();
    double t_parallel = tp1 - tp0;

    /* Get the maximum parallel time across all processes */
    double max_parallel_time = 0.0;
    MPI_Reduce(&t_parallel, &max_parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double pi_parallel = 4.0 / (double)N * global_sum;

        /* ---- Serial computation for speedup reference ---- */
        double ts0 = MPI_Wtime();
        double serial_sum = 0.0;
        for (long long i = 0; i < N; i++) {
            double x = (i + 0.5) / (double)N;
            serial_sum += 1.0 / (1.0 + x * x);
        }
        double pi_serial = 4.0 / (double)N * serial_sum;
        double ts1 = MPI_Wtime();
        double t_serial = ts1 - ts0;

        /* Use pi_serial to prevent compiler from optimizing away the loop */
        double error = fabs(pi_parallel - M_PI);
        double serial_error = fabs(pi_serial - M_PI);
        double speedup = t_serial / max_parallel_time;
        double efficiency = speedup / size;

        printf("N=%lld P=%d pi=%.15f error=%.2e serial_error=%.2e serial=%.6e parallel=%.6e speedup=%.4f efficiency=%.4f\n",
               N, size, pi_parallel, error, serial_error, t_serial, max_parallel_time, speedup, efficiency);

        /* Append to CSV for plotting */
        FILE *f = fopen("timings.csv", "a");
        if (f) {
            fprintf(f, "%lld,%d,%.12e,%.12e,%.6f,%.6f,%.15f\n",
                    N, size, t_serial, max_parallel_time, speedup, efficiency, pi_parallel);
            fclose(f);
        }
    }

    MPI_Finalize();
    return 0;
}
