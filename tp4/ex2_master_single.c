/**
 * TP4 - Exercise 2: Exclusive Execution - Master vs Single
 *
 * - A master thread initializes a matrix.
 * - A single thread prints the matrix.
 * - All threads compute the sum of all elements in parallel.
 * - Compare execution time with and without OpenMP.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000

void init_matrix(int n, double *A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i*n + j] = (double)(i + j);
        }
    }
}

void print_matrix(int n, double *A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%6.1f ", A[i*n + j]);
        }
        printf("\n");
    }
}

double sum_matrix(int n, double *A) {
    double sum = 0.0;
    for (int i = 0; i < n*n; i++) {
        sum += A[i];
    }
    return sum;
}

int main() {
    double *A;
    double sum;
    double start, end;

    A = (double*) malloc(N * N * sizeof(double));
    if (A == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    /* ============ Sequential Version ============ */
    printf("=== Sequential Version ===\n");
    start = omp_get_wtime();

    init_matrix(N, A);
    /* print_matrix(N, A); */ /* Commented: N is large */
    sum = sum_matrix(N, A);

    end = omp_get_wtime();
    printf("Sum = %lf\n", sum);
    printf("Execution time = %lf seconds\n\n", end - start);
    double seq_time = end - start;

    /* ============ Parallel Version (Master + Single + Parallel Sum) ============ */
    printf("=== Parallel Version (master/single) ===\n");
    printf("Using %d threads\n", omp_get_max_threads());
    
    sum = 0.0;
    start = omp_get_wtime();

    #pragma omp parallel
    {
        /* Master thread initializes the matrix */
        #pragma omp master
        {
            init_matrix(N, A);
            printf("Matrix initialized by master thread (thread %d)\n",
                   omp_get_thread_num());
        }
        /* Need a barrier so all threads wait for initialization to complete */
        #pragma omp barrier

        /* Single thread prints the matrix (only for small N) */
        #pragma omp single
        {
            if (N <= 10) {
                printf("Matrix printed by thread %d:\n", omp_get_thread_num());
                print_matrix(N, A);
            } else {
                printf("Matrix printing skipped (N=%d too large), done by thread %d\n",
                       N, omp_get_thread_num());
            }
        }
        /* Implicit barrier after single ensures print is done before sum */

        /* All threads compute the sum in parallel */
        #pragma omp for reduction(+:sum)
        for (int i = 0; i < N*N; i++) {
            sum += A[i];
        }
    }

    end = omp_get_wtime();
    printf("Sum = %lf\n", sum);
    printf("Execution time = %lf seconds\n\n", end - start);
    double par_time = end - start;

    printf("Speedup = %.2fx\n", seq_time / par_time);

    /* ============ Explanation of Master vs Single ============ */
    printf("\n=== Master vs Single ===\n");
    printf("master: Only thread 0 executes the block. NO implicit barrier.\n");
    printf("single: Any ONE thread executes the block. HAS implicit barrier.\n");
    printf("Use master when only thread 0 should act and you manage sync yourself.\n");
    printf("Use single when any thread can do it and others should wait.\n");

    free(A);
    return 0;
}
