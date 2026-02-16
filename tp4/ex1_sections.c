/**
 * TP4 - Exercise 1: Work Distribution with Parallel Sections
 * 
 * Uses #pragma omp sections to divide work:
 *   Section 1: Compute the sum of all elements
 *   Section 2: Compute the maximum value
 *   Section 3: Compute the standard deviation (uses sum from Section 1)
 *
 * Key challenge: Section 3 depends on the result of Section 1 (the sum/mean).
 * We use a barrier between the sum/max computation and the stddev computation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 1000000

int main() {
    double *A = malloc(N * sizeof(double));
    if (A == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    double sum = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    double max_val;

    // Initialization
    srand(0);
    for (int i = 0; i < N; i++)
        A[i] = (double)rand() / RAND_MAX;

    /* ============ Sequential Version ============ */
    double t_seq_start = omp_get_wtime();
    
    double seq_sum = 0.0;
    double seq_max = A[0];
    for (int i = 0; i < N; i++) {
        seq_sum += A[i];
        if (A[i] > seq_max) seq_max = A[i];
    }
    double seq_mean = seq_sum / N;
    double seq_stddev = 0.0;
    for (int i = 0; i < N; i++)
        seq_stddev += (A[i] - seq_mean) * (A[i] - seq_mean);
    seq_stddev = sqrt(seq_stddev / N);
    
    double t_seq_end = omp_get_wtime();

    printf("=== Sequential Results ===\n");
    printf("Sum     = %f\n", seq_sum);
    printf("Max     = %f\n", seq_max);
    printf("Std Dev = %f\n", seq_stddev);
    printf("Time    = %f seconds\n\n", t_seq_end - t_seq_start);

    /* ============ Parallel Version with Sections ============ */
    double t_par_start = omp_get_wtime();

    sum = 0.0;
    max_val = A[0];
    stddev = 0.0;

    /*
     * Strategy: We use two phases.
     * Phase 1: Sections for sum and max (independent).
     * Phase 2: After barrier (implicit at end of sections), compute stddev.
     * 
     * Note: We cannot run all 3 truly in parallel using sections alone because
     * Section 3 (stddev) depends on Section 1 (sum). So we split into two
     * parallel regions or use a two-phase approach within a single parallel region.
     */
    #pragma omp parallel
    {
        /* Phase 1: Compute sum and max in parallel sections */
        #pragma omp sections
        {
            #pragma omp section
            {
                /* Section 1: Compute sum */
                double local_sum = 0.0;
                for (int i = 0; i < N; i++)
                    local_sum += A[i];
                sum = local_sum;
            }

            #pragma omp section
            {
                /* Section 2: Compute maximum */
                double local_max = A[0];
                for (int i = 1; i < N; i++) {
                    if (A[i] > local_max)
                        local_max = A[i];
                }
                max_val = local_max;
            }
        }
        /* Implicit barrier here ensures sum is ready */

        /* Compute mean (single thread, to avoid redundant work) */
        #pragma omp single
        {
            mean = sum / N;
        }
        /* Implicit barrier ensures mean is ready for all threads */

        /* Phase 2: Section for standard deviation (uses mean from Phase 1) */
        #pragma omp sections
        {
            #pragma omp section
            {
                /* Section 3: Compute standard deviation using mean from Section 1 */
                double local_stddev = 0.0;
                for (int i = 0; i < N; i++)
                    local_stddev += (A[i] - mean) * (A[i] - mean);
                stddev = sqrt(local_stddev / N);
            }
        }
    }

    double t_par_end = omp_get_wtime();

    printf("=== Parallel Results (Sections) ===\n");
    printf("Sum     = %f\n", sum);
    printf("Max     = %f\n", max_val);
    printf("Std Dev = %f\n", stddev);
    printf("Time    = %f seconds\n\n", t_par_end - t_par_start);

    double speedup = (t_seq_end - t_seq_start) / (t_par_end - t_par_start);
    printf("Speedup = %.2fx\n", speedup);
    printf("Threads = %d\n", omp_get_max_threads());

    /* Verify correctness */
    printf("\n=== Verification ===\n");
    printf("Sum diff     = %e\n", fabs(sum - seq_sum));
    printf("Max diff     = %e\n", fabs(max_val - seq_max));
    printf("Stddev diff  = %e\n", fabs(stddev - seq_stddev));

    free(A);
    return 0;
}
