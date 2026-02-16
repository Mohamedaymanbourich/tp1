/**
 * TP4 - Exercise 3: Load Balancing with Parallel Sections
 *
 * Implement task scheduling using parallel sections.
 * Three different workloads:
 *   Task A: light computation
 *   Task B: moderate computation
 *   Task C: heavy computation
 * Measure execution time and optimize workload distribution.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Light computation: N iterations */
double task_light(int N) {
    double x = 0.0;
    for (int i = 0; i < N; i++) {
        x += sin(i * 0.001);
    }
    return x;
}

/* Moderate computation: 5*N iterations */
double task_moderate(int N) {
    double x = 0.0;
    for (int i = 0; i < 5*N; i++) {
        x += sqrt(i * 0.5) * cos(i * 0.001);
    }
    return x;
}

/* Heavy computation: 20*N iterations */
double task_heavy(int N) {
    double x = 0.0;
    for (int i = 0; i < 20*N; i++) {
        x += sqrt(i * 0.5) * cos(i * 0.001) * sin(i * 0.0001);
    }
    return x;
}

int main() {
    int N_WORK = 1000000;
    double r1, r2, r3;

    printf("TP4 Exercise 3: Load Balancing with Parallel Sections\n");
    printf("=====================================================\n");
    printf("Task A (light):    %d iterations\n", N_WORK);
    printf("Task B (moderate): %d iterations\n", 5*N_WORK);
    printf("Task C (heavy):    %d iterations\n\n", 20*N_WORK);

    /* ============ Sequential Version ============ */
    double t_seq_start = omp_get_wtime();
    r1 = task_light(N_WORK);
    r2 = task_moderate(N_WORK);
    r3 = task_heavy(N_WORK);
    double t_seq_end = omp_get_wtime();
    double t_seq = t_seq_end - t_seq_start;

    printf("=== Sequential Execution ===\n");
    printf("Time = %f seconds\n\n", t_seq);

    /* ============ Naive Parallel Sections (one task per section) ============ */
    double t_naive_start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            r1 = task_light(N_WORK);
            printf("[Naive] Task A (light) on thread %d\n", omp_get_thread_num());
        }
        #pragma omp section
        {
            r2 = task_moderate(N_WORK);
            printf("[Naive] Task B (moderate) on thread %d\n", omp_get_thread_num());
        }
        #pragma omp section
        {
            r3 = task_heavy(N_WORK);
            printf("[Naive] Task C (heavy) on thread %d\n", omp_get_thread_num());
        }
    }
    double t_naive_end = omp_get_wtime();
    double t_naive = t_naive_end - t_naive_start;

    printf("Time = %f seconds\n", t_naive);
    printf("Speedup = %.2fx\n\n", t_seq / t_naive);

    /* ============ Optimized: Split heavy task into sub-tasks ============ */
    /*
     * The naive approach is limited because the heavy task (20x) dominates.
     * With 3 sections taking 1x, 5x, 20x the total is bounded by max(1,5,20) = 20x.
     * Ideal: total_work/num_threads = 26x/3 ≈ 8.7x per thread.
     *
     * Optimization: Split the heavy task into multiple sub-sections so that
     * work is more evenly distributed. We split heavy into 4 chunks,
     * giving sections of: 1x, 5x, 5x, 5x, 5x, 5x (total 26x).
     */
    double heavy_parts[4] = {0.0, 0.0, 0.0, 0.0};
    int chunk = 20 * N_WORK / 4;

    double t_opt_start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            /* Light task + part of heavy */
            r1 = task_light(N_WORK);
            double x = 0.0;
            int start = 0, end = chunk;
            for (int i = start; i < end; i++)
                x += sqrt(i * 0.5) * cos(i * 0.001) * sin(i * 0.0001);
            heavy_parts[0] = x;
            printf("[Optimized] Task A + Heavy[0] on thread %d\n", omp_get_thread_num());
        }
        #pragma omp section
        {
            /* Moderate task */
            r2 = task_moderate(N_WORK);
            printf("[Optimized] Task B on thread %d\n", omp_get_thread_num());
        }
        #pragma omp section
        {
            /* Heavy part 1+2 */
            double x1 = 0.0, x2 = 0.0;
            int start1 = chunk, end1 = 2*chunk;
            int start2 = 2*chunk, end2 = 3*chunk;
            for (int i = start1; i < end1; i++)
                x1 += sqrt(i * 0.5) * cos(i * 0.001) * sin(i * 0.0001);
            for (int i = start2; i < end2; i++)
                x2 += sqrt(i * 0.5) * cos(i * 0.001) * sin(i * 0.0001);
            heavy_parts[1] = x1;
            heavy_parts[2] = x2;
            printf("[Optimized] Heavy[1+2] on thread %d\n", omp_get_thread_num());
        }
        #pragma omp section
        {
            /* Heavy part 3 */
            double x = 0.0;
            int start = 3*chunk, end = 20*N_WORK;
            for (int i = start; i < end; i++)
                x += sqrt(i * 0.5) * cos(i * 0.001) * sin(i * 0.0001);
            heavy_parts[3] = x;
            printf("[Optimized] Heavy[3] on thread %d\n", omp_get_thread_num());
        }
    }
    r3 = heavy_parts[0] + heavy_parts[1] + heavy_parts[2] + heavy_parts[3];
    double t_opt_end = omp_get_wtime();
    double t_opt = t_opt_end - t_opt_start;

    printf("Time = %f seconds\n", t_opt);
    printf("Speedup vs Sequential = %.2fx\n", t_seq / t_opt);
    printf("Speedup vs Naive      = %.2fx\n\n", t_naive / t_opt);

    /* ============ Summary ============ */
    printf("=== Summary ===\n");
    printf("%-25s %10s %10s\n", "Version", "Time (s)", "Speedup");
    printf("%-25s %10.4f %10s\n", "Sequential", t_seq, "1.00x");
    printf("%-25s %10.4f %9.2fx\n", "Naive Sections", t_naive, t_seq / t_naive);
    printf("%-25s %10.4f %9.2fx\n", "Optimized Sections", t_opt, t_seq / t_opt);
    printf("\nConclusion: Naive sections are limited by the heaviest task.\n");
    printf("Splitting heavy work across sections improves load balance.\n");

    return 0;
}
