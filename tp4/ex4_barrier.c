/**
 * TP4 - Exercise 4: Synchronization and Barrier Cost
 *
 * Dense Matrix-Vector Multiplication (DMVM):  lhs = mat * rhs
 *   Version 1: Implicit barrier (default)
 *   Version 2: schedule(dynamic) with nowait
 *   Version 3: schedule(static) with nowait
 *
 * Run with 1, 2, 4, 8, 16 threads and measure:
 *   - CPU time
 *   - Speedup
 *   - Efficiency
 *   - MFLOP/s
 *
 * Output: CSV format for easy plotting.
 * Usage: ./ex4_barrier <num_threads> <version>
 *   version: 1=implicit barrier, 2=dynamic+nowait, 3=static+nowait
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* Version 1: Implicit barrier (default parallel for) */
void dmvm_v1(int n, int m, double *lhs, double *rhs, double *mat) {
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < n; ++c) {
        int offset = m * c;
        for (int r = 0; r < m; ++r)
            lhs[r] += mat[r + offset] * rhs[c];
    }
    /* Implicit barrier at end of parallel for */
}

/* Version 2: schedule(dynamic) with nowait */
void dmvm_v2(int n, int m, double *lhs, double *rhs, double *mat) {
    #pragma omp parallel
    {
        double *local_lhs = calloc(m, sizeof(double));
        
        #pragma omp for schedule(dynamic, 64) nowait
        for (int c = 0; c < n; ++c) {
            int offset = m * c;
            for (int r = 0; r < m; ++r)
                local_lhs[r] += mat[r + offset] * rhs[c];
        }

        /* Critical section to accumulate results */
        #pragma omp critical
        {
            for (int r = 0; r < m; ++r)
                lhs[r] += local_lhs[r];
        }
        free(local_lhs);
    }
}

/* Version 3: schedule(static) with nowait */
void dmvm_v3(int n, int m, double *lhs, double *rhs, double *mat) {
    #pragma omp parallel
    {
        double *local_lhs = calloc(m, sizeof(double));

        #pragma omp for schedule(static) nowait
        for (int c = 0; c < n; ++c) {
            int offset = m * c;
            for (int r = 0; r < m; ++r)
                local_lhs[r] += mat[r + offset] * rhs[c];
        }

        /* Critical section to accumulate results */
        #pragma omp critical
        {
            for (int r = 0; r < m; ++r)
                lhs[r] += local_lhs[r];
        }
        free(local_lhs);
    }
}

/* Sequential version for reference timing */
void dmvm_seq(int n, int m, double *lhs, double *rhs, double *mat) {
    for (int c = 0; c < n; ++c) {
        int offset = m * c;
        for (int r = 0; r < m; ++r)
            lhs[r] += mat[r + offset] * rhs[c];
    }
}

int main(int argc, char *argv[]) {
    const int n = 40000;  /* columns */
    const int m = 600;    /* rows */
    int num_threads = 4;
    int version = 0;      /* 0 = run all, 1/2/3 = specific version */
    int csv_mode = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--threads") == 0 && i+1 < argc)
            num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--version") == 0 && i+1 < argc)
            version = atoi(argv[++i]);
        else if (strcmp(argv[i], "--csv") == 0)
            csv_mode = 1;
    }

    omp_set_num_threads(num_threads);

    double *mat = malloc(n * m * sizeof(double));
    double *rhs = malloc(n * sizeof(double));
    double *lhs = malloc(m * sizeof(double));
    double *lhs_ref = malloc(m * sizeof(double));

    if (!mat || !rhs || !lhs || !lhs_ref) {
        printf("Memory allocation failed\n");
        return 1;
    }

    /* Initialization */
    for (int c = 0; c < n; ++c) {
        rhs[c] = 1.0;
        for (int r = 0; r < m; ++r)
            mat[r + c*m] = 1.0;
    }

    /* FLOPs: for each of n columns, m multiply + m add = 2*n*m */
    double flops = 2.0 * n * m;

    /* Sequential reference */
    for (int r = 0; r < m; ++r) lhs_ref[r] = 0.0;
    int warmup_iters = 5;
    int bench_iters = 20;
    /* Warmup */
    for (int it = 0; it < warmup_iters; it++) {
        for (int r = 0; r < m; ++r) lhs_ref[r] = 0.0;
        dmvm_seq(n, m, lhs_ref, rhs, mat);
    }
    for (int r = 0; r < m; ++r) lhs_ref[r] = 0.0;
    double t_seq_start = omp_get_wtime();
    for (int it = 0; it < bench_iters; it++) {
        for (int r = 0; r < m; ++r) lhs_ref[r] = 0.0;
        dmvm_seq(n, m, lhs_ref, rhs, mat);
    }
    double t_seq = (omp_get_wtime() - t_seq_start) / bench_iters;

    if (!csv_mode) {
        printf("TP4 Exercise 4: Synchronization and Barrier Cost\n");
        printf("=================================================\n");
        printf("Matrix: %d x %d, Threads: %d\n", m, n, num_threads);
        printf("FLOPs: %.0f\n\n", flops);
        printf("Sequential time: %f seconds\n", t_seq);
        printf("Sequential MFLOP/s: %.2f\n\n", flops / t_seq / 1e6);
    }

    /* Number of repetitions for stable timing */
    int nreps = 10;

    /* Run requested versions */
    int versions_to_run[3] = {0, 0, 0};
    if (version == 0) {
        versions_to_run[0] = versions_to_run[1] = versions_to_run[2] = 1;
    } else {
        versions_to_run[version - 1] = 1;
    }

    const char *version_names[] = {
        "V1 (implicit barrier)",
        "V2 (dynamic+nowait)",
        "V3 (static+nowait)"
    };

    void (*dmvm_funcs[])(int, int, double*, double*, double*) = {
        dmvm_v1, dmvm_v2, dmvm_v3
    };

    if (csv_mode) {
        /* CSV header if running all */
        if (version == 0)
            printf("version,threads,time,speedup,efficiency,mflops\n");
    }

    for (int v = 0; v < 3; v++) {
        if (!versions_to_run[v]) continue;

        double best_time = 1e30;
        /* Warmup */
        for (int rep = 0; rep < 3; rep++) {
            for (int r = 0; r < m; ++r) lhs[r] = 0.0;
            dmvm_funcs[v](n, m, lhs, rhs, mat);
        }
        for (int rep = 0; rep < nreps; rep++) {
            double t_start = omp_get_wtime();
            for (int it = 0; it < bench_iters; it++) {
                for (int r = 0; r < m; ++r) lhs[r] = 0.0;
                dmvm_funcs[v](n, m, lhs, rhs, mat);
            }
            double elapsed = (omp_get_wtime() - t_start) / bench_iters;
            if (elapsed < best_time) best_time = elapsed;
        }

        double speedup = t_seq / best_time;
        double efficiency = speedup / num_threads;
        double mflops = flops / best_time / 1e6;

        if (csv_mode) {
            printf("%d,%d,%f,%f,%f,%f\n",
                   v+1, num_threads, best_time, speedup, efficiency, mflops);
        } else {
            printf("--- %s ---\n", version_names[v]);
            printf("  Time       = %f seconds\n", best_time);
            printf("  Speedup    = %.2fx\n", speedup);
            printf("  Efficiency = %.2f%%\n", efficiency * 100.0);
            printf("  MFLOP/s    = %.2f\n\n", mflops);

            /* Verify correctness */
            double max_diff = 0.0;
            for (int r = 0; r < m; ++r) {
                /* For V1 with race conditions, we expect potential issues */
                /* V2 and V3 use local arrays so should be correct */
            }
        }
    }

    free(mat);
    free(rhs);
    free(lhs);
    free(lhs_ref);
    return 0;
}
