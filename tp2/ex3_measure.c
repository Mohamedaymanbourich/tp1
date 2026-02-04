#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Test with different values of N
#ifndef N
#define N 100000000
#endif

void add_noise(double *a, int n) {
    a[0] = 1.0;
    for (int i = 1; i < n; i++) {
        a[i] = a[i-1] * 1.0000001;
    }
}

void init_b(double *b, int n) {
    for (int i = 0; i < n; i++) {
        b[i] = i * 0.5;
    }
}

void compute_addition(double *a, double *b, double *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

double reduction(double *c, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += c[i];
    }
    return sum;
}

int main() {
    int n = N;
    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *c = malloc(n * sizeof(double));

    clock_t start, end;
    double time_noise, time_init, time_add, time_reduce;

    // Measure add_noise (sequential)
    start = clock();
    add_noise(a, n);
    end = clock();
    time_noise = (double)(end - start) / CLOCKS_PER_SEC;

    // Measure init_b (parallelizable)
    start = clock();
    init_b(b, n);
    end = clock();
    time_init = (double)(end - start) / CLOCKS_PER_SEC;

    // Measure compute_addition (parallelizable)
    start = clock();
    compute_addition(a, b, c, n);
    end = clock();
    time_add = (double)(end - start) / CLOCKS_PER_SEC;

    // Measure reduction (parallelizable)
    start = clock();
    double sum = reduction(c, n);
    end = clock();
    time_reduce = (double)(end - start) / CLOCKS_PER_SEC;

    double total_time = time_noise + time_init + time_add + time_reduce;
    double parallel_time = time_init + time_add + time_reduce;
    double fs = time_noise / total_time;

    printf("N = %d\n", n);
    printf("Sum = %f\n", sum);
    printf("\nExecution times:\n");
    printf("  add_noise (sequential):     %.6f s\n", time_noise);
    printf("  init_b (parallelizable):    %.6f s\n", time_init);
    printf("  compute_addition (par):     %.6f s\n", time_add);
    printf("  reduction (parallelizable): %.6f s\n", time_reduce);
    printf("  TOTAL:                      %.6f s\n", total_time);
    printf("\nSequential fraction fs = %.6f (%.2f%%)\n", fs, fs * 100);

    free(a);
    free(b);
    free(c);
    return 0;
}
