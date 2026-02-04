#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000

// U = 1 (baseline)
double test_u1(double *a) {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += a[i];
    }
    return sum;
}

// U = 2
double test_u2(double *a) {
    double sum = 0.0;
    int i;
    for (i = 0; i < N - 1; i += 2) {
        sum += a[i] + a[i + 1];
    }
    // Handle remainder
    for (; i < N; i++) {
        sum += a[i];
    }
    return sum;
}

// U = 4
double test_u4(double *a) {
    double sum = 0.0;
    int i;
    for (i = 0; i < N - 3; i += 4) {
        sum += a[i] + a[i + 1] + a[i + 2] + a[i + 3];
    }
    // Handle remainder
    for (; i < N; i++) {
        sum += a[i];
    }
    return sum;
}

// U = 8
double test_u8(double *a) {
    double sum = 0.0;
    int i;
    for (i = 0; i < N - 7; i += 8) {
        sum += a[i] + a[i + 1] + a[i + 2] + a[i + 3] +
               a[i + 4] + a[i + 5] + a[i + 6] + a[i + 7];
    }
    // Handle remainder
    for (; i < N; i++) {
        sum += a[i];
    }
    return sum;
}

// U = 16
double test_u16(double *a) {
    double sum = 0.0;
    int i;
    for (i = 0; i < N - 15; i += 16) {
        sum += a[i] + a[i + 1] + a[i + 2] + a[i + 3] +
               a[i + 4] + a[i + 5] + a[i + 6] + a[i + 7] +
               a[i + 8] + a[i + 9] + a[i + 10] + a[i + 11] +
               a[i + 12] + a[i + 13] + a[i + 14] + a[i + 15];
    }
    // Handle remainder
    for (; i < N; i++) {
        sum += a[i];
    }
    return sum;
}

// U = 32
double test_u32(double *a) {
    double sum = 0.0;
    int i;
    for (i = 0; i < N - 31; i += 32) {
        sum += a[i] + a[i + 1] + a[i + 2] + a[i + 3] +
               a[i + 4] + a[i + 5] + a[i + 6] + a[i + 7] +
               a[i + 8] + a[i + 9] + a[i + 10] + a[i + 11] +
               a[i + 12] + a[i + 13] + a[i + 14] + a[i + 15] +
               a[i + 16] + a[i + 17] + a[i + 18] + a[i + 19] +
               a[i + 20] + a[i + 21] + a[i + 22] + a[i + 23] +
               a[i + 24] + a[i + 25] + a[i + 26] + a[i + 27] +
               a[i + 28] + a[i + 29] + a[i + 30] + a[i + 31];
    }
    // Handle remainder
    for (; i < N; i++) {
        sum += a[i];
    }
    return sum;
}

int main() {
    double *a = malloc(N * sizeof(double));
    clock_t start, end;
    double sum, elapsed;

    // Initialize array
    for (int i = 0; i < N; i++)
        a[i] = 1.0;

    printf("Unrolling Factor Benchmarks (N=%d, type=double, -O0)\n", N);
    printf("%-15s %-15s %-15s\n", "Unroll Factor", "Time (ms)", "Sum");
    printf("-----------------------------------------------\n");

    // U = 1
    start = clock();
    sum = test_u1(a);
    end = clock();
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("%-15d %-15.6f %-15.2f\n", 1, elapsed, sum);

    // U = 2
    start = clock();
    sum = test_u2(a);
    end = clock();
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("%-15d %-15.6f %-15.2f\n", 2, elapsed, sum);

    // U = 4
    start = clock();
    sum = test_u4(a);
    end = clock();
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("%-15d %-15.6f %-15.2f\n", 4, elapsed, sum);

    // U = 8
    start = clock();
    sum = test_u8(a);
    end = clock();
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("%-15d %-15.6f %-15.2f\n", 8, elapsed, sum);

    // U = 16
    start = clock();
    sum = test_u16(a);
    end = clock();
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("%-15d %-15.6f %-15.2f\n", 16, elapsed, sum);

    // U = 32
    start = clock();
    sum = test_u32(a);
    end = clock();
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("%-15d %-15.6f %-15.2f\n", 32, elapsed, sum);

    free(a);
    return 0;
}
