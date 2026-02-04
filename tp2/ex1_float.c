#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000

float test_u1(float *a) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += a[i];
    }
    return sum;
}

float test_u4(float *a) {
    float sum = 0.0f;
    int i;
    for (i = 0; i < N - 3; i += 4) {
        sum += a[i] + a[i + 1] + a[i + 2] + a[i + 3];
    }
    for (; i < N; i++) {
        sum += a[i];
    }
    return sum;
}

float test_u8(float *a) {
    float sum = 0.0f;
    int i;
    for (i = 0; i < N - 7; i += 8) {
        sum += a[i] + a[i + 1] + a[i + 2] + a[i + 3] +
               a[i + 4] + a[i + 5] + a[i + 6] + a[i + 7];
    }
    for (; i < N; i++) {
        sum += a[i];
    }
    return sum;
}

float test_u16(float *a) {
    float sum = 0.0f;
    int i;
    for (i = 0; i < N - 15; i += 16) {
        sum += a[i] + a[i + 1] + a[i + 2] + a[i + 3] +
               a[i + 4] + a[i + 5] + a[i + 6] + a[i + 7] +
               a[i + 8] + a[i + 9] + a[i + 10] + a[i + 11] +
               a[i + 12] + a[i + 13] + a[i + 14] + a[i + 15];
    }
    for (; i < N; i++) {
        sum += a[i];
    }
    return sum;
}

int main() {
    float *a = malloc(N * sizeof(float));
    clock_t start, end;
    float sum;
    double elapsed;

    for (int i = 0; i < N; i++)
        a[i] = 1.0f;

    printf("Float Type Benchmarks (N=%d)\n", N);
    printf("%-15s %-15s %-15s\n", "Unroll Factor", "Time (ms)", "Sum");
    printf("-----------------------------------------------\n");

    start = clock();
    sum = test_u1(a);
    end = clock();
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("%-15d %-15.6f %-15.2f\n", 1, elapsed, sum);

    start = clock();
    sum = test_u4(a);
    end = clock();
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("%-15d %-15.6f %-15.2f\n", 4, elapsed, sum);

    start = clock();
    sum = test_u8(a);
    end = clock();
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("%-15d %-15.6f %-15.2f\n", 8, elapsed, sum);

    start = clock();
    sum = test_u16(a);
    end = clock();
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("%-15d %-15.6f %-15.2f\n", 16, elapsed, sum);

    free(a);
    return 0;
}
