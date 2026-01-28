#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void mxm_bloc(double *A, double *B, double *C, int n, int tileSize) {
    clock_t start, end;
    start = clock();

    // Tiled (blocked) matrix multiplication
    for (int i0 = 0; i0 < n; i0 += tileSize) {
        for (int j0 = 0; j0 < n; j0 += tileSize) {
            for (int k0 = 0; k0 < n; k0 += tileSize) {
                for (int i = i0; i < i0 + tileSize && i < n; i++) {
                    for (int k = k0; k < k0 + tileSize && k < n; k++) {
                        double a = A[i * n + k]; // Load element of A
                        for (int j = j0; j < j0 + tileSize && j < n; j++) {
                            C[i * n + j] += a * B[k * n + j]; // Update C
                        }
                    }
                }
            }
        }
    }

    end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Approximate memory traffic:
    // - A: read once
    // - B: read once
    // - C: read + write
    double total_bytes = (1.0 * n * n   // A read
                          + 1.0 * n * n // B read
                          + 2.0 * n * n // C read + write
                         ) * sizeof(double);

    double bandwidth = (total_bytes / time_taken) / 1e9; // GB/s

    printf("Tile size %d: Time = %.6f s, Memory Bandwidth = %.3f GB/s\n",
           tileSize, time_taken, bandwidth);
}

int main() {
    int N = 512; // Matrix size
    int Blocks[4] = {16, 32, 64, 128}; // Tile sizes to test

    // Allocate matrices
    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));

    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize A, B, and C
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (double)(i + j);
            B[i * N + j] = (double)(i - j);
            C[i * N + j] = 0.0;
        }
    }

    // Run tiled multiplication for each block size
    for (int b = 0; b < 4; b++) {
        // Reset C to zero before each run
        for (int i = 0; i < N * N; i++) C[i] = 0.0;

        mxm_bloc(A, B, C, N, Blocks[b]);
    }

    // Print a few elements of C to verify correctness
    printf("C[0][0] = %f\n", C[0]);
    printf("C[N-1][N-1] = %f\n", C[(N-1) * N + (N-1)]);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
