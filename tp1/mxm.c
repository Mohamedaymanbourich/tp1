#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void mxm(int N, double **A, double **B, double **C) {
    
    time_t start, end;
    start = clock();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
            
        }
    }

    end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    double total_bytes = (2.0 * N * N * N + 2.0 * N * N) * sizeof(double);
    double bandwidth = (total_bytes / time_taken) / 1e9; // GB/s

    printf("Time taken for mxm: %f seconds\n", time_taken);
    printf("Memory Bandwidth for mxm: %f GB/s\n", bandwidth);
}

void mxm_2(int N, double **A, double **B, double **C) {

    time_t start, end;
    start = clock();
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i][k];
            for (int j = 0; j < N; j++) {
                C[i][j] += r * B[k][j];
            }
        }
    }
    end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    double total_bytes = (3.0 * N * N * N + N * N) * sizeof(double);
    double bandwidth = (total_bytes / time_taken) / 1e9; // GB/s

    printf("Time taken for mxm_2: %f seconds\n", time_taken);
    printf("Memory Bandwidth for mxm_2: %f GB/s\n", bandwidth);
}



int main() {
    int N = 512; // Example size
    double **A = malloc(N * sizeof(double *));
    double **B = malloc(N *sizeof(double * ));
    double **C = malloc(N * sizeof(double * ));

    // Initialize matrices A and B
    for (int i = 0; i < N; i++) {
        A[i] = malloc(N * sizeof(double));
        B[i] = malloc(N * sizeof(double));
        C[i] = calloc(N, sizeof(double)); // Allocate and zero-init C
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 1.0;
            B[i][j] = 1.0;
        }
    }

    mxm(N, A, B, C);
    mxm_2(N, A, B, C);

    free(A);
    free(B);
    free(C);
    return 0;
}

