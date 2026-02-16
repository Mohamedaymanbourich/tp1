#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
	int m = 800, n = 800;
	int num_threads = 1;
	char *schedule_type = "STATIC";
	int chunk_size = 100;
	int num_runs = 5;
	
	// Parse command line arguments
	if (argc >= 2) num_threads = atoi(argv[1]);
	if (argc >= 3) schedule_type = argv[2];
	if (argc >= 4) chunk_size = atoi(argv[3]);
	if (argc >= 5) num_runs = atoi(argv[4]);
	
	omp_set_num_threads(num_threads);
	
	double *a = (double *)malloc(m * n * sizeof(double));
	double *b = (double *)malloc(n * m * sizeof(double));
	double *c = (double *)malloc(m * m * sizeof(double));

	// Initialize matrices with collapse
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			a[i * n + j] = (i + 1) + (j + 1);
		}
	}

	#pragma omp parallel for collapse(2)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			b[i * m + j] = (i + 1) - (j + 1);
		}
	}

	#pragma omp parallel for collapse(2)
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			c[i * m + j] = 0;
		}
	}

	// Warmup run
	#pragma omp parallel for collapse(2) schedule(static, chunk_size)
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			for (int k = 0; k < n; k++) {
				c[i * m + j] += a[i * n + k] * b[k * m + j];
			}
		}
	}

	// Reset and benchmark
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			c[i * m + j] = 0;
		}
	}

	double total_time = 0.0;
	for (int run = 0; run < num_runs; run++) {
		// Reset matrix c
		#pragma omp parallel for collapse(2)
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				c[i * m + j] = 0;
			}
		}
		
		double start = omp_get_wtime();
		
		// Matrix multiplication with chosen schedule
		if (strcmp(schedule_type, "STATIC") == 0) {
			#pragma omp parallel for collapse(2) schedule(static, chunk_size)
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < m; j++) {
					for (int k = 0; k < n; k++) {
						c[i * m + j] += a[i * n + k] * b[k * m + j];
					}
				}
			}
		} else if (strcmp(schedule_type, "DYNAMIC") == 0) {
			#pragma omp parallel for collapse(2) schedule(dynamic, chunk_size)
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < m; j++) {
					for (int k = 0; k < n; k++) {
						c[i * m + j] += a[i * n + k] * b[k * m + j];
					}
				}
			}
		} else if (strcmp(schedule_type, "GUIDED") == 0) {
			#pragma omp parallel for collapse(2) schedule(guided, chunk_size)
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < m; j++) {
					for (int k = 0; k < n; k++) {
						c[i * m + j] += a[i * n + k] * b[k * m + j];
					}
				}
			}
		}
		
		double end = omp_get_wtime();
		total_time += (end - start);
	}

	double avg_time = total_time / num_runs;
	printf("%d %s %d %.6f\n", num_threads, schedule_type, chunk_size, avg_time);

	free(a);
	free(b);
	free(c);
	return 0;
}