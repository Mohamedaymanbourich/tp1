#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>

#ifndef VAL_N
#define VAL_N 120
#endif
#ifndef VAL_D
#define VAL_D 80
#endif

void random_number(double* array, int size) {
	for (int i = 0; i < size; i++) {
		array[i] = (double)rand() / (double)(RAND_MAX - 1);
	}
}

int main() {
	int n = VAL_N, diag = VAL_D;
	int i, j, iteration = 0;
	double norme;

	double *a = (double*)malloc(n * n * sizeof(double));
	double *x = (double*)malloc(n * sizeof(double));
	double *x_courant = (double*)malloc(n * sizeof(double));
	double *b = (double*)malloc(n * sizeof(double));

	if (!a || !x || !x_courant || !b) {
		fprintf(stderr, "Memory allocation failed!\n");
		exit(EXIT_FAILURE);
	}

	struct timeval t_elapsed_0, t_elapsed_1;
	double t_elapsed;

	srand(421);
	random_number(a, n * n);
	random_number(b, n);

	for (i = 0; i < n; i++) {
		a[i * n + i] += diag;
	}

	for (i = 0; i < n; i++) {
		x[i] = 1.0;
	}

	gettimeofday(&t_elapsed_0, NULL);

	while (1) {
		iteration++;

		// Sequential computation of x_courant
		for (i = 0; i < n; i++) {
			x_courant[i] = 0;
			for (j = 0; j < i; j++) {
				x_courant[i] += a[j * n + i] * x[j];
			}
			for (j = i + 1; j < n; j++) {
				x_courant[i] += a[j * n + i] * x[j];
			}
			x_courant[i] = (b[i] - x_courant[i]) / a[i * n + i];
		}

		// Sequential computation of the maximum difference
		double absmax = 0;
		for (i = 0; i < n; i++) {
			double curr = fabs(x[i] - x_courant[i]);
			if (curr > absmax)
				absmax = curr;
		}
		norme = absmax / n;

		if ((norme <= DBL_EPSILON) || (iteration >= n)) break;

		memcpy(x, x_courant, n * sizeof(double));
	}

	gettimeofday(&t_elapsed_1, NULL);
	t_elapsed = (t_elapsed_1.tv_sec - t_elapsed_0.tv_sec) +
		(t_elapsed_1.tv_usec - t_elapsed_0.tv_usec) / 1e6;

	fprintf(stdout, "\n\nSystem size            : %5d\n"
		"Iterations             : %4d\n"
		"Norme                  : %10.3E\n"
		"Elapsed time           : %10.3E sec.\n",
		n, iteration, norme, t_elapsed);

	free(a); free(x); free(x_courant); free(b);
	return EXIT_SUCCESS;
}
