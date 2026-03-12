#include <stdlib.h>
#include <stdio.h>

/* Global variables defined in poisson.c */
extern int sx, sy, ex, ey, ntx, nty;

/* Right-hand side */
static double *f;
static double coef[3];

/*
 * u : nbligne = ex-sx+3 nbcolonne = ey-sy+3
 */

#define IDX(i, j) ( ((i)-(sx-1))*(ey-sy+3) + (j)-(sy-1) )

void initialization(double **pu, double **pu_new, double **pu_exact) {
  double hx, hy;
  int iterx, itery;
  double x, y;

  *pu = calloc((ex-sx+3) * (ey-sy+3), sizeof(double));
  *pu_new = calloc((ex-sx+3) * (ey-sy+3), sizeof(double));
  *pu_exact = calloc((ex-sx+3) * (ey-sy+3), sizeof(double));
  f = calloc((ex-sx+3) * (ey-sy+3), sizeof(double));

  /* Grid spacing */
  hx = 1 / (ntx + 1.0);
  hy = 1 / (nty + 1.0);

  /* Compute coefficients */
  coef[0] = (0.5 * hx * hx * hy * hy) / (hx * hx + hy * hy);
  coef[1] = 1.0 / (hx * hx);
  coef[2] = 1.0 / (hy * hy);

  /* Initialize right-hand side and compute exact solution */
  for (iterx = sx; iterx < ex + 1; iterx++) {
    for (itery = sy; itery < ey + 1; itery++) {
      x = iterx * hx;
      y = itery * hy;
      f[IDX(iterx, itery)] = 2 * (x * x - x + y * y - y);
      (*pu_exact)[IDX(iterx, itery)] = x * y * (x - 1) * (y - 1);
    }
  }
}

/*
 * Compute the solution u_new at iteration n+1
 */
void compute(double *u, double *u_new) {
  int iterx, itery;

  for (iterx = sx; iterx < ex + 1; iterx++) {
    for (itery = sy; itery < ey + 1; itery++) {
      u_new[IDX(iterx, itery)] =
        coef[0] * (
          coef[1] * (u[IDX(iterx + 1, itery)] + u[IDX(iterx - 1, itery)]) +
          coef[2] * (u[IDX(iterx, itery + 1)] + u[IDX(iterx, itery - 1)]) -
          f[IDX(iterx, itery)]
        );
    }
  }
}

/*
 * Print results
 */
void output_results(double *u, double *u_exact) {
  int itery;

  printf("Exact solution u_exact - Computed solution u\n");
  for (itery = sy; itery < ey + 1; itery++)
    printf("%12.5e - %12.5e\n", u_exact[IDX(1, itery)], u[IDX(1, itery)]);
}
