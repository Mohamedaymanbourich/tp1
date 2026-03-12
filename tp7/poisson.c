/*
 * TP7 - Exercise 2: Poisson Equation Solver with MPI 2D Cartesian Topology
 *
 * Solves: Laplacian(u) = f(x,y) = 2*(x*x - x + y*y - y)  on [0,1]x[0,1]
 * BC:     u = 0  on the boundary
 * Exact:  u(x,y) = x*y*(x-1)*(y-1)
 *
 * Uses Jacobi iterative method with domain decomposition via MPI_Cart_create.
 * Ghost-layer exchange with MPI_Sendrecv and MPI_Type_vector.
 *
 * Compile: mpicc -O2 -Wall -std=c99 -o poisson poisson.c compute.c -lm
 * Run:     mpirun -np 4 ./poisson [ntx] [nty]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

/* ---------------------------------------------------------------
 * Global variables shared with compute.c
 * sx, sy : start indices of the local subdomain (global, 1-based)
 * ex, ey : end indices of the local subdomain (global, 1-based)
 * ntx, nty : number of interior grid points in each direction
 * --------------------------------------------------------------- */
int sx, sy, ex, ey, ntx, nty;

/* Function declarations from compute.c */
extern void initialization(double **pu, double **pu_new, double **pu_exact);
extern void compute(double *u, double *u_new);
extern void output_results(double *u, double *u_exact);

/* Redefine IDX to match compute.c for local use */
#define IDX(i, j) (((i) - (sx - 1)) * (ey - sy + 3) + (j) - (sy - 1))

int main(int argc, char **argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Default domain size */
    ntx = 12;
    nty = 10;
    int max_iter = 10000;
    double tolerance = 1e-14;

    /* Parse optional command-line arguments */
    if (argc > 1) ntx = atoi(argv[1]);
    if (argc > 2) nty = atoi(argv[2]);

    if (rank == 0) {
        printf("Poisson execution with %d MPI processes\n", size);
        printf("Domain size: ntx=%d nty=%d\n", ntx, nty);
    }

    /* ---------------------------------------------------------------
     * 1. Create 2D Cartesian topology (non-periodic for Dirichlet BC)
     * --------------------------------------------------------------- */
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int Px = dims[0], Py = dims[1];

    if (rank == 0) {
        printf("Topology dimensions: %d along x, %d along y\n", Px, Py);
        printf("-----------------------------------------\n");
    }

    int periods[2] = {0, 0};  /* non-periodic: u=0 on boundary */
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);
    int coords[2];
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    /* ---------------------------------------------------------------
     * 2. Domain decomposition: compute local indices (1-based, inclusive)
     * --------------------------------------------------------------- */
    int local_nx = ntx / Px;
    int extra_x  = ntx % Px;
    if (coords[0] < extra_x) {
        local_nx++;
        sx = coords[0] * local_nx + 1;
    } else {
        sx = extra_x * (local_nx + 1) + (coords[0] - extra_x) * local_nx + 1;
    }
    ex = sx + local_nx - 1;

    int local_ny = nty / Py;
    int extra_y  = nty % Py;
    if (coords[1] < extra_y) {
        local_ny++;
        sy = coords[1] * local_ny + 1;
    } else {
        sy = extra_y * (local_ny + 1) + (coords[1] - extra_y) * local_ny + 1;
    }
    ey = sy + local_ny - 1;

    /* ---------------------------------------------------------------
     * 3. Find neighbors using MPI_Cart_shift
     * --------------------------------------------------------------- */
    int north, south, east, west;
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);  /* dim 0: x-direction */
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);    /* dim 1: y-direction */

    /* Print topology information (ordered by rank) */
    for (int r = 0; r < size; r++) {
        if (cart_rank == r) {
            printf("Rank in the topology: %d  Array indices: x from %d to %d, y from %d to %d\n",
                   cart_rank, sx, ex, sy, ey);
            printf("Process %d has neighbors: N %d E %d S %d W %d\n",
                   cart_rank, north, east, south, west);
        }
        MPI_Barrier(cart_comm);
    }

    /* ---------------------------------------------------------------
     * 4. Initialize arrays (u, u_new, u_exact, f) via compute.c
     * --------------------------------------------------------------- */
    double *u, *u_new, *u_exact;
    initialization(&u, &u_new, &u_exact);

    /* ---------------------------------------------------------------
     * 5. Create MPI derived type for column exchange (non-contiguous)
     *    A column has local_nx elements, spaced (ey-sy+3) apart.
     * --------------------------------------------------------------- */
    int ncols = ey - sy + 3;  /* stride in the local array */
    MPI_Datatype col_type;
    MPI_Type_vector(local_nx, 1, ncols, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    /* ---------------------------------------------------------------
     * 6. Jacobi iteration loop
     * --------------------------------------------------------------- */
    double start_time = MPI_Wtime();
    int iter;
    double global_error;

    for (iter = 1; iter <= max_iter; iter++) {

        /* --- Exchange halo layers --- */

        /* North/South: exchange rows (contiguous in memory) */
        /* Send first interior row (sx) to north, receive south ghost (ex+1) from south */
        MPI_Sendrecv(&u[IDX(sx, sy)], local_ny, MPI_DOUBLE, north, 0,
                     &u[IDX(ex + 1, sy)], local_ny, MPI_DOUBLE, south, 0,
                     cart_comm, MPI_STATUS_IGNORE);
        /* Send last interior row (ex) to south, receive north ghost (sx-1) from north */
        MPI_Sendrecv(&u[IDX(ex, sy)], local_ny, MPI_DOUBLE, south, 1,
                     &u[IDX(sx - 1, sy)], local_ny, MPI_DOUBLE, north, 1,
                     cart_comm, MPI_STATUS_IGNORE);

        /* West/East: exchange columns (non-contiguous, use col_type) */
        /* Send first interior column (sy) to west, receive east ghost (ey+1) from east */
        MPI_Sendrecv(&u[IDX(sx, sy)], 1, col_type, west, 2,
                     &u[IDX(sx, ey + 1)], 1, col_type, east, 2,
                     cart_comm, MPI_STATUS_IGNORE);
        /* Send last interior column (ey) to east, receive west ghost (sy-1) from west */
        MPI_Sendrecv(&u[IDX(sx, ey)], 1, col_type, east, 3,
                     &u[IDX(sx, sy - 1)], 1, col_type, west, 3,
                     cart_comm, MPI_STATUS_IGNORE);

        /* --- Jacobi update (done inside compute.c) --- */
        compute(u, u_new);

        /* --- Compute local convergence error: max |u_new - u| --- */
        double local_error = 0.0;
        for (int i = sx; i <= ex; i++) {
            for (int j = sy; j <= ey; j++) {
                double diff = fabs(u_new[IDX(i, j)] - u[IDX(i, j)]);
                if (diff > local_error) local_error = diff;
            }
        }

        /* Global max error via MPI_Allreduce */
        MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, cart_comm);

        /* Print every 100 iterations */
        if (iter % 100 == 0 && cart_rank == 0) {
            printf("Iteration %d global_error = %g\n", iter, global_error);
        }

        /* Swap u and u_new for next iteration */
        double *tmp = u;
        u = u_new;
        u_new = tmp;

        /* Check convergence */
        if (global_error < tolerance) break;
    }

    double end_time = MPI_Wtime();

    if (cart_rank == 0) {
        printf("Converged after %d iterations in %f seconds\n", iter, end_time - start_time);
    }

    /* ---------------------------------------------------------------
     * 7. Output results: exact vs computed solution (from rank with sx<=1)
     * --------------------------------------------------------------- */
    if (sx <= 1 && 1 <= ex) {
        output_results(u, u_exact);
    }

    /* ---------------------------------------------------------------
     * 8. Cleanup
     * --------------------------------------------------------------- */
    free(u);
    free(u_new);
    free(u_exact);
    MPI_Type_free(&col_type);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    return 0;
}
