/*
 * TP7 - Exercise 1: Conway's Game of Life with MPI 2D Cartesian Topology
 *
 * Parallel implementation using MPI_Cart_create, MPI_Cart_shift,
 * and MPI_Sendrecv for ghost-layer exchange with periodic boundaries.
 *
 * Compile: mpicc -O2 -Wall -std=c99 -o game_of_life game_of_life.c
 * Run:     mpirun -np 4 ./game_of_life [global_nx] [global_ny] [generations]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/* Default parameters */
#define DEFAULT_NX   20
#define DEFAULT_NY   20
#define DEFAULT_GENS 10

/* Macro for 2D indexing into a 1D array (row-major, with halo) */
#define CELL(grid, i, j) ((grid)[(i) * (local_ny + 2) + (j)])

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Parse optional command-line arguments */
    int global_nx = (argc > 1) ? atoi(argv[1]) : DEFAULT_NX;
    int global_ny = (argc > 2) ? atoi(argv[2]) : DEFAULT_NY;
    int num_gens  = (argc > 3) ? atoi(argv[3]) : DEFAULT_GENS;

    /* ----------------------------------------------------------------
     * 1. Create 2D Cartesian topology with periodic boundary conditions
     * ---------------------------------------------------------------- */
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int Px = dims[0], Py = dims[1];

    int periods[2] = {1, 1}; /* periodic wrap-around */
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);
    int coords[2];
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    /* ----------------------------------------------------------------
     * 2. Compute local subgrid dimensions (handle uneven division)
     * ---------------------------------------------------------------- */
    int local_nx, local_ny;
    int start_x, start_y;

    /* X dimension */
    local_nx = global_nx / Px;
    int extra_x = global_nx % Px;
    if (coords[0] < extra_x) {
        local_nx++;
        start_x = coords[0] * local_nx;
    } else {
        start_x = extra_x * (local_nx + 1) + (coords[0] - extra_x) * local_nx;
    }

    /* Y dimension */
    local_ny = global_ny / Py;
    int extra_y = global_ny % Py;
    if (coords[1] < extra_y) {
        local_ny++;
        start_y = coords[1] * local_ny;
    } else {
        start_y = extra_y * (local_ny + 1) + (coords[1] - extra_y) * local_ny;
    }

    /* ----------------------------------------------------------------
     * 3. Allocate local grids with halo border (local_nx+2) x (local_ny+2)
     * ---------------------------------------------------------------- */
    int grid_size = (local_nx + 2) * (local_ny + 2);
    int *grid     = calloc(grid_size, sizeof(int));
    int *new_grid = calloc(grid_size, sizeof(int));

    /* ----------------------------------------------------------------
     * 4. Identify four neighbors using MPI_Cart_shift
     * ---------------------------------------------------------------- */
    int north, south, east, west;
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);

    /* ----------------------------------------------------------------
     * 5. Initialize the grid with some known patterns
     *    - Glider at (1,1)
     *    - Blinker at (10,10)
     *    - Glider at (5,15) going in another direction
     * ---------------------------------------------------------------- */

    /* Glider (pattern relative to top-left corner): */
    int glider[][2] = {{1,0}, {2,1}, {0,2}, {1,2}, {2,2}};
    int glider_offset[2] = {1, 1};
    for (int g = 0; g < 5; g++) {
        int gi = glider[g][0] + glider_offset[0];
        int gj = glider[g][1] + glider_offset[1];
        if (gi >= start_x && gi < start_x + local_nx &&
            gj >= start_y && gj < start_y + local_ny) {
            CELL(grid, gi - start_x + 1, gj - start_y + 1) = 1;
        }
    }

    /* Blinker (horizontal, period 2) */
    int blinker[][2] = {{10,9}, {10,10}, {10,11}};
    for (int g = 0; g < 3; g++) {
        int gi = blinker[g][0], gj = blinker[g][1];
        if (gi >= start_x && gi < start_x + local_nx &&
            gj >= start_y && gj < start_y + local_ny) {
            CELL(grid, gi - start_x + 1, gj - start_y + 1) = 1;
        }
    }

    if (cart_rank == 0) {
        printf("Game of Life: %dx%d grid, %d processes (%dx%d), %d generations\n",
               global_nx, global_ny, size, Px, Py, num_gens);
        printf("Periodic boundary conditions enabled\n\n");
    }

    /* ----------------------------------------------------------------
     * 6. MPI derived type for column exchange (non-contiguous)
     *    Includes ghost rows (0..local_nx+1) so corners are handled.
     * ---------------------------------------------------------------- */
    MPI_Datatype col_type;
    MPI_Type_vector(local_nx + 2, 1, local_ny + 2, MPI_INT, &col_type);
    MPI_Type_commit(&col_type);

    /* ----------------------------------------------------------------
     * 7. Main simulation loop
     * ---------------------------------------------------------------- */
    for (int gen = 1; gen <= num_gens; gen++) {

        /* --- Ghost layer exchange --- */

        /* Step 1: Exchange rows (N/S) — interior columns only */
        /* Send first interior row to north, receive south ghost from south */
        MPI_Sendrecv(&CELL(grid, 1, 1), local_ny, MPI_INT, north, 0,
                     &CELL(grid, local_nx + 1, 1), local_ny, MPI_INT, south, 0,
                     cart_comm, MPI_STATUS_IGNORE);
        /* Send last interior row to south, receive north ghost from north */
        MPI_Sendrecv(&CELL(grid, local_nx, 1), local_ny, MPI_INT, south, 1,
                     &CELL(grid, 0, 1), local_ny, MPI_INT, north, 1,
                     cart_comm, MPI_STATUS_IGNORE);

        /* Step 2: Exchange columns (E/W) — full height including ghost rows
         * This automatically fills corner ghost cells. */
        /* Send first interior column to west, receive east ghost from east */
        MPI_Sendrecv(&CELL(grid, 0, 1), 1, col_type, west, 2,
                     &CELL(grid, 0, local_ny + 1), 1, col_type, east, 2,
                     cart_comm, MPI_STATUS_IGNORE);
        /* Send last interior column to east, receive west ghost from west */
        MPI_Sendrecv(&CELL(grid, 0, local_ny), 1, col_type, east, 3,
                     &CELL(grid, 0, 0), 1, col_type, west, 3,
                     cart_comm, MPI_STATUS_IGNORE);

        /* --- Apply Game of Life rules --- */
        for (int i = 1; i <= local_nx; i++) {
            for (int j = 1; j <= local_ny; j++) {
                int neighbors =
                    CELL(grid, i-1, j-1) + CELL(grid, i-1, j) + CELL(grid, i-1, j+1) +
                    CELL(grid, i,   j-1) +                        CELL(grid, i,   j+1) +
                    CELL(grid, i+1, j-1) + CELL(grid, i+1, j) + CELL(grid, i+1, j+1);

                if (CELL(grid, i, j) == 1)
                    CELL(new_grid, i, j) = (neighbors == 2 || neighbors == 3) ? 1 : 0;
                else
                    CELL(new_grid, i, j) = (neighbors == 3) ? 1 : 0;
            }
        }

        /* Swap grids */
        int *tmp = grid;
        grid = new_grid;
        new_grid = tmp;

        /* Optional: print at certain generations */
        if (gen == num_gens || gen == 1) {
            for (int r = 0; r < size; r++) {
                if (cart_rank == r) {
                    printf("Rank %d - Generation %d:\n", cart_rank, gen);
                    for (int i = 1; i <= local_nx; i++) {
                        for (int j = 1; j <= local_ny; j++) {
                            printf("%d ", CELL(grid, i, j));
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
                MPI_Barrier(cart_comm);
            }
        }
    }

    /* ----------------------------------------------------------------
     * 8. Gather the final global grid at rank 0
     * ---------------------------------------------------------------- */
    if (cart_rank == 0) {
        int *global_grid = calloc(global_nx * global_ny, sizeof(int));

        /* Copy rank 0's own data */
        for (int i = 0; i < local_nx; i++)
            for (int j = 0; j < local_ny; j++)
                global_grid[(start_x + i) * global_ny + (start_y + j)] =
                    CELL(grid, i + 1, j + 1);

        /* Receive from other processes */
        for (int r = 1; r < size; r++) {
            int r_coords[2];
            MPI_Cart_coords(cart_comm, r, 2, r_coords);

            int r_local_nx = global_nx / Px;
            int r_extra_x = global_nx % Px;
            int r_start_x;
            if (r_coords[0] < r_extra_x) {
                r_local_nx++;
                r_start_x = r_coords[0] * r_local_nx;
            } else {
                r_start_x = r_extra_x * (r_local_nx + 1) +
                             (r_coords[0] - r_extra_x) * r_local_nx;
            }

            int r_local_ny = global_ny / Py;
            int r_extra_y = global_ny % Py;
            int r_start_y;
            if (r_coords[1] < r_extra_y) {
                r_local_ny++;
                r_start_y = r_coords[1] * r_local_ny;
            } else {
                r_start_y = r_extra_y * (r_local_ny + 1) +
                             (r_coords[1] - r_extra_y) * r_local_ny;
            }

            int r_size = r_local_nx * r_local_ny;
            int *buf = malloc(r_size * sizeof(int));
            MPI_Recv(buf, r_size, MPI_INT, r, 99, cart_comm, MPI_STATUS_IGNORE);

            for (int i = 0; i < r_local_nx; i++)
                for (int j = 0; j < r_local_ny; j++)
                    global_grid[(r_start_x + i) * global_ny + (r_start_y + j)] =
                        buf[i * r_local_ny + j];
            free(buf);
        }

        printf("=== Final Global Grid (Generation %d) ===\n", num_gens);
        for (int i = 0; i < global_nx; i++) {
            for (int j = 0; j < global_ny; j++)
                printf("%d ", global_grid[i * global_ny + j]);
            printf("\n");
        }

        free(global_grid);

    } else {
        /* Pack local interior data and send to rank 0 */
        int *buf = malloc(local_nx * local_ny * sizeof(int));
        for (int i = 0; i < local_nx; i++)
            for (int j = 0; j < local_ny; j++)
                buf[i * local_ny + j] = CELL(grid, i + 1, j + 1);
        MPI_Send(buf, local_nx * local_ny, MPI_INT, 0, 99, cart_comm);
        free(buf);
    }

    /* ----------------------------------------------------------------
     * 9. Cleanup
     * ---------------------------------------------------------------- */
    free(grid);
    free(new_grid);
    MPI_Type_free(&col_type);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
