/*
 * TP6 - Exercise 1: Matrix Transposition using MPI Derived Types
 *
 * A 4x5 matrix A is initialized on process 0 with values 1..20.
 * It is sent to process 1, which receives the transposed matrix A^T (5x4)
 * directly in memory using MPI derived datatypes:
 *   - MPI_Type_vector to define a column layout
 *   - MPI_Type_create_hvector to build the full transpose structure
 *
 * Compile: mpicc -O2 -Wall -o ex1_transpose ex1_transpose.c
 * Run:     mpirun -np 2 ./ex1_transpose
 */

#include <stdio.h>
#include <mpi.h>

#define ROWS 4
#define COLS 5

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0)
            fprintf(stderr, "Error: this program requires exactly 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        /* ---- Process 0: initialize and send the matrix ---- */
        double a[ROWS][COLS];
        int val = 1;
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++)
                a[i][j] = val++;

        /* Display matrix a */
        printf("Process 0 - Matrix a:\n");
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++)
                printf(" %2.0f", a[i][j]);
            printf("\n");
        }
        fflush(stdout);

        /*
         * Send the entire matrix as a contiguous block of ROWS*COLS doubles.
         * The transposition is handled on the receiver side via derived types.
         */
        MPI_Send(&a[0][0], ROWS * COLS, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

    } else {
        /* ---- Process 1: receive the transposed matrix ---- */
        double at[COLS][ROWS];  /* transposed: 5 x 4 */

        /*
         * Build a derived datatype so that MPI_Recv places the incoming
         * row-major 4x5 data into at[5][4] in transposed order.
         *
         * Strategy:
         *   1) column_type: picks one column of at (COLS elements, stride = ROWS).
         *      This corresponds to one ROW of the original matrix being
         *      scattered into a column of at.
         *
         *      MPI_Type_vector(count=COLS, blocklength=1, stride=ROWS, ...)
         *        -> selects elements at[0][col], at[1][col], ..., at[COLS-1][col]
         *
         *   2) transpose_type: replicate column_type ROWS times, each shifted
         *      by sizeof(double) — i.e., moving one element to the right in
         *      each row of at.
         *
         *      MPI_Type_create_hvector(count=ROWS, blocklength=1,
         *                              stride=sizeof(double), column_type)
         *
         * The resulting type maps the incoming ROWS*COLS doubles into the
         * transposed layout inside at.
         */

        MPI_Datatype column_type, transpose_type;

        /* Step 1: a "column" in at — COLS elements spaced ROWS apart */
        MPI_Type_vector(COLS,           /* count: number of blocks          */
                        1,              /* blocklength: 1 element per block */
                        ROWS,           /* stride: ROWS elements apart      */
                        MPI_DOUBLE,     /* old type                         */
                        &column_type);

        /* Step 2: replicate for each of the ROWS columns of at */
        MPI_Type_create_hvector(ROWS,               /* count                */
                                1,                   /* blocklength          */
                                sizeof(double),      /* stride in bytes      */
                                column_type,         /* old type             */
                                &transpose_type);

        MPI_Type_commit(&transpose_type);

        /* Receive using the derived type — automatic transposition */
        MPI_Recv(&at[0][0], 1, transpose_type, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        /* Display transposed matrix */
        printf("Process 1 - Matrix transposee at:\n");
        for (int i = 0; i < COLS; i++) {
            for (int j = 0; j < ROWS; j++)
                printf(" %2.0f", at[i][j]);
            printf("\n");
        }
        fflush(stdout);

        MPI_Type_free(&transpose_type);
        MPI_Type_free(&column_type);
    }

    MPI_Finalize();
    return 0;
}
