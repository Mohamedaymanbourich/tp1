/*
 * TP6 - Exercise 2: Distributed Gradient Descent with MPI Derived Types
 *
 * Batch gradient descent parallelized over MPI processes.
 * Uses MPI_Type_create_struct to define a custom datatype for Sample
 * (feature vector + label).  The dataset is generated on process 0 and
 * scattered with MPI_Scatterv.  Each process computes a local gradient
 * and loss; global reduction updates the weights synchronously.
 *
 * Model:  y = w[0]*x[0] + w[1]*x[1] + ... + w[N_FEATURES-1]*x[N_FEATURES-1] + bias
 *         (bias absorbed into w[N_FEATURES] with a constant feature = 1)
 *
 * Compile: mpicc -O2 -Wall -o distrib_grad distrib_grad.c -lm
 * Run:     mpirun -np 4 ./distrib_grad [N_SAMPLES]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* ---------- tunables ---------- */
#define N_FEATURES   2          /* number of features (excluding bias) */
#define DIM          (N_FEATURES + 1)   /* weights dimension (features + bias) */
#define MAX_EPOCHS   5000
#define LR           0.01       /* learning rate */
#define THRESHOLD    1.0e-2     /* early-stopping MSE threshold */
#define PRINT_EVERY  10         /* print every N epochs */

/* ---------- sample structure ---------- */
typedef struct {
    double x[N_FEATURES];       /* feature vector */
    double y;                   /* label */
} Sample;

/* ---------- data generation (process 0 only) ---------- */
/*
 * True model: y = 2.0*x[0] - 1.0*x[1] + 0.5  (with some noise)
 */
static void generate_data(Sample *data, int n)
{
    double true_w[N_FEATURES] = {2.0, -1.0};
    double true_bias = 0.5;
    for (int i = 0; i < n; i++) {
        for (int f = 0; f < N_FEATURES; f++)
            data[i].x[f] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        double noise = ((double)rand() / RAND_MAX - 0.5) * 0.2;
        data[i].y = true_bias;
        for (int f = 0; f < N_FEATURES; f++)
            data[i].y += true_w[f] * data[i].x[f];
        data[i].y += noise;
    }
}

/* ---------- build MPI derived type for Sample ---------- */
static MPI_Datatype create_sample_type(void)
{
    MPI_Datatype sample_type;
    int          block_lengths[2] = {N_FEATURES, 1};
    MPI_Aint     displacements[2];
    MPI_Datatype types[2]        = {MPI_DOUBLE, MPI_DOUBLE};

    /* Compute displacements relative to the start of the struct */
    Sample dummy;
    MPI_Aint base_addr, x_addr, y_addr;
    MPI_Get_address(&dummy,    &base_addr);
    MPI_Get_address(&dummy.x,  &x_addr);
    MPI_Get_address(&dummy.y,  &y_addr);
    displacements[0] = x_addr - base_addr;
    displacements[1] = y_addr - base_addr;

    MPI_Type_create_struct(2, block_lengths, displacements, types, &sample_type);

    /* Resize to match the actual struct size (handles padding) */
    MPI_Datatype resized;
    MPI_Type_create_resized(sample_type, 0, sizeof(Sample), &resized);
    MPI_Type_commit(&resized);
    MPI_Type_free(&sample_type);

    return resized;
}

/* ================================================================== */
int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Number of samples (optional command-line argument) */
    int N_SAMPLES = 1000;
    if (argc > 1)
        N_SAMPLES = atoi(argv[1]);

    /* ---- Create MPI derived type for Sample ---- */
    MPI_Datatype mpi_sample = create_sample_type();

    /* ---- Generate dataset on rank 0 ---- */
    Sample *all_data = NULL;
    if (rank == 0) {
        all_data = (Sample *)malloc(N_SAMPLES * sizeof(Sample));
        srand(42);
        generate_data(all_data, N_SAMPLES);
    }

    /* ---- Compute scatterv counts and displacements ---- */
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));
    {
        int base  = N_SAMPLES / size;
        int extra = N_SAMPLES % size;
        int offset = 0;
        for (int p = 0; p < size; p++) {
            sendcounts[p] = base + (p < extra ? 1 : 0);
            displs[p]     = offset;
            offset        += sendcounts[p];
        }
    }
    int local_n = sendcounts[rank];
    Sample *local_data = (Sample *)malloc(local_n * sizeof(Sample));

    /* ---- Scatter samples ---- */
    MPI_Scatterv(all_data, sendcounts, displs, mpi_sample,
                 local_data, local_n, mpi_sample,
                 0, MPI_COMM_WORLD);

    /* ---- Initialize weights to zero ---- */
    double w[DIM];
    memset(w, 0, sizeof(w));

    /* ---- Gradient descent loop ---- */
    double start_time = MPI_Wtime();
    int    epoch;
    double global_loss = 0.0;

    for (epoch = 0; epoch < MAX_EPOCHS; epoch++) {

        /* Local gradient and loss */
        double local_grad[DIM];
        memset(local_grad, 0, sizeof(local_grad));
        double local_loss = 0.0;

        for (int i = 0; i < local_n; i++) {
            /* prediction: w[0..N_FEATURES-1] . x  +  w[N_FEATURES] (bias) */
            double pred = w[DIM - 1];               /* bias term */
            for (int f = 0; f < N_FEATURES; f++)
                pred += w[f] * local_data[i].x[f];

            double err = pred - local_data[i].y;
            local_loss += err * err;

            /* gradient contributions */
            for (int f = 0; f < N_FEATURES; f++)
                local_grad[f] += 2.0 * err * local_data[i].x[f];
            local_grad[DIM - 1] += 2.0 * err;       /* bias gradient */
        }

        /* ---- Allreduce gradient and loss ---- */
        double global_grad[DIM];
        MPI_Allreduce(local_grad, global_grad, DIM, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_loss, &global_loss, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);

        /* MSE */
        global_loss /= N_SAMPLES;

        /* Normalize gradient by N_SAMPLES */
        for (int f = 0; f < DIM; f++)
            global_grad[f] /= N_SAMPLES;

        /* Weight update */
        for (int f = 0; f < DIM; f++)
            w[f] -= LR * global_grad[f];

        /* Reporting */
        if (rank == 0 && (epoch + 1) % PRINT_EVERY == 0) {
            printf("Epoch %4d | Loss (MSE): %f |", epoch + 1, global_loss);
            for (int f = 0; f < N_FEATURES; f++)
                printf(" w[%d]: %.4f,", f, w[f]);
            printf(" bias: %.4f\n", w[DIM - 1]);
        }

        /* Early stopping */
        if (global_loss < THRESHOLD) {
            if (rank == 0)
                printf("Early stopping at epoch %d — loss %f < %.1e\n",
                       epoch + 1, global_loss, THRESHOLD);
            break;
        }
    }

    double elapsed = MPI_Wtime() - start_time;

    if (rank == 0) {
        if (epoch == MAX_EPOCHS)
            printf("Reached maximum epochs (%d). Final loss: %f\n",
                   MAX_EPOCHS, global_loss);
        printf("\nTraining time: %.3f seconds (MPI, %d procs, %d samples)\n",
               elapsed, size, N_SAMPLES);
        printf("Final weights:");
        for (int f = 0; f < N_FEATURES; f++)
            printf(" w[%d]=%.4f", f, w[f]);
        printf("  bias=%.4f\n", w[DIM - 1]);
    }

    /* Cleanup */
    free(local_data);
    free(sendcounts);
    free(displs);
    if (all_data) free(all_data);
    MPI_Type_free(&mpi_sample);
    MPI_Finalize();
    return 0;
}
