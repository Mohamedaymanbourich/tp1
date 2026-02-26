#!/usr/bin/env bash
# Benchmark script for Exercise 4: Matrix-Vector Product
# Usage: bash benchmark.sh
set -euo pipefail

EXE="./ex4"

# Compile
mpicc -O2 -o ex4 ex4.c -lm

# Clear previous results
rm -f timings.csv
echo "N,P,t_serial,t_parallel,speedup,efficiency,max_error" > timings.csv

SIZES=(64 128 256 512 1024 2048)
PROCS=(1 2 4 8)

for N in "${SIZES[@]}"; do
    for P in "${PROCS[@]}"; do
        echo "Running: N=$N  P=$P"
        mpirun --oversubscribe -np "$P" "$EXE" "$N"
    done
done

echo ""
echo "Results saved to timings.csv"
cat timings.csv
