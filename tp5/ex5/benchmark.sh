#!/usr/bin/env bash
# Benchmark script for Exercise 5: Pi Calculation
# Usage: bash benchmark.sh
set -euo pipefail

EXE="./ex5"

# Compile
mpicc -O2 -o ex5 ex5.c -lm

# Clear previous results
rm -f timings.csv
echo "N,P,t_serial,t_parallel,speedup,efficiency,pi" > timings.csv

N_VALUES=(1000000 10000000 100000000)
PROCS=(1 2 4 8)

for N in "${N_VALUES[@]}"; do
    for P in "${PROCS[@]}"; do
        echo "Running: N=$N  P=$P"
        mpirun --oversubscribe -np "$P" "$EXE" "$N"
    done
done

echo ""
echo "Results saved to timings.csv"
cat timings.csv
