#!/bin/bash
# TP6 - Exercise 2: Benchmark script for speedup & efficiency
#
# Runs the distributed gradient descent with increasing core counts
# and records timing to compute speedup and efficiency.
#
# Usage:  bash run_benchmark.sh [N_SAMPLES]
#         (Default: N_SAMPLES=100000)
#
# Output: benchmark_results.csv

N_SAMPLES=${1:-100000}
BINARY="./distrib_grad"
OUTPUT="benchmark_results.csv"
PROCS_LIST="1 2 4 7 8 14 16 28 32 56"

# Compile if necessary
if [ ! -f "$BINARY" ]; then
    echo "Compiling distrib_grad..."
    mpicc -O2 -Wall -std=c99 -o distrib_grad distrib_grad.c -lm
fi

echo "Benchmark: N_SAMPLES=$N_SAMPLES"
echo "procs,time_s" > "$OUTPUT"

T1=""

for NP in $PROCS_LIST; do
    echo -n "Running with $NP process(es)... "
    # Extract the training time from the program output
    TIME=$(mpirun --oversubscribe -np "$NP" "$BINARY" "$N_SAMPLES" 2>/dev/null \
           | grep "Training time" | awk '{print $3}')
    if [ -z "$TIME" ]; then
        echo "FAILED (no output)"
        continue
    fi
    echo "${TIME}s"
    echo "$NP,$TIME" >> "$OUTPUT"

    if [ "$NP" -eq 1 ]; then
        T1=$TIME
    fi
done

echo ""
echo "Results written to $OUTPUT"

# Compute speedup & efficiency table
if [ -n "$T1" ]; then
    echo ""
    echo "=== Speedup & Efficiency (T1 = ${T1}s) ==="
    printf "%-8s %-12s %-12s %-12s\n" "Procs" "Time(s)" "Speedup" "Efficiency"
    tail -n +2 "$OUTPUT" | while IFS=',' read -r procs time; do
        speedup=$(echo "scale=3; $T1 / $time" | bc)
        efficiency=$(echo "scale=3; $speedup / $procs" | bc)
        printf "%-8s %-12s %-12s %-12s\n" "$procs" "$time" "$speedup" "$efficiency"
    done
fi
