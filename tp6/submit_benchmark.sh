#!/bin/bash
#SBATCH --job-name=tp6_benchmark
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=56
#SBATCH --time=00:30:00
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

# TP6 - Exercise 2: Slurm benchmark job
# Runs distributed gradient descent with 1 to 56 cores on one Toubkal node

module purge
module load OpenMPI/4.1.5-GCC-12.3.0
module load matplotlib/3.7.2-gfbf-2023a

cd $SLURM_SUBMIT_DIR

# Compile
mpicc -O2 -Wall -std=c99 -o distrib_grad distrib_grad.c -lm

N_SAMPLES=${1:-10000000}
BINARY="./distrib_grad"
OUTPUT="benchmark_results.csv"
PROCS_LIST="1 2 4 7 8 14 16 28 32 56"

echo "Benchmark: N_SAMPLES=$N_SAMPLES on $(hostname)"
echo "procs,time_s" > "$OUTPUT"

T1=""

for NP in $PROCS_LIST; do
    echo -n "Running with $NP process(es)... "
    TIME=$(mpirun -np "$NP" "$BINARY" "$N_SAMPLES" 2>/dev/null \
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

# Print speedup & efficiency table
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

# Generate the plot
echo ""
echo "Generating speedup/efficiency plot..."
python3 plot_speedup.py "$OUTPUT"
