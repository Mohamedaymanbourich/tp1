#!/bin/bash
#SBATCH --job-name=my_job      # Job name
#SBATCH --output=slurm-hpl.out  # Standard output and error log
#SBATCH --error=slurm-hpl.err   # Standard error log (optional)



# HPL Benchmark Script
# This script runs HPL with different matrix sizes (N) and block sizes (NB)
# and collects the results

# Output directory and file
OUTPUT_DIR="/home/mohamed.bourich/hpl-2.3/results"
mkdir -p "$OUTPUT_DIR"
RESULTS_FILE="$OUTPUT_DIR/hpl_results.csv"

# Initialize results file with header
echo "N,NB,Time(s),GFLOPS,Status" > "$RESULTS_FILE"

# Matrix sizes to test
MATRIX_SIZES=(1000 5000 10000 20000)

# Block sizes to test
BLOCK_SIZES=(1 2 4 8 16 32 64 128 256)

# HPL binary and template
HPL_BIN="/home/mohamed.bourich/hpl-2.3/bin/Linux/xhpl"
HPL_TEMPLATE="/home/mohamed.bourich/hpl-2.3/bin/Linux/HPL.dat.template"
HPL_DAT="/home/mohamed.bourich/hpl-2.3/bin/Linux/HPL.dat"

# Create HPL.dat template
cat > "$HPL_TEMPLATE" << 'EOF'
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
__N__        Ns
1            # of NBs
__NB__       NBs
0            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
1            Ps
1            Qs
16.0         threshold
1            # of panel fact
2            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
4            NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
2            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
1            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
1            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
EOF

# Counter for progress
total_runs=$((${#MATRIX_SIZES[@]} * ${#BLOCK_SIZES[@]}))
current_run=0

echo "=========================================="
echo "HPL Benchmark Suite"
echo "=========================================="
echo "Matrix sizes: ${MATRIX_SIZES[@]}"
echo "Block sizes: ${BLOCK_SIZES[@]}"
echo "Total runs: $total_runs"
echo "=========================================="
echo ""

# Loop through all combinations
for N in "${MATRIX_SIZES[@]}"; do
    for NB in "${BLOCK_SIZES[@]}"; do
        current_run=$((current_run + 1))
        
        echo "[$current_run/$total_runs] Running HPL: N=$N, NB=$NB"
        
        # Create HPL.dat for this configuration
        sed "s/__N__/$N/g; s/__NB__/$NB/g" "$HPL_TEMPLATE" > "$HPL_DAT"
        
        # Change to bin directory and run HPL
        cd "/home/mohamed.bourich/hpl-2.3/bin/Linux"
        
        # Run HPL and capture output
        OUTPUT_FILE="$OUTPUT_DIR/hpl_N${N}_NB${NB}.out"
        mpirun -np 1 --bind-to core ./xhpl > "$OUTPUT_FILE" 2>&1
        
        # Extract results from output
        # Look for the line with results: WR00L2L2 N NB P Q Time Gflops
        RESULT_LINE=$(grep "WR" "$OUTPUT_FILE" | grep -E "^\s*WR" | head -1)
        
        if [ -n "$RESULT_LINE" ]; then
            # Extract Time and Gflops
            TIME=$(echo "$RESULT_LINE" | awk '{print $6}')
            GFLOPS=$(echo "$RESULT_LINE" | awk '{print $7}')
            
            # Check if test passed
            if grep -q "PASSED" "$OUTPUT_FILE"; then
                STATUS="PASSED"
            else
                STATUS="FAILED"
            fi
        else
            TIME="N/A"
            GFLOPS="N/A"
            STATUS="ERROR"
        fi
        
        # Save to results file
        echo "$N,$NB,$TIME,$GFLOPS,$STATUS" >> "$RESULTS_FILE"
        
        echo "  -> Time: $TIME s, GFLOPS: $GFLOPS, Status: $STATUS"
        echo ""
    done
done

echo "=========================================="
echo "Benchmarks completed!"
echo "Results saved to: $RESULTS_FILE"
echo "=========================================="
