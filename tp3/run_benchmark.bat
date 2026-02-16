@echo off
REM Simple benchmark script for OpenMP matrix multiplication

echo Compiling...
gcc -O2 -fopenmp ex4.c -o ex4_bench.exe -lm

if errorlevel 1 (
    echo Compilation failed!
    exit /b 1
)

echo.
echo Running benchmarks...
(
    echo Threads Schedule ChunkSize AvgTime
    
    for %%t in (1 2 4 8 16) do (
        for %%s in (STATIC DYNAMIC GUIDED) do (
            for %%c in (1 10 50 100 500) do (
                echo Testing: threads=%%t schedule=%%s chunk=%%c
                .\ex4_bench.exe %%t %%s %%c 10
            )
        )
    )
) > benchmark_results.txt

echo.
echo Results saved to benchmark_results.txt
echo Run: python analyze_results.py
