#!/usr/bin/env python3
"""
Amdahl's and Gustafson's Law Analysis for Exercise 4 (Matrix Multiplication)
"""

import numpy as np
import matplotlib.pyplot as plt

def amdahl_speedup(fs, p):
    """Amdahl's Law speedup"""
    return 1.0 / (fs + (1 - fs) / p)

def gustafson_speedup(fs, p):
    """Gustafson's Law speedup"""
    return fs + p * (1 - fs)

# Measured sequential fractions for different matrix sizes
# For matrix multiplication, sequential part (generate_noise) is O(N)
# while parallel part (matmul) is O(N^3), so fs decreases as N increases
measurements = {
    256: 0.000172,  # Measured: 0.02% (5μs / 29ms)
    512: 0.000009,  # Measured: 0.00% (2μs / 223ms) 
    1024: 0.000001  # Measured: 0.00% (3μs / 5635ms)
}

processors = [1, 2, 4, 8, 16, 32, 64]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Amdahl's Law for different matrix sizes
ax1 = axes[0, 0]
for n, fs in measurements.items():
    speedups = [amdahl_speedup(fs, p) for p in processors]
    ax1.plot(processors, speedups, marker='o', label=f'N={n} (fs={fs:.5f})')

ax1.plot(processors, processors, 'k--', alpha=0.3, label='Linear (ideal)')
ax1.set_xlabel('Number of Processors (p)')
ax1.set_ylabel('Speedup S(p)')
ax1.set_title("Amdahl's Law: Matrix Multiplication Strong Scaling")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)
ax1.set_ylim(0, 70)

# Plot 2: Gustafson's Law for different matrix sizes
ax2 = axes[0, 1]
for n, fs in measurements.items():
    speedups = [gustafson_speedup(fs, p) for p in processors]
    ax2.plot(processors, speedups, marker='s', label=f'N={n} (fs={fs:.5f})')

ax2.plot(processors, processors, 'k--', alpha=0.3, label='Linear (ideal)')
ax2.set_xlabel('Number of Processors (p)')
ax2.set_ylabel('Speedup S(p)')
ax2.set_title("Gustafson's Law: Matrix Multiplication Weak Scaling")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)

# Plot 3: Comparison Ex3 vs Ex4 using Amdahl's Law
ax3 = axes[1, 0]
fs_ex3 = 0.277  # From Exercise 3
fs_ex4 = measurements[512]

amdahl_ex3 = [amdahl_speedup(fs_ex3, p) for p in processors]
amdahl_ex4 = [amdahl_speedup(fs_ex4, p) for p in processors]

ax3.plot(processors, amdahl_ex3, marker='o', label=f"Ex3: Vector ops (fs={fs_ex3:.3f})", linewidth=2)
ax3.plot(processors, amdahl_ex4, marker='s', label=f"Ex4: MatMul (fs={fs_ex4:.5f})", linewidth=2)
ax3.plot(processors, processors, 'k--', alpha=0.3, label='Linear (ideal)')
ax3.set_xlabel('Number of Processors (p)')
ax3.set_ylabel('Speedup S(p)')
ax3.set_title("Amdahl's Law: Exercise 3 vs Exercise 4")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log', base=2)

# Plot 4: Sequential fraction impact
ax4 = axes[1, 1]
fs_values = [0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001]
for fs in fs_values:
    speedups = [amdahl_speedup(fs, p) for p in processors]
    ax4.plot(processors, speedups, marker='o', label=f'fs={fs:.4f}')

ax4.plot(processors, processors, 'k--', alpha=0.3, label='Linear (ideal)')
ax4.set_xlabel('Number of Processors (p)')
ax4.set_ylabel('Speedup S(p)')
ax4.set_title("Impact of Sequential Fraction on Speedup")
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.3)
ax4.set_xscale('log', base=2)

plt.tight_layout()
plt.savefig('ex4_analysis.png', dpi=300, bbox_inches='tight')
print("Plot saved as ex4_analysis.png")

# Print numerical results
print("\n" + "="*80)
print("EXERCISE 4: MATRIX MULTIPLICATION - AMDAHL'S AND GUSTAFSON'S LAW ANALYSIS")
print("="*80)

for n, fs in measurements.items():
    print(f"\nMatrix Size N = {n}x{n}")
    print(f"Sequential fraction fs = {fs:.6f} ({fs*100:.4f}%)")
    print(f"Maximum theoretical speedup (Amdahl) = {1/fs:.2f}" if fs > 0 else "Infinite (perfectly parallelizable)")
    print("\n{:<8} {:<15} {:<15} {:<15} {:<15}".format(
        "p", "Amdahl S(p)", "Amdahl Eff%", "Gustafson S(p)", "Gustafson Eff%"))
    print("-" * 80)
    
    for p in processors:
        s_amdahl = amdahl_speedup(fs, p)
        s_gustafson = gustafson_speedup(fs, p)
        eff_amdahl = s_amdahl / p * 100
        eff_gustafson = s_gustafson / p * 100
        
        print(f"{p:<8} {s_amdahl:<15.3f} {eff_amdahl:<15.2f} "
              f"{s_gustafson:<15.3f} {eff_gustafson:<15.2f}")

print("\n" + "="*80)
print("COMPARISON: EXERCISE 3 vs EXERCISE 4")
print("="*80)
print("\nExercise 3 (Vector Operations):")
print(f"  - Sequential fraction: fs ≈ 27.7%")
print(f"  - Sequential part: add_noise() - O(N) with data dependency")
print(f"  - Parallel part: init, addition, reduction - O(N)")
print(f"  - Max speedup: ~3.6x (limited by sequential part)")
print(f"  - Problem: Sequential fraction does NOT decrease with N")

print("\nExercise 4 (Matrix Multiplication):")
print(f"  - Sequential fraction: fs < 0.1% (for N=512)")
print(f"  - Sequential part: generate_noise() - O(N)")
print(f"  - Parallel part: matrix multiplication - O(N³)")
print(f"  - Max speedup: Nearly linear (hundreds to thousands)")
print(f"  - Problem: Sequential fraction DECREASES as N increases")

print("\n" + "="*80)
print("KEY OBSERVATIONS:")
print("="*80)
print("1. Matrix Multiplication is Highly Parallelizable:")
print("   - Sequential overhead (generate_noise) is O(N)")
print("   - Computational work (matmul) is O(N³)")
print("   - As N increases, fs → 0, speedup → p (linear)\n")

print("2. Comparison with Exercise 3:")
print("   - Ex3: Sequential and parallel parts both O(N) → fs constant")
print("   - Ex4: Sequential O(N), parallel O(N³) → fs decreases with N")
print("   - Ex4 has MUCH better scalability!\n")

print("3. Practical Implications:")
print("   - Ex4 benefits greatly from parallelization")
print("   - Even with 64 cores, can achieve near-linear speedup")
print("   - Larger matrices → better parallel efficiency")
print("   - This is why HPC systems excel at matrix operations!\n")

print("4. Gustafson's Law:")
print("   - Both exercises scale well under Gustafson's model")
print("   - Weak scaling: increase problem size with processors")
print("   - More realistic for many real-world applications")
print("="*80)

plt.show()
