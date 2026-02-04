#!/usr/bin/env python3
"""
Amdahl's and Gustafson's Law Analysis for Exercise 3
"""

import numpy as np
import matplotlib.pyplot as plt

def amdahl_speedup(fs, p):
    """
    Amdahl's Law speedup
    S(p) = 1 / (fs + (1-fs)/p)
    fs: sequential fraction
    p: number of processors
    """
    return 1.0 / (fs + (1 - fs) / p)

def gustafson_speedup(fs, p):
    """
    Gustafson's Law speedup
    S(p) = fs + p * (1 - fs)
    fs: sequential fraction
    p: number of processors
    """
    return fs + p * (1 - fs)

# Measured sequential fractions for different N values
# These should be updated based on actual measurements or Callgrind profiling
measurements = {
    5_000_000: 0.276,    # Example: ~27.6% sequential (add_noise)
    10_000_000: 0.276,
    100_000_000: 0.277   # Should be close for this problem
}

# Number of processors
processors = [1, 2, 4, 8, 16, 32, 64]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Amdahl's Law for different N
ax1 = axes[0, 0]
for n, fs in measurements.items():
    speedups = [amdahl_speedup(fs, p) for p in processors]
    ax1.plot(processors, speedups, marker='o', label=f'N={n:,} (fs={fs:.3f})')
    max_speedup = 1.0 / fs
    ax1.axhline(y=max_speedup, linestyle='--', alpha=0.5, 
                label=f'Max={max_speedup:.2f}')

ax1.plot(processors, processors, 'k--', alpha=0.3, label='Linear (ideal)')
ax1.set_xlabel('Number of Processors (p)')
ax1.set_ylabel('Speedup S(p)')
ax1.set_title("Amdahl's Law: Strong Scaling")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)

# Plot 2: Gustafson's Law for different N
ax2 = axes[0, 1]
for n, fs in measurements.items():
    speedups = [gustafson_speedup(fs, p) for p in processors]
    ax2.plot(processors, speedups, marker='s', label=f'N={n:,} (fs={fs:.3f})')

ax2.plot(processors, processors, 'k--', alpha=0.3, label='Linear (ideal)')
ax2.set_xlabel('Number of Processors (p)')
ax2.set_ylabel('Speedup S(p)')
ax2.set_title("Gustafson's Law: Weak Scaling")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)

# Plot 3: Comparison for N=100M
ax3 = axes[1, 0]
fs = measurements[100_000_000]
amdahl = [amdahl_speedup(fs, p) for p in processors]
gustafson = [gustafson_speedup(fs, p) for p in processors]

ax3.plot(processors, amdahl, marker='o', label="Amdahl's Law", linewidth=2)
ax3.plot(processors, gustafson, marker='s', label="Gustafson's Law", linewidth=2)
ax3.plot(processors, processors, 'k--', alpha=0.3, label='Linear (ideal)')
ax3.set_xlabel('Number of Processors (p)')
ax3.set_ylabel('Speedup S(p)')
ax3.set_title(f"Amdahl vs Gustafson (N=100M, fs={fs:.3f})")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log', base=2)

# Plot 4: Efficiency
ax4 = axes[1, 1]
efficiency_amdahl = [amdahl_speedup(fs, p) / p * 100 for p in processors]
efficiency_gustafson = [gustafson_speedup(fs, p) / p * 100 for p in processors]

ax4.plot(processors, efficiency_amdahl, marker='o', label="Amdahl Efficiency", linewidth=2)
ax4.plot(processors, efficiency_gustafson, marker='s', label="Gustafson Efficiency", linewidth=2)
ax4.axhline(y=100, color='k', linestyle='--', alpha=0.3, label='100% Efficiency')
ax4.set_xlabel('Number of Processors (p)')
ax4.set_ylabel('Efficiency (%)')
ax4.set_title(f"Parallel Efficiency (N=100M, fs={fs:.3f})")
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xscale('log', base=2)

plt.tight_layout()
plt.savefig('ex3_analysis.png', dpi=300, bbox_inches='tight')
print("Plot saved as ex3_analysis.png")

# Print numerical results
print("\n" + "="*70)
print("EXERCISE 3: AMDAHL'S AND GUSTAFSON'S LAW ANALYSIS")
print("="*70)

for n, fs in measurements.items():
    print(f"\nN = {n:,}")
    print(f"Sequential fraction fs = {fs:.4f} ({fs*100:.2f}%)")
    print(f"Maximum theoretical speedup (Amdahl) = {1/fs:.2f}")
    print("\n{:<8} {:<15} {:<15} {:<15} {:<15}".format(
        "p", "Amdahl S(p)", "Amdahl Eff%", "Gustafson S(p)", "Gustafson Eff%"))
    print("-" * 70)
    
    for p in processors:
        s_amdahl = amdahl_speedup(fs, p)
        s_gustafson = gustafson_speedup(fs, p)
        eff_amdahl = s_amdahl / p * 100
        eff_gustafson = s_gustafson / p * 100
        
        print(f"{p:<8} {s_amdahl:<15.3f} {eff_amdahl:<15.2f} "
              f"{s_gustafson:<15.3f} {eff_gustafson:<15.2f}")

print("\n" + "="*70)
print("KEY OBSERVATIONS:")
print("="*70)
print("1. Amdahl's Law (Strong Scaling):")
print("   - Speedup saturates as p increases")
print(f"   - Maximum speedup limited by sequential fraction: ~{1/fs:.2f}x")
print("   - Efficiency decreases rapidly with more processors\n")
print("2. Gustafson's Law (Weak Scaling):")
print("   - Speedup scales nearly linearly with p")
print("   - Assumes problem size grows with number of processors")
print("   - Better scalability for larger problems\n")
print("3. The sequential part (add_noise) limits strong scaling")
print("   - add_noise is inherently sequential (data dependency)")
print("   - Other parts (init_b, compute_addition, reduction) are parallelizable")
print("="*70)

plt.show()
