#!/usr/bin/env python3
"""
TP6 - Exercise 2: Plot Speedup & Efficiency from benchmark_results.csv

Usage: python3 plot_speedup.py [benchmark_results.csv]
Output: speedup_efficiency.png
"""

import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"

    procs = []
    times = []
    with open(fname, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            procs.append(int(row['procs']))
            times.append(float(row['time_s']))

    if not procs:
        print("No data to plot.")
        return

    T1 = times[0]  # sequential time (1 process)
    speedup    = [T1 / t for t in times]
    efficiency = [s / p for s, p in zip(speedup, procs)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Speedup plot
    ax1.plot(procs, speedup, 'bo-', label='Measured Speedup')
    ax1.plot(procs, procs, 'r--', alpha=0.5, label='Ideal (linear)')
    ax1.set_xlabel('Number of Processes')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup vs Number of Processes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Efficiency plot
    ax2.plot(procs, efficiency, 'gs-', label='Efficiency')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Ideal (100%)')
    ax2.set_xlabel('Number of Processes')
    ax2.set_ylabel('Efficiency (Speedup / P)')
    ax2.set_title('Efficiency vs Number of Processes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig('speedup_efficiency.png', dpi=150)
    print("Plot saved to speedup_efficiency.png")

if __name__ == '__main__':
    main()
