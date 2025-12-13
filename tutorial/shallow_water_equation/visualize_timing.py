#!/usr/bin/env python3
"""
Visualize timing data from CPU and GPU benchmarks.
Plots time (right y-axis) as line plots and speedup (left y-axis) as histogram.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# N_CPU and N_GPU values from easier_swe.sh
N_CPU=[500, 1000, 2000, 3000, 4000, 5000]
N_GPU=[500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]

# Read CSV files
script_dir = os.path.dirname(os.path.abspath(__file__))
res_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'res')

cpu_cpu_file = os.path.join(res_dir, 'timing_cpu_cpu.csv')
cpu_torch_file = os.path.join(res_dir, 'timing_cpu_torch.csv')
cuda_cuda_file = os.path.join(res_dir, 'timing_cuda_cuda.csv')
cuda_torch_file = os.path.join(res_dir, 'timing_cuda_torch.csv')
kokkos_cpu_file = os.path.join(res_dir, 'timing_cpu_kokkos.csv')
kokkos_gpu_file = os.path.join(res_dir, 'timing_cuda_kokkos.csv')

# Load CPU data
df_cpu_cpu = pd.read_csv(cpu_cpu_file)
df_cpu_torch = pd.read_csv(cpu_torch_file)

# Load GPU data
df_cuda_cuda = pd.read_csv(cuda_cuda_file)
df_cuda_torch = pd.read_csv(cuda_torch_file)

# Load Kokkos data
df_kokkos_cpu = pd.read_csv(kokkos_cpu_file)
df_kokkos_gpu = pd.read_csv(kokkos_gpu_file)

# Extract time data (seconds column)
# Only use rows that have data (exclude header and empty rows)
cpu_cpu_times = df_cpu_cpu['seconds'].dropna().values
cpu_torch_times = df_cpu_torch['seconds'].dropna().values
cuda_cuda_times = df_cuda_cuda['seconds'].dropna().values
cuda_torch_times = df_cuda_torch['seconds'].dropna().values
kokkos_cpu_times = df_kokkos_cpu['seconds'].dropna().values
kokkos_gpu_times = df_kokkos_gpu['seconds'].dropna().values

# Match N_CPU and N_GPU values to available data
n_cpu_data = N_CPU[:len(cpu_cpu_times)]
n_gpu_data = N_GPU[:len(cuda_cuda_times)]

# Calculate bar width based on minimum spacing for each dataset
min_spacing_cpu = min([n_cpu_data[i+1] - n_cpu_data[i] for i in range(len(n_cpu_data)-1)]) if len(n_cpu_data) > 1 else 500
min_spacing_gpu = min([n_gpu_data[i+1] - n_gpu_data[i] for i in range(len(n_gpu_data)-1)]) if len(n_gpu_data) > 1 else 250
bar_width_cpu = min_spacing_cpu * 0.4  # 40% of minimum spacing for CPU
bar_width_gpu = min_spacing_gpu * 0.4  # 40% of minimum spacing for GPU

# Calculate speedup (torch time / backend time)
speedup_torch_cpu = cpu_torch_times / cpu_cpu_times
speedup_torch_gpu = cuda_torch_times / cuda_cuda_times

# Calculate Kokkos speedup (kokkos time / backend time)
speedup_kokkos_cpu = kokkos_cpu_times / cpu_cpu_times
speedup_kokkos_gpu = kokkos_gpu_times / cuda_cuda_times

# ========== CPU Comparison Plot ==========
fig1, ax1 = plt.subplots(figsize=(8, 6))

# Left y-axis: Speedup histogram (bar chart)
# Use steelblue to match Torch time line color
color_speedup_torch = 'steelblue'
ax1.set_xlabel('n', fontsize=18)
ax1.set_ylabel('Speedup', color='black', fontsize=18)
bars_torch = ax1.bar(
    n_cpu_data, speedup_torch_cpu, width=bar_width_cpu, alpha=0.7,
    color=color_speedup_torch, 
    edgecolor='black', linewidth=1.2, label='Speedup (Torch/Easier)')
ax1.tick_params(axis='y', labelsize=14, 
                labelcolor='black')
ax1.tick_params(axis='x', labelsize=14)
ax1.set_xticks(n_cpu_data)
ax1.set_xticklabels(n_cpu_data)
ax1.grid(True, alpha=0.3, axis='y')
# Increase y-range to avoid legend overlap
max_speedup = max(speedup_torch_cpu) * 1.4
if speedup_kokkos_cpu is not None:
    max_speedup = max(max_speedup, max(speedup_kokkos_cpu))
ax1.set_ylim(0, max_speedup * 1.15)  # Add 15% padding at top

# Calculate text offset to avoid overlap (2% of y-axis range)
y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
text_offset = y_range * 0.02

# Add value labels on top of each bar
for bar in bars_torch:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + text_offset,
             f'{height:.2f}x',
             ha='center', va='bottom', fontsize=12, color=color_speedup_torch, fontweight='bold')

# Add kokkos speedup bars if available
if speedup_kokkos_cpu is not None:
    color_speedup_kokkos = 'purple'
    bars_kokkos = ax1.bar([
        x + bar_width_cpu for x in n_cpu_data], speedup_kokkos_cpu, width=bar_width_cpu, 
        alpha=0.7, color=color_speedup_kokkos,
        edgecolor='black', linewidth=1.2, label='Speedup (Kokkos/Easier)')
    # Add value labels on top of each kokkos bar
    for bar in bars_kokkos:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + text_offset,
                 f'{height:.2f}x',
                 ha='center', va='bottom', fontsize=12, color=color_speedup_kokkos, fontweight='bold')

# Right y-axis: Time line plots
ax2 = ax1.twinx()
color_cpu = 'green'
color_torch = 'steelblue'
ax2.set_ylabel('Time (seconds)', fontsize=18)

# Plot CPU time
line1 = ax2.plot(n_cpu_data, cpu_cpu_times, marker='o', linestyle='-', 
                 color=color_cpu, linewidth=2, markersize=8, alpha=0.7, 
                 markerfacecolor=color_cpu, markeredgecolor=color_cpu, label='Easier')

# Plot Torch time
line2 = ax2.plot(n_cpu_data, cpu_torch_times, marker='s', linestyle='--', 
                 color=color_torch, linewidth=2, markersize=8, alpha=0.7,
                 markerfacecolor=color_torch, markeredgecolor=color_torch, label='Torch')

# Plot Kokkos time if available
if kokkos_cpu_times is not None:
    color_kokkos = 'purple'
    line3 = ax2.plot(n_cpu_data, kokkos_cpu_times, marker='^', linestyle='-.', 
                     color=color_kokkos, linewidth=2, markersize=8, alpha=0.7,
                     markerfacecolor=color_kokkos, markeredgecolor=color_kokkos, label='Kokkos')

ax2.tick_params(axis='y', labelsize=14)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
           fontsize=14, ncol=2)

# Adjust layout
plt.tight_layout()

# Save CPU figure
output_file_cpu = os.path.join(script_dir, 'timing_visualization_cpu.pdf')
plt.savefig(output_file_cpu, dpi=300, bbox_inches='tight')
print(f"CPU visualization saved to: {output_file_cpu}")

# ========== GPU Comparison Plot ==========
fig2, ax1_gpu = plt.subplots(figsize=(8, 6))

# Left y-axis: Speedup histogram (bar chart)
# Use steelblue to match Torch time line color
color_speedup_torch_gpu = 'steelblue'
ax1_gpu.set_xlabel('n', fontsize=18)
ax1_gpu.set_ylabel('Speedup', color='black', fontsize=18)
bars_torch_gpu = ax1_gpu.bar(
    n_gpu_data, speedup_torch_gpu, width=bar_width_gpu, alpha=0.7,
    color=color_speedup_torch_gpu, 
    edgecolor='black', linewidth=1.2, label='Speedup (Torch/Easier)')
ax1_gpu.tick_params(axis='y', labelsize=14, 
                    labelcolor='black')
ax1_gpu.tick_params(axis='x', labelsize=14)
ax1_gpu.set_xticks(n_gpu_data)
ax1_gpu.set_xticklabels(n_gpu_data)
ax1_gpu.grid(True, alpha=0.3, axis='y')
# Increase y-range to avoid legend overlap
max_speedup_gpu = max(speedup_torch_gpu) * 1.2
if speedup_kokkos_gpu is not None:
    max_speedup_gpu = max(max_speedup_gpu, max(speedup_kokkos_gpu))
ax1_gpu.set_ylim(0, max_speedup_gpu * 1.15)  # Add 15% padding at top

# Calculate text offset to avoid overlap (2% of y-axis range)
y_range_gpu = ax1_gpu.get_ylim()[1] - ax1_gpu.get_ylim()[0]
text_offset_gpu = y_range_gpu * 0.02

# Add value labels on top of each bar
for bar in bars_torch_gpu:
    height = bar.get_height()
    ax1_gpu.text(bar.get_x() + bar.get_width()/2., height + text_offset_gpu,
                 f'{height:.2f}x',
                 ha='center', va='bottom', fontsize=12, color=color_speedup_torch_gpu, fontweight='bold')

# Add kokkos speedup bars if available
if speedup_kokkos_gpu is not None:
    color_speedup_kokkos_gpu = 'purple'
    bars_kokkos_gpu = ax1_gpu.bar([
        x + bar_width_gpu for x in n_gpu_data], speedup_kokkos_gpu, width=bar_width_gpu, 
        alpha=0.7, color=color_speedup_kokkos_gpu,
        edgecolor='black', linewidth=1.2, label='Speedup (Kokkos/Easier)')
    # Add value labels on top of each kokkos bar
    for bar in bars_kokkos_gpu:
        height = bar.get_height()
        ax1_gpu.text(bar.get_x() + bar.get_width()/2., height + text_offset_gpu,
                     f'{height:.2f}x',
                     ha='center', va='bottom', fontsize=12, color=color_speedup_kokkos_gpu, fontweight='bold')

# Right y-axis: Time line plots
ax2_gpu = ax1_gpu.twinx()
color_cuda = 'green'
color_torch_gpu = 'steelblue'
ax2_gpu.set_ylabel('Time (seconds)', fontsize=18)

# Plot CUDA time
line1_gpu = ax2_gpu.plot(n_gpu_data, cuda_cuda_times, marker='o', linestyle='-', 
                         color=color_cuda, linewidth=2, markersize=8, alpha=0.7,
                         markerfacecolor=color_cuda, markeredgecolor=color_cuda, label='Easier')

# Plot Torch time
line2_gpu = ax2_gpu.plot(
    n_gpu_data, cuda_torch_times, marker='s', linestyle='--', 
    color=color_torch_gpu, linewidth=2, markersize=8, alpha=0.7,
    markerfacecolor=color_torch_gpu, markeredgecolor=color_torch_gpu, label='Torch')

# Plot Kokkos time if available
if kokkos_gpu_times is not None:
    color_kokkos_gpu = 'purple'
    line3_gpu = ax2_gpu.plot(n_gpu_data, kokkos_gpu_times, marker='^', linestyle='-.', 
                             color=color_kokkos_gpu, linewidth=2, markersize=8, alpha=0.7,
                             markerfacecolor=color_kokkos_gpu, markeredgecolor=color_kokkos_gpu, label='Kokkos')

ax2_gpu.tick_params(axis='y', labelsize=14)

# Combine legends
lines1_gpu, labels1_gpu = ax1_gpu.get_legend_handles_labels()
lines2_gpu, labels2_gpu = ax2_gpu.get_legend_handles_labels()
# Arrange the legend into 4 columns
ax1_gpu.legend(lines1_gpu + lines2_gpu, labels1_gpu + labels2_gpu, 
               loc='upper left', fontsize=14, ncol=2)

# Adjust layout
plt.tight_layout()

# Save GPU figure
output_file_gpu = os.path.join(script_dir, 'timing_visualization_gpu.pdf')
plt.savefig(output_file_gpu, dpi=300, bbox_inches='tight')
print(f"GPU visualization saved to: {output_file_gpu}")

# Also display
plt.show()

