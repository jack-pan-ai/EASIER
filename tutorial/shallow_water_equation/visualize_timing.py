#!/usr/bin/env python3
"""
Visualize timing data from CPU and GPU benchmarks.
Plots time (right y-axis) as line plots and speedup (left y-axis) as histogram.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Publication-style palette: blue vs orange (high contrast, colorblind-friendly; prints well).
# Within Kokkos / Taichi: simple = darker, fused = lighter (same hue).
# Based on Tableau 10–style pairs (common in scientific figures; CMYK-friendly).
COLOR_KOKKOS_SIMPLE = '#4E79A7'
COLOR_KOKKOS_FUSED = '#F28E2B'
COLOR_TAICHI_SIMPLE = '#A0CBE8'
COLOR_TAICHI_FUSED = '#FFBE7D'
COLOR_TORCH = '#59A14F'  # Distinct, colorblind-friendly green (Tableau 10 palette)


# N_CPU and N_GPU values from easier_swe.sh
# N_CPU=[500, 1000, 1500, 2000, 2500, 3000]
# N_GPU=[500, 1000, 1500, 2000, 2500, 3000]
N_CPU=[500, 1000, 1500, 2000]
N_GPU=[500, 1000, 1500, 2000]
# Show only the first N points in plots (set to 5 or 4, etc.; None keeps all)
PLOT_FIRST_N = 5


def _slice_head(arr, n):
    return None if arr is None else arr[:n]


def _resolve_plot_len(n_values, plot_first_n, *time_arrays):
    """Pick a plot length that all *existing* series can provide."""
    max_len = len(n_values)
    if plot_first_n is not None:
        max_len = min(max_len, int(plot_first_n))
    existing_lengths = [len(a) for a in time_arrays if a is not None]
    return min([max_len] + existing_lengths) if existing_lengths else max_len
# Small helper: load timing CSVs (expects an `ms_per_iteration` column)
def _load_seconds(path: str, *, required: bool):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Missing required timing CSV: {path}")
        return None
    df = pd.read_csv(path)
    if 'ms_per_iteration' not in df.columns:
        raise ValueError(f"Expected an 'ms_per_iteration' column in: {path}")
    return df['ms_per_iteration'].dropna().values/1000.0

# Read CSV files
script_dir = os.path.dirname(os.path.abspath(__file__))
res_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'res')

cpu_cpu_file = os.path.join(res_dir, 'timing_cpu_cpu.csv')
cpu_torch_file = os.path.join(res_dir, 'timing_cpu_torch.csv')
cuda_cuda_file = os.path.join(res_dir, 'timing_cuda_cuda.csv')
cuda_torch_file = os.path.join(res_dir, 'timing_cuda_torch.csv')
kokkos_cpu_file = os.path.join(res_dir, 'swe_profile_cpu_simple/timing.csv')
kokkos_gpu_file = os.path.join(res_dir, 'swe_profile_cuda/timing.csv')
kokkos_cpu_fused_file = os.path.join(res_dir, 'swe_profile_cpu_fused/timing_fused.csv')
kokkos_gpu_fused_file = os.path.join(res_dir, 'swe_profile_cuda_fused/timing_fused.csv')
taichi_cpu_file = os.path.join(res_dir, 'res_simple/timing_cpu_taichi.csv')
taichi_gpu_file = os.path.join(res_dir, 'res_simple/timing_cuda_taichi.csv')
taichi_cpu_fused_file = os.path.join(res_dir, 'res_fused/timing_cpu_taichi_fused.csv')
taichi_gpu_fused_file = os.path.join(res_dir, 'res_fused/timing_cuda_taichi_fused.csv')

# Load + extract time data (seconds column)
# Only use rows that have data (exclude header and empty rows)
cpu_cpu_times = _load_seconds(cpu_cpu_file, required=True)
cpu_torch_times = _load_seconds(cpu_torch_file, required=False)
cuda_cuda_times = _load_seconds(cuda_cuda_file, required=True)
cuda_torch_times = _load_seconds(cuda_torch_file, required=False)
kokkos_cpu_times = _load_seconds(kokkos_cpu_file, required=False)
kokkos_gpu_times = _load_seconds(kokkos_gpu_file, required=False)
kokkos_cpu_fused_times = _load_seconds(kokkos_cpu_fused_file, required=False)
kokkos_gpu_fused_times = _load_seconds(kokkos_gpu_fused_file, required=False)
taichi_cpu_times = _load_seconds(taichi_cpu_file, required=False)
taichi_gpu_times = _load_seconds(taichi_gpu_file, required=False)
taichi_cpu_fused_times = _load_seconds(taichi_cpu_fused_file, required=False)
taichi_gpu_fused_times = _load_seconds(taichi_gpu_fused_file, required=False)

# Match N_CPU and N_GPU values to available data, then keep only first PLOT_FIRST_N.
# The plot length is the min across all series that exist so we don't get shape mismatches.
cpu_plot_len = _resolve_plot_len(
    N_CPU, PLOT_FIRST_N,
    cpu_cpu_times,
    cpu_torch_times,
    kokkos_cpu_times,
    kokkos_cpu_fused_times,
    taichi_cpu_times,
    taichi_cpu_fused_times,
)
gpu_plot_len = _resolve_plot_len(
    N_GPU, PLOT_FIRST_N,
    cuda_cuda_times,
    cuda_torch_times,
    kokkos_gpu_times,
    kokkos_gpu_fused_times,
    taichi_gpu_times,
    taichi_gpu_fused_times,
)

n_cpu_data = N_CPU[:cpu_plot_len]
n_gpu_data = N_GPU[:gpu_plot_len]

cpu_cpu_times = cpu_cpu_times[:cpu_plot_len]
cuda_cuda_times = cuda_cuda_times[:gpu_plot_len]

cpu_torch_times = _slice_head(cpu_torch_times, cpu_plot_len)
cuda_torch_times = _slice_head(cuda_torch_times, gpu_plot_len)
kokkos_cpu_times = _slice_head(kokkos_cpu_times, cpu_plot_len)
kokkos_gpu_times = _slice_head(kokkos_gpu_times, gpu_plot_len)
kokkos_cpu_fused_times = _slice_head(kokkos_cpu_fused_times, cpu_plot_len)
kokkos_gpu_fused_times = _slice_head(kokkos_gpu_fused_times, gpu_plot_len)
taichi_cpu_times = _slice_head(taichi_cpu_times, cpu_plot_len)
taichi_gpu_times = _slice_head(taichi_gpu_times, gpu_plot_len)
taichi_cpu_fused_times = _slice_head(taichi_cpu_fused_times, cpu_plot_len)
taichi_gpu_fused_times = _slice_head(taichi_gpu_fused_times, gpu_plot_len)

# Calculate bar width based on minimum spacing for each dataset and number of bar series
min_spacing_cpu = min([n_cpu_data[i+1] - n_cpu_data[i] for i in range(len(n_cpu_data)-1)]) if len(n_cpu_data) > 1 else 500
min_spacing_gpu = min([n_gpu_data[i+1] - n_gpu_data[i] for i in range(len(n_gpu_data)-1)]) if len(n_gpu_data) > 1 else 250
bar_series_cpu = (
    int(cpu_torch_times is not None)
    + int(kokkos_cpu_times is not None)
    + int(kokkos_cpu_fused_times is not None)
    + int(taichi_cpu_times is not None)
    + int(taichi_cpu_fused_times is not None)
)
bar_series_gpu = (
    int(cuda_torch_times is not None)
    + int(kokkos_gpu_times is not None)
    + int(kokkos_gpu_fused_times is not None)
    + int(taichi_gpu_times is not None)
    + int(taichi_gpu_fused_times is not None)
)
bar_width_cpu = min_spacing_cpu * (0.8 / max(bar_series_cpu, 1))
bar_width_gpu = min_spacing_gpu * (0.8 / max(bar_series_gpu, 1))

# Calculate speedup (torch time / backend time)
speedup_torch_cpu = cpu_torch_times / cpu_cpu_times if cpu_torch_times is not None else None
speedup_torch_gpu = cuda_torch_times / cuda_cuda_times if cuda_torch_times is not None else None

# Calculate Kokkos speedup (kokkos time / backend time)
speedup_kokkos_cpu = None if kokkos_cpu_times is None else (kokkos_cpu_times / cpu_cpu_times)
speedup_kokkos_gpu = None if kokkos_gpu_times is None else (kokkos_gpu_times / cuda_cuda_times)
speedup_kokkos_cpu_fused = None if kokkos_cpu_fused_times is None else (kokkos_cpu_fused_times / cpu_cpu_times)
speedup_kokkos_gpu_fused = None if kokkos_gpu_fused_times is None else (kokkos_gpu_fused_times / cuda_cuda_times)

# Calculate Taichi speedup (taichi time / backend time)
speedup_taichi_cpu = None if taichi_cpu_times is None else (taichi_cpu_times / cpu_cpu_times)
speedup_taichi_gpu = None if taichi_gpu_times is None else (taichi_gpu_times / cuda_cuda_times)
speedup_taichi_cpu_fused = None if taichi_cpu_fused_times is None else (taichi_cpu_fused_times / cpu_cpu_times)
speedup_taichi_gpu_fused = None if taichi_gpu_fused_times is None else (taichi_gpu_fused_times / cuda_cuda_times)


def _max_speedup_bar_height(*arrays):
    """Max height across optional speedup arrays (for y-axis limits)."""
    vals = []
    for a in arrays:
        if a is not None:
            vals.extend(np.asarray(a, dtype=float).ravel())
    return float(max(vals)) if vals else 1.0


# ========== CPU Comparison Plot ==========
fig1, ax1 = plt.subplots(figsize=(9, 6))

# Left y-axis: Speedup histogram (bar chart)
# Use steelblue to match Torch time line color
# color_speedup_torch = 'green'
ax1.set_xlabel('n', fontsize=18)
ax1.set_ylabel('Speedup', color='black', fontsize=18)
bars_torch = None
if speedup_torch_cpu is not None:
    bars_torch = ax1.bar(
        n_cpu_data, speedup_torch_cpu, width=bar_width_cpu, alpha=0.7,
        color=COLOR_TORCH,
        edgecolor='black', linewidth=1.2, label='Speedup (Torch/Easier)')
ax1.tick_params(axis='y', labelsize=14,
                labelcolor='black')
ax1.tick_params(axis='x', labelsize=14)
ax1.set_xticks(n_cpu_data)
ax1.set_xticklabels(n_cpu_data)
ax1.grid(True, alpha=0.3, axis='y')
max_speedup = _max_speedup_bar_height(
    speedup_torch_cpu, speedup_kokkos_cpu, speedup_kokkos_cpu_fused,
    speedup_taichi_cpu, speedup_taichi_cpu_fused,
)
ax1.set_ylim(0, max_speedup * 1.15)  # Add 15% padding at top

# Calculate text offset to avoid overlap (2% of y-axis range)
y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
text_offset = y_range * 0.02

# Add value labels on top of each bar
if bars_torch is not None:
    for bar in bars_torch:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + text_offset,
                 f'{height:.2f}x',
                 ha='center', va='bottom', fontsize=14,color=COLOR_TORCH, fontweight='bold')

# Speedup bars: Kokkos, Taichi, fused Kokkos, fused Taichi (after Torch if present)
bar_offset_cpu = bar_width_cpu if speedup_torch_cpu is not None else 0
if speedup_kokkos_cpu is not None:
    bars_kokkos = ax1.bar([
        x + bar_offset_cpu for x in n_cpu_data], speedup_kokkos_cpu, width=bar_width_cpu,
        alpha=0.7, color=COLOR_KOKKOS_SIMPLE,
        edgecolor='black', linewidth=1.2, label='Speedup (Kokkos/Easier)')
    for bar in bars_kokkos:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + text_offset,
                 f'{height:.2f}x',
                 ha='center', va='bottom', fontsize=14,color=COLOR_KOKKOS_SIMPLE, fontweight='bold')
    bar_offset_cpu += bar_width_cpu

if speedup_taichi_cpu is not None:
    bars_taichi = ax1.bar(
        [x + bar_offset_cpu for x in n_cpu_data], speedup_taichi_cpu, width=bar_width_cpu,
        alpha=0.7, color=COLOR_TAICHI_SIMPLE,
        edgecolor='black', linewidth=1.2, label='Speedup (Taichi/Easier)')
    # for bar in bars_taichi:
    #     height = bar.get_height()
    #     ax1.text(bar.get_x() + bar.get_width()/2., height + text_offset,
    #              f'{height:.2f}x',
    #              ha='center', va='bottom', fontsize=14,color=COLOR_TAICHI_SIMPLE, fontweight='bold')
    bar_offset_cpu += bar_width_cpu

if speedup_kokkos_cpu_fused is not None:
    bars_kokkos_fused = ax1.bar([
        x + bar_offset_cpu for x in n_cpu_data], speedup_kokkos_cpu_fused, width=bar_width_cpu,
        alpha=0.7, color=COLOR_KOKKOS_FUSED,
        edgecolor='black', linewidth=1.2, label='Speedup (Kokkos Fused/Easier)')
    for bar in bars_kokkos_fused:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + text_offset,
                 f'{height:.2f}x',
                 ha='center', va='bottom', fontsize=14,color=COLOR_KOKKOS_FUSED, fontweight='bold')
    bar_offset_cpu += bar_width_cpu

if speedup_taichi_cpu_fused is not None:
    bars_taichi_fused = ax1.bar(
        [x + bar_offset_cpu for x in n_cpu_data], speedup_taichi_cpu_fused, width=bar_width_cpu,
        alpha=0.7, color=COLOR_TAICHI_FUSED,
        edgecolor='black', linewidth=1.2, label='Speedup (Taichi Fused/Easier)')
    # for bar in bars_taichi_fused:
    #     height = bar.get_height()
    #     ax1.text(bar.get_x() + bar.get_width()/2., height + text_offset,
    #              f'{height:.2f}x',
    #              ha='center', va='bottom', fontsize=14,color=COLOR_TAICHI_FUSED, fontweight='bold')

# Right y-axis: Time line plots
ax2 = ax1.twinx()
color_cpu = 'green'
color_torch = 'steelblue'
ax2.set_ylabel('Time (seconds)', fontsize=18)

# Plot CPU time
line1 = ax2.plot(n_cpu_data, cpu_cpu_times, marker='o', linestyle='-', 
                 color=color_cpu, linewidth=2, markersize=8, alpha=0.7, 
                 markerfacecolor=color_cpu, markeredgecolor=color_cpu, label='Easier')

# Plot Torch time if available
if cpu_torch_times is not None:
    ax2.plot(n_cpu_data, cpu_torch_times, marker='s', linestyle='--',
             color=color_torch, linewidth=2, markersize=8, alpha=0.7,
             markerfacecolor=color_torch, markeredgecolor=color_torch, label='Torch')

# Plot Kokkos time if available
if kokkos_cpu_times is not None:
    line3 = ax2.plot(n_cpu_data, kokkos_cpu_times, marker='^', linestyle='-.',
                     color=COLOR_KOKKOS_SIMPLE, linewidth=2, markersize=8, alpha=0.7,
                     markerfacecolor=COLOR_KOKKOS_SIMPLE, markeredgecolor=COLOR_KOKKOS_SIMPLE, label='Kokkos')

# # Plot Taichi time if available
# if taichi_cpu_times is not None:
#     line4 = ax2.plot(n_cpu_data, taichi_cpu_times, marker='D', linestyle=':',
#                      color=COLOR_TAICHI_SIMPLE, linewidth=2, markersize=7, alpha=0.7,
#                      markerfacecolor=COLOR_TAICHI_SIMPLE, markeredgecolor=COLOR_TAICHI_SIMPLE, label='Taichi')

# Plot Kokkos fused time if available
if kokkos_cpu_fused_times is not None:
    line3_fused = ax2.plot(n_cpu_data, kokkos_cpu_fused_times, marker='v', linestyle='-.',
                           color=COLOR_KOKKOS_FUSED, linewidth=2, markersize=8, alpha=0.7,
                           markerfacecolor=COLOR_KOKKOS_FUSED, markeredgecolor=COLOR_KOKKOS_FUSED, label='Kokkos Fused')

# Plot Taichi fused time if available
if taichi_cpu_fused_times is not None:
    line4_fused = ax2.plot(n_cpu_data, taichi_cpu_fused_times, marker='X', linestyle=':',
                           color=COLOR_TAICHI_FUSED, linewidth=2, markersize=7, alpha=0.7,
                           markerfacecolor=COLOR_TAICHI_FUSED, markeredgecolor=COLOR_TAICHI_FUSED, label='Taichi Fused')

ax2.tick_params(axis='y', labelsize=14)

# Combine legends and place at bottom outside figure
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, 
           loc='upper center', bbox_to_anchor=(0.5, -0.15),
           fontsize=12, ncol=3, frameon=True)

# Adjust layout to make room for legend
plt.tight_layout(rect=[0, 0.05, 1, 1])

# Save CPU figure
output_file_cpu = os.path.join(script_dir, 'timing_visualization_cpu.pdf')
plt.savefig(output_file_cpu, dpi=300, bbox_inches='tight')
print(f"CPU visualization saved to: {output_file_cpu}")

# ========== GPU Comparison Plot ==========
fig2, ax1_gpu = plt.subplots(figsize=(9, 6))

# Left y-axis: Speedup histogram (bar chart)
# Use steelblue to match Torch time line color
# color_speedup_torch_gpu = 'steelblue'
ax1_gpu.set_xlabel('n', fontsize=18)
ax1_gpu.set_ylabel('Speedup', color='black', fontsize=18)
bars_torch_gpu = None
if speedup_torch_gpu is not None:
    bars_torch_gpu = ax1_gpu.bar(
        n_gpu_data, speedup_torch_gpu, width=bar_width_gpu, alpha=0.7,
        color=COLOR_TORCH,
        edgecolor='black', linewidth=1.2, label='Speedup (Torch/Easier)')
ax1_gpu.tick_params(axis='y', labelsize=14,
                    labelcolor='black')
ax1_gpu.tick_params(axis='x', labelsize=14)
ax1_gpu.set_xticks(n_gpu_data)
ax1_gpu.set_xticklabels(n_gpu_data)
ax1_gpu.grid(True, alpha=0.3, axis='y')
max_speedup_gpu = _max_speedup_bar_height(
    speedup_torch_gpu, speedup_kokkos_gpu, speedup_kokkos_gpu_fused,
    speedup_taichi_gpu, speedup_taichi_gpu_fused,
)
ax1_gpu.set_ylim(0, max_speedup_gpu * 1.15)  # Add 15% padding at top

# Calculate text offset to avoid overlap (2% of y-axis range)
y_range_gpu = ax1_gpu.get_ylim()[1] - ax1_gpu.get_ylim()[0]
text_offset_gpu = y_range_gpu * 0.02

# Add value labels on top of each bar
if bars_torch_gpu is not None:
    for bar in bars_torch_gpu:
        height = bar.get_height()
        ax1_gpu.text(bar.get_x() + bar.get_width()/2., height + text_offset_gpu,
                     f'{height:.2f}x',
                     ha='center', va='bottom', fontsize=12, color=COLOR_TORCH, fontweight='bold')

# Speedup bars: Kokkos, Taichi, fused Kokkos, fused Taichi (after Torch if present)
bar_offset_gpu = bar_width_gpu if speedup_torch_gpu is not None else 0
if speedup_kokkos_gpu is not None:
    bars_kokkos_gpu = ax1_gpu.bar([
        x + bar_offset_gpu for x in n_gpu_data], speedup_kokkos_gpu, width=bar_width_gpu,
        alpha=0.7, color=COLOR_KOKKOS_SIMPLE,
        edgecolor='black', linewidth=1.2, label='Speedup (Kokkos/Easier)')
    for bar in bars_kokkos_gpu:
        height = bar.get_height()
        ax1_gpu.text(bar.get_x() + bar.get_width()/2., height + text_offset_gpu,
                     f'{height:.2f}x',
                     ha='center', va='bottom', fontsize=12, color=COLOR_KOKKOS_SIMPLE, fontweight='bold')
    bar_offset_gpu += bar_width_gpu

if speedup_taichi_gpu is not None:
    bars_taichi_gpu = ax1_gpu.bar(
        [x + bar_offset_gpu for x in n_gpu_data], speedup_taichi_gpu, width=bar_width_gpu,
        alpha=0.7, color=COLOR_TAICHI_SIMPLE,
        edgecolor='black', linewidth=1.2, label='Speedup (Taichi/Easier)')
    # for bar in bars_taichi_gpu:
    #     height = bar.get_height()
    #     ax1_gpu.text(bar.get_x() + bar.get_width()/2., height + text_offset_gpu,
    #                  f'{height:.2f}x',
    #                  ha='center', va='bottom', fontsize=12, color=COLOR_TAICHI_SIMPLE, fontweight='bold')
    bar_offset_gpu += bar_width_gpu

if speedup_kokkos_gpu_fused is not None:
    bars_kokkos_gpu_fused = ax1_gpu.bar([
        x + bar_offset_gpu for x in n_gpu_data], speedup_kokkos_gpu_fused, width=bar_width_gpu,
        alpha=0.7, color=COLOR_KOKKOS_FUSED,
        edgecolor='black', linewidth=1.2, label='Speedup (Kokkos Fused/Easier)')
    for bar in bars_kokkos_gpu_fused:
        height = bar.get_height()
        ax1_gpu.text(bar.get_x() + bar.get_width()/2., height + text_offset_gpu,
                     f'{height:.2f}x',
                     ha='center', va='bottom', fontsize=12, color=COLOR_KOKKOS_FUSED, fontweight='bold')
    bar_offset_gpu += bar_width_gpu

if speedup_taichi_gpu_fused is not None:
    bars_taichi_gpu_fused = ax1_gpu.bar(
        [x + bar_offset_gpu for x in n_gpu_data], speedup_taichi_gpu_fused, width=bar_width_gpu,
        alpha=0.7, color=COLOR_TAICHI_FUSED,
        edgecolor='black', linewidth=1.2, label='Speedup (Taichi Fused/Easier)')
    # for bar in bars_taichi_gpu_fused:
    #     height = bar.get_height()
    #     ax1_gpu.text(bar.get_x() + bar.get_width()/2., height + text_offset_gpu,
    #                  f'{height:.2f}x',
    #                  ha='center', va='bottom', fontsize=12, color=COLOR_TAICHI_FUSED, fontweight='bold')

# Right y-axis: Time line plots
ax2_gpu = ax1_gpu.twinx()
color_cuda = 'green'
color_torch_gpu = 'steelblue'
ax2_gpu.set_ylabel('Time (seconds)', fontsize=18)

# Plot CUDA time
line1_gpu = ax2_gpu.plot(n_gpu_data, cuda_cuda_times, marker='o', linestyle='-', 
                         color=color_cuda, linewidth=2, markersize=8, alpha=0.7,
                         markerfacecolor=color_cuda, markeredgecolor=color_cuda, label='Easier')

# Plot Torch time if available
if cuda_torch_times is not None:
    ax2_gpu.plot(
        n_gpu_data, cuda_torch_times, marker='s', linestyle='--',
        color=color_torch_gpu, linewidth=2, markersize=8, alpha=0.7,
        markerfacecolor=color_torch_gpu, markeredgecolor=color_torch_gpu, label='Torch')

# Plot Kokkos time if available
if kokkos_gpu_times is not None:
    line3_gpu = ax2_gpu.plot(n_gpu_data, kokkos_gpu_times, marker='^', linestyle='-.',
                             color=COLOR_KOKKOS_SIMPLE, linewidth=2, markersize=8, alpha=0.7,
                             markerfacecolor=COLOR_KOKKOS_SIMPLE, markeredgecolor=COLOR_KOKKOS_SIMPLE, label='Kokkos')

# Plot Taichi time if available
if taichi_gpu_times is not None:
    line4_gpu = ax2_gpu.plot(n_gpu_data, taichi_gpu_times, marker='D', linestyle=':',
                             color=COLOR_TAICHI_SIMPLE, linewidth=2, markersize=7, alpha=0.7,
                             markerfacecolor=COLOR_TAICHI_SIMPLE, markeredgecolor=COLOR_TAICHI_SIMPLE, label='Taichi')

# Plot Kokkos fused time if available
if kokkos_gpu_fused_times is not None:
    line3_gpu_fused = ax2_gpu.plot(n_gpu_data, kokkos_gpu_fused_times, marker='v', linestyle='-.',
                                    color=COLOR_KOKKOS_FUSED, linewidth=2, markersize=8, alpha=0.7,
                                    markerfacecolor=COLOR_KOKKOS_FUSED, markeredgecolor=COLOR_KOKKOS_FUSED, label='Kokkos Fused')

# Plot Taichi fused time if available
if taichi_gpu_fused_times is not None:
    line4_gpu_fused = ax2_gpu.plot(n_gpu_data, taichi_gpu_fused_times, marker='X', linestyle=':',
                                    color=COLOR_TAICHI_FUSED, linewidth=2, markersize=7, alpha=0.7,
                                    markerfacecolor=COLOR_TAICHI_FUSED, markeredgecolor=COLOR_TAICHI_FUSED, label='Taichi Fused')

ax2_gpu.tick_params(axis='y', labelsize=14)

# Combine legends and place at bottom outside figure
lines1_gpu, labels1_gpu = ax1_gpu.get_legend_handles_labels()
lines2_gpu, labels2_gpu = ax2_gpu.get_legend_handles_labels()
ax1_gpu.legend(lines1_gpu + lines2_gpu, labels1_gpu + labels2_gpu, 
               loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fontsize=12, ncol=3, frameon=True)

# Adjust layout to make room for legend
plt.tight_layout(rect=[0, 0.05, 1, 1])

# Save GPU figure
output_file_gpu = os.path.join(script_dir, 'timing_visualization_gpu.pdf')
plt.savefig(output_file_gpu, dpi=300, bbox_inches='tight')
print(f"GPU visualization saved to: {output_file_gpu}")

# Also display
plt.show()

