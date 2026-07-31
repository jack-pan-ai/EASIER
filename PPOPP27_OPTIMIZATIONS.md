# PPoPP 2027 merged-SPMV framework snapshot

This branch isolates the generic EASIER optimization work that was developed
against `4773b73bfdc38df117bf699a4cfd856d3562909a`. It is source-only: datasets,
JIT caches, timing logs, and generated benchmark results belong under the Coder
profile paths in `/home/qpan/workspace/coder/{data,cache,tmp}`.

## Generic CUDA paths

- A direct-map kernel handles eligible selector/map graphs with full-overwrite
  outputs without merge-path temporary storage or carry fixup.
- A register-fed merge path keeps the pure single-reducer evaluator in the
  merged kernel, emits compact carry state, stores completed rows directly, and
  limits fixup to tile boundaries. Set `EASIER_CUDA_REGISTER_FED_MERGE=0` to
  disable it; it is enabled by default.
- Merge-path generation caches immutable tile coordinates and avoids redundant
  output initialization/allocation where write semantics permit forwarding.
- FP64 merge tile width is selected with
  `EASIER_CUDA_F64_ITEMS_PER_THREAD` (default `6`, valid range `1..16`). The
  value and target CUDA architecture are part of the JIT build identity.

## Runtime and compilation

The JIT source snapshots and persistent cache are content/build keyed and
architecture namespaced. Optional whole-module CUDA graph replay is controlled
by `EASIER_CUDA_GRAPH_MODULE_REPLAY=1`. The single-rank sparse-encoding path
avoids distributed reorder/halo work when `EASIER_SINGLE_RANK_FAST_PATH=1`.

## Tests carried with this snapshot

- `test_cuda_codegen_binding.py`: direct-map and register-fed eligibility,
  output forwarding, carry layout, launch, and int64 coordinate generation.
- `test_cuda_policy_target_arch.py`: target architecture, FP64 policy override,
  build fingerprint, and cache namespace behavior.
- `test_sparse_encoding_single_rank.py`: reducer/selector single-rank rewrites.
- `test_codegen_snapshot_isolation.py`: CPU snapshots exclude mutable CUDA
  generated includes.

## Deliberate exclusions

The following mutable generated scratch artifacts remain exactly at the base
commit and are not part of this branch's optimization source:

- `easier/core/passes/code_gen/merged_binding.cu`
- `easier/core/passes/code_gen/merged_binding_cpu.cpp`
- `easier/core/passes/code_gen/merged_spmv.h`

Regenerate these through EASIER codegen for the graph under test. Do not treat a
scratch rendering as the authoritative generic implementation; the Python
generators and `*_template` sources are authoritative.
