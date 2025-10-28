# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import tempfile
import shutil
import hashlib
import concurrent.futures
import torch
import fcntl
import contextlib
import easier.core.module as esr

from torch.utils.cpp_extension import load
from typing import List, Tuple, Dict

from torch.fx.node import Node
from easier.core.runtime.modules import HaloExchanger
from easier.core.passes.utils import get_called_module, FX
# from easier.core.runtime.metadata import is_node_skipped
from easier.core.passes.code_gen.python_wrapper.dynamic_replace_submodule import dynamic_replace_submodule
from easier.core.passes.code_gen.merged_gen_gpu import generate_cuda_code_from_graph
from easier.core.passes.code_gen.merged_gen_cpu import generate_cpu_code_from_graph
from easier.core.utils import logger

def _is_node_halo_exchanger(root: esr.Module, node: Node) -> bool:
    if node.op == FX.CALL_MODULE:
        submod = get_called_module(root, node)
        if isinstance(submod, HaloExchanger):
            return True
    return False

def _repo_root() -> str:
    # this file: EASIER/easier/core/passes/codegen.py -> easier/core/passes
    # project root for generated files/templates: EASIER/easier/core/passes/code_gen
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "passes", "code_gen"))


def _hash_directory_tree(root_dir: str) -> str:
    sha1 = hashlib.sha1()
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        sha1.update(chunk)
            except Exception:
                continue
    return sha1.hexdigest()[:12]


@contextlib.contextmanager
def _codegen_lock():
    """Serialize code generation across threads/processes to avoid file clobbering."""
    lock_file = os.path.join("/tmp", "easier_codegen.lock")
    # Ensure directory exists; _repo_root should already exist
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _snapshot_generated_source_cuda(suffix: str) -> Tuple[str, str, str]:
    project_root = _repo_root()
    src = os.path.join(project_root, "merged_binding.cu")
    inc_dir = os.path.join(project_root, "include")
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    tmp_dir = tempfile.mkdtemp(prefix=f"easier_src_{suffix}_")
    dst_src = os.path.join(tmp_dir, f"merged_binding_{suffix}.cu")
    shutil.copyfile(src, dst_src)
    dst_inc = os.path.join(tmp_dir, "include")
    shutil.copytree(inc_dir, dst_inc)
    try:
        with open(dst_src, "rb") as f:
            src_bytes = f.read()
        src_hash = hashlib.sha1(src_bytes).hexdigest()[:12]
    except Exception:
        src_hash = "nohash"
    dir_hash = _hash_directory_tree(dst_inc)
    combined_hash = hashlib.sha1((src_hash + dir_hash).encode()).hexdigest()[:12]
    return dst_src, dst_inc, combined_hash


def _snapshot_generated_source_cpu(suffix: str) -> Tuple[str, str, str]:
    project_root = _repo_root()
    src = os.path.join(project_root, "merged_binding_cpu.cpp")
    gen_hdr = os.path.join(project_root, "merged_spmv.h")
    # Ensure the generated header exists before snapshot
    if not os.path.exists(gen_hdr):
        raise FileNotFoundError(f"Generated header not found: {gen_hdr}")
    shared_hdr = os.path.join(project_root, "data_struct_shared.cuh")
    inc_dir = os.path.join(project_root, "include")
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    tmp_dir = tempfile.mkdtemp(prefix=f"easier_src_cpu_{suffix}_")
    dst_src = os.path.join(tmp_dir, f"merged_binding_cpu_{suffix}.cpp")
    shutil.copyfile(src, dst_src)
    if os.path.exists(gen_hdr):
        shutil.copyfile(gen_hdr, os.path.join(tmp_dir, "merged_spmv.h"))
    if os.path.exists(shared_hdr):
        shutil.copyfile(shared_hdr, os.path.join(tmp_dir, "data_struct_shared.cuh"))
    dst_inc = os.path.join(tmp_dir, "include")
    shutil.copytree(inc_dir, dst_inc)
    try:
        with open(dst_src, "rb") as f:
            src_bytes = f.read()
        src_hash = hashlib.sha1(src_bytes).hexdigest()[:12]
    except Exception:
        src_hash = "nohash"
    dir_hash = _hash_directory_tree(dst_inc)
    combined_hash = hashlib.sha1(("cpu:" + src_hash + dir_hash).encode()).hexdigest()[:12]
    return dst_src, dst_inc, combined_hash


def _nvcc_threads_flag() -> str:
    try:
        default_threads = min(4, os.cpu_count() or 1)
        n = max(1, int(os.getenv("NVCC_THREADS", default_threads)))
    except Exception:
        n = min(4, os.cpu_count() or 1)
    return f"--threads={n}"


def _build_cuda_extension_from_src(name: str, src_path: str, snapshot_include_dir: str, hash_key: str):
    logger.info(f"Building CUDA extension from snapshot: {src_path}")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    if shutil.which("sccache"):
        os.environ.setdefault("CMAKE_C_COMPILER_LAUNCHER", "sccache")
        os.environ.setdefault("CMAKE_CXX_COMPILER_LAUNCHER", "sccache")
        os.environ.setdefault("CMAKE_CUDA_COMPILER_LAUNCHER", "sccache")
    elif shutil.which("ccache"):
        os.environ.setdefault("CMAKE_C_COMPILER_LAUNCHER", "ccache")
        os.environ.setdefault("CMAKE_CXX_COMPILER_LAUNCHER", "ccache")
        os.environ.setdefault("CMAKE_CUDA_COMPILER_LAUNCHER", "ccache")
    if "MAX_JOBS" not in os.environ and os.cpu_count():
        os.environ["MAX_JOBS"] = str(min(8, os.cpu_count()))

    code_hash = hash_key or "nohash"
    ext_name = f"new_{name}_{code_hash}"

    ext = load(
        name=ext_name,
        sources=[src_path],
        extra_include_paths=[snapshot_include_dir, _repo_root()],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            _nvcc_threads_flag(),
            "-DNDEBUG",
        ],
        build_directory=os.path.dirname(src_path),
        keep_intermediates=False,
        verbose=False,
    )
    return ext


def _build_cpu_extension_from_src(name: str, src_path: str, snapshot_include_dir: str, hash_key: str):
    logger.info(f"Building CPU extension from snapshot: {src_path}")
    if shutil.which("sccache"):
        os.environ.setdefault("CMAKE_C_COMPILER_LAUNCHER", "sccache")
        os.environ.setdefault("CMAKE_CXX_COMPILER_LAUNCHER", "sccache")
    elif shutil.which("ccache"):
        os.environ.setdefault("CMAKE_C_COMPILER_LAUNCHER", "ccache")
        os.environ.setdefault("CMAKE_CXX_COMPILER_LAUNCHER", "ccache")
    if "MAX_JOBS" not in os.environ and os.cpu_count():
        os.environ["MAX_JOBS"] = str(min(8, os.cpu_count()))

    code_hash = hash_key or "nohash"
    ext_name = f"new_cpu_{name}_{code_hash}"

    ext = load(
        name=ext_name,
        sources=[src_path],
        extra_include_paths=[os.path.dirname(src_path), snapshot_include_dir, _repo_root()],
        extra_cflags=["-O3", "-fopenmp", "-DNDEBUG", "-iquote", os.path.dirname(src_path)],
        extra_ldflags=["-fopenmp"],
        build_directory=os.path.dirname(src_path),
        keep_intermediates=False,
        verbose=False,
    )
    return ext


def _collect_fused_call_modules(module, graph) -> List[Tuple[str, object]]:
    callmods: List[Tuple[str, object]] = []
    for node in graph.nodes:
        if node.op == 'call_module':
            if _is_node_halo_exchanger(module, node):
                continue
            submod = get_called_module(module, node)
            callmods.append((node.name, node, submod))
    return callmods


def code_generation(ms: List[object], gs: List[object]) -> Tuple[List[object], List[object]]:
    """
    Generate backend-specific code for each fused FX submodule, build extensions,
    and replace the submodules with dynamic wrappers that call the compiled kernels.
    """
    assert len(ms) == len(gs)

    # Collect build jobs across modules first (generation + snapshot)
    build_jobs: List[Tuple[str, str, str, str, object]] = []
    
    logger.info("================================================")
    for m, g in zip(ms, gs):
        backend = getattr(m, 'easier_jit_backend', 'torch')
        logger.info(f"Generating code for {backend} backend")
        if backend not in ['cuda', 'cpu']:
            logger.info(f"Skipping module with backend {backend}")
            continue

        callmods = _collect_fused_call_modules(m, g)

        # Generate per submodule and snapshot immediately to avoid file clobbering
        for node_name, node, submod in callmods:
            logger.info(f"Node: {node_name}")
            # submod.graph.print_tabular()
            if backend == 'cuda':
                # Serialize only the codegen + snapshot to avoid shared-file contention
                with _codegen_lock():
                    generate_cuda_code_from_graph(submod, g)
                    snap_src, snap_inc, combined_hash = _snapshot_generated_source_cuda(node_name)
            elif backend == 'cpu':
                # Serialize only the codegen + snapshot to avoid shared-file contention
                with _codegen_lock():
                    generate_cpu_code_from_graph(submod, g)
                    snap_src, snap_inc, combined_hash = _snapshot_generated_source_cpu(node_name)
            elif backend == 'torch':
                continue
            else:
                raise ValueError(f"Invalid backend {backend}")
            build_jobs.append((backend, node_name, snap_src, snap_inc, combined_hash))

    # Build in parallel per job
    logger.info("================================================")
    logger.info("Building extensions in parallel")
    built_extensions: Dict[str, object] = {}
    if build_jobs:
        n_workers = min(len(build_jobs), max(1, (os.cpu_count() or 2) // 2))
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_map = {}
            for backend, node_name, snap_src, snap_inc, h in build_jobs:
                if backend == 'cuda':
                    fut = executor.submit(_build_cuda_extension_from_src, node_name, snap_src, snap_inc, h)
                elif backend == 'cpu':
                    fut = executor.submit(_build_cpu_extension_from_src, node_name, snap_src, snap_inc, h)
                elif backend == 'torch':
                    continue
                else:
                    raise ValueError(f"Invalid backend {backend}")
                future_map[node_name] = fut
            for node_name, fut in future_map.items():
                built_extensions[node_name] = fut.result()

    # Replace submodules sequentially to preserve semantics
    logger.info("================================================")
    logger.info("Replacing submodules sequentially to preserve semantics")
    for m, g in zip(ms, gs):
        backend = getattr(m, 'easier_jit_backend', 'torch')
        if backend not in ['cuda', 'cpu']:
            continue
        for node in g.nodes:
            if node.op != 'call_module':
                continue
            if _is_node_halo_exchanger(m, node):
                continue
            ext = built_extensions.get(node.name)
            if ext is None:
                continue
            ok = dynamic_replace_submodule(m, g, ext, node.name)
            if not ok:
                raise RuntimeError(f"Failed to replace submodule {node.name}")
    logger.info("================================================")
    return ms, gs


