# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import json
import operator
import platform
import tempfile
import shutil
import hashlib
import importlib.util
import concurrent.futures
import torch
import fcntl
import contextlib
import easier.core.module as esr

from torch.utils.cpp_extension import load
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from types import ModuleType

from torch.fx.node import Node
from easier.core.runtime.metadata import (
    ViewSrc,
    collect_meta,
    get_node_view_src,
    set_node_view_src,
)
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


@dataclass(frozen=True)
class _ForwardedReducerOutput:
    output_index: int
    destination: Node
    getitem_node: Node
    setitem_node: Node


def _is_full_overwrite_index(index) -> bool:
    if index is Ellipsis:
        return True
    if isinstance(index, slice):
        return (
            index.start is None
            and index.stop is None
            and index.step is None
        )
    if isinstance(index, tuple):
        return bool(index) and all(
            _is_full_overwrite_index(item) for item in index
        )
    return False


def _same_tensor_source(lhs: Node, rhs: Node) -> bool:
    if lhs is rhs:
        return True
    return (
        lhs.op == FX.GET_ATTR
        and rhs.op == FX.GET_ATTR
        and lhs.target == rhs.target
    )


def _tensor_sources_may_alias(lhs: Node, rhs: Node) -> bool:
    """Use runtime allocator metadata to reject output-forwarding aliases."""
    try:
        lhs_sources = set(collect_meta(
            get_node_view_src(lhs), leaf_type=ViewSrc
        ))
        rhs_sources = set(collect_meta(
            get_node_view_src(rhs), leaf_type=ViewSrc
        ))
    except KeyError:
        # Small graph-only tests may not run metadata propagation.  Retain the
        # original syntactic fallback there; production code has ViewSrcs.
        return _same_tensor_source(lhs, rhs)

    # Tensor-valued fused inputs and destinations must have an allocator
    # source.  If metadata is incomplete, preserve the copy instead of trying
    # to prove non-aliasing from node names.
    if not lhs_sources or not rhs_sources:
        return True
    return not lhs_sources.isdisjoint(rhs_sources)


def _find_forwarded_reducer_outputs(
    submod, graph, call_node: Node
) -> List[_ForwardedReducerOutput]:
    """Find safe ``dst[:] = fused_reducer(...)`` output forwarding.

    Forwarding is deliberately conservative.  The destination must cover the
    complete output, must not also feed the fused kernel, and no computation
    may occur between the fused call and the assignment.  All rejected cases
    retain the existing allocate-then-copy behavior.
    """
    inner_output = next(
        (node for node in submod.graph.nodes if node.op == FX.OUTPUT),
        None,
    )
    if inner_output is None:
        return []
    output_items = inner_output.args[0]
    if not isinstance(output_items, (list, tuple)):
        output_items = (output_items,)

    call_inputs = list(call_node.all_input_nodes)
    candidates: List[_ForwardedReducerOutput] = []
    for getitem_node in list(call_node.users):
        if (
            getitem_node.op != FX.CALL_FUNCTION
            or getitem_node.target is not operator.getitem
            or len(getitem_node.args) != 2
            or not isinstance(getitem_node.args[1], int)
        ):
            continue
        output_index = getitem_node.args[1]
        if output_index < 0 or output_index >= len(output_items):
            continue
        inner_item = output_items[output_index]
        if not (
            isinstance(inner_item, Node)
            and inner_item.op == FX.CALL_MODULE
            and isinstance(get_called_module(submod, inner_item), esr.Reducer)
        ):
            continue
        users = list(getitem_node.users)
        if len(users) != 1:
            continue
        setitem_node = users[0]
        if (
            setitem_node.op != FX.CALL_FUNCTION
            or setitem_node.target is not operator.setitem
            or len(setitem_node.args) < 3
            or setitem_node.args[2] is not getitem_node
            or not _is_full_overwrite_index(setitem_node.args[1])
            or len(setitem_node.users) != 0
        ):
            continue
        destination = setitem_node.args[0]
        if not isinstance(destination, Node):
            continue
        if any(
            _tensor_sources_may_alias(destination, input_node)
            for input_node in call_inputs
        ):
            continue
        candidates.append(_ForwardedReducerOutput(
            output_index=output_index,
            destination=destination,
            getitem_node=getitem_node,
            setitem_node=setitem_node,
        ))

    # One generated output pointer cannot represent two assignments, and two
    # reducer outputs must not be forwarded to the same destination.
    if len({item.output_index for item in candidates}) != len(candidates):
        return []
    destinations = [
        (item.destination.op, item.destination.target)
        for item in candidates
    ]
    if len(set(destinations)) != len(destinations):
        return []
    for i, lhs in enumerate(candidates):
        if any(
            _tensor_sources_may_alias(lhs.destination, rhs.destination)
            for rhs in candidates[i + 1:]
        ):
            return []

    graph_nodes = list(graph.nodes)
    node_position = {node: idx for idx, node in enumerate(graph_nodes)}
    syntactic_nodes = {
        user
        for user in call_node.users
        if (
            user.op == FX.CALL_FUNCTION
            and user.target is operator.getitem
        )
    }
    syntactic_nodes.update(item.setitem_node for item in candidates)
    safe_candidates = []
    for item in candidates:
        between = graph_nodes[
            node_position[call_node] + 1:
            node_position[item.setitem_node]
        ]
        if all(node in syntactic_nodes for node in between):
            safe_candidates.append(item)

    return sorted(safe_candidates, key=lambda item: item.output_index)


def _apply_forwarded_reducer_outputs(
    graph, call_node: Node, forwarding: List[_ForwardedReducerOutput]
) -> None:
    if not forwarding:
        return

    # The compiled wrapper returns each forwarded destination to preserve its
    # existing tuple ABI.  Teach the runtime allocator validator that those
    # tuple items alias the destination's established storage rather than a
    # fresh allocation by the fused call.
    try:
        call_view_srcs = list(get_node_view_src(call_node))
    except KeyError:
        # Pure graph unit tests do not run metadata propagation.
        call_view_srcs = []
    if call_view_srcs:
        for item in forwarding:
            destination_view_src = get_node_view_src(item.destination)
            if not isinstance(destination_view_src, ViewSrc):
                raise RuntimeError(
                    "forwarded reducer destination must have one ViewSrc"
                )
            call_view_srcs[item.output_index] = destination_view_src
        set_node_view_src(call_node, call_view_srcs)

    call_node.args = tuple(call_node.args) + tuple(
        item.destination for item in forwarding
    )
    for item in forwarding:
        graph.erase_node(item.setitem_node)
        graph.erase_node(item.getitem_node)
    graph.lint()

def _repo_root() -> str:
    # this file: EASIER/easier/core/passes/codegen.py -> easier/core/passes
    # project root for generated files/templates: EASIER/easier/core/passes/code_gen
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "passes", "code_gen"))


def _jit_cache_requested() -> bool:
    # Keep the historical (inverted) switch as a compatibility alias, but the
    # cache below is now content-addressed.  It never bypasses source hashing.
    truthy = {"1", "true", "yes", "on"}
    explicit = os.getenv("EASIER_JIT_CACHE", "").strip().lower()
    legacy = os.getenv("EASIER_DISABLE_JIT_HASH", "").strip().lower()
    return explicit in truthy or legacy in truthy


def _cache_root_dir() -> str:
    root = os.getenv(
        "EASIER_JIT_CACHE_ROOT",
        os.path.join(tempfile.gettempdir(), "easier_codegen_cache"),
    )
    root = os.path.abspath(os.path.expanduser(root))
    os.makedirs(root, exist_ok=True)
    return root


def _tool_identity(executable: Optional[str]) -> Optional[dict]:
    if not executable:
        return None
    resolved = shutil.which(executable) or executable
    try:
        stat = os.stat(resolved)
    except OSError:
        return {"path": resolved, "missing": True}
    return {
        "path": os.path.realpath(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _cuda_target_architecture() -> Tuple[int, str]:
    """Return one CUDA target for flags, policy selection, and cache identity."""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
    else:
        major, minor = 7, 0
    return major * 100 + minor * 10, f"sm_{major}{minor}"


def _cuda_f64_items_per_thread() -> int:
    """Return the validated FP64 merge-tile depth for CUDA code generation."""
    raw_value = os.getenv("EASIER_CUDA_F64_ITEMS_PER_THREAD", "6")
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            "EASIER_CUDA_F64_ITEMS_PER_THREAD must be an integer"
        ) from error
    if not 1 <= value <= 16:
        raise ValueError(
            "EASIER_CUDA_F64_ITEMS_PER_THREAD must be between 1 and 16"
        )
    return value


def _driver_source_hash() -> str:
    """Fingerprint this cache/build driver so flag changes invalidate entries."""
    with open(__file__, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def _build_fingerprint(backend: str) -> str:
    """Hash build/runtime dependencies not present in generated sources."""
    payload = {
        "schema": 2,
        "backend": backend,
        # Generated sources cover the kernel.  Hashing this driver additionally
        # covers changes to compiler/linker flags and extension-loading logic,
        # even when those changes leave the emitted kernel text unchanged.
        "driver_source": _driver_source_hash(),
        "python_abi": getattr(sys.implementation, "cache_tag", None),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cxx11_abi": getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None),
        "cxx": _tool_identity(os.getenv("CXX", "c++")),
        "nvcc": _tool_identity(
            os.path.join(os.getenv("CUDA_HOME", ""), "bin", "nvcc")
            if os.getenv("CUDA_HOME") else shutil.which("nvcc")
        ),
        "index_dtype": os.getenv("EASIER_CODEGEN_INDEX_DTYPE", "int32"),
        "cuda_arch": _cuda_target_architecture()[1] if backend == "cuda" else None,
        "cuda_f64_items_per_thread": (
            _cuda_f64_items_per_thread() if backend == "cuda" else None
        ),
        "cflags": ["-O3", "-DNDEBUG"],
        "cuda_flags": ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-O3"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _verified_cache_key(backend: str, source_hash: str) -> str:
    value = f"schema=2|source={source_hash}|build={_build_fingerprint(backend)}"
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def _cache_entry_dir(backend: str, node_name: str, cache_key: str) -> str:
    safe_name = node_name.replace(os.sep, "_")
    if backend == "cuda":
        _, target = _cuda_target_architecture()
        return os.path.join(
            _cache_root_dir(), "v2", backend, target, safe_name, cache_key
        )
    return os.path.join(
        _cache_root_dir(), "v2", backend, safe_name, cache_key
    )


def _cache_snapshot_filename(backend: str, node_name: str) -> str:
    if backend == "cpu":
        return f"merged_binding_cpu_{node_name}.cpp"
    return f"merged_binding_{node_name}.cu"


def _load_cached_snapshot(
    backend: str, node_name: str, cache_key: str
) -> Optional[Tuple[str, str, str]]:
    if not _jit_cache_requested():
        return None
    cache_dir = _cache_entry_dir(backend, node_name, cache_key)
    hash_file = os.path.join(cache_dir, ".hash")
    src_file = os.path.join(cache_dir, _cache_snapshot_filename(backend, node_name))
    inc_dir = os.path.join(cache_dir, "include")
    if not (os.path.isfile(src_file) and os.path.isdir(inc_dir) and os.path.isfile(hash_file)):
        return None
    try:
        with open(hash_file, "r") as f:
            combined_hash = f.read().strip() or "nohash"
    except OSError:
        return None
    return src_file, inc_dir, combined_hash


def _persist_snapshot_cache(
    backend: str, node_name: str, cache_key: str,
    snap_src: str, snap_inc: str, combined_hash: str
) -> Optional[Tuple[str, str, str]]:
    if not _jit_cache_requested():
        return None
    cache_dir = _cache_entry_dir(backend, node_name, cache_key)
    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    snapshot_root = os.path.dirname(snap_src)
    try:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        shutil.copytree(snapshot_root, cache_dir)
        hash_file = os.path.join(cache_dir, ".hash")
        with open(hash_file, "w") as f:
            f.write(combined_hash)
    except OSError:
        logger.warning("Failed to persist snapshot cache for %s:%s", backend, node_name)
        return None
    rel_inc = os.path.relpath(snap_inc, snapshot_root)
    if rel_inc.startswith(".."):
        rel_inc = os.path.basename(snap_inc)
    cached_src = os.path.join(cache_dir, os.path.basename(snap_src))
    cached_inc = os.path.join(cache_dir, rel_inc)
    return cached_src, cached_inc, combined_hash


def _cached_extension_meta_path(
    backend: str, node_name: str, cache_key: str
) -> str:
    return os.path.join(
        _cache_entry_dir(backend, node_name, cache_key), ".ext_meta"
    )


def _persist_compiled_extension(
    backend: str, node_name: str, cache_key: str, module: ModuleType
):
    if not _jit_cache_requested():
        return
    cache_dir = _cache_entry_dir(backend, node_name, cache_key)
    so_path = getattr(module, "__file__", None)
    if not so_path or not os.path.isfile(so_path):
        return
    bin_name = os.path.basename(so_path)
    dst_path = os.path.join(cache_dir, bin_name)
    try:
        if os.path.abspath(so_path) != os.path.abspath(dst_path):
            shutil.copyfile(so_path, dst_path)
        meta = {"module_name": module.__name__, "binary": bin_name}
        with open(
            _cached_extension_meta_path(backend, node_name, cache_key), "w"
        ) as f:
            json.dump(meta, f)
    except OSError:
        logger.warning("Failed to persist compiled extension cache for %s:%s", backend, node_name)


def _load_cached_extension_module(
    backend: str, node_name: str, cache_key: str
) -> Optional[ModuleType]:
    if not _jit_cache_requested():
        return None
    cache_dir = _cache_entry_dir(backend, node_name, cache_key)
    meta_path = _cached_extension_meta_path(backend, node_name, cache_key)
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        module_name = meta.get("module_name", "")
        bin_name = meta.get("binary", "")
    except (OSError, json.JSONDecodeError):
        return None
    if not module_name or not bin_name:
        return None
    so_path = os.path.join(cache_dir, bin_name)
    if not os.path.isfile(so_path):
        return None
    if module_name in sys.modules:
        return sys.modules[module_name]
    try:
        spec = importlib.util.spec_from_file_location(module_name, so_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
        return module
    except Exception:
        logger.warning("Failed to load cached extension for %s:%s", backend, node_name, exc_info=True)
        return None


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
    shared_hdr = os.path.join(project_root, "include", "data_struct_shared.cuh")
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    tmp_dir = tempfile.mkdtemp(prefix=f"easier_src_cpu_{suffix}_")
    dst_src = os.path.join(tmp_dir, f"merged_binding_cpu_{suffix}.cpp")
    shutil.copyfile(src, dst_src)
    dst_hdr = os.path.join(tmp_dir, "merged_spmv.h")
    if os.path.exists(gen_hdr):
        shutil.copyfile(gen_hdr, dst_hdr)
    dst_inc = os.path.join(tmp_dir, "include")
    # CPU generated code includes only the generated ``merged_spmv.h`` and
    # its sibling ``data_struct_shared.cuh``.  Do not snapshot the CUDA
    # ``include/`` tree: CUDA generation rewrites several files there, which
    # otherwise makes identical CPU rebuilds depend on which backend ran
    # immediately beforehand.  The shared Tensor definition included by
    # ``merged_spmv.h`` is a real CPU dependency, so copy only that stable
    # header into the include directory expected by the build helper.
    os.makedirs(dst_inc)
    if not os.path.exists(shared_hdr):
        raise FileNotFoundError(shared_hdr)
    shutil.copyfile(shared_hdr, os.path.join(dst_inc, "data_struct_shared.cuh"))
    try:
        with open(dst_src, "rb") as f:
            src_bytes = f.read()
        src_hash = hashlib.sha1(src_bytes).hexdigest()[:12]
        with open(dst_hdr, "rb") as f:
            hdr_bytes = f.read()
        hdr_hash = hashlib.sha1(hdr_bytes).hexdigest()[:12]
    except Exception:
        src_hash = "nohash"
        hdr_hash = "nohash"
    dir_hash = _hash_directory_tree(dst_inc)
    combined_hash = hashlib.sha1(
        ("cpu:" + src_hash + hdr_hash + dir_hash).encode()
    ).hexdigest()[:12]
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
    target_arch, arch_code = _cuda_target_architecture()
    ext_name = f"new_{name}_{arch_code}_{code_hash}"

    ext = load(
        name=ext_name,
        sources=[src_path],
        extra_include_paths=[snapshot_include_dir, _repo_root()],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "-Xptxas=-O3",
            # "--disable-warnings",
            # "--extra-device-vectorization",
            f"-arch={arch_code}",
            # Unlike CUB_PTX_ARCH, an explicit nvcc -D has the same value in
            # host and device passes.  The generated host launch geometry and
            # device kernel therefore instantiate the same policy type.
            f"-DEASIER_CUDA_TARGET_ARCH={target_arch}",
            (
                "-DEASIER_CUDA_F64_ITEMS_PER_THREAD="
                f"{_cuda_f64_items_per_thread()}"
            ),
            # _nvcc_threads_flag(),
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
        extra_cflags=["-O3", "-fopenmp", "-DNDEBUG", "-iquote",
                      os.path.dirname(src_path)],
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
    logger.info(f"{len(ms)} graphs have been processing")
    
    # Collect build jobs across modules first (generation + snapshot)
    build_jobs: List[Tuple[str, str, str, str, str, str]] = []
    prebuilt_extensions: Dict[str, object] = {}
    forwarding_plans: Dict[
        Tuple[int, str], List[_ForwardedReducerOutput]
    ] = {}
    
    logger.info("================================================")
    for m, g in zip(ms, gs):
        backend = getattr(m, 'easier_jit_backend', 'torch')
        logger.info(f"Generating code for {backend} backend")
        if backend not in ['cuda', 'cpu']:
            logger.info(f"Skipping module with backend {backend}")
            continue

        callmods = _collect_fused_call_modules(m, g)
        # print(f"callmods: {callmods}")
        # Generate per submodule and snapshot immediately to avoid file clobbering
        for node_name, node, submod in callmods:
            logger.info(f"Node: {node_name}")
            forwarding = _find_forwarded_reducer_outputs(submod, g, node)
            forwarding_plans[(id(g), node_name)] = forwarding
            # check each submodule details for debugs 
            # submod.graph.print_tabular()
            # Always regenerate and snapshot first.  The former cache loaded a
            # node-name-keyed .so before observing current generated code,
            # which could silently execute stale kernels after a code change.
            # The v2 cache can only be queried with a key derived from the
            # current complete source snapshot and build fingerprint.
            if backend == 'cuda':
                with _codegen_lock():
                    generate_cuda_code_from_graph(
                        submod,
                        g,
                        forwarded_output_indices=[
                            item.output_index for item in forwarding
                        ],
                    )
                    snap_src, snap_inc, source_hash = \
                        _snapshot_generated_source_cuda(node_name)
            elif backend == 'cpu':
                with _codegen_lock():
                    generate_cpu_code_from_graph(
                        submod,
                        g,
                        forwarded_output_indices=[
                            item.output_index for item in forwarding
                        ],
                    )
                    snap_src, snap_inc, source_hash = \
                        _snapshot_generated_source_cpu(node_name)
            else:
                raise ValueError(f"Invalid backend {backend}")

            cache_key = _verified_cache_key(backend, source_hash)
            cached_module = _load_cached_extension_module(
                backend, node_name, cache_key
            )
            if cached_module is not None:
                logger.info(
                    "Verified JIT cache hit for %s (source=%s key=%s)",
                    node_name, source_hash, cache_key,
                )
                prebuilt_extensions[node_name] = cached_module
                shutil.rmtree(os.path.dirname(snap_src), ignore_errors=True)
                continue

            logger.info(
                "Verified JIT cache miss for %s (source=%s key=%s)",
                node_name, source_hash, cache_key,
            )
            cached_snapshot = _load_cached_snapshot(
                backend, node_name, cache_key
            )
            if cached_snapshot is not None:
                shutil.rmtree(os.path.dirname(snap_src), ignore_errors=True)
                snap_src, snap_inc, source_hash = cached_snapshot
            else:
                cached = _persist_snapshot_cache(
                    backend, node_name, cache_key,
                    snap_src, snap_inc, source_hash,
                )
                if cached is not None:
                    shutil.rmtree(os.path.dirname(snap_src), ignore_errors=True)
                    snap_src, snap_inc, source_hash = cached
            build_jobs.append((
                backend, node_name, snap_src, snap_inc, source_hash, cache_key
            ))

    # Build in parallel per job
    logger.info("================================================")
    logger.info("Building extensions in parallel")
    built_extensions: Dict[str, object] = dict(prebuilt_extensions)
    if build_jobs:
        n_workers = min(len(build_jobs), max(1, (os.cpu_count() or 2) // 2))
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_map: Dict[
                str, Tuple[str, str, concurrent.futures.Future]
            ] = {}
            for backend, node_name, snap_src, snap_inc, source_hash, cache_key \
                    in build_jobs:
                if backend == 'cuda':
                    fut = executor.submit(
                        _build_cuda_extension_from_src,
                        node_name, snap_src, snap_inc, cache_key,
                    )
                elif backend == 'cpu':
                    fut = executor.submit(
                        _build_cpu_extension_from_src,
                        node_name, snap_src, snap_inc, cache_key,
                    )
                elif backend == 'torch':
                    continue
                else:
                    raise ValueError(f"Invalid backend {backend}")
                future_map[node_name] = (backend, cache_key, fut)
            for node_name, (backend, cache_key, fut) in future_map.items():
                ext = fut.result()
                built_extensions[node_name] = ext
                _persist_compiled_extension(
                    backend, node_name, cache_key, ext
                )

    # Replace submodules sequentially to preserve semantics
    logger.info("================================================")
    logger.info("Replacing submodules sequentially to preserve semantics")
    for m, g in zip(ms, gs):
        backend = getattr(m, 'easier_jit_backend', 'torch')
        if backend not in ['cuda', 'cpu']:
            continue
        for node in list(g.nodes):
            if node.op != 'call_module':
                continue
            if _is_node_halo_exchanger(m, node):
                continue
            ext = built_extensions.get(node.name)
            if ext is None:
                continue
            forwarding = forwarding_plans.get((id(g), node.name), [])
            ok = dynamic_replace_submodule(
                m,
                g,
                ext,
                node.name,
                forwarded_output_count=len(forwarding),
            )
            if not ok:
                raise RuntimeError(f"Failed to replace submodule {node.name}")
            _apply_forwarded_reducer_outputs(g, node, forwarding)
    logger.info("================================================")
    return ms, gs
