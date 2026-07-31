# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
from string import Template as StrTemplate

from easier.core.passes.code_gen.utils import get_dim_length
from easier.core.utils import logger


def _cuda_output_allocation_factory(target) -> str:
    """Return the Torch constructor required by a generated CUDA output."""
    target_name = str(target)
    if "sum" in target_name or "norm" in target_name:
        # Global aggregators accumulate block results with atomicAdd, so their
        # output buffers must start at zero.
        return "torch::zeros"
    # Merge reducers directly initialize every completed row before their
    # carry-only fixup kernel runs. Map kernels also assign every output
    # element. Neither path needs a full output zero-fill.
    return "torch::empty"


def generate_binding_code(
        project_root,
        inputs,
        outputs,
        selector_register,
        direct_map=False,
        forwarded_output_indices=(),
    ):
    """
    Generate the CUDA/Torch extension 
    binding (merged_binding.cu) from graph metadata.

    Args:
        project_root: absolute path to repo root
        inputs: list of input node dicts from trace_graph
        outputs: list of output node dicts from trace_graph
        selector_register: list of selector metadata from trace_graph
        direct_map: launch the no-temporary-storage gather-map-store path
        forwarded_output_indices: outputs whose storage is supplied by the
            caller because the fused result completely overwrites an existing
            destination tensor
    """
    forwarded_output_indices = tuple(
        sorted(set(forwarded_output_indices))
    )
    if any(i < 0 or i >= len(outputs) for i in forwarded_output_indices):
        raise ValueError("forwarded output index is out of range")
    has_reducer = any(str(output["target"]) == "reducer" for output in outputs)

    # Infer argument ordering compatible with current driver:
    #  - value inputs excluding any selector target(s)
    #  - selector index tensors
    #  - selector target value tensors (gather sources)
    #  - row_end_offsets, num_rows, num_cols
    gather_target_names = set(
        inter["target"] for inter in selector_register if inter.get("selector") == 1
    )

    value_inputs = [inp for inp in inputs if inp["dtype"] != "int"]
    # Only keep unique selector inputs by 'target'
    _tensor_target_selector_set = set()
    selector_inputs = []
    for inp in inputs:
        if inp["dtype"] == "int" and inp["target"] not in _tensor_target_selector_set:
            selector_inputs.append(inp)
            _tensor_target_selector_set.add(inp["target"])

    non_gather_value_inputs = [
        inp for inp in value_inputs if inp["name"] not in gather_target_names
    ]
    _set_gather_value_inputs = set()
    gather_value_inputs = []
    for _inp_node in inputs:
        # in the input, the call_module node is the selector node
        if _inp_node["op"] == "call_module" and \
            _inp_node["args"][0].name in gather_target_names and \
            _inp_node["args"][0].name not in _set_gather_value_inputs:
            def append_by_name(target_name, source_list, dest_list):
                for item in source_list:
                    if item["name"] == target_name:
                        dest_list.append(item)
                        break
            append_by_name(_inp_node["args"][0].name, inputs, gather_value_inputs)
            _set_gather_value_inputs.add(_inp_node["args"][0].name)
    # Pick a representative value tensor for dtype dispatch
    dispatch_tensor_name = (
        non_gather_value_inputs[0]["name"]
        if non_gather_value_inputs
        else (gather_value_inputs[0]["name"] \
            if gather_value_inputs else value_inputs[0]["name"])  # type: ignore[index]
    )

    # Build parameter list for the binding function
    bind_params = []
    for vi in non_gather_value_inputs:
        bind_params.append((vi["name"], "value"))
    for gi in gather_value_inputs:
        bind_params.append((gi["name"], "value"))
    for si in selector_inputs:
        bind_params.append((si["target"] + "_idx", "index"))
    # Static CSR pointer and sizes
    bind_params.extend([
        ("row_end_offsets", "index"), 
        ("num_rows", "size"), 
        ("num_cols", "size")
    ])
    forwarded_output_params = {
        idx: f"forward_out_{idx}_{outputs[idx]['name']}"
        for idx in forwarded_output_indices
    }
    for idx in forwarded_output_indices:
        bind_params.append((forwarded_output_params[idx], "output"))

    def make_tensor_checks(name, kind):
        lines = []
        if kind in ("value", "index", "output"):
            lines.append(f"  TORCH_CHECK({name}.is_cuda(), \"{name} must be a CUDA tensor\");\n")
            lines.append(f"  TORCH_CHECK({name}.is_contiguous(), \"{name} must be contiguous\");\n")
        return lines

    # Load binding template
    template_binding_path = os.path.join(
        project_root, "cuda_template", "merged_binding_template.cu"
    )
    with open(template_binding_path, "r") as f:
        binding_template = f.read()

    # Prepare substitutions: outputs (reducers, aggregators, maps)
    tuple_types = []
    output_allocations = []
    output_tuple_returns_list = []
    params_output_ptrs = []
    output_checks = []

    def add_output(
            out_name: str,
            dim: int,
            size_expr: str,
            numel_expr: str,
            idx: int,
            allocation_factory: str,
        ):
        if idx in forwarded_output_params:
            var_name = forwarded_output_params[idx]
            output_checks.append(
                f"  TORCH_CHECK({var_name}.numel() == {numel_expr}, "
                f"\"{var_name} has the wrong number of elements\");\n"
            )
            output_checks.append(
                f"  TORCH_CHECK({var_name}.device() == "
                f"{dispatch_tensor_name}.device(), "
                f"\"{var_name} must be on the same device as inputs\");\n"
            )
        else:
            var_name = f"out_{idx}_{out_name}"
            output_allocations.append(
                f"  torch::Tensor {var_name} = {allocation_factory}"
                f"({{{size_expr}}}, options_val);\n"
            )
        output_tuple_returns_list.append(var_name)
        params_output_ptrs.append(
            f"  params.output_y_{out_name}_ptr = \
                reinterpret_cast<ValueT*>({var_name}.data_ptr());\n"
        )
        tuple_types.append("torch::Tensor")

    def make_shape_expr(base: str, d: int) -> str:
        return base if d == 1 else f"{base}, {d}"

    for idx, o in enumerate(outputs):
        dim = get_dim_length(o["shape"])
        out_name = o["name"]
        allocation_factory = _cuda_output_allocation_factory(o["target"])
        if str(o["target"]) == "reducer":
            add_output(
                out_name,
                dim,
                make_shape_expr("num_rows", dim),
                "num_rows" if dim == 1 else f"num_rows * {dim}",
                idx,
                allocation_factory,
            )
        elif 'sum' in str(o["target"]) or 'norm' in str(o["target"]):
            # aggregator
            add_output(
                out_name, dim, f"{dim}", f"{dim}", idx, allocation_factory
            )
        else:
            add_output(
                out_name,
                dim,
                make_shape_expr("ne", dim),
                "ne" if dim == 1 else f"ne * {dim}",
                idx,
                allocation_factory,
            )

    if output_tuple_returns_list:
        output_tuple_returns = ", ".join(output_tuple_returns_list)
        output_tuple_returns = f"return std::make_tuple({output_tuple_returns});"
    else:
        output_tuple_returns = ""
    if tuple_types:
        _tuple_types_list = ", ".join(tuple_types)
        tuple_type_return = f"static std::tuple<{_tuple_types_list}>"
    else:
        tuple_type_return = "void"

    # Function params
    def ctype_of(kind: str) -> str:
        return "torch::Tensor" if kind != "size" else "int64_t"

    function_params = ",\n".join([f"    {ctype_of(k)} {n}" for n, k in bind_params])
    function_call_args = ", ".join([n for n, _ in bind_params])

    # Build optional torch::kLong dispatch case string (do not modify the template here)
    optional_long_case = ""
    try:
        _dispatch_input = (
            non_gather_value_inputs[0]
            if non_gather_value_inputs
            else (gather_value_inputs[0] if gather_value_inputs else value_inputs[0])  # type: ignore[index]
        )
        if _dispatch_input.get("dtype_data") == torch.int64:
            optional_long_case = (
                "    case torch::kLong:\n" +
                f"      return merged_spmv_launch_index_dispatch<long>({function_call_args}, temp_storage);\n"
            )
    except Exception:
        optional_long_case = ""

    # Checks
    basic_checks = []
    for n, k in bind_params:
        basic_checks.extend(make_tensor_checks(n, k))
    dtype_checks = []
    for vi in non_gather_value_inputs + gather_value_inputs:
        n = vi["name"]
        dtype_checks.append(
            f"  TORCH_CHECK({n}.scalar_type() == \
            c10::CppTypeToScalarType<ValueT>::value, \"{n} dtype mismatch\");\n"
        )
    for si in selector_inputs:
        n = si["target"] + "_idx"
        dtype_checks.append(
            f"  TORCH_CHECK({n}.scalar_type() == \
            c10::CppTypeToScalarType<OffsetT>::value, \"{n} dtype mismatch\");\n"
        )
    dtype_checks.append(
        "  TORCH_CHECK(row_end_offsets.scalar_type() == \
            c10::CppTypeToScalarType<OffsetT>::value, \"row_end_offsets dtype mismatch\");\n"
    )
    for idx in forwarded_output_indices:
        n = forwarded_output_params[idx]
        dtype_checks.append(
            f"  TORCH_CHECK({n}.scalar_type() == "
            f"c10::CppTypeToScalarType<ValueT>::value, "
            f"\"{n} dtype mismatch\");\n"
        )

    # ne expr and bsx/bsy size checks
    if selector_inputs:
        gi0 = selector_inputs[0]["target"] + "_idx"
        ne_expr = f"{gi0}.numel()"
    else:
        ne_expr = "num_cols"
    ne_multiple_checks = []
    for selector in selector_inputs[1:]:
        selector_arg = selector["target"] + "_idx"
        ne_multiple_checks.append(
            f"  TORCH_CHECK({selector_arg}.numel() == ne, "
            f"\"{selector_arg} must have the same length as the first selector\");\n"
        )
    if any(v["name"] == "bsx" for v in non_gather_value_inputs + gather_value_inputs):
        ne_multiple_checks.append(
            "  TORCH_CHECK(bsx.numel() % ne == \
            0, \"bsx.numel() must be a multiple of ne\");\n"
        )
    if any(v["name"] == "bsy" for v in non_gather_value_inputs + gather_value_inputs):
        ne_multiple_checks.append(
            "  TORCH_CHECK(bsy.numel() % ne == \
            0, \"bsy.numel() must be a multiple of ne\");\n"
        )

    # Params mapping
    params_value_ptrs = [
        f"  params.{vi['name']}_ptr = \
        reinterpret_cast<ValueT*>({vi['name']}.data_ptr());\n"
        for vi in (non_gather_value_inputs + gather_value_inputs)
    ]
    params_index_ptrs = [
        f"  params.{si['target']}_ptr = \
        reinterpret_cast<OffsetT*>(({si['target']}_idx).data_ptr());\n"
        for si in selector_inputs
    ]

    if direct_map:
        launch_code = """
  size_t temp_storage_bytes = 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t err = merged::merged_spmv_launch<ValueT, OffsetT>(
      params, /*d_temp_storage=*/nullptr, temp_storage_bytes,
      /*debug_synchronous=*/false, stream.stream());
  TORCH_CHECK(err == cudaSuccess, "direct gather-map-store launch failed: ",
              cudaGetErrorString(err));
"""
    elif has_reducer:
        launch_code = """
  TORCH_CHECK(temp_storage.defined(),
              "temp_storage must be a CUDA uint8 tensor");
  TORCH_CHECK(temp_storage.scalar_type() == torch::kUInt8,
              "temp_storage must use uint8 storage");
  TORCH_CHECK(temp_storage.device() == ${dispatch_tensor}.device(),
              "temp_storage must be on the same device as inputs");

  const bool initialize_tile_coordinates = temp_storage.numel() == 0;
  size_t temp_storage_bytes =
      static_cast<size_t>(temp_storage.numel());
  auto stream = at::cuda::getCurrentCUDAStream();
  void* d_temp_storage = nullptr;

  if (initialize_tile_coordinates) {
    cudaError_t err = merged::merged_spmv_launch<ValueT, OffsetT>(
        params, /*d_temp_storage=*/nullptr, temp_storage_bytes,
        /*initialize_tile_coordinates=*/false,
        /*debug_synchronous=*/false, stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "merged_spmv_launch (size query) failed: ",
                cudaGetErrorString(err));
    TORCH_CHECK(temp_storage_bytes > 0,
                "merge reducer requested empty temporary storage");
    temp_storage.resize_(
        {static_cast<long long>(temp_storage_bytes)});
  }

  d_temp_storage = temp_storage.data_ptr();
  cudaError_t err = merged::merged_spmv_launch<ValueT, OffsetT>(
      params, d_temp_storage, temp_storage_bytes,
      initialize_tile_coordinates,
      /*debug_synchronous=*/false, stream.stream());
  TORCH_CHECK(err == cudaSuccess, "merged_spmv_launch failed: ",
              cudaGetErrorString(err));
"""
        launch_code = StrTemplate(launch_code).substitute(
            dispatch_tensor=dispatch_tensor_name
        )
    else:
        launch_code = """
  size_t temp_storage_bytes = 0;
  auto stream = at::cuda::getCurrentCUDAStream();

  // Always do a clean size query with nullptr to avoid any invalid-pointer paths
  void* d_temp_storage = nullptr;
  cudaError_t err = merged::merged_spmv_launch<ValueT, OffsetT>(
      params, /*d_temp_storage=*/nullptr, temp_storage_bytes, /*debug_synchronous=*/false, stream.stream());
  TORCH_CHECK(err == cudaSuccess, "merged_spmv_launch (size query) failed: ", cudaGetErrorString(err));

  if (temp_storage_bytes > 0) {
    // In-place resize only: do not rebind to a new tensor or change device
    TORCH_CHECK(temp_storage.defined(), "temp_storage must be provided as a CUDA uint8 tensor");
    TORCH_CHECK(temp_storage.device() == ${dispatch_tensor}.device(), "temp_storage must be on the same device as inputs");
    if (static_cast<size_t>(temp_storage.numel()) < temp_storage_bytes) {
      temp_storage.resize_({static_cast<long long>(temp_storage_bytes)});
    }
    d_temp_storage = temp_storage.data_ptr();
  }

  err = merged::merged_spmv_launch<ValueT, OffsetT>(
      params, d_temp_storage, temp_storage_bytes, /*debug_synchronous=*/false, stream.stream());
  TORCH_CHECK(err == cudaSuccess, "merged_spmv_launch failed: ", cudaGetErrorString(err));
"""
        launch_code = StrTemplate(launch_code).substitute(
            dispatch_tensor=dispatch_tensor_name
        )

    # pybind args
    pybind_args = "\n".join([f"      , py::arg(\"{n}\")" for n, _ in bind_params])

    binding_code = StrTemplate(binding_template).substitute(
        tuple_type_return=tuple_type_return,
        function_params=function_params,
        function_call_args=function_call_args,
        basic_checks="".join(basic_checks),
        dtype_checks="".join(dtype_checks),
        ne_expr=ne_expr,
        ne_multiple_checks="".join(ne_multiple_checks),
        output_checks="".join(output_checks),
        dispatch_tensor=dispatch_tensor_name,
        output_allocations="".join(output_allocations),
        params_value_ptrs="".join(params_value_ptrs),
        params_index_ptrs="".join(params_index_ptrs),
        params_output_ptrs="".join(params_output_ptrs),
        output_tuple_returns=output_tuple_returns,
        pybind_args=pybind_args,
        optional_long_case=optional_long_case,
        launch_code=launch_code,
    )

    # Write binding file at project root
    binding_path = os.path.join(project_root, "merged_binding.cu")
    with open(binding_path, "w") as f:
        f.write(binding_code)
    logger.info(f"Binding code generated successfully at {binding_path}!")
