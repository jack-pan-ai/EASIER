# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
from string import Template as StrTemplate

from easier.core.passes.code_gen.utils import get_dim_length
from easier.core.utils import logger


def generate_cpu_binding_code(
        project_root, inputs, outputs, selector_register
    ):
    """
    Generate the CPU/Torch extension binding (merged_binding_cpu.cpp) from graph metadata.

    Args:
        project_root: absolute path to repo root
        inputs: list of input node dicts from trace_graph
        outputs: list of output node dicts from trace_graph
        selector_register: list of selector metadata from trace_graph
    """
    # Infer argument ordering compatible with runtime wrapper:
    #  - value inputs excluding any selector target(s)
    #  - selector index tensors (unique per target)
    #  - selector target value tensors (gather sources) after non-gather to keep parity with CUDA
    #  - row_end_offsets, num_rows, num_cols
    gather_target_names = set(
        inter["target"] for inter in selector_register if inter.get("selector") == 1
    )

    value_inputs = [inp for inp in inputs if inp["dtype"] != "int"]

    # Unique selector inputs by 'target'
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
        ("num_cols", "size"),
    ])

    # Load CPU binding template
    template_binding_path = os.path.join(
        project_root, "cpu_template", "merged_binding_template_cpu.cpp"
    )
    with open(template_binding_path, "r") as f:
        binding_template = f.read()

    # Prepare substitutions: outputs (reducers, aggregators, maps)
    tuple_types = []
    output_allocations = []
    output_tuple_returns_list = []
    output_ptrs = []

    def add_output(out_name: str, dim: int, size_expr: str, idx: int):
        var_name = f"out_{idx}_{out_name}"
        # Uninitialized buffer: OmpMergeSystem writes every output element (map
        # per edge, reducer per row, aggregators in fixup). torch::zeros would
        # call aten::fill_ on the full tensor and dominated CPU profiles.
        output_allocations.append(
            f"  torch::Tensor {var_name} = torch::empty({{{size_expr}}}, options_val);\n"
        )
        # output_allocations.append(
        #     f"  torch::Tensor {var_name} = torch::zeros({{{size_expr}}}, options_val);\n"
        # )
        output_tuple_returns_list.append(var_name)
        output_ptrs.append(
            f"  auto* output_y_{out_name}_ptr = reinterpret_cast<ValueT*>({var_name}.data_ptr());\n"
        )
        tuple_types.append("torch::Tensor")

    def make_shape_expr(base: str, d: int) -> str:
        return base if d == 1 else f"{base}, {d}"

    for idx, o in enumerate(outputs):
        dim = get_dim_length(o["shape"])
        out_name = o["name"]
        if str(o["target"]) == "reducer":
            add_output(out_name, dim, make_shape_expr("num_rows", dim), idx)
        elif 'sum' in str(o["target"]) or 'norm' in str(o["target"]):
            add_output(out_name, dim, f"{dim}", idx)
        else:
            add_output(out_name, dim, make_shape_expr("ne", dim), idx)

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

    # Optional long case for ValueT
    optional_long_case = ""
    try:
        _dispatch_input = (
            non_gather_value_inputs[0]
            if non_gather_value_inputs
            else (gather_value_inputs[0] if gather_value_inputs else value_inputs[0])  # type: ignore[index]
        )
        if _dispatch_input.get("dtype_data") == torch.int64:
            optional_long_case = (
                "    case torch::kLong:\n"
                f"      return merged_spmv_launch_cpu_index_dispatch<long>({function_call_args});\n"
            )
    except Exception:
        optional_long_case = ""

    # Checks
    basic_checks = []
    for n, k in bind_params:
        if k in ("value", "index"):
            basic_checks.append(f"  TORCH_CHECK({n}.device().is_cpu(), \"{n} must be a CPU tensor\");\n")
            basic_checks.append(f"  TORCH_CHECK({n}.is_contiguous(), \"{n} must be contiguous\");\n")

    dtype_checks = []
    for vi in non_gather_value_inputs + gather_value_inputs:
        n = vi["name"]
        if str(vi['dtype_data']) == 'torch.int32' and vi['shape_ahead'] == 1:
            # this is very speicial case in the GMRES solver
            continue
        else:
            dtype_checks.append(
                f"  TORCH_CHECK({n}.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, \"{n} dtype mismatch\");\n"
            )
    for si in selector_inputs:
        n = si["target"] + "_idx"
        dtype_checks.append(
            f"  TORCH_CHECK({n}.scalar_type() == c10::CppTypeToScalarType<OffsetT>::value, \"{n} dtype mismatch\");\n"
        )
    dtype_checks.append(
        "  TORCH_CHECK(row_end_offsets.scalar_type() == c10::CppTypeToScalarType<OffsetT>::value, \"row_end_offsets dtype mismatch\");\n"
    )

    # ne expr and bsx/bsy size checks
    if selector_inputs:
        gi0 = selector_inputs[0]["target"] + "_idx"
        ne_expr = f"{gi0}.numel()"
    else:
        ne_expr = "num_cols"
    ne_multiple_checks = []
    if any(v["name"] == "bsx" for v in non_gather_value_inputs + gather_value_inputs):
        ne_multiple_checks.append(
            "  TORCH_CHECK(bsx.numel() % ne == 0, \"bsx.numel() must be a multiple of ne\");\n"
        )
    if any(v["name"] == "bsy" for v in non_gather_value_inputs + gather_value_inputs):
        ne_multiple_checks.append(
            "  TORCH_CHECK(bsy.numel() % ne == 0, \"bsy.numel() must be a multiple of ne\");\n"
        )

    # Build call inputs exactly in the same order as declarations_gen emits
    # (iterate original inputs list and map to pointer expressions)
    raw_call_inputs_exprs = []
    _tensor_target_selector_set = set()
    for inp in inputs:
        if inp["dtype"] == "int":
            # find selector entry to map selector_name -> target
            sel = None
            for inter in selector_register:
                if inter.get("name") == inp["name"]\
                    and inter.get("selector_name") not in _tensor_target_selector_set:
                    sel = inter
                    _tensor_target_selector_set.add(inter.get("selector_name"))
                    break
            if sel is not None:
                target_name = sel["selector_name"]
                raw_call_inputs_exprs.append(
                    f"reinterpret_cast<OffsetT*>(({target_name}_idx).data_ptr())"
                )
        else:
            if str(inp['dtype_data']) == 'torch.int32' and inp['shape_ahead'] == 1:
                # this is very speicial case in the GMRES solver
                raw_call_inputs_exprs.append(
                    f"{inp['name']}_const.item<int>()"
                )
            else:
                raw_call_inputs_exprs.append(
                    f"reinterpret_cast<ValueT*>({inp['name']}.data_ptr())"
                )

    # reducer presence determines OmpMergeSystem signature
    reducer_present = any(str(o["target"]) == "reducer" for o in outputs)

    # Build raw call argument order from merged_spmv.h
    raw_args = []
    # For reducer mode, row_end_offsets pointer comes first
    if reducer_present:
        raw_args.append("reinterpret_cast<OffsetT*>(row_end_offsets.data_ptr())")
    # Then all input pointers in the order of `inputs`
    raw_args.extend(raw_call_inputs_exprs)
    # Then outputs in order
    for o in outputs:
        raw_args.append(f"output_y_{o['name']}_ptr")
    # Finally sizes
    raw_args.append("static_cast<int>(num_rows)")
    raw_args.append("static_cast<int>(ne)")

    if reducer_present:
        omp_call = (
            "  OmpMergeSystem<ValueT, OffsetT>(\n"
            "    num_threads, " + ", ".join(raw_args) + ");\n"
        )
    else:
        # Remove the row_end_offsets for aggregator signature
        args_wo_row = raw_args.copy()
        omp_call = (
            "  OmpMergeSystem<ValueT, OffsetT>(\n"
            "    num_threads, " + ", ".join(args_wo_row) + ");\n"
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
        dispatch_tensor=dispatch_tensor_name,
        output_allocations="".join(output_allocations),
        value_ptrs="",
        index_ptrs="",
        output_ptrs="".join(output_ptrs),
        omp_call=omp_call,
        output_tuple_returns=output_tuple_returns,
        pybind_args=pybind_args,
        optional_long_case=optional_long_case,
    )

    # Write binding file at project root
    binding_path = os.path.join(project_root, "merged_binding_cpu.cpp")
    with open(binding_path, "w") as f:
        f.write(binding_code)
    logger.info(f"CPU binding code generated successfully at {binding_path}!")

