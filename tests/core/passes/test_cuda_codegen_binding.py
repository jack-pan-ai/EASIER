# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from pathlib import Path
import shutil

import torch
import torch.fx as fx

from easier.core import module as esr
from easier.core.passes import codegen
from easier.core.passes.code_gen import merged_gen_binding, merged_gen_gpu
from easier.core.runtime.metadata import (
    ViewSrc,
    get_node_view_src,
    set_node_view_src,
)


def test_cuda_output_allocation_matches_kernel_write_semantics(tmp_path):
    template_dir = tmp_path / "cuda_template"
    template_dir.mkdir()
    codegen_dir = Path(merged_gen_binding.__file__).parent
    shutil.copyfile(
        codegen_dir / "cuda_template" / "merged_binding_template.cu",
        template_dir / "merged_binding_template.cu",
    )

    inputs = [{
        "name": "input_x",
        "target": "input_x",
        "dtype": "scalar_t",
        "op": "placeholder",
        "args": (),
        "dtype_data": torch.float64,
    }]
    outputs = [
        {"name": "reduce_out", "target": "reducer", "shape": (1,)},
        {"name": "sum_out", "target": "sum", "shape": (1,)},
        {"name": "norm_out", "target": "norm", "shape": (1,)},
        {"name": "map_out", "target": "map", "shape": (1,)},
    ]

    merged_gen_binding.generate_binding_code(tmp_path, inputs, outputs, [])
    binding = (tmp_path / "merged_binding.cu").read_text()

    assert "out_0_reduce_out = torch::empty({num_rows}, options_val);" \
        in binding
    assert "out_0_reduce_out = torch::zeros" not in binding
    assert "out_1_sum_out = torch::zeros({1}, options_val);" in binding
    assert "out_2_norm_out = torch::zeros({1}, options_val);" in binding
    assert "out_3_map_out = torch::empty({ne}, options_val);" in binding
    assert "out_3_map_out = torch::zeros" not in binding


def test_forwarded_reducer_output_uses_caller_storage_without_allocation(
    tmp_path,
):
    template_dir = tmp_path / "cuda_template"
    template_dir.mkdir()
    codegen_dir = Path(merged_gen_binding.__file__).parent
    shutil.copyfile(
        codegen_dir / "cuda_template" / "merged_binding_template.cu",
        template_dir / "merged_binding_template.cu",
    )
    inputs = [{
        "name": "input_x",
        "target": "input_x",
        "dtype": "scalar_t",
        "op": "placeholder",
        "args": (),
        "dtype_data": torch.float64,
    }]
    outputs = [{
        "name": "reduce_out",
        "target": "reducer",
        "shape": (4,),
    }]

    merged_gen_binding.generate_binding_code(
        tmp_path,
        inputs,
        outputs,
        [],
        forwarded_output_indices=[0],
    )
    binding = (tmp_path / "merged_binding.cu").read_text()

    destination = "forward_out_0_reduce_out"
    assert f"torch::Tensor {destination}" in binding
    assert "torch::empty({num_rows, 4}" not in binding
    assert f"params.output_y_reduce_out_ptr" in binding
    assert f"reinterpret_cast<ValueT*>({destination}.data_ptr())" in binding
    assert f"{destination}.numel() == num_rows * 4" in binding
    assert f"{destination}.device() == input_x.device()" in binding
    assert f"{destination} must be a CUDA tensor" in binding
    assert f"{destination} must be contiguous" in binding
    assert f"{destination} dtype mismatch" in binding
    assert f"return std::make_tuple({destination});" in binding


def _make_outer_reducer_overwrite(index=slice(None), destination_is_input=False):
    reducer_root = torch.nn.Module()
    reducer_root.reducer = esr.Reducer(
        torch.tensor([0, 1], dtype=torch.int64),
        n=2,
    )
    inner_graph = fx.Graph()
    inner_input = inner_graph.placeholder("input_x")
    inner_reducer = inner_graph.call_module(
        "reducer", args=(inner_input,)
    )
    inner_graph.output([inner_reducer])
    fused = fx.GraphModule(reducer_root, inner_graph)

    outer_graph = fx.Graph()
    input_x = outer_graph.get_attr("input_x")
    destination = outer_graph.get_attr("rhs")
    call_args = (
        (input_x, destination)
        if destination_is_input
        else (input_x,)
    )
    fused_call = outer_graph.call_module("_fused", args=call_args)
    result = outer_graph.call_function(
        operator.getitem, args=(fused_call, 0)
    )
    outer_graph.call_function(
        operator.setitem, args=(destination, index, result)
    )
    outer_graph.output(None)
    return fused, outer_graph, fused_call, destination


def test_full_overwrite_reducer_result_is_forwarded_and_copy_node_removed():
    fused, graph, fused_call, destination = \
        _make_outer_reducer_overwrite()
    set_node_view_src(destination, ViewSrc(destination, None))
    set_node_view_src(fused_call, [ViewSrc(fused_call, 0)])

    forwarding = codegen._find_forwarded_reducer_outputs(
        fused, graph, fused_call
    )
    assert [item.output_index for item in forwarding] == [0]

    codegen._apply_forwarded_reducer_outputs(
        graph, fused_call, forwarding
    )
    assert fused_call.args[-1] is destination
    assert get_node_view_src(fused_call) == [
        ViewSrc(destination, None)
    ]
    assert not any(
        node.op == "call_function"
        and node.target in (operator.getitem, operator.setitem)
        for node in graph.nodes
    )


def test_reducer_output_forwarding_rejects_partial_or_input_alias():
    fused, graph, fused_call, _ = _make_outer_reducer_overwrite(
        index=slice(1, None)
    )
    assert not codegen._find_forwarded_reducer_outputs(
        fused, graph, fused_call
    )

    fused, graph, fused_call, _ = _make_outer_reducer_overwrite(
        destination_is_input=True
    )
    assert not codegen._find_forwarded_reducer_outputs(
        fused, graph, fused_call
    )


def test_merge_reducer_directly_initializes_rows_then_fixes_tile_carries():
    codegen_dir = Path(merged_gen_binding.__file__).parent
    agent = (
        codegen_dir
        / "cuda_template"
        / "reducer"
        / "merged_agent_spmv_template.cuh"
    ).read_text()
    dispatch = (
        codegen_dir
        / "cuda_template"
        / "reducer"
        / "merged_spmv_template.cuh"
    ).read_text()
    utils = (
        codegen_dir / "cuda_template" / "merged_utils_template.cuh"
    ).read_text()

    assert "output_vector_y[" in agent
    assert "] = s_partials[item].values[i];" in agent
    assert "d_tile_carry_keys[tile_idx] = tile_carry.key;" in agent
    assert "d_tile_carry_values[" in agent
    assert "atomicAdd(" not in agent

    assert "ReducerCarryFixupKernel" in dispatch
    assert "${reducer_fixup_launch_code}" in dispatch
    assert "${total_reducer_dim}" in dispatch
    assert "if (initialize_tile_coordinates)" in dispatch
    assert "OffsetT *d_tile_carry_keys;" in utils
    assert "ValueT *d_tile_carry_values;" in utils


def test_reducer_codegen_renders_carry_layout_and_empty_output(tmp_path):
    codegen_dir = Path(merged_gen_gpu.__file__).parent
    shutil.copytree(
        codegen_dir / "cuda_template", tmp_path / "cuda_template"
    )
    (tmp_path / "include").mkdir()

    reducer = {
        "name": "reduce_out",
        "op": "reducer",
        "args": ["weighted"],
        "shape": (2,),
    }
    traced_fixture = (
        [
            {
                "name": "coeff",
                "target": "coeff",
                "dtype": "float",
                "dtype_data": torch.float64,
                "shape": (1,),
                "shape_ahead": 4,
                "op": "placeholder",
                "args": (),
            },
            {
                "name": "rates",
                "target": "rates",
                "dtype": "float",
                "dtype_data": torch.float64,
                "shape": (2,),
                "shape_ahead": 3,
                "op": "placeholder",
                "args": (),
            },
            {
                "name": "selector_idx",
                "target": "selector",
                "dtype": "int",
                "dtype_data": torch.int32,
                "shape": (1,),
                "shape_ahead": 4,
                "op": "placeholder",
                "args": (),
            },
        ],
        [{"name": "reduce_out", "target": "reducer", "shape": (2,)}],
        [
            {
                "selector": 0,
                "shape": (1,),
                "name": "coeff",
                "target": "coeff",
                "selector_name": "coeff",
                "shape_ahead": 4,
                "shape_all": (4, 1),
            },
            {
                "selector": 1,
                "shape": (2,),
                "name": "selected",
                "target": "rates",
                "selector_name": "selector",
                "shape_ahead": 4,
                "shape_all": (4, 2),
            },
        ],
        [
            {
                "name": "weighted",
                "op": "mul",
                "args": ["coeff", "selected"],
                "shape": (2,),
            },
            reducer,
        ],
        [reducer],
        [],
    )

    original_file = merged_gen_gpu.__file__
    original_trace = merged_gen_gpu.trace_graph
    original_coindexed = (
        merged_gen_gpu._coindexed_reducer_selector_targets
    )
    try:
        merged_gen_gpu.__file__ = str(tmp_path / "merged_gen_gpu.py")
        merged_gen_gpu.trace_graph = lambda *_: traced_fixture
        merged_gen_gpu._coindexed_reducer_selector_targets = (
            lambda *_: {"selector"}
        )
        merged_gen_gpu.generate_cuda_code_from_graph(None, None)
    finally:
        merged_gen_gpu.__file__ = original_file
        merged_gen_gpu.trace_graph = original_trace
        merged_gen_gpu._coindexed_reducer_selector_targets = (
            original_coindexed
        )

    agent = (tmp_path / "include" / "merged_agent_spmv.cuh").read_text()
    dispatch = (tmp_path / "include" / "merged_spmv.cuh").read_text()
    binding = (tmp_path / "merged_binding.cu").read_text()
    assert "atomicAdd(" not in agent
    assert "ReducerCarryFixupKernel<2>" in dispatch
    assert "static_cast<size_t>(0) * num_merge_tiles" in dispatch
    assert "torch::empty({num_rows, 2}, options_val)" in binding
    assert "torch::zeros({num_rows, 2}, options_val)" not in binding
    assert "evaluate_reducer_nonzero(" in agent
    assert "return weighted;" in agent
    assert (
        "s_tile_value_reducer[nonzero_idx] = weighted"
        not in agent
    )
    assert (
        "TensorT nonzero = evaluate_reducer_nonzero("
        in agent
    )
    assert (
        "const bool initialize_tile_coordinates = "
        "temp_storage.numel() == 0;"
    ) in binding
    assert "if (initialize_tile_coordinates)" in binding
    assert "initialize_tile_coordinates," in binding


def test_cuda_int64_dispatch_keeps_counts_and_merge_coordinates_int64():
    codegen_dir = Path(merged_gen_binding.__file__).parent
    binding = (codegen_dir / "cuda_template" / "merged_binding_template.cu").read_text()
    utils = (codegen_dir / "cuda_template" / "merged_utils_template.cuh").read_text()
    kernels = (codegen_dir / "include" / "merged_spmv_kernels.cuh").read_text()
    agent = (codegen_dir / "include" / "merged_agent_spmv.cuh").read_text()

    assert "params.num_rows = static_cast<OffsetT>(num_rows);" in binding
    assert "params.num_nonzeros = static_cast<OffsetT>(ne);" in binding
    assert "OffsetT num_rows;" in utils
    assert "OffsetT num_nonzeros;" in utils
    assert "using CoordinateT = typename cub::CubVector<OffsetT, 2>::Type;" in (
        codegen_dir / "include" / "merged_spmv.cuh"
    ).read_text()
    assert "spmv_params.num_rows," in kernels
    assert "static_cast<OffsetT>(tile_num_rows)" in agent
    assert "static_cast<OffsetT>(tile_num_nonzeros)" in agent


def test_direct_map_eligibility_requires_selectors_full_overwrite_and_no_reduce():
    selectors = [{"selector": 1}]
    full_map = [
        {"op": "mul", "args": ["x", "y"], "shape": (8,)},
        {
            "op": "setitem",
            "args": ["out", slice(None), "value"],
            "shape": (8,),
        },
    ]
    selectors[0]["shape"] = (8,)

    assert merged_gen_gpu._should_use_direct_map_kernel(
        selectors, full_map, [], []
    )
    assert not merged_gen_gpu._should_use_direct_map_kernel(
        selectors,
        [{
            "op": "setitem",
            "args": ["out", slice(1, None), "value"],
            "shape": (8,),
        }],
        [],
        [],
    )
    assert not merged_gen_gpu._should_use_direct_map_kernel(
        selectors, full_map, [{"op": "reducer"}], []
    )
    assert not merged_gen_gpu._should_use_direct_map_kernel(
        selectors, full_map, [], [{"op": "sum"}]
    )
    assert not merged_gen_gpu._should_use_direct_map_kernel(
        [{"selector": 0}], full_map, [], []
    )
    assert not merged_gen_gpu._should_use_direct_map_kernel(
        selectors,
        full_map + [{
            "op": "add_",
            "args": ["out", "value"],
            "shape": (8,),
        }],
        [],
        [],
    )


def test_register_fed_merge_requires_one_pure_reducer(monkeypatch):
    outputs = [{"target": "reducer"}]
    reducer = {"name": "reduce_out", "op": "reducer"}
    pure_map = [
        {"name": "weighted", "op": "mul"},
        reducer,
    ]

    assert merged_gen_gpu._should_use_register_fed_merge(
        outputs, pure_map, [reducer], [], {"selector"}
    )
    for side_effect in ("copy_", "add_", "setitem"):
        assert not merged_gen_gpu._should_use_register_fed_merge(
            outputs,
            [{"name": "write", "op": side_effect}, reducer],
            [reducer],
            [],
            {"selector"},
        )
    assert not merged_gen_gpu._should_use_register_fed_merge(
        outputs, pure_map, [reducer], [{"op": "sum"}], {"selector"}
    )
    assert not merged_gen_gpu._should_use_register_fed_merge(
        [{"target": "map"}], pure_map, [reducer], [], {"selector"}
    )
    assert not merged_gen_gpu._should_use_register_fed_merge(
        outputs, pure_map, [reducer], [], set()
    )

    monkeypatch.setenv("EASIER_CUDA_REGISTER_FED_MERGE", "0")
    assert not merged_gen_gpu._should_use_register_fed_merge(
        outputs, pure_map, [reducer], [], {"selector"}
    )


def test_register_fed_merge_reuses_exact_shared_reducer_row_index():
    shared = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    other = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    root = torch.nn.Module()
    root.shared_selector = esr.Selector(shared)
    root.other_selector = esr.Selector(other)
    root.reducer = esr.Reducer(shared, n=2)
    root.shared_selector.idx = shared
    root.other_selector.idx = other
    root.reducer.idx = shared

    graph = fx.Graph()
    receiver_values = graph.placeholder("receiver_values")
    source_values = graph.placeholder("source_values")
    selected_receiver = graph.call_module(
        "shared_selector", (receiver_values,)
    )
    selected_source = graph.call_module(
        "other_selector", (source_values,)
    )
    summed = graph.call_function(
        operator.add, (selected_receiver, selected_source)
    )
    reduced = graph.call_module(
        "reducer", (summed,)
    )
    graph.output([reduced])
    graph_module = fx.GraphModule(root, graph)

    assert merged_gen_gpu._coindexed_reducer_selector_targets(
        graph_module
    ) == {"shared_selector"}


def test_direct_map_binding_launches_once_without_merge_temp_query(tmp_path):
    template_dir = tmp_path / "cuda_template"
    template_dir.mkdir()
    codegen_dir = Path(merged_gen_binding.__file__).parent
    shutil.copyfile(
        codegen_dir / "cuda_template" / "merged_binding_template.cu",
        template_dir / "merged_binding_template.cu",
    )
    inputs = [{
        "name": "input_x",
        "target": "input_x",
        "dtype": "scalar_t",
        "op": "placeholder",
        "args": (),
        "dtype_data": torch.float32,
    }]

    merged_gen_binding.generate_binding_code(
        tmp_path, inputs, [], [], direct_map=True
    )
    binding = (tmp_path / "merged_binding.cu").read_text()

    assert binding.count(
        "merged::merged_spmv_launch<ValueT, OffsetT>"
    ) == 1
    assert "direct gather-map-store launch failed" in binding
    assert "size query" not in binding
