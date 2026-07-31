# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import string
import numpy as np

from easier.core.module import Reducer, Selector
from easier.core.passes.code_gen.utils import get_dim_length
from easier.core.passes.code_gen.merged_gen_binding import generate_binding_code
from easier.core.passes.code_gen.trace.graph_trace import trace_graph
from easier.core.utils import logger


def _is_full_overwrite_index(index):
    """Return whether an FX setitem index covers the complete destination."""
    if index is Ellipsis:
        return True
    if isinstance(index, slice):
        return index.start is None and index.stop is None and index.step is None
    if isinstance(index, tuple):
        return bool(index) and all(_is_full_overwrite_index(i) for i in index)
    return False


def _should_use_direct_map_kernel(
    selector_register,
    map_operations,
    reducer_operations,
    aggregator_operations,
):
    """Recognize selector/map/full-overwrite regions that need no merge path.

    These regions have no segmented or global reduction.  Their selector
    indices already define an independent mapped item, so scheduling the work
    through the merge-SpMV traversal only adds row-offset staging, merge-path
    search, shared-memory synchronization, and temporary coordinates.
    """
    if reducer_operations or aggregator_operations:
        return False
    if not any(inter.get("selector") == 1 for inter in selector_register):
        return False

    setitems = [op for op in map_operations if op.get("op") == "setitem"]
    if not setitems:
        return False
    pointwise_ops = {
        "add", "sub", "mul", "pow", "truediv", "neg", "exp", "clone",
        "squeeze", "setitem", "abs", "sign", "lt", "gt", "where",
    }
    if any(op.get("op") not in pointwise_ops for op in map_operations):
        return False
    if not all(
        len(op.get("args", ())) >= 3
        and _is_full_overwrite_index(op["args"][1])
        for op in setitems
    ):
        return False

    # Scalarize only uniform elementwise regions.  This makes consecutive CUDA
    # threads access consecutive tensor lanes without changing broadcasting or
    # slicing semantics for more general maps, which remain on merge SpMV.
    widths = {
        get_dim_length(op["shape"])
        for op in map_operations
        if "shape" in op
    }
    widths.update(
        get_dim_length(inter["shape"])
        for inter in selector_register
        if "shape" in inter
    )
    return len(widths) == 1 and next(iter(widths), 0) > 0


def _register_fed_merge_enabled():
    value = os.getenv("EASIER_CUDA_REGISTER_FED_MERGE", "1")
    return value.strip().lower() not in ("0", "false", "no", "off")


def _should_use_register_fed_merge(
    outputs,
    map_operations,
    reducer_operations,
    aggregator_operations,
    coindexed_reducer_selectors,
):
    """Recognize a pure single-reducer region that can feed merge in registers.

    The original merge template first evaluates every edge in a striped
    thread assignment, stores the mapped values in shared memory, synchronizes,
    and then reloads them in merge-path order.  A pure single-reducer region
    has no observable per-edge side effect, so the merge-path thread can
    evaluate an edge exactly when it takes the corresponding downward step.
    This preserves the merge-path schedule and segmented reduction while
    removing the shared-memory store/reload round trip and its preparation
    barrier.
    """
    if not _register_fed_merge_enabled():
        return False
    if len(reducer_operations) != 1 or aggregator_operations:
        return False
    if not coindexed_reducer_selectors:
        return False
    if len(outputs) != 1 or str(outputs[0].get("target")) != "reducer":
        return False
    if not map_operations or map_operations[-1].get("op") != "reducer":
        return False
    if sum(op.get("op") == "reducer" for op in map_operations) != 1:
        return False
    side_effecting_ops = {"copy_", "add_", "setitem"}
    return not any(
        op.get("op") in side_effecting_ops for op in map_operations
    )


def _coindexed_reducer_selector_targets(submodule):
    """Return Selector targets that share the sole Reducer's rewritten index.

    Sparse encoding deliberately preserves exact tensor sharing when a
    Selector and Reducer originate from the same loader.  In a register-fed
    merge step the reducer row is already known, so those selectors can index
    the receiver tensor with that row directly instead of reloading the
    receiver ID from the edge map.
    """
    if submodule is None or not hasattr(submodule, "graph"):
        return set()

    reducers = []
    selectors = []
    for node in submodule.graph.nodes:
        if node.op != "call_module":
            continue
        called = submodule.get_submodule(str(node.target))
        if isinstance(called, Reducer):
            reducers.append(called)
        elif isinstance(called, Selector):
            selectors.append((str(node.target).replace(".", "_"), called))
    if len(reducers) != 1:
        return set()
    reducer_index = reducers[0].idx
    return {
        target
        for target, selector in selectors
        if selector.idx is reducer_index
    }


def generate_cuda_code_from_graph(
        submodule, traced_model, forwarded_output_indices=()
    ):
    """
    Generate CUDA code from the graph

    Args:
        submodule: the submodule -> GraphModule type
        traced_model: the traced model FullGraph
    """

    # Analyze the graph and extract operations
    inputs = []
    _input_keys = set()
    selector_register = []
    map_operations = []
    reducer_operations = []
    aggregator_operations = []
    outputs = []

    # Analyze the graph and extract operations, 
    # 1) input and output, 2) selector, 
    # 3) map 4) reducer and aggregator
    inputs, outputs, selector_register, map_operations, \
        reducer_operations, aggregator_operations = trace_graph(submodule, traced_model)
    use_direct_map_kernel = _should_use_direct_map_kernel(
        selector_register,
        map_operations,
        reducer_operations,
        aggregator_operations,
    )
    coindexed_reducer_selectors = (
        _coindexed_reducer_selector_targets(submodule)
        if not use_direct_map_kernel
        else set()
    )
    use_register_fed_merge = (
        not use_direct_map_kernel
        and _should_use_register_fed_merge(
            outputs,
            map_operations,
            reducer_operations,
            aggregator_operations,
            coindexed_reducer_selectors,
        )
    )
    if not use_register_fed_merge:
        coindexed_reducer_selectors = set()
    direct_map_lane_width = (
        get_dim_length(map_operations[-1]["shape"])
        if use_direct_map_kernel
        else None
    )
    item_index_expr = (
        "item_idx"
        if use_direct_map_kernel
        else (
            "nonzero_idx"
            if use_register_fed_merge
            else "(tile_start_coord.y + nonzero_idx)"
        )
    )
    if use_direct_map_kernel:
        logger.info(
            "Using direct gather-map-store CUDA kernel for selector/map "
            "full-overwrite region"
        )
    if use_register_fed_merge:
        logger.info(
            "Using register-fed merge CUDA kernel for pure single-reducer "
            "region%s",
            (
                " with reducer-row selector reuse"
                if coindexed_reducer_selectors
                else ""
            ),
        )
    direct_overwrite_only_targets = set()
    if use_direct_map_kernel:
        for operation in map_operations:
            if operation.get("op") != "setitem":
                continue
            target = operation["args"][0]
            used_elsewhere = False
            for other in map_operations:
                args = list(other.get("args", ()))
                if other is operation:
                    args = args[1:]
                if target in args:
                    used_elsewhere = True
                    break
            if not used_elsewhere:
                direct_overwrite_only_targets.add(target)

    # Generate the code
    # Generate the input/output code
    input_declarations_code = []
    input_init_code = []
    input_declarations_utils_code = []
    input_agent_tenosrs_code = []
    output_agent_tenosrs_code = []
    output_agent_SMEM_code = []
    output_agent_forloop_code = []
    reducer_outputs = [
        (op["name"], get_dim_length(op["shape"]))
        for op in reducer_operations
    ]
    _tensor_names = set()
    _tensor_target_selector_set = set() # some gather may share the same selector
    for inp in inputs:
        name = inp['name']
        target = inp['target']
        if inp['dtype'] == 'int':
            # column indices, here target is used
            # considering that multiple selectors might have the same target
            if target not in _tensor_target_selector_set:
                input_declarations_utils_code.append(
                    f"  OffsetT *{target}_ptr; \n")
                input_declarations_code.append(
                    f"  ColumnIndicesIteratorT {target}_ptr; \n")
                input_init_code.append(
                    f"    {target}_ptr(spmv_params.{target}_ptr), \n")
                _tensor_target_selector_set.add(target)
        else:
            if inp['dtype_data'] == 'torch.int32' and inp['shape_ahead'] == 1:
                # this is very speicial case in the GMRES solver
                continue
            else:
                # spm and vector x
                input_declarations_utils_code.append(
                    f"  ValueT *{name}_ptr; \n")
                input_declarations_code.append(
                    f"  VectorValueIteratorT {name}_ptr; \n")
                input_init_code.append(
                    f"    {name}_ptr(spmv_params.{name}_ptr), \n")
                _dim = get_dim_length(inp['shape'])
                input_agent_tenosrs_code.append(
                    f"  typedef Tensor<ValueT, "
                    f"{1 if use_direct_map_kernel else _dim}> "
                    f"TensorInput_{name}_T; \n")
                _tensor_names.add(name)

    # output code in the declarations utils file
    for out in outputs:
        out_name = out['name']
        target_name = out['target']
        input_declarations_utils_code.append(
            f"  ValueT *output_y_{out_name}_ptr; \n")
        _dim = get_dim_length(out['shape'])
        if 'reducer' in str(out['target']):
            # reducer outside the forloop
            output_agent_tenosrs_code.append(
                f"  // Tensor and TensorKey for reducers \n")
            output_agent_tenosrs_code.append(
                f"  typedef TensorKey<OffsetT, ValueT, {_dim}> \
                    TensorKeyOutput_{out_name}_T; \n")
            output_agent_tenosrs_code.append(
                f"  typedef Tensor<ValueT, {_dim}> TensorOutput_{out_name}_T; \n")
            output_agent_tenosrs_code.append(
                f"  // Reduce-value-by-segment scan operator \n")
            output_agent_tenosrs_code.append(
                f"  typedef ReduceTensorByKeyOp<TensorKeyOutput_{out_name}_T>\
                        ReduceBySegmentOp_{out_name}_T; \n")
            output_agent_tenosrs_code.append(f"  typedef BlockScan< \n")
            output_agent_tenosrs_code.append(
                f"            TensorKeyOutput_{out_name}_T, \n")
            output_agent_tenosrs_code.append(f"            BLOCK_THREADS, \n")
            output_agent_tenosrs_code.append(
                f"            AgentSpmvPolicyT::SCAN_ALGORITHM> \n")
            output_agent_tenosrs_code.append(
                f"            BlockScan_{out_name}_T; \n")
            output_agent_SMEM_code.append(
                f"               SmemReuseReducer<{_dim}, \
                    BlockScan_{out_name}_T> smem_{out_name}; \n")
        elif 'sum' in str(out['target']):
            # aggregator
            _name = out_name
            output_agent_tenosrs_code.append(
                f"  // Tensor type and block reduce for output \n")
            output_agent_tenosrs_code.append(
                f"  typedef Tensor<ValueT, {_dim}> TensorOutput_{_name}_T; \n")
            output_agent_tenosrs_code.append(f"  typedef BlockReduce< \n")
            output_agent_tenosrs_code.append(
                f"            TensorOutput_{_name}_T, \n")
            output_agent_tenosrs_code.append(f"            BLOCK_THREADS, \n")
            output_agent_tenosrs_code.append(
                f"            BLOCK_REDUCE_WARP_REDUCTIONS> \n")
            output_agent_tenosrs_code.append(
                f"            BlockReduce_{_name}_T; \n")
            output_agent_SMEM_code.append(
                f"               typename BlockReduce_{_name}_T::TempStorage smem_{_name}; \n")
        elif 'norm' in str(out['target']):
            # aggregator
            # # norm is special case that it is not a tensor, but a value
            # output_agent_forloop_code.append(
            #     f"  for (int i = 0; i < {_dim}; i++) \n")
            # output_agent_forloop_code.append(f"  {{ \n")
            # output_agent_forloop_code.append(
            #     f"    spmv_params.output_y_{out_name}_ptr[(tile_start_coord.y + nonzero_idx) \
            #         * {_dim} + i] = {out_name}; \n")
            # output_agent_forloop_code.append(f"  }} \n")
            _name = out_name
            output_agent_tenosrs_code.append(
                f"  // Tensor type and block reduce for output \n")
            output_agent_tenosrs_code.append(
                f"  typedef Tensor<ValueT, {_dim}> TensorOutput_{_name}_T; \n")
            output_agent_tenosrs_code.append(f"  typedef BlockReduce< \n")
            output_agent_tenosrs_code.append(
                f"            TensorOutput_{_name}_T, \n")
            output_agent_tenosrs_code.append(f"            BLOCK_THREADS, \n")
            output_agent_tenosrs_code.append(
                f"            BLOCK_REDUCE_WARP_REDUCTIONS> \n")
            output_agent_tenosrs_code.append(
                f"            BlockReduce_{_name}_T; \n")
            output_agent_SMEM_code.append(
                f"               typename BlockReduce_{_name}_T::TempStorage smem_{_name}; \n")
        else:
            # map inside the forloop
            if use_direct_map_kernel:
                output_agent_forloop_code.append(
                    f"    spmv_params.output_y_{out_name}_ptr[flat_idx] = "
                    f"{out_name}.values[0]; \n"
                )
            else:
                output_agent_forloop_code.append(f"  #pragma unroll \n")
                output_agent_forloop_code.append(
                    f"  for (int i = 0; i < {_dim}; i++) \n")
                output_agent_forloop_code.append(f"  {{ \n")
                output_agent_forloop_code.append(
                    f"    spmv_params.output_y_{out_name}_ptr[{item_index_expr} \
                        * {_dim} + i] = {out_name}.values[i]; \n")
                output_agent_forloop_code.append(f"  }} \n")

    # debug print
    # if os.getenv("EASIER_VERBOSE_CODEGEN") in ("1", "", "true", "True"):
    for inp in input_declarations_utils_code:
        logger.debug(f"Input declarations utils: {inp}")
    for inp in input_declarations_code:
        logger.debug(f"Input declarations: {inp}")
    for inp in input_init_code:
        logger.debug(f"Input init code: {inp}")
    for inp in input_agent_tenosrs_code:
        logger.debug(f"Input agent tenosrs code: {inp}")
    for out in output_agent_tenosrs_code:
        logger.debug(f"Output agent tenosrs code: {out}")
    for out in output_agent_SMEM_code:
        logger.debug(f"Output agent SMEM code: {out}")
    for out in output_agent_forloop_code:
        logger.debug(f"Output agent forloop code: {out}")

    # Generate the selector register code
    selector_code = []
    for inter in selector_register:
        # obtain the dimension of the selector
        _dim = get_dim_length(inter['shape'])
        _name = inter['name']
        _target = inter['target'] # target is the args[0]
        _selector_name = inter['selector_name'] # selector_name is the target
        _shape_ahead = inter['shape_ahead']
        _shape_all = inter['shape_all']
        if use_direct_map_kernel and _name in direct_overwrite_only_targets:
            continue
        if inter['selector'] == 1:
            # load the selector register
            current = f"{_name}_ptr_current"
            if use_direct_map_kernel:
                selector_code.append(
                    f"    const OffsetT {current} = "
                    f"spmv_params.{_selector_name}_ptr[{item_index_expr}]; \n"
                )
                selector_code.append(
                    f"    TensorInput_{_target}_T {_name}("
                    f"spmv_params.{_target}_ptr + "
                    f"static_cast<size_t>({current}) * {_dim} + lane_idx); \n"
                )
            elif (
                use_register_fed_merge
                and _selector_name in coindexed_reducer_selectors
            ):
                selector_code.append(
                    f"    const OffsetT {current} = row_idx; \n"
                )
                selector_code.append(
                    f"    TensorInput_{_target}_T {_name}("
                    f"{_target}_ptr + "
                    f"static_cast<size_t>({current}) * {_dim}); \n"
                )
            else:
                selector_code.append(
                    f"    ColumnIndicesIteratorT {current} = \
                        {_selector_name}_ptr + {item_index_expr}; \n")
                selector_code.append(
                    f"    TensorInput_{_target}_T \
                        {_name}({_target}_ptr + *{current} * {_dim}); \n")
        else:
            # spm loading
            
            if _shape_ahead > 1:
                # for vector
                current = f"{_name}_ptr_current"
                if use_direct_map_kernel:
                    selector_code.append(
                        f"    TensorInput_{_target}_T {_selector_name}("
                        f"spmv_params.{_selector_name}_ptr + "
                        f"static_cast<size_t>({item_index_expr}) * {_dim} "
                        f"+ lane_idx); \n"
                    )
                else:
                    selector_code.append(
                        f"    VectorValueIteratorT {current} = {_selector_name}_ptr + \
                            {item_index_expr} * {_dim}; \n")
                    selector_code.append(
                        f"    TensorInput_{_target}_T {_selector_name}({_target}_ptr_current); \n")
            else: 
                # for scalar
                current = f"{_name}_ptr_current"
                if use_direct_map_kernel:
                    selector_code.append(
                        f"    TensorInput_{_target}_T {_selector_name}("
                        f"spmv_params.{_selector_name}_ptr); \n"
                    )
                else:
                    selector_code.append(
                        f"    VectorValueIteratorT {current} = {_selector_name}_ptr; \n")
                    selector_code.append(
                        f"    TensorInput_{_target}_T {_selector_name}({_target}_ptr_current); \n")

    # debug print
    # if os.getenv("EASIER_VERBOSE_CODEGEN") in ("1", "", "true", "True"):
    for inter in selector_code:
        logger.debug(f"Selector code: {inter}")

    # Generate the CUDA kernel code (map operations)
    map_code = []
    map_agent_tenosrs_code = []
    for op in map_operations:
        _name = op['name']
        _op = op['op']
        _dim = get_dim_length(op['shape'])

        # add the declarations for the map agent tenosrs
        if _op != 'reducer' and 'sum' not in _op and 'norm' not in _op:
            map_agent_tenosrs_code.append(
                f"  typedef Tensor<ValueT, "
                f"{1 if use_direct_map_kernel else _dim}> "
                f"TensorOutput_{_name}_T; \n")

        if _op == 'add':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = {op['args'][0]} + \
                    {op['args'][1]}; \n")
        elif _op == 'sub':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = {op['args'][0]} - \
                    {op['args'][1]}; \n")
        elif _op == 'mul':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = {op['args'][0]} * \
                    {op['args'][1]}; \n")
        elif _op == 'pow':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = {op['args'][0]} ^ \
                    {op['args'][1]}; \n")
        elif _op == 'truediv':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = {op['args'][0]} / \
                    {op['args'][1]}; \n")
        elif _op == 'neg':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = -{op['args'][0]}; \n")
        elif _op == 'exp':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = {op['args'][0]}.exp(); \n")
        elif _op == 'getitem':
            _variable_name = op['args'][0]
            _index = op['args'][1][1]
            _shape = op['shape']
            if str(_index) == 'Ellipsis':
                # this is very speicial case in the GMRES solver
                # here is the slicing operation for a tensor
                # [TODO: check it for high dimesion, such as slice dimension > 3]
                # index is changed to slicing index
                _index = op['args'][1][2]
                if len(_shape) == 1 or _index == None:
                    # this is only the pure copy operation
                    map_code.append(
                        f"    TensorOutput_{_name}_T {_name}({_variable_name}); \n")
                else:
                    # this is the slicing operation for a tensor
                    _len_shape_slicing = np.prod(_shape[1:])
                    map_code.append(
                        f"    TensorOutput_{_name}_T {_name}\
                            ({_variable_name}.values[{_index * _len_shape_slicing}]); \n")
            else:
                # this is only for the normal case
                map_code.append(
                    f"    TensorOutput_{_name}_T \
                        {_name}({_variable_name}.values[{_index}]); \n")
        elif _op == 'clone':
            _variable_name = op['args'][0]
            map_code.append(
                f"    TensorOutput_{_name}_T {_name}({_variable_name}); \n")
        elif _op == 'squeeze':
            # squeeze is no-op considering continuous memory layout
            _variable_name = op['args'][0]
            map_code.append(
                f"    TensorOutput_{_name}_T {_name}({_variable_name}); \n")
        elif _op == 'copy_':
            _output_name = op['args'][0]
            _variable_name = op['args'][1]
            map_code.append(f"  #pragma unroll \n")
            map_code.append(f"  for (int i = 0; i < {_dim}; i++) \n")
            map_code.append(f"  {{ \n")
            map_code.append(f"    spmv_params.{_output_name}_ptr[\
                {item_index_expr} * {_dim} + i] = \
                    {_variable_name}.values[i]; \n")
            map_code.append(f"  }} \n")
        elif _op == 'add_':
            map_code.append(
                f"{op['args'][0]} = {op['args'][0]} + \
                    {op['args'][1]}; \n")
            map_code.append(f"  #pragma unroll \n")
            map_code.append(f"  for (int i = 0; i < {_dim}; i++) \n")
            map_code.append(f"  {{ \n")
            map_code.append(f"    spmv_params.{op['args'][0]}_ptr[\
                {item_index_expr} * {_dim} + i] = \
                    {op['args'][0]}.values[i]; \n")
            map_code.append(f"  }} \n")
        elif _op == 'setitem':
            if use_direct_map_kernel:
                map_code.append(
                    f"    spmv_params.{op['args'][0]}_ptr[flat_idx] = "
                    f"{op['args'][2]}.values[0]; \n"
                )
                if op['args'][0] not in direct_overwrite_only_targets:
                    map_code.append(
                        f"    {op['args'][0]}.values[0] = "
                        f"{op['args'][2]}.values[0]; \n"
                    )
            else:
                map_code.append(f"  #pragma unroll \n")
                map_code.append(f"  for (int i = 0; i < {_dim}; i++) \n")
                map_code.append(f"  {{ \n")
                map_code.append(f"    spmv_params.{op['args'][0]}_ptr[\
                    {item_index_expr} * {_dim} + i] = \
                        {op['args'][2]}.values[i]; \n")
                map_code.append(f"  }} \n")
                # the updated output may serves as input for a new function
                map_code.append(f"  #pragma unroll \n")
                map_code.append(f"  for (int i = 0; i < {_dim}; i++) \n")
                map_code.append(f"  {{ \n")
                map_code.append(f"    {op['args'][0]}.values[i] = {op['args'][2]}.values[i];\n")
                map_code.append(f"  }} \n")
        elif _op == 'reducer':
            if use_register_fed_merge:
                map_code.append(
                    f"    return {op['args'][0]}; \n"
                )
            else:
                map_code.append(
                    f"    temp_storage.smem_{_name}.s_tile_value_reducer[nonzero_idx] = \
                        {op['args'][0]}; \n")
        elif _op == 'sum':
            # aggregator
            # map_code.append(f"    {_name} = {_name} + {op['args'][0]}; \n")
            map_code.append(f"    {_name} = {op['args'][0]}; \n")
        elif _op == 'norm':
            # map_code.append(
            #     f"    ValueT {_name} = {op['args'][0]}.l2Norm(); \n")
            # aggregator
            map_code.append(f"    {_name} = {op['args'][0]} * {op['args'][0]}; \n")
        elif _op == 'abs':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = {op['args'][0]}.abs(); \n")
        elif _op == 'sign':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = {op['args'][0]}.sign(); \n")
        elif _op == 'lt':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = {op['args'][0]} < {op['args'][1]}; \n")
        elif _op == 'gt':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = {op['args'][0]} > {op['args'][1]}; \n")
        elif _op == 'where':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = \
                    _where({op['args'][0]}, {op['args'][1]}, {op['args'][2]}); \n")
        else:
            # error
            raise ValueError(f"Operation {_op} not supported")

    # # Debug print to check kernel operations
    # if os.getenv("EASIER_VERBOSE_CODEGEN") in ("1", "", "true", "True"):
    for op in map_code:
        logger.debug(f"Map code: {op}")
    for op in map_agent_tenosrs_code:
        logger.debug(f"Map agent tenosrs code: {op}")

    # Generate the aggregator code
    aggregator_code = []
    aggregator_reg_definitions = []
    aggregator_norm_code_sqrt_after_reduction_code = []

    for op in aggregator_operations:
        _name = op['name']
        _dim = get_dim_length(op['shape'])
        aggregator_reg_definitions.append(
            f"   // each aggregator need a register to store the non-zero values \n")
        aggregator_reg_definitions.append(
            f"   TensorOutput_{_name}_T {_name}; \n")
        if op['op'] == 'sum':
            aggregator_code.append(f"   // blockReduce \n")
            aggregator_code.append(
                f"   TensorOutput_{_name}_T {_name}_result = \
                    BlockReduce_{_name}_T(temp_storage.smem_{_name}).Sum({_name}); \n")
            aggregator_code.append(f"   if (threadIdx.x == 0) \n")
            aggregator_code.append(f"   {{ \n")
            aggregator_code.append(f"     #pragma unroll \n")
            aggregator_code.append(f"     for (int i = 0; i < {_dim}; i++) \n")
            aggregator_code.append(f"     {{ \n")
            aggregator_code.append(
                f"       atomicAdd(&spmv_params.output_y_{_name}_ptr[i], \
                    {_name}_result.values[i]); \n")
            aggregator_code.append(f"     }} \n")
            aggregator_code.append(f"   }} \n")
        elif op['op'] == 'norm':
            aggregator_code.append(f"   // blockReduce \n")
            aggregator_code.append(
                f"   TensorOutput_{_name}_T {_name}_result = \
                    BlockReduce_{_name}_T(temp_storage.smem_{_name}).Sum({_name}); \n")
            aggregator_code.append(f"   if (threadIdx.x == 0) \n")
            aggregator_code.append(f"   {{ \n")
            aggregator_code.append(f"     #pragma unroll \n")
            aggregator_code.append(f"     for (int i = 0; i < {_dim}; i++) \n")
            aggregator_code.append(f"     {{ \n")
            aggregator_code.append(
                f"       atomicAdd(&spmv_params.output_y_{_name}_ptr[i], \
                    {_name}_result.values[i]); \n")
            aggregator_code.append(f"     }} \n")
            aggregator_code.append(f"   }} \n")
            # add sqrt operation
            aggregator_norm_code_sqrt_after_reduction_code.append(
                f"   sqrt_tensor_kernel<<<1, 1, 0, stream>>>( \n")
            aggregator_norm_code_sqrt_after_reduction_code.append(
                f"     spmv_params.output_y_{_name}_ptr, \n")
            aggregator_norm_code_sqrt_after_reduction_code.append(
                f"     {_dim} \n")
            aggregator_norm_code_sqrt_after_reduction_code.append(
                f"   ); \n")
        else:
            raise ValueError(f"Operation {op['op']} not supported")

    # debug print
    # if os.getenv("EASIER_VERBOSE_CODEGEN") in ("1", "", "true", "True"):
    for op in aggregator_reg_definitions:
        logger.debug(f"Aggregator reg definitions: {op}")
    for op in aggregator_code:
        logger.debug(f"Aggregator code: {op}")
    for op in aggregator_norm_code_sqrt_after_reduction_code:
        logger.debug(f"Aggregator norm code sqrt after reduction code: {op}")

    # Generate the reducer code
    reducer_code = []
    reducer_carry_lane_offset = 0
    register_fed_evaluator_code = []
    nonzero_preparation_code = []
    nonzero_initialization_code = []
    nonzero_load_code = []
    nonzero_advance_code = []

    if use_register_fed_merge:
        reducer_name = reducer_operations[0]["name"]
        register_fed_evaluator_code.extend([
            "        __device__ __forceinline__\n",
            f"        TensorOutput_{reducer_name}_T "
            "evaluate_reducer_nonzero(\n",
            "            OffsetT nonzero_idx, OffsetT row_idx)\n",
            "        {\n",
            *selector_code,
            *map_code,
            "        }\n\n",
        ])
        nonzero_load_code.extend([
            "                    TensorT nonzero = "
            "evaluate_reducer_nonzero(\n",
            "                        tile_start_coord.y + "
            "thread_current_coord.y,\n",
            "                        tile_start_coord.x + "
            "thread_current_coord.x);\n",
        ])
    else:
        nonzero_preparation_code.extend([
            "// Select\n",
            "// Gather the nonzeros for the merge tile into shared memory\n",
            "#pragma unroll\n",
            "            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)\n",
            "            {\n",
            "                int nonzero_idx = threadIdx.x + "
            "(ITEM * BLOCK_THREADS);\n\n",
            "                if (nonzero_idx < tile_num_nonzeros)\n",
            "                {\n",
            "                    // [code generation]\n",
            *selector_code,
            "\n                    // map\n",
            *map_code,
            "\n                    //output for map\n",
            *output_agent_forloop_code,
            "                }\n",
            "            }\n",
            "            CTA_SYNC();\n",
        ])
        nonzero_load_code.append(
            ""
        )
        nonzero_initialization_code.append(
            "            TensorT nonzero = "
            "s_tile_value_nonzeros[thread_current_coord.y];\n"
        )
        nonzero_advance_code.append(
            "                    nonzero = "
            "s_tile_value_nonzeros[thread_current_coord.y];\n"
        )

    # generate the reducer code for each reducer
    for op in reducer_operations:
        # get the dimension length of the shape
        _dim = get_dim_length(op['shape'])
        _name = op['name']
        # generate the SMEM definitions and the reducer code
        reducer_code.append(
            f"   reduce<{_dim}, BlockScan_{_name}_T, TensorOutput_{_name}_T, \
                ReduceBySegmentOp_{_name}_T>( \n")
        reducer_code.append(
            f"                temp_storage.smem_{_name}.s_tile_value_reducer,          \
                ///< [in, code gen] Shared memory array of non-zero values for the merge tile \n")
        reducer_code.append(
            f"                temp_storage.s_tile_row_end_offsets,         \
                ///< [in, code gen] Shared memory array of row end offsets for the merge tile \n")
        reducer_code.append(
            f"                tile_start_coord,               \
                ///< [in] Starting coordinate of the merge tile \n")
        reducer_code.append(
            f"                tile_end_coord,                 \
                ///< [in] Ending coordinate of the merge tile \n")
        reducer_code.append(
            f"                thread_start_coord,             \
                ///< [in] Starting coordinate of the thread \n")
        reducer_code.append(
            f"                tile_num_rows,                  \
                ///< [in] Number of rows in the merge tile \n")
        reducer_code.append(
            f"                tile_num_nonzeros,               \
                ///< [in] Number of non-zeros in the merge tile \n")
        reducer_code.append(
            f"                spmv_params.output_y_{_name}_ptr,      \
                 ///< [out] Output vector y \n")
        reducer_code.append(
            f"                temp_storage.smem_{_name}.scan,        \
                ///< [in] Scan storage for BlockScanT \n")
        reducer_code.append(
            f"                tile_idx, num_merge_tiles,             \
                ///< [in] Tile identity and carry-buffer stride \n")
        reducer_code.append(
            f"                {reducer_carry_lane_offset}             \
                ///< [in] First carry lane for this reducer \n")
        reducer_code.append(f"            ); \n")
        reducer_code.append(f"   CTA_SYNC(); \n")
        reducer_carry_lane_offset += _dim

    if reducer_carry_lane_offset != sum(dim for _, dim in reducer_outputs):
        raise RuntimeError(
            "reducer operation/output dimensions disagree during CUDA codegen"
        )

    reducer_fixup_launch_code = []
    reducer_carry_lane_offset = 0
    for name, dim in reducer_outputs:
        reducer_fixup_launch_code.append(
            f"            ReducerCarryFixupKernel<{dim}>"
            f"<<<carry_grid_size, carry_block_size, 0, stream>>>(\n"
        )
        reducer_fixup_launch_code.append(
            "                spmv_params.d_tile_carry_keys,\n"
        )
        reducer_fixup_launch_code.append(
            "                spmv_params.d_tile_carry_values + "
            f"static_cast<size_t>({reducer_carry_lane_offset}) * "
            "num_merge_tiles,\n"
        )
        reducer_fixup_launch_code.append(
            f"                spmv_params.output_y_{name}_ptr,\n"
        )
        reducer_fixup_launch_code.append(
            "                num_merge_tiles, spmv_params.num_rows);\n"
        )
        reducer_fixup_launch_code.append(
            "            if (CubDebug(error = cudaPeekAtLastError()))\n"
            "                break;\n"
        )
        reducer_carry_lane_offset += dim

    # # debug print
    # if os.getenv("EASIER_VERBOSE_CODEGEN") in ("1", "", "true", "True"):
    for op in reducer_code:
        logger.debug(f"Reducer code: {op}")

    # Create directories for generated code if they don't exist
    project_root = os.path.abspath(os.path.dirname(__file__))
    include_dir = os.path.join(project_root, "include")

    # Read template files
    # Map-only regions normally use the aggregator template.  Eligible
    # selector/map/full-overwrite regions use a direct flat kernel instead.
    _folder = 'reducer' if reducer_operations != [] else 'aggregator'
    kernel_agent_template = None
    if not use_direct_map_kernel:
        template_agent_path = os.path.join(
            project_root,
            "cuda_template",
            _folder,
            "merged_agent_spmv_template.cuh"
        )
        with open(template_agent_path, "r") as f:
            kernel_agent_template = f.read()

    template_utils_path = os.path.join(
        project_root, 
        "cuda_template", 
        "merged_utils_template.cuh"
    )
    with open(template_utils_path, "r") as f:
        utils_template = f.read()

    template_spmv_path = os.path.join(
        project_root,
        "cuda_template",
        "direct_map" if use_direct_map_kernel else _folder,
        "merged_spmv_template.cuh"
    )
    with open(template_spmv_path, "r") as f:
        kernel_spmv_template = f.read()

    # Apply templates using string.Template
    def trans_str(code):
        if code == []:
            return ''
        else:
            return ''.join(code)

    input_declarations_str = trans_str(input_declarations_code)
    input_init_str = trans_str(input_init_code)
    input_agent_tenosrs_code_str = trans_str(input_agent_tenosrs_code)
    selector_str = trans_str(selector_code)
    map_str = trans_str(map_code)
    reducer_code_str = trans_str(reducer_code)
    aggregator_reg_definitions_str = trans_str(aggregator_reg_definitions)
    aggregator_code_str = trans_str(aggregator_code)
    aggregator_norm_code_sqrt_after_reduction_code_str = \
        trans_str(aggregator_norm_code_sqrt_after_reduction_code)
    output_agent_tenosrs_code_str = trans_str(output_agent_tenosrs_code)
    output_agent_SMEM_code_str = trans_str(output_agent_SMEM_code)
    output_agent_forloop_code_str = trans_str(output_agent_forloop_code)
    map_agent_tenosrs_code_str = trans_str(map_agent_tenosrs_code)
    input_declarations_utils_str = trans_str(input_declarations_utils_code)

    if kernel_agent_template is not None:
        agent_kernel_code = string.Template(kernel_agent_template).substitute(
            input_declarations_code=input_declarations_str,
            input_init_code=input_init_str,
            input_agent_tenosrs_code=input_agent_tenosrs_code_str,
            selector_code=selector_str,
            map_code=map_str,
            reducer_code=reducer_code_str,
            aggregator_reg_definitions=aggregator_reg_definitions_str,
            aggregator_code=aggregator_code_str,
            output_agent_tenosrs_code=output_agent_tenosrs_code_str,
            output_agent_SMEM_code=output_agent_SMEM_code_str,
            output_agent_forloop_code=output_agent_forloop_code_str,
            map_agent_tenosrs_code=map_agent_tenosrs_code_str,
            register_fed_evaluator_code=trans_str(
                register_fed_evaluator_code
            ),
            nonzero_preparation_code=trans_str(
                nonzero_preparation_code
            ),
            nonzero_initialization_code=trans_str(
                nonzero_initialization_code
            ),
            nonzero_load_code=trans_str(nonzero_load_code),
            nonzero_advance_code=trans_str(nonzero_advance_code),
        )
    else:
        agent_kernel_code = (
            "// Direct gather-map-store code is emitted in merged_spmv.cuh.\n"
        )

    # norm aggregator
    if use_direct_map_kernel:
        spmv_kernel_code = string.Template(kernel_spmv_template).substitute(
            input_agent_tenosrs_code=input_agent_tenosrs_code_str,
            output_agent_tenosrs_code=output_agent_tenosrs_code_str,
            map_agent_tenosrs_code=map_agent_tenosrs_code_str,
            selector_code=selector_str,
            map_code=map_str,
            output_agent_forloop_code=output_agent_forloop_code_str,
            direct_map_lane_width=direct_map_lane_width,
        )
    else:
        spmv_kernel_code = string.Template(kernel_spmv_template).substitute(
            aggregator_norm_code_sqrt_after_reduction=\
                aggregator_norm_code_sqrt_after_reduction_code_str,
            total_reducer_dim=sum(dim for _, dim in reducer_outputs),
            reducer_fixup_launch_code=trans_str(
                reducer_fixup_launch_code
            ),
        )

    utils_code = string.Template(utils_template).substitute(
        input_declarations_utils_code=input_declarations_utils_str
    )

    # Write files
    with open(os.path.join(include_dir, "merged_agent_spmv.cuh"), "w") as f:
        f.write(agent_kernel_code)

    with open(os.path.join(include_dir, "merged_spmv.cuh"), "w") as f:
        f.write(spmv_kernel_code)

    with open(os.path.join(include_dir, "merged_utils.cuh"), "w") as f:
        f.write(utils_code)

    logger.info("CUDA code generated successfully!")

    # Delegate binding and wrapper generation to dedicated modules
    generate_binding_code(
        project_root,
        inputs,
        outputs,
        selector_register,
        direct_map=use_direct_map_kernel,
        forwarded_output_indices=forwarded_output_indices,
    )
