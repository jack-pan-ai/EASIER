# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import string
import numpy as np

from easier.core.codegen.utils import get_dim_length
from easier.core.codegen.merged_gen_binding import generate_binding_code
from easier.core.codegen.trace.graph_trace import trace_graph


def generate_cuda_code_from_graph(submodule, traced_model):
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

    # Generate the code
    # Generate the input/output code
    input_declarations_code = []
    input_init_code = []
    input_declarations_utils_code = []
    input_agent_tenosrs_code = []
    output_agent_tenosrs_code = []
    output_agent_SMEM_code = []
    output_agent_forloop_code = []
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
                    f"  typedef Tensor<ValueT, {_dim}> TensorInput_{name}_T; \n")
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
            _name = out_name
            output_agent_tenosrs_code.append(
                f"  // Tensor type and block reducefor output \n")
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
            # norm is special case that it is not a tensor, but a value
            output_agent_forloop_code.append(
                f"  for (int i = 0; i < {_dim}; i++) \n")
            output_agent_forloop_code.append(f"  {{ \n")
            output_agent_forloop_code.append(
                f"    spmv_params.output_y_{out_name}_ptr[(tile_start_coord.y + nonzero_idx) \
                    * {_dim} + i] = {out_name}; \n")
            output_agent_forloop_code.append(f"  }} \n")
        else:
            # map inside the forloop
            output_agent_forloop_code.append(f"  #pragma unroll \n")
            output_agent_forloop_code.append(
                f"  for (int i = 0; i < {_dim}; i++) \n")
            output_agent_forloop_code.append(f"  {{ \n")
            output_agent_forloop_code.append(
                f"    spmv_params.output_y_{out_name}_ptr[(tile_start_coord.y + nonzero_idx) \
                    * {_dim} + i] = {out_name}.values[i]; \n")
            output_agent_forloop_code.append(f"  }} \n")

    # debug print
    if os.getenv("EASIER_VERBOSE_CODEGEN") in ("1", "", "true", "True"):
        for inp in input_declarations_utils_code:
            print(f"Input declarations utils: {inp}")
        for inp in input_declarations_code:
            print(f"Input declarations: {inp}")
        for inp in input_init_code:
            print(f"Input init code: {inp}")
        for inp in input_agent_tenosrs_code:
            print(f"Input agent tenosrs code: {inp}")
        for out in output_agent_tenosrs_code:
            print(f"Output agent tenosrs code: {out}")
        for out in output_agent_SMEM_code:
            print(f"Output agent SMEM code: {out}")
        for out in output_agent_forloop_code:
            print(f"Output agent forloop code: {out}")

    # Generate the selector register code
    selector_code = []
    for inter in selector_register:
        # obtain the dimension of the selector
        _dim = get_dim_length(inter['shape'])
        _name = inter['name']
        _target = inter['target'] # target is the args[0]
        _selector_name = inter['selector_name'] # selector_name is the target
        if inter['selector'] == 1:
            # load the selector register
            current = f"{_name}_ptr_current"
            selector_code.append(
                f"    ColumnIndicesIteratorT {current} = \
                    {_selector_name}_ptr + tile_start_coord.y + nonzero_idx; \n")
            selector_code.append(
                f"    TensorInput_{_target}_T \
                    {_name}({_target}_ptr + *{current} * {_dim}); \n")
        else:
            # spm loading
            current = f"{_name}_ptr_current"
            selector_code.append(
                f"    VectorValueIteratorT {current} = {_selector_name}_ptr + \
                    (tile_start_coord.y + nonzero_idx) * {_dim}; \n")
            selector_code.append(
                f"    TensorInput_{_target}_T {_selector_name}({_target}_ptr_current); \n")

    # debug print
    if os.getenv("EASIER_VERBOSE_CODEGEN") in ("1", "", "true", "True"):
        for inter in selector_code:
            print(f"Selector code: {inter}")

    # Generate the CUDA kernel code (map operations)
    map_code = []
    map_agent_tenosrs_code = []
    for op in map_operations:
        _name = op['name']
        _op = op['op']
        _dim = get_dim_length(op['shape'])

        # add the declarations for the map agent tenosrs
        if _op != 'reducer' and 'sum' not in _op:
            map_agent_tenosrs_code.append(
                f"  typedef Tensor<ValueT, {_dim}> TensorOutput_{_name}_T; \n")

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
                (tile_start_coord.y + nonzero_idx) * {_dim} + i] = \
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
                (tile_start_coord.y + nonzero_idx) * {_dim} + i] = \
                    {op['args'][0]}.values[i]; \n")
            map_code.append(f"  }} \n")
        elif _op == 'setitem':
            map_code.append(f"  #pragma unroll \n")
            map_code.append(f"  for (int i = 0; i < {_dim}; i++) \n")
            map_code.append(f"  {{ \n")
            map_code.append(f"    spmv_params.{op['args'][0]}_ptr[\
                (tile_start_coord.y + nonzero_idx) * {_dim} + i] = \
                    {op['args'][2]}.values[i]; \n")
            map_code.append(f"  }} \n")
            # the updated output may serves as input for a new function
            map_code.append(f"  #pragma unroll \n")
            map_code.append(f"  for (int i = 0; i < {_dim}; i++) \n")
            map_code.append(f"  {{ \n")
            map_code.append(f"    {op['args'][0]}.values[i] = {op['args'][2]}.values[i];\n")
            map_code.append(f"  }} \n")
        elif _op == 'norm':
            map_code.append(
                f"    ValueT {_name} = {op['args'][0]}.l2Norm(); \n")
        elif _op == 'reducer':
            map_code.append(
                f"    temp_storage.smem_{_name}.s_tile_value_reducer[nonzero_idx] = \
                    {op['args'][0]}; \n")
        elif _op == 'sum':
            map_code.append(f"    {_name} = {_name} + {op['args'][0]}; \n")
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
    if os.getenv("EASIER_VERBOSE_CODEGEN") in ("1", "", "true", "True"):
        for op in map_code:
            print("Map code: ", op)
        for op in map_agent_tenosrs_code:
            print("Map agent tenosrs code: ", op)

    # Generate the aggregator code
    aggregator_code = []
    aggregator_reg_definitions = []

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

    # debug print
    if os.getenv("EASIER_VERBOSE_CODEGEN") in ("1", "", "true", "True"):
        for op in aggregator_reg_definitions:
            print(f"Aggregator reg definitions: {op}")
        for op in aggregator_code:
            print(f"Aggregator code: {op}")

    # Generate the reducer code
    reducer_code = []

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
            f"                temp_storage.smem_{_name}.scan         \
                ///< [in] Scan storage for BlockScanT \n")
        reducer_code.append(f"            ); \n")
        reducer_code.append(f"   CTA_SYNC(); \n")

    # # debug print
    if os.getenv("EASIER_VERBOSE_CODEGEN") in ("1", "", "true", "True"):
        for op in reducer_code:
            print(f"Reducer code: {op}")

    # Create directories for generated code if they don't exist
    project_root = os.path.abspath(os.path.dirname(__file__))
    include_dir = os.path.join(project_root, "include")

    # Read template files
    # map only will use the aggregator template
    _folder = 'reducer' if reducer_operations != [] else 'aggregator'
    template_agent_path = os.path.join(project_root, "cuda_template", _folder, "merged_agent_spmv_template.cuh")
    with open(template_agent_path, "r") as f:
        kernel_agent_template = f.read()

    template_utils_path = os.path.join(project_root, "cuda_template", "merged_utils_template.cuh")
    with open(template_utils_path, "r") as f:
        utils_template = f.read()

    template_spmv_path = os.path.join(project_root, "cuda_template", _folder, "merged_spmv_template.cuh")
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
    output_agent_tenosrs_code_str = trans_str(output_agent_tenosrs_code)
    output_agent_SMEM_code_str = trans_str(output_agent_SMEM_code)
    output_agent_forloop_code_str = trans_str(output_agent_forloop_code)
    map_agent_tenosrs_code_str = trans_str(map_agent_tenosrs_code)

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
    )

    spmv_kernel_code = string.Template(kernel_spmv_template).substitute(
    )

    # Join the list of input declarations into a single string
    input_declarations_utils_str = ''.join(input_declarations_utils_code)

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

    print("CUDA code generated successfully!")

    # Delegate binding and wrapper generation to dedicated modules
    generate_binding_code(project_root, inputs, outputs, selector_register)
