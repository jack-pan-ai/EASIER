# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import string
import numpy as np
from easier.core.passes.code_gen.utils import get_dim_length
from easier.core.utils import logger

# debug print
def debug_print(code, code_name):
    if code != []:
        for op in code:
            logger.debug(f"{code_name}: {op}")
    else:
        logger.debug(f"{code_name} is empty")


def declarations_gen(
        inputs,
        outputs,
        selector_register):
    """
    Generate the code for the variables declarations

    Args:
        inputs: list of input variables
        outputs: list of output variables
        selector_register: list of selector variables

    Returns:
        input_parameters_code: list of code for the input parameters
        input_agent_tenosrs_code: list of code for the input agent tensors
        output_agent_tenosrs_code: list of code for the output agent tensors
        output_agent_forloop_code: list of code for the output agent forloop
        selector_code: list of code for the selector
    """

    # Generate the input/output code
    input_parameters_code = []
    input_agent_tenosrs_code = []
    output_agent_tenosrs_code = []
    output_agent_forloop_code = []
    _tensor_names = set()
    _tensor_target_selector_set = set()
    for inp in inputs:
        name = inp['name']
        target = inp['target']
        if inp['dtype'] == 'int':
            if target not in _tensor_target_selector_set:
                # column indices
                input_parameters_code.append(
                    f"  OffsetT *__restrict {target}_ptr, \n")
                _tensor_target_selector_set.add(target)
        else:
            if str(inp['dtype_data']) == 'torch.int32' and inp['shape_ahead'] == 1:
                # this is very speicial case in the GMRES solver
                input_parameters_code.append(
                    f"  int {name}_const, \n")
            else:
                # spm and vector x
                input_parameters_code.append(
                    f"  ValueT *__restrict {name}_ptr, \n")

                if inp['target'] not in _tensor_names:
                    _dim = get_dim_length(inp['shape'])
                    target = inp['target']
                    input_agent_tenosrs_code.append(
                        f"  typedef Tensor<ValueT, {_dim}> TensorInput_{target}_T; \n")
                    _tensor_names.add(target)

    # output code in the declarations utils file
    for out in outputs:
        _name = out['name']
        target_name = out['target']
        input_parameters_code.append(
            f"  ValueT *__restrict output_y_{_name}_ptr, \n")

        _dim = get_dim_length(out['shape'])
        if 'reducer' in str(out['target']):
            # reducer outside the forloop
            output_agent_tenosrs_code.append(
                f"  typedef Tensor<ValueT, {_dim}> TensorOutput_{_name}_T; \n")
            output_agent_tenosrs_code.append(
                f" OffsetT row_carry_out_{_name}[256]; \n")
            output_agent_tenosrs_code.append(
                f" TensorOutput_{_name}_T value_carry_out_{_name}[256]; \n")
        elif 'sum' in str(out['target']) or 'norm' in str(out['target']):
            # aggregator
            output_agent_tenosrs_code.append(
                f"  typedef Tensor<ValueT, {_dim}> TensorOutput_{_name}_T; \n")
        # elif 'norm' in str(out['target']):
        #     # norm is special case that it is not a tensor, but a value
        #     output_agent_forloop_code.append(
        #         f"  for (int i = 0; i < {_dim}; i++) \n")
        #     output_agent_forloop_code.append(f"  {{ \n")
        #     output_agent_forloop_code.append(
        #         f"    output_y_{_name}_ptr[thread_coord.y * {_dim} + i] = {_name}; \n")
        #     output_agent_forloop_code.append(f"  }} \n")
        else:
            # map consume the main part
            output_agent_forloop_code.append(
                f"  for (int i = 0; i < {_dim}; i++) \n")
            output_agent_forloop_code.append(f"  {{ \n")
            output_agent_forloop_code.append(
                f"    output_y_{_name}_ptr[thread_coord.y * {_dim} + i] = {_name}.values[i]; \n")
            output_agent_forloop_code.append(f"  }} \n")

    # Generate the selector register code
    selector_code = []
    for inter in selector_register:
        # obtain the dimension of the selector
        _dim = get_dim_length(inter['shape'])
        _name = inter['name']
        _target = inter['target']
        _selector_name = inter['selector_name']
        _shape_ahead = inter['shape_ahead']
        _shape_all = inter['shape_all']
        if inter['selector'] == 1:
            # load the selector register
            column_indices = f"column_indices_{_name}"
            selector_ptr = f"{_selector_name}_ptr"
            target_ptr = f"{_target}_ptr"
            selector_code.append(
                f"  OffsetT {column_indices} = {selector_ptr}[thread_coord.y]; \n")
            selector_code.append(
                f"  TensorInput_{_target}_T {_name}({target_ptr} + \
                    {column_indices} * {_dim}); \n")
        else:
            # # spm loading
            # target_ptr = f"{_target}_ptr"
            # selector_code.append(
            #     f"  TensorInput_{_target}_T {_selector_name}({target_ptr} + \
            #         thread_coord.y * {_dim}); \n")
            if _shape_ahead > 1:
                # for vector
                target_ptr = f"{_target}_ptr"
                selector_code.append(
                    f"  TensorInput_{_target}_T {_selector_name}({target_ptr} + \
                        thread_coord.y * {_dim}); \n")
            else: 
                # for scalar
                target_ptr = f"{_target}_ptr"
                selector_code.append(
                    f"  TensorInput_{_target}_T {_selector_name}({target_ptr}); \n")

    return input_parameters_code, input_agent_tenosrs_code, \
        output_agent_tenosrs_code, \
        output_agent_forloop_code, selector_code



def map_gen(map_operations):
    """
    Generate the code for map

    Args:
        map_operations: list of map operations

    Returns:
        map_code: list of code for the map
        map_agent_tenosrs_code: list of code for the map agent tenosrs
    """
    map_code = []
    map_agent_tenosrs_code = []

    for op in map_operations:
        _name = op['name']
        _op = op['op']
        _dim = get_dim_length(op['shape'])
        # add the declarations for the map agent tenosrs
        if _op != 'reducer' and ('sum' not in _op and 'norm' not in _op):
            map_agent_tenosrs_code.append(
                f"  typedef Tensor<ValueT, {_dim}> TensorOutput_{_name}_T; \n")

        if _op == 'add':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = \
                    {op['args'][0]} + {op['args'][1]}; \n")
        elif _op == 'sub':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = \
                {op['args'][0]} - {op['args'][1]}; \n")
        elif _op == 'mul':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = \
                    {op['args'][0]} * {op['args'][1]}; \n")
        elif _op == 'pow':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = \
                    {op['args'][0]} ^ {op['args'][1]}; \n")
        elif _op == 'truediv':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = \
                    {op['args'][0]} / {op['args'][1]}; \n")
        elif _op == 'neg':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = \
                    -{op['args'][0]}; \n")
        elif _op == 'add_':
            map_code.append(
                # f"    {op['args'][0]} = {op['args'][0]} + \
                #     {op['args'][1]}; \n")
                f"    {op['args'][0]} += \
                    {op['args'][1]}; \n")
            map_code.append(f"  for (int i = 0; i < {_dim}; i++) \n")
            map_code.append(f"  {{ \n")
            map_code.append(f"    {op['args'][0]}_ptr[thread_coord.y * {_dim} + i] = {op['args'][0]}.values[i]; \n")
            map_code.append(f"  }} \n")
        elif _op == 'exp':
            map_code.append(
                f"    TensorOutput_{_name}_T {_name} = \
                    {op['args'][0]}.exp(); \n")
        elif _op == 'reducer':
            pass
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
                    # [TODO: check it for high dimesion, such as slice dimension > 2]
                    # [here may lead to the mismatch input]
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
            # the squeeze is no-op considering continuous memory layout
            _variable_name = op['args'][0]
            map_code.append(
                f"    TensorOutput_{_name}_T {_name}({_variable_name}); \n")
        elif _op == 'copy_':
            _output_name = op['args'][0]
            _variable_name = op['args'][1]
            map_code.append(f"  for (int i = 0; i < {_dim}; i++) \n")
            map_code.append(f"  {{ \n")
            map_code.append(f"    {_output_name}_ptr[thread_coord.y * {_dim} + i] = {_variable_name}.values[i]; \n")
            map_code.append(f"  }} \n")
        elif _op == 'setitem':
            _variable_name = op['args'][0]
            _value_name = op['args'][2]
            map_code.append(f"  for (int i = 0; i < {_dim}; i++) \n")
            map_code.append(f"  {{ \n")
            map_code.append(f"    {_variable_name}_ptr[thread_coord.y * {_dim} + i] = {_value_name}.values[i]; \n")
            map_code.append(f"  }} \n")
            # the updated output may serves as input for a new function
            map_code.append(f"  for (int i = 0; i < {_dim}; i++) \n")
            map_code.append(f"  {{ \n")
            map_code.append(f"    {_variable_name}.values[i] = {_value_name}.values[i];\n")
            map_code.append(f"  }} \n")
        elif _op == 'sum':
            # aggregator
            pass
        elif _op == 'norm':
            # aggregator
            # map_code.append(
            #     f"    ValueT {_name} = {op['args'][0]}.l2Norm(); \n")
            pass
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

    return map_code, map_agent_tenosrs_code


def aggregator_gen(aggregator_operations):
    """
    Generate the code for aggregators

    Args:
        aggregator_operations: list of aggregator operations

    Returns:
        aggregator_partial_carry_fixup_code: list of code for the aggregator partial carry fixup
        aggregator_partial_forloop_code: list of code for the aggregator partial forloop
        aggregator_diagonal_code_search: list of code for the aggregator diagonal code search
        aggregator_tenosrs_carry_out_code: list of code for the aggregator tenosrs carry out
    """
    aggregator_partial_carry_fixup_code = []
    aggregator_partial_forloop_code = []
    aggregator_diagonal_code_search = []
    aggregator_tenosrs_carry_out_code = []

    if aggregator_operations != []:
        aggregator_diagonal_code_search.append(f"    int2 thread_coord; \n")
        aggregator_diagonal_code_search.append(
            f"    int2 thread_coord_end; \n")
        aggregator_diagonal_code_search.append(
            f"    thread_coord.y = tid * items_per_thread; \n")
        aggregator_diagonal_code_search.append(
            f"    thread_coord_end.y = std::min(tid * items_per_thread + \
                items_per_thread, num_nonzeros); \n")

    for op in aggregator_operations:
        _dim = get_dim_length(op['shape'])
        _name = op['name']
        if 'sum' in _name:
            aggregator_partial_carry_fixup_code.append(
                f"    ApplyCarryOutFixupSum<ValueT, OffsetT, {_dim}>(num_threads, \
                    {_name}_running_carry_out, output_y_{_name}_ptr); \n")
            aggregator_partial_forloop_code.append(
            f"    {_name}_running_carry_out[tid] += {op['args'][0]}; \n")
        elif 'norm' in _name:
            aggregator_partial_carry_fixup_code.append(
                f"    ApplyCarryOutFixupNorm<ValueT, OffsetT, {_dim}>(num_threads, \
                    {_name}_running_carry_out, output_y_{_name}_ptr); \n")
            aggregator_partial_forloop_code.append(
            f"    {_name}_running_carry_out[tid] += {op['args'][0]} * {op['args'][0]}; \n")
        else:
            raise ValueError(f"Operation {_name} not supported")
        aggregator_partial_carry_fixup_code.append(
            f"    delete[] {_name}_running_carry_out; \n")
        
        aggregator_tenosrs_carry_out_code.append(
            f"    TensorOutput_{_name}_T* {_name}_running_carry_out=new \
                TensorOutput_{_name}_T[num_threads]; \n")

    # # debug
    # debug_print(aggregator_partial_carry_fixup_code, "aggregator_partial_carry_fixup_code")
    return aggregator_partial_carry_fixup_code, aggregator_partial_forloop_code, \
        aggregator_diagonal_code_search, aggregator_tenosrs_carry_out_code


def reducer_gen(reducer_operations):
    """
    Generate the code for reducers

    Args:
        reducer_operations: list of reducer operations

    Returns:
        reducer_consume_init_code: list of code for the reducer consume init
        reducer_consume_forloop_code: list of code for the reducer consume forloop
        reducer_consume_forloop_add_code: list of code for the reducer consume forloop add
        reducer_partial_init_code: list of code for the reducer partial init
        reducer_partial_forloop_code: list of code for the reducer partial forloop
        reducer_partial_carry_code: list of code for the reducer partial carry
        reducer_partial_carry_fixup_code: list of code for the reducer partial carry fixup
    """

    reducer_consume_init_code = []
    reducer_consume_forloop_code = []
    reducer_consume_forloop_add_code = []
    reducer_partial_init_code = []
    reducer_partial_forloop_code = []
    reducer_partial_carry_code = []
    reducer_partial_carry_fixup_code = []

    # generate the reducer code for each reducer
    for op in reducer_operations:
        # get the dimension length of the shape
        _dim = get_dim_length(op['shape'])
        _name = op['name']
        _param = op['args'][0]
        # generate the SMEM definitions and the reducer code
        reducer_consume_init_code.append(
            f"   TensorOutput_{_name}_T reducer_{_name}_running_total; \n")
        reducer_consume_forloop_code.append(
            f"   reducer_{_name}_running_total += {_param}; \n")
        reducer_consume_forloop_add_code.append(
            f"   for (int i = 0; i < {_dim}; i++) \n")
        reducer_consume_forloop_add_code.append(f"   {{ \n")
        reducer_consume_forloop_add_code.append(
            f"     output_y_{_name}_ptr[thread_coord.x * {_dim} + i] = \
                reducer_{_name}_running_total.values[i]; \n")
        reducer_consume_forloop_add_code.append(f"   }} \n")
        reducer_partial_init_code.append(
            f"   TensorOutput_{_name}_T {_name}_running_total_partial; \n")
        reducer_partial_forloop_code.append(
            f"   {_name}_running_total_partial += {_param}; \n")
        reducer_partial_carry_code.append(
            f"   row_carry_out_{_name}[tid] = thread_coord_end.x; \n")
        reducer_partial_carry_code.append(
            f"   value_carry_out_{_name}[tid] = {_name}_running_total_partial; \n")
        reducer_partial_carry_fixup_code.append(
            f"   ApplyCarryOutFixup<ValueT, OffsetT, {_dim}>( \n")
        reducer_partial_carry_fixup_code.append(
            f"     num_threads, num_rows, row_carry_out_{_name},\
                 value_carry_out_{_name}, output_y_{_name}_ptr); \n")

    return reducer_consume_init_code, reducer_consume_forloop_code, \
            reducer_consume_forloop_add_code, reducer_partial_init_code, \
            reducer_partial_forloop_code, reducer_partial_carry_code, \
            reducer_partial_carry_fixup_code
