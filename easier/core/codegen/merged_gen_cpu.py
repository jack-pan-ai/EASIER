# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import string
import os

from easier.core.codegen.trace.graph_trace import trace_graph
from easier.core.codegen.merged_gen_core import declarations_gen, reducer_gen, map_gen, aggregator_gen
from easier.core.codegen.merged_gen_binding_cpu import generate_cpu_binding_code

# debug print
def debug_print(code, code_name):
    if code != []:
        for op in code:
            print(f"{code_name}: {op}")
    else:
        print(f"{code_name} is empty")

def generate_cpu_code_from_graph(submodule, traced_model):
    """
    Generate CPU code from the graph

    Args:
        submodule: the submodule -> GraphModule type
        traced_model: the traced model
    """

    # Analyze the graph and extract operations
    inputs = []
    _input_keys = set()
    selector_register = []
    map_operations = []
    reducer_operations = []
    aggregator_operations = []
    outputs = []

    inputs, outputs, selector_register, map_operations, \
        reducer_operations, aggregator_operations = \
            trace_graph(submodule, traced_model)

    # Generate the code
    # Generate the code for input and output and selector
    input_parameters_code = []
    input_agent_tenosrs_code = []
    output_agent_tenosrs_code = []
    output_agent_forloop_code = []
    selector_code = []

    input_parameters_code, input_agent_tenosrs_code, output_agent_tenosrs_code, \
        output_agent_forloop_code, selector_code = \
            declarations_gen(inputs, outputs, selector_register)

    if os.getenv("EASIER_VERBOSE_CODEGEN") in ("1", "", "true", "True"):
        debug_print(input_parameters_code, "input_parameters_code")
        debug_print(input_agent_tenosrs_code, "input_agent_tenosrs_code")
        debug_print(output_agent_tenosrs_code, "output_agent_tenosrs_code")
        debug_print(output_agent_forloop_code, "output_agent_forloop_code")
        debug_print(selector_code, "selector_code")

    # Generate the code for map
    map_code, map_agent_tenosrs_code = map_gen(map_operations)
    if os.getenv("EASIER_VERBOSE_CODEGEN") in ("1", "", "true", "True"):
        debug_print(map_code, "map_code")
        debug_print(map_agent_tenosrs_code, "map_agent_tenosrs_code")

    # Generate the code for reducers and aggregators

    reducer_consume_init_code, reducer_consume_forloop_code, \
        reducer_consume_forloop_add_code, reducer_partial_init_code, \
            reducer_partial_forloop_code, reducer_partial_carry_code, \
                reducer_partial_carry_fixup_code = reducer_gen(reducer_operations)
    if os.getenv("EASIER_VERBOSE_CODEGEN") in ("1", "", "true", "True"):
        debug_print(reducer_consume_init_code, "reducer_consume_init_code")
        debug_print(reducer_consume_forloop_code, "reducer_consume_forloop_code")
        debug_print(reducer_consume_forloop_add_code, "reducer_consume_forloop_add_code")
        debug_print(reducer_partial_init_code, "reducer_partial_init_code")
        debug_print(reducer_partial_forloop_code, "reducer_partial_forloop_code")
        debug_print(reducer_partial_carry_code, "reducer_partial_carry_code")
        debug_print(reducer_partial_carry_fixup_code, "reducer_partial_carry_fixup_code")
    # Aggregators
    # debug
    # debug_print(aggregator_operations, "aggregator_operations")
    aggregator_partial_carry_fixup_code, aggregator_partial_forloop_code, \
        aggregator_diagonal_code_search, aggregator_tenosrs_carry_out_code = aggregator_gen(
        aggregator_operations)
    if os.getenv("EASIER_VERBOSE_CODEGEN") in ("1", "", "true", "True"):
        debug_print(aggregator_partial_carry_fixup_code, "aggregator_partial_carry_fixup_code")
        debug_print(aggregator_partial_forloop_code, "aggregator_partial_forloop_code")
        debug_print(aggregator_diagonal_code_search, "aggregator_diagonal_code_search")
        debug_print(aggregator_tenosrs_carry_out_code, "aggregator_tenosrs_carry_out_code")
    # Read template files
    _tag = "reducer" if reducer_operations != [] else "aggregator"
    project_root = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(project_root, f"cpu_template/merged_spmv_template_{_tag}.h"), "r") as f:
        kernel_spmv_template = f.read()

    # Apply templates using string.Template
    def trans_str(code):
        if code == []:
            return ''
        else:
            return ''.join(code)

    input_parameters_str = trans_str(input_parameters_code)
    input_agent_tenosrs_str = trans_str(input_agent_tenosrs_code)
    output_agent_tenosrs_str = trans_str(output_agent_tenosrs_code)
    output_agent_forloop_str = trans_str(output_agent_forloop_code)
    selector_str = trans_str(selector_code)
    map_str = trans_str(map_code)
    map_agent_tenosrs_str = trans_str(map_agent_tenosrs_code)
    reducer_consume_init_str = trans_str(reducer_consume_init_code)
    reducer_consume_forloop_str = trans_str(reducer_consume_forloop_code)
    reducer_consume_forloop_add_str = trans_str(
        reducer_consume_forloop_add_code)
    reducer_partial_init_str = trans_str(reducer_partial_init_code)
    reducer_partial_forloop_str = trans_str(reducer_partial_forloop_code)
    reducer_partial_carry_str = trans_str(reducer_partial_carry_code)
    reducer_partial_carry_fixup_str = trans_str(
        reducer_partial_carry_fixup_code)
    aggregator_partial_carry_fixup_code_str = trans_str(
        aggregator_partial_carry_fixup_code)
    aggregator_partial_forloop_code_str = trans_str(
        aggregator_partial_forloop_code)
    aggregator_diagonal_code_search_str = trans_str(
        aggregator_diagonal_code_search)
    aggregator_tenosrs_carry_out_code_str = trans_str(
        aggregator_tenosrs_carry_out_code)
    spmv_kernel_code = string.Template(kernel_spmv_template).substitute(
        input_parameters_code=input_parameters_str,
        input_agent_tenosrs_code=input_agent_tenosrs_str,
        output_agent_tenosrs_code=output_agent_tenosrs_str,
        output_agent_forloop_code=output_agent_forloop_str,
        selector_code=selector_str,
        map_code=map_str,
        map_agent_tenosrs_code=map_agent_tenosrs_str,
        reducer_consume_init_code=reducer_consume_init_str,
        reducer_consume_forloop_code=reducer_consume_forloop_str,
        reducer_consume_forloop_add_code=reducer_consume_forloop_add_str,
        reducer_partial_init_code=reducer_partial_init_str,
        reducer_partial_forloop_code=reducer_partial_forloop_str,
        reducer_partial_carry_code=reducer_partial_carry_str,
        reducer_partial_carry_fixup_code=reducer_partial_carry_fixup_str,
        aggregator_partial_carry_fixup_code=aggregator_partial_carry_fixup_code_str,
        aggregator_partial_forloop_code=aggregator_partial_forloop_code_str,
        aggregator_diagonal_code_search=aggregator_diagonal_code_search_str,
        aggregator_tenosrs_carry_out_code=aggregator_tenosrs_carry_out_code_str,
    )

    # Write files
    with open(os.path.join(project_root, "merged_spmv.h"), "w") as f:
        f.write(spmv_kernel_code)

    print("CPU code generated successfully!")

    # Also generate the CPU binding cpp file next to project root
    generate_cpu_binding_code(project_root, inputs, outputs, selector_register)
