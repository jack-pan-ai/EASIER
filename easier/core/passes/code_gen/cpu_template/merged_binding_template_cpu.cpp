// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <ATen/Parallel.h>

#include "merged_spmv.h" // generated CPU header

namespace {

template <typename ValueT, typename OffsetT>
${tuple_type_return} merged_spmv_launch_cpu_typed(
${function_params}
) {
${basic_checks}
${dtype_checks}

  const int64_t ne = ${ne_expr};
  TORCH_CHECK(ne > 0, "selector index must be non-empty (ne > 0)");
${ne_multiple_checks}

  auto options_val = torch::TensorOptions().dtype(${dispatch_tensor}.scalar_type()).device(${dispatch_tensor}.device());
${output_allocations}

  // Build raw pointers in OmpMergeSystem parameter order
${value_ptrs}
${index_ptrs}
${output_ptrs}

  const int num_threads = at::get_num_threads();

  // Invoke generated kernel
${omp_call}

  ${output_tuple_returns}
}

template <typename ValueT>
${tuple_type_return} merged_spmv_launch_cpu_index_dispatch(
${function_params}
) {
  switch (row_end_offsets.scalar_type()) {
    case torch::kInt:
      return merged_spmv_launch_cpu_typed<ValueT, int>(${function_call_args});
    case torch::kLong:
      return merged_spmv_launch_cpu_typed<ValueT, int64_t>(${function_call_args});
    default:
      TORCH_CHECK(false, "Unsupported index dtype. Use int32 or int64.");
  }
}

${tuple_type_return} merged_spmv_launch_bind_cpu(
${function_params}
) {
  TORCH_CHECK(${dispatch_tensor}.device().is_cpu(), "CPU device required");
  switch (${dispatch_tensor}.scalar_type()) {
    case torch::kFloat:
      return merged_spmv_launch_cpu_index_dispatch<float>(${function_call_args});
    case torch::kDouble:
      return merged_spmv_launch_cpu_index_dispatch<double>(${function_call_args});
    ${optional_long_case}
    default:
      TORCH_CHECK(false, "Unsupported dtype for ValueT. Use float32 or float64.");
  }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "merged_spmv_launch",
      &merged_spmv_launch_bind_cpu,
      "Run merged SpMV CPU kernel"
${pybind_args}
  );
}

