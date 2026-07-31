// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "include/merged_spmv.cuh"
#include "include/merged_utils.cuh"

namespace {

template <typename ValueT, typename OffsetT>
${tuple_type_return} merged_spmv_launch_typed(
${function_params}
  , torch::Tensor temp_storage
) {
${basic_checks}
${dtype_checks}

  const int64_t ne = ${ne_expr};
  TORCH_CHECK(ne > 0, "selector index must be non-empty");
  // TORCH_CHECK(row_end_offsets.numel() == num_rows + 1, "row_end_offsets must have length num_rows + 1");
${ne_multiple_checks}
${output_checks}

  auto options_val = torch::TensorOptions().dtype(${dispatch_tensor}.scalar_type()).device(${dispatch_tensor}.device());
${output_allocations}

  FlexParams<ValueT, OffsetT> params;
${params_value_ptrs}
${params_index_ptrs}
${params_output_ptrs}
  params.d_row_end_offsets = reinterpret_cast<OffsetT*>(row_end_offsets.data_ptr());
  params.num_rows = static_cast<OffsetT>(num_rows);
  params.num_cols = static_cast<OffsetT>(num_cols);
  params.num_nonzeros = static_cast<OffsetT>(ne);

${launch_code}

  ${output_tuple_returns}
}

template <typename ValueT>
${tuple_type_return} merged_spmv_launch_index_dispatch(
${function_params}
  , torch::Tensor temp_storage
) {
  switch (row_end_offsets.scalar_type()) {
    case torch::kInt:
      return merged_spmv_launch_typed<ValueT, int>(${function_call_args}, temp_storage);
    case torch::kLong:
      return merged_spmv_launch_typed<ValueT, int64_t>(${function_call_args}, temp_storage);
    default:
      TORCH_CHECK(false, "Unsupported index dtype. Use int32 or int64.");
  }
}

${tuple_type_return} merged_spmv_launch_bind(
${function_params}
  , torch::Tensor temp_storage
) {
  TORCH_CHECK(${dispatch_tensor}.device().is_cuda(), "CUDA device required");
  switch (${dispatch_tensor}.scalar_type()) {
    case torch::kFloat:
      return merged_spmv_launch_index_dispatch<float>(${function_call_args}, temp_storage);
    case torch::kDouble:
      return merged_spmv_launch_index_dispatch<double>(${function_call_args}, temp_storage);
    ${optional_long_case}    
    default:
      std::cerr << "Unsupported dtype for ValueT. dtype code: " << static_cast<int>(${dispatch_tensor}.scalar_type()) << std::endl;
      // TORCH_CHECK(false, "Unsupported dtype for ValueT. Use float32 or float64.");
  }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "merged_spmv_launch",
      &merged_spmv_launch_bind,
      "Run merged SpMV kernel"
${pybind_args}
      , py::arg("temp_storage")
  );
}
