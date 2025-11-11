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

  auto options_val = torch::TensorOptions().dtype(${dispatch_tensor}.scalar_type()).device(${dispatch_tensor}.device());
${output_allocations}

  FlexParams<ValueT, OffsetT> params;
${params_value_ptrs}
${params_index_ptrs}
${params_output_ptrs}
  params.d_row_end_offsets = reinterpret_cast<OffsetT*>(row_end_offsets.data_ptr());
  params.num_rows = static_cast<int>(num_rows);
  params.num_cols = static_cast<int>(num_cols);
  params.num_nonzeros = static_cast<int>(ne);

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

  ${output_tuple_returns}
}

${tuple_type_return} merged_spmv_launch_bind(
${function_params}
  , torch::Tensor temp_storage
) {
  TORCH_CHECK(${dispatch_tensor}.device().is_cuda(), "CUDA device required");
  switch (${dispatch_tensor}.scalar_type()) {
    case torch::kFloat:
      return merged_spmv_launch_typed<float, int>(${function_call_args}, temp_storage);
    case torch::kDouble:
      return merged_spmv_launch_typed<double, int>(${function_call_args}, temp_storage);
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


