// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "include/merged_spmv.cuh"
#include "include/merged_utils.cuh"

namespace {

template <typename ValueT, typename OffsetT>
void merged_spmv_launch_typed(
    torch::Tensor v,
    torch::Tensor clone_1,
    torch::Tensor x,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(clone_1.is_cuda(), "clone_1 must be a CUDA tensor");
  TORCH_CHECK(clone_1.is_contiguous(), "clone_1 must be contiguous");
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(row_end_offsets.is_cuda(), "row_end_offsets must be a CUDA tensor");
  TORCH_CHECK(row_end_offsets.is_contiguous(), "row_end_offsets must be contiguous");

  TORCH_CHECK(v.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "v dtype mismatch");
  TORCH_CHECK(clone_1.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "clone_1 dtype mismatch");
  TORCH_CHECK(x.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "x dtype mismatch");
  TORCH_CHECK(row_end_offsets.scalar_type() ==             c10::CppTypeToScalarType<OffsetT>::value, "row_end_offsets dtype mismatch");


  const int64_t ne = num_cols;
  TORCH_CHECK(ne > 0, "selector index must be non-empty");
  // TORCH_CHECK(row_end_offsets.numel() == num_rows + 1, "row_end_offsets must have length num_rows + 1");


  auto options_val = torch::TensorOptions().dtype(v.scalar_type()).device(v.device());


  FlexParams<ValueT, OffsetT> params;
  params.v_ptr =         reinterpret_cast<ValueT*>(v.data_ptr());
  params.clone_1_ptr =         reinterpret_cast<ValueT*>(clone_1.data_ptr());
  params.x_ptr =         reinterpret_cast<ValueT*>(x.data_ptr());



  params.d_row_end_offsets = reinterpret_cast<OffsetT*>(row_end_offsets.data_ptr());
  params.num_rows = static_cast<int>(num_rows);
  params.num_cols = static_cast<int>(num_cols);
  params.num_nonzeros = static_cast<int>(ne);

  size_t temp_storage_bytes = 0;
  void* d_temp_storage = nullptr;
  auto stream = at::cuda::getCurrentCUDAStream();

  cudaError_t err = merged::merged_spmv_launch<ValueT, OffsetT>(
      params, d_temp_storage, temp_storage_bytes, /*debug_synchronous=*/false, stream.stream());
  TORCH_CHECK(err == cudaSuccess, "merged_spmv_launch (size query) failed: ", cudaGetErrorString(err));

  if (temp_storage_bytes > 0) {
    cudaError_t alloc_err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    TORCH_CHECK(alloc_err == cudaSuccess, "cudaMalloc temp storage failed: ", cudaGetErrorString(alloc_err));
  }

  err = merged::merged_spmv_launch<ValueT, OffsetT>(
      params, d_temp_storage, temp_storage_bytes, /*debug_synchronous=*/false, stream.stream());
  if (d_temp_storage) { cudaFree(d_temp_storage); }
  TORCH_CHECK(err == cudaSuccess, "merged_spmv_launch failed: ", cudaGetErrorString(err));

  
}

void merged_spmv_launch_bind(
    torch::Tensor v,
    torch::Tensor clone_1,
    torch::Tensor x,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(v.device().is_cuda(), "CUDA device required");
  switch (v.scalar_type()) {
    case torch::kFloat:
      return merged_spmv_launch_typed<float, int>(v, clone_1, x, row_end_offsets, num_rows, num_cols);
    case torch::kDouble:
      return merged_spmv_launch_typed<double, int>(v, clone_1, x, row_end_offsets, num_rows, num_cols);
        
    default:
      std::cerr << "Unsupported dtype for ValueT. dtype code: " << static_cast<int>(v.scalar_type()) << std::endl;
      // TORCH_CHECK(false, "Unsupported dtype for ValueT. Use float32 or float64.");
  }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "merged_spmv_launch",
      &merged_spmv_launch_bind,
      "Run merged SpMV kernel"
      , py::arg("v")
      , py::arg("clone_1")
      , py::arg("x")
      , py::arg("row_end_offsets")
      , py::arg("num_rows")
      , py::arg("num_cols")
  );
}


