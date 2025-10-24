// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <ATen/Parallel.h>

#include "merged_spmv.h" // generated CPU header

namespace {

template <typename ValueT, typename OffsetT>
void merged_spmv_launch_cpu_typed(
    torch::Tensor v,
    torch::Tensor clone_1,
    torch::Tensor x,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(v.device().is_cpu(), "v must be a CPU tensor");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(clone_1.device().is_cpu(), "clone_1 must be a CPU tensor");
  TORCH_CHECK(clone_1.is_contiguous(), "clone_1 must be contiguous");
  TORCH_CHECK(x.device().is_cpu(), "x must be a CPU tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(row_end_offsets.device().is_cpu(), "row_end_offsets must be a CPU tensor");
  TORCH_CHECK(row_end_offsets.is_contiguous(), "row_end_offsets must be contiguous");

  TORCH_CHECK(v.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "v dtype mismatch");
  TORCH_CHECK(clone_1.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "clone_1 dtype mismatch");
  TORCH_CHECK(x.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "x dtype mismatch");
  TORCH_CHECK(row_end_offsets.scalar_type() == c10::CppTypeToScalarType<OffsetT>::value, "row_end_offsets dtype mismatch");


  const int64_t ne = num_cols;
  TORCH_CHECK(ne > 0, "selector index must be non-empty (ne > 0)");


  auto options_val = torch::TensorOptions().dtype(v.scalar_type()).device(v.device());


  // Build raw pointers in OmpMergeSystem parameter order




  const int num_threads = at::get_num_threads();

  // Invoke generated kernel
  OmpMergeSystem<ValueT, OffsetT>(
    num_threads, reinterpret_cast<ValueT*>(v.data_ptr()), reinterpret_cast<ValueT*>(clone_1.data_ptr()), reinterpret_cast<ValueT*>(x.data_ptr()), static_cast<int>(num_rows), static_cast<int>(ne));


  
}

void merged_spmv_launch_bind_cpu(
    torch::Tensor v,
    torch::Tensor clone_1,
    torch::Tensor x,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(v.device().is_cpu(), "CPU device required");
  switch (v.scalar_type()) {
    case torch::kFloat:
      return merged_spmv_launch_cpu_typed<float, int>(v, clone_1, x, row_end_offsets, num_rows, num_cols);
    case torch::kDouble:
      return merged_spmv_launch_cpu_typed<double, int>(v, clone_1, x, row_end_offsets, num_rows, num_cols);
    
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
      , py::arg("v")
      , py::arg("clone_1")
      , py::arg("x")
      , py::arg("row_end_offsets")
      , py::arg("num_rows")
      , py::arg("num_cols")
  );
}


