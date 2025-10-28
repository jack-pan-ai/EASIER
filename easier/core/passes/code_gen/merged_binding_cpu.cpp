// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <ATen/Parallel.h>

#include "merged_spmv.h" // generated CPU header

namespace {

template <typename ValueT, typename OffsetT>
void merged_spmv_launch_cpu_typed(
    torch::Tensor truediv,
    torch::Tensor p,
    torch::Tensor r,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(truediv.device().is_cpu(), "truediv must be a CPU tensor");
  TORCH_CHECK(truediv.is_contiguous(), "truediv must be contiguous");
  TORCH_CHECK(p.device().is_cpu(), "p must be a CPU tensor");
  TORCH_CHECK(p.is_contiguous(), "p must be contiguous");
  TORCH_CHECK(r.device().is_cpu(), "r must be a CPU tensor");
  TORCH_CHECK(r.is_contiguous(), "r must be contiguous");
  TORCH_CHECK(row_end_offsets.device().is_cpu(), "row_end_offsets must be a CPU tensor");
  TORCH_CHECK(row_end_offsets.is_contiguous(), "row_end_offsets must be contiguous");

  TORCH_CHECK(truediv.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "truediv dtype mismatch");
  TORCH_CHECK(p.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "p dtype mismatch");
  TORCH_CHECK(r.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "r dtype mismatch");
  TORCH_CHECK(row_end_offsets.scalar_type() == c10::CppTypeToScalarType<OffsetT>::value, "row_end_offsets dtype mismatch");


  const int64_t ne = num_cols;
  TORCH_CHECK(ne > 0, "selector index must be non-empty (ne > 0)");


  auto options_val = torch::TensorOptions().dtype(truediv.scalar_type()).device(truediv.device());


  // Build raw pointers in OmpMergeSystem parameter order




  const int num_threads = at::get_num_threads();

  // Invoke generated kernel
  OmpMergeSystem<ValueT, OffsetT>(
    num_threads, reinterpret_cast<ValueT*>(truediv.data_ptr()), reinterpret_cast<ValueT*>(p.data_ptr()), reinterpret_cast<ValueT*>(r.data_ptr()), static_cast<int>(num_rows), static_cast<int>(ne));


  
}

void merged_spmv_launch_bind_cpu(
    torch::Tensor truediv,
    torch::Tensor p,
    torch::Tensor r,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(truediv.device().is_cpu(), "CPU device required");
  switch (truediv.scalar_type()) {
    case torch::kFloat:
      return merged_spmv_launch_cpu_typed<float, int>(truediv, p, r, row_end_offsets, num_rows, num_cols);
    case torch::kDouble:
      return merged_spmv_launch_cpu_typed<double, int>(truediv, p, r, row_end_offsets, num_rows, num_cols);
    
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
      , py::arg("truediv")
      , py::arg("p")
      , py::arg("r")
      , py::arg("row_end_offsets")
      , py::arg("num_rows")
      , py::arg("num_cols")
  );
}


