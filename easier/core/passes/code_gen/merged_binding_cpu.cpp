// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <ATen/Parallel.h>

#include "merged_spmv.h" // generated CPU header

namespace {

template <typename ValueT, typename OffsetT>
void merged_spmv_launch_cpu_typed(
    torch::Tensor truediv_3,
    torch::Tensor truediv_8,
    torch::Tensor truediv_13,
    torch::Tensor scatter_10,
    torch::Tensor scatter_b_6,
    torch::Tensor area,
    torch::Tensor uh,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(truediv_3.device().is_cpu(), "truediv_3 must be a CPU tensor");
  TORCH_CHECK(truediv_3.is_contiguous(), "truediv_3 must be contiguous");
  TORCH_CHECK(truediv_8.device().is_cpu(), "truediv_8 must be a CPU tensor");
  TORCH_CHECK(truediv_8.is_contiguous(), "truediv_8 must be contiguous");
  TORCH_CHECK(truediv_13.device().is_cpu(), "truediv_13 must be a CPU tensor");
  TORCH_CHECK(truediv_13.is_contiguous(), "truediv_13 must be contiguous");
  TORCH_CHECK(scatter_10.device().is_cpu(), "scatter_10 must be a CPU tensor");
  TORCH_CHECK(scatter_10.is_contiguous(), "scatter_10 must be contiguous");
  TORCH_CHECK(scatter_b_6.device().is_cpu(), "scatter_b_6 must be a CPU tensor");
  TORCH_CHECK(scatter_b_6.is_contiguous(), "scatter_b_6 must be contiguous");
  TORCH_CHECK(area.device().is_cpu(), "area must be a CPU tensor");
  TORCH_CHECK(area.is_contiguous(), "area must be contiguous");
  TORCH_CHECK(uh.device().is_cpu(), "uh must be a CPU tensor");
  TORCH_CHECK(uh.is_contiguous(), "uh must be contiguous");
  TORCH_CHECK(row_end_offsets.device().is_cpu(), "row_end_offsets must be a CPU tensor");
  TORCH_CHECK(row_end_offsets.is_contiguous(), "row_end_offsets must be contiguous");

  TORCH_CHECK(truediv_3.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "truediv_3 dtype mismatch");
  TORCH_CHECK(truediv_8.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "truediv_8 dtype mismatch");
  TORCH_CHECK(truediv_13.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "truediv_13 dtype mismatch");
  TORCH_CHECK(scatter_10.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "scatter_10 dtype mismatch");
  TORCH_CHECK(scatter_b_6.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "scatter_b_6 dtype mismatch");
  TORCH_CHECK(area.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "area dtype mismatch");
  TORCH_CHECK(uh.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "uh dtype mismatch");
  TORCH_CHECK(row_end_offsets.scalar_type() == c10::CppTypeToScalarType<OffsetT>::value, "row_end_offsets dtype mismatch");


  const int64_t ne = num_cols;
  TORCH_CHECK(ne > 0, "selector index must be non-empty (ne > 0)");


  auto options_val = torch::TensorOptions().dtype(truediv_3.scalar_type()).device(truediv_3.device());


  // Build raw pointers in OmpMergeSystem parameter order




  const int num_threads = at::get_num_threads();

  // Invoke generated kernel
  OmpMergeSystem<ValueT, OffsetT>(
    num_threads, reinterpret_cast<ValueT*>(truediv_3.data_ptr()), reinterpret_cast<ValueT*>(truediv_8.data_ptr()), reinterpret_cast<ValueT*>(truediv_13.data_ptr()), reinterpret_cast<ValueT*>(scatter_10.data_ptr()), reinterpret_cast<ValueT*>(scatter_b_6.data_ptr()), reinterpret_cast<ValueT*>(area.data_ptr()), reinterpret_cast<ValueT*>(uh.data_ptr()), static_cast<int>(num_rows), static_cast<int>(ne));


  
}

void merged_spmv_launch_bind_cpu(
    torch::Tensor truediv_3,
    torch::Tensor truediv_8,
    torch::Tensor truediv_13,
    torch::Tensor scatter_10,
    torch::Tensor scatter_b_6,
    torch::Tensor area,
    torch::Tensor uh,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(truediv_3.device().is_cpu(), "CPU device required");
  switch (truediv_3.scalar_type()) {
    case torch::kFloat:
      return merged_spmv_launch_cpu_typed<float, int>(truediv_3, truediv_8, truediv_13, scatter_10, scatter_b_6, area, uh, row_end_offsets, num_rows, num_cols);
    case torch::kDouble:
      return merged_spmv_launch_cpu_typed<double, int>(truediv_3, truediv_8, truediv_13, scatter_10, scatter_b_6, area, uh, row_end_offsets, num_rows, num_cols);
    
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
      , py::arg("truediv_3")
      , py::arg("truediv_8")
      , py::arg("truediv_13")
      , py::arg("scatter_10")
      , py::arg("scatter_b_6")
      , py::arg("area")
      , py::arg("uh")
      , py::arg("row_end_offsets")
      , py::arg("num_rows")
      , py::arg("num_cols")
  );
}


