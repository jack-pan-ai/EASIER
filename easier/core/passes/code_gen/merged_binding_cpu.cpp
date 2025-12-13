// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <ATen/Parallel.h>

#include "merged_spmv.h" // generated CPU header

namespace {

template <typename ValueT, typename OffsetT>
void merged_spmv_launch_cpu_typed(
    torch::Tensor bsx,
    torch::Tensor bsy,
    torch::Tensor points,
    torch::Tensor truediv_5,
    torch::Tensor selector_bp_0_idx,
    torch::Tensor selector_bp_1_idx,
    torch::Tensor bselector_idx,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(bsx.device().is_cpu(), "bsx must be a CPU tensor");
  TORCH_CHECK(bsx.is_contiguous(), "bsx must be contiguous");
  TORCH_CHECK(bsy.device().is_cpu(), "bsy must be a CPU tensor");
  TORCH_CHECK(bsy.is_contiguous(), "bsy must be contiguous");
  TORCH_CHECK(points.device().is_cpu(), "points must be a CPU tensor");
  TORCH_CHECK(points.is_contiguous(), "points must be contiguous");
  TORCH_CHECK(truediv_5.device().is_cpu(), "truediv_5 must be a CPU tensor");
  TORCH_CHECK(truediv_5.is_contiguous(), "truediv_5 must be contiguous");
  TORCH_CHECK(selector_bp_0_idx.device().is_cpu(), "selector_bp_0_idx must be a CPU tensor");
  TORCH_CHECK(selector_bp_0_idx.is_contiguous(), "selector_bp_0_idx must be contiguous");
  TORCH_CHECK(selector_bp_1_idx.device().is_cpu(), "selector_bp_1_idx must be a CPU tensor");
  TORCH_CHECK(selector_bp_1_idx.is_contiguous(), "selector_bp_1_idx must be contiguous");
  TORCH_CHECK(bselector_idx.device().is_cpu(), "bselector_idx must be a CPU tensor");
  TORCH_CHECK(bselector_idx.is_contiguous(), "bselector_idx must be contiguous");
  TORCH_CHECK(row_end_offsets.device().is_cpu(), "row_end_offsets must be a CPU tensor");
  TORCH_CHECK(row_end_offsets.is_contiguous(), "row_end_offsets must be contiguous");

  TORCH_CHECK(bsx.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "bsx dtype mismatch");
  TORCH_CHECK(bsy.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "bsy dtype mismatch");
  TORCH_CHECK(points.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "points dtype mismatch");
  TORCH_CHECK(truediv_5.scalar_type() == c10::CppTypeToScalarType<ValueT>::value, "truediv_5 dtype mismatch");
  TORCH_CHECK(selector_bp_0_idx.scalar_type() == c10::CppTypeToScalarType<OffsetT>::value, "selector_bp_0_idx dtype mismatch");
  TORCH_CHECK(selector_bp_1_idx.scalar_type() == c10::CppTypeToScalarType<OffsetT>::value, "selector_bp_1_idx dtype mismatch");
  TORCH_CHECK(bselector_idx.scalar_type() == c10::CppTypeToScalarType<OffsetT>::value, "bselector_idx dtype mismatch");
  TORCH_CHECK(row_end_offsets.scalar_type() == c10::CppTypeToScalarType<OffsetT>::value, "row_end_offsets dtype mismatch");


  const int64_t ne = selector_bp_0_idx.numel();
  TORCH_CHECK(ne > 0, "selector index must be non-empty (ne > 0)");
  TORCH_CHECK(bsx.numel() % ne == 0, "bsx.numel() must be a multiple of ne");
  TORCH_CHECK(bsy.numel() % ne == 0, "bsy.numel() must be a multiple of ne");


  auto options_val = torch::TensorOptions().dtype(bsx.scalar_type()).device(bsx.device());


  // Build raw pointers in OmpMergeSystem parameter order




  const int num_threads = at::get_num_threads();

  // Invoke generated kernel
  OmpMergeSystem<ValueT, OffsetT>(
    num_threads, reinterpret_cast<ValueT*>(points.data_ptr()), reinterpret_cast<ValueT*>(truediv_5.data_ptr()), reinterpret_cast<ValueT*>(bsx.data_ptr()), reinterpret_cast<ValueT*>(bsy.data_ptr()), reinterpret_cast<OffsetT*>((selector_bp_0_idx).data_ptr()), reinterpret_cast<OffsetT*>((selector_bp_1_idx).data_ptr()), reinterpret_cast<OffsetT*>((bselector_idx).data_ptr()), static_cast<int>(num_rows), static_cast<int>(ne));


  
}

void merged_spmv_launch_bind_cpu(
    torch::Tensor bsx,
    torch::Tensor bsy,
    torch::Tensor points,
    torch::Tensor truediv_5,
    torch::Tensor selector_bp_0_idx,
    torch::Tensor selector_bp_1_idx,
    torch::Tensor bselector_idx,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols
) {
  TORCH_CHECK(bsx.device().is_cpu(), "CPU device required");
  switch (bsx.scalar_type()) {
    case torch::kFloat:
      return merged_spmv_launch_cpu_typed<float, int>(bsx, bsy, points, truediv_5, selector_bp_0_idx, selector_bp_1_idx, bselector_idx, row_end_offsets, num_rows, num_cols);
    case torch::kDouble:
      return merged_spmv_launch_cpu_typed<double, int>(bsx, bsy, points, truediv_5, selector_bp_0_idx, selector_bp_1_idx, bselector_idx, row_end_offsets, num_rows, num_cols);
    
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
      , py::arg("bsx")
      , py::arg("bsy")
      , py::arg("points")
      , py::arg("truediv_5")
      , py::arg("selector_bp_0_idx")
      , py::arg("selector_bp_1_idx")
      , py::arg("bselector_idx")
      , py::arg("row_end_offsets")
      , py::arg("num_rows")
      , py::arg("num_cols")
  );
}


