// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "data_struct_shared.cuh"

// Minimal CPU helpers
template <typename OffsetT> struct CountingInputIterator {
  OffsetT start;
  inline CountingInputIterator(OffsetT s) : start(s) {}
  inline OffsetT operator[](OffsetT idx) const { return start + idx; }
};

struct int2 {
  int x;
  int y;
};

/**
 * Computes the begin offsets into A and B for the specific diagonal from CUB
 */
template <typename AIteratorT, typename BIteratorT, typename OffsetT,
          typename CoordinateT>
inline void MergePathSearch(
    OffsetT diagonal,             ///< [in]The diagonal to search
    AIteratorT a,                 ///< [in]List A
    BIteratorT b,                 ///< [in]List B
    OffsetT a_len,                ///< [in]Length of A
    OffsetT b_len,                ///< [in]Length of B
    CoordinateT &path_coordinate) ///< [out] (x,y) coordinate where diagonal
                                  ///< intersects the merge path
{
  OffsetT x_min = std::max(diagonal - b_len, 0);
  OffsetT x_max = std::min(diagonal, a_len);

  while (x_min < x_max) {
    OffsetT x_pivot = (x_min + x_max) >> 1;
    if (a[x_pivot] <= b[diagonal - x_pivot - 1])
      x_min = x_pivot + 1; // Contract range up A (down B)
    else
      x_max = x_pivot; // Contract range down A (up B)
  }

  path_coordinate.x = std::min(x_min, a_len);
  path_coordinate.y = diagonal - x_min;
}

/**
 * Apply carry-out fix-up for rows spanning multiple threads for reducers
 */
template <typename ValueT, typename OffsetT, int dim>
void ApplyCarryOutFixup(int num_threads, int num_rows,
                        OffsetT *row_carry_out_reducer,
                        Tensor<ValueT, dim> *value_carry_out_reducer,
                        ValueT *output_y_reducer_ptr) {

  for (int tid = 0; tid < num_threads - 1; ++tid) {
    if (row_carry_out_reducer[tid] < num_rows) {
      for (int i = 0; i < dim; i++) {
        output_y_reducer_ptr[row_carry_out_reducer[tid] * dim + i] +=
            value_carry_out_reducer[tid].values[i];
      }
    }
  }
}

/**
 * Apply carry-out fix-up for rows spanning multiple threads for aggregators
 */
template <typename ValueT, typename OffsetT, int dim>
void ApplyCarryOutFixupSum(int num_threads,
                        Tensor<ValueT, dim> *value_carry_out_sum,
                        ValueT *output_y_sum_ptr) {
  typedef Tensor<ValueT, dim> TensorOutput_sum_T;
  TensorOutput_sum_T sum_result;
  for (int tid = 0; tid < num_threads; ++tid) {
    sum_result += value_carry_out_sum[tid];
  }
  for (int i = 0; i < dim; i++) {
    output_y_sum_ptr[i] = sum_result.values[i];
  }
}

/**
 * Apply carry-out fix-up for rows spanning multiple threads for aggregators
 */
 template <typename ValueT, typename OffsetT, int dim>
 void ApplyCarryOutFixupNorm(int num_threads,
                         Tensor<ValueT, dim> *value_carry_out_norm,
                         ValueT *output_y_norm_ptr) {
   typedef Tensor<ValueT, dim> TensorOutput_norm_T;
   TensorOutput_norm_T norm_result;
   for (int tid = 0; tid < num_threads; ++tid) {
    norm_result += value_carry_out_norm[tid];
   }
   for (int i = 0; i < dim; i++) {
     output_y_norm_ptr[i] = sqrt(norm_result.values[i]);
   }
 }

/**
 * OpenMP CPU merge-based SpMV from CUB
 */
template <typename ValueT, typename OffsetT>
void OmpMergeSystem(
    int num_threads,
    // [code generation]
      ValueT *__restrict truediv_3_ptr, 
  ValueT *__restrict truediv_8_ptr, 
  ValueT *__restrict truediv_13_ptr, 
  ValueT *__restrict scatter_10_ptr, 
  ValueT *__restrict scatter_b_6_ptr, 
  ValueT *__restrict area_ptr, 
  ValueT *__restrict uh_ptr, 
 int num_rows, int num_nonzeros) {
  // [code generation]
  // input and output tensors types
    typedef Tensor<ValueT, 1> TensorInput_truediv_3_T; 
  typedef Tensor<ValueT, 1> TensorInput_truediv_8_T; 
  typedef Tensor<ValueT, 1> TensorInput_truediv_13_T; 
  typedef Tensor<ValueT, 1> TensorInput_scatter_10_T; 
  typedef Tensor<ValueT, 1> TensorInput_scatter_b_6_T; 
  typedef Tensor<ValueT, 1> TensorInput_area_T; 
  typedef Tensor<ValueT, 1> TensorInput_uh_T; 
   typedef Tensor<ValueT, 1> TensorOutput_add_45_T; 
  typedef Tensor<ValueT, 1> TensorOutput_neg_10_T; 
  typedef Tensor<ValueT, 1> TensorOutput_truediv_18_T; 
  typedef Tensor<ValueT, 1> TensorOutput_add_52_T; 
  typedef Tensor<ValueT, 1> TensorOutput_add_53_T; 
  typedef Tensor<ValueT, 1> TensorOutput_add_54_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_86_T; 
  typedef Tensor<ValueT, 1> TensorOutput_add__1_T; 
  

#pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int tid = 0; tid < num_threads; tid++) {
    OffsetT num_merge_items =
        num_rows + num_nonzeros; // Merge path total length
    OffsetT items_per_thread = (num_merge_items + num_threads - 1) /
                               num_threads; // Merge items per thread

    // Find starting and ending MergePath coordinates (row-idx, nonzero-idx) for
    // [code generation]
    // Merge list B (NZ indices)
    int2 thread_coord;
    int2 thread_coord_end;
    thread_coord.y = tid * items_per_thread;
    thread_coord_end.y =
        std::min(tid * items_per_thread + items_per_thread, num_nonzeros);

    // Consume whole rows

    
    for (; thread_coord.y < thread_coord_end.y; ++thread_coord.y) {
      // selector
        TensorInput_truediv_3_T truediv_3(truediv_3_ptr +                         thread_coord.y * 1); 
  TensorInput_truediv_8_T truediv_8(truediv_8_ptr +                         thread_coord.y * 1); 
  TensorInput_truediv_13_T truediv_13(truediv_13_ptr +                         thread_coord.y * 1); 
  TensorInput_scatter_10_T scatter_10(scatter_10_ptr +                         thread_coord.y * 1); 
  TensorInput_scatter_b_6_T scatter_b_6(scatter_b_6_ptr +                         thread_coord.y * 1); 
  TensorInput_area_T area(area_ptr +                         thread_coord.y * 1); 
  TensorInput_uh_T uh(uh_ptr +                         thread_coord.y * 1); 


      // mapping
          TensorOutput_add_45_T add_45 =                     scatter_10 + scatter_b_6; 
    TensorOutput_neg_10_T neg_10 =                     -add_45; 
    TensorOutput_truediv_18_T truediv_18 =                     neg_10 / area; 
    TensorOutput_add_52_T add_52 =                     truediv_3 + truediv_8; 
    TensorOutput_add_53_T add_53 =                     add_52 + truediv_13; 
    TensorOutput_add_54_T add_54 =                     add_53 + truediv_18; 
    TensorOutput_mul_86_T mul_86 =                     4.1666666666666665e-05 * add_54; 
    uh +=                     mul_86; 
  for (int i = 0; i < 1; i++) 
  { 
    uh_ptr[thread_coord.y * 1 + i] = uh.values[i]; 
  } 


      // output for aggregator
      

      // output for map
      
    }
  }
  // carry-out fix-up for aggregators
  
}
