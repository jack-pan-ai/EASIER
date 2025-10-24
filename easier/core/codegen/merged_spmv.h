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
void ApplyCarryOutFixup(int num_threads,
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
 * OpenMP CPU merge-based SpMV from CUB
 */
template <typename ValueT, typename OffsetT>
void OmpMergeSystem(
    int num_threads,
    // [code generation]
      ValueT *__restrict v_ptr, 
  ValueT *__restrict clone_1_ptr, 
  ValueT *__restrict x_ptr, 
 int num_rows, int num_nonzeros) {
  // [code generation]
  // input and output tensors types
    typedef Tensor<ValueT, 20> TensorInput_v_T; 
  typedef Tensor<ValueT, 1> TensorInput_clone_1_T; 
  typedef Tensor<ValueT, 1> TensorInput_x_T; 
   typedef Tensor<ValueT, 20> TensorOutput_getitem_T; 
  typedef Tensor<ValueT, 20> TensorOutput_clone_T; 
  typedef Tensor<ValueT, 1> TensorOutput_matmul_T; 
  typedef Tensor<ValueT, 1> TensorOutput_squeeze_T; 
  typedef Tensor<ValueT, 1> TensorOutput_clone_2_T; 
  typedef Tensor<ValueT, 1> TensorOutput_add__T; 
  

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
        TensorInput_v_T v(v_ptr +                     thread_coord.y * 20); 
  TensorInput_clone_1_T clone_1(clone_1_ptr +                     thread_coord.y * 1); 
  TensorInput_x_T x(x_ptr +                     thread_coord.y * 1); 


      // mapping
          TensorOutput_getitem_T getitem(v); 
    TensorOutput_clone_T clone(getitem); 
    TensorOutput_matmul_T matmul =                     clone * clone_1; 
    TensorOutput_squeeze_T squeeze(matmul); 
    TensorOutput_clone_2_T clone_2(squeeze); 
    x = x +                     clone_2; 
  for (int i = 0; i < 1; i++) 
  { 
    x_ptr[thread_coord.y * 1 + i] = x.values[i]; 
  } 


      // output for aggregator
      

      // output for map
      
    }
  }
  // carry-out fix-up for aggregators
  
}
