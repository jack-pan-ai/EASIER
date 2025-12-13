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
      ValueT *__restrict points_ptr, 
  ValueT *__restrict truediv_5_ptr, 
  ValueT *__restrict bsx_ptr, 
  ValueT *__restrict bsy_ptr, 
  OffsetT *__restrict selector_bp_0_ptr, 
  OffsetT *__restrict selector_bp_1_ptr, 
  OffsetT *__restrict bselector_ptr, 
 int num_rows, int num_nonzeros) {
  // [code generation]
  // input and output tensors types
    typedef Tensor<ValueT, 2> TensorInput_points_T; 
  typedef Tensor<ValueT, 2> TensorInput_truediv_5_T; 
  typedef Tensor<ValueT, 1> TensorInput_bsx_T; 
  typedef Tensor<ValueT, 1> TensorInput_bsy_T; 
   typedef Tensor<ValueT, 1> TensorOutput_getitem_68_T; 
  typedef Tensor<ValueT, 1> TensorOutput_clone_68_T; 
  typedef Tensor<ValueT, 1> TensorOutput_getitem_69_T; 
  typedef Tensor<ValueT, 1> TensorOutput_clone_69_T; 
  typedef Tensor<ValueT, 1> TensorOutput_getitem_70_T; 
  typedef Tensor<ValueT, 1> TensorOutput_clone_70_T; 
  typedef Tensor<ValueT, 1> TensorOutput_getitem_71_T; 
  typedef Tensor<ValueT, 1> TensorOutput_clone_71_T; 
  typedef Tensor<ValueT, 1> TensorOutput_getitem_72_T; 
  typedef Tensor<ValueT, 1> TensorOutput_clone_72_T; 
  typedef Tensor<ValueT, 1> TensorOutput_getitem_73_T; 
  typedef Tensor<ValueT, 1> TensorOutput_clone_73_T; 
  typedef Tensor<ValueT, 1> TensorOutput_sub_71_T; 
  typedef Tensor<ValueT, 1> TensorOutput_sub_72_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_48_T; 
  typedef Tensor<ValueT, 1> TensorOutput_sub_73_T; 
  typedef Tensor<ValueT, 1> TensorOutput_sub_74_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_49_T; 
  typedef Tensor<ValueT, 1> TensorOutput_sub_75_T; 
  typedef Tensor<ValueT, 1> TensorOutput_sign_6_T; 
  typedef Tensor<ValueT, 1> TensorOutput_sub_76_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_50_T; 
  typedef Tensor<ValueT, 1> TensorOutput_neg_6_T; 
  typedef Tensor<ValueT, 1> TensorOutput_sub_77_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_51_T; 
  typedef Tensor<ValueT, 1> TensorOutput_neg_7_T; 
  typedef Tensor<ValueT, 1> TensorOutput_setitem_13_T; 
  typedef Tensor<ValueT, 1> TensorOutput_neg_8_T; 
  typedef Tensor<ValueT, 1> TensorOutput_setitem_14_T; 
  

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
        TensorInput_bsx_T bsx(bsx_ptr +                     thread_coord.y * 1); 
  TensorInput_bsy_T bsy(bsy_ptr +                     thread_coord.y * 1); 
  OffsetT column_indices_selector_bp_0 = selector_bp_0_ptr[thread_coord.y]; 
  TensorInput_points_T selector_bp_0(points_ptr +                     column_indices_selector_bp_0 * 2); 
  OffsetT column_indices_selector_bp_1 = selector_bp_1_ptr[thread_coord.y]; 
  TensorInput_points_T selector_bp_1(points_ptr +                     column_indices_selector_bp_1 * 2); 
  OffsetT column_indices_bselector = bselector_ptr[thread_coord.y]; 
  TensorInput_truediv_5_T bselector(truediv_5_ptr +                     column_indices_bselector * 2); 


      // mapping
          TensorOutput_getitem_68_T                         getitem_68(bselector.values[0]); 
    TensorOutput_clone_68_T clone_68(getitem_68); 
    TensorOutput_getitem_69_T                         getitem_69(bselector.values[1]); 
    TensorOutput_clone_69_T clone_69(getitem_69); 
    TensorOutput_getitem_70_T                         getitem_70(selector_bp_0.values[0]); 
    TensorOutput_clone_70_T clone_70(getitem_70); 
    TensorOutput_getitem_71_T                         getitem_71(selector_bp_0.values[1]); 
    TensorOutput_clone_71_T clone_71(getitem_71); 
    TensorOutput_getitem_72_T                         getitem_72(selector_bp_1.values[0]); 
    TensorOutput_clone_72_T clone_72(getitem_72); 
    TensorOutput_getitem_73_T                         getitem_73(selector_bp_1.values[1]); 
    TensorOutput_clone_73_T clone_73(getitem_73); 
    TensorOutput_sub_71_T sub_71 =                 clone_70 - clone_72; 
    TensorOutput_sub_72_T sub_72 =                 clone_69 - clone_73; 
    TensorOutput_mul_48_T mul_48 =                     sub_71 * sub_72; 
    TensorOutput_sub_73_T sub_73 =                 clone_71 - clone_73; 
    TensorOutput_sub_74_T sub_74 =                 clone_68 - clone_72; 
    TensorOutput_mul_49_T mul_49 =                     sub_73 * sub_74; 
    TensorOutput_sub_75_T sub_75 =                 mul_48 - mul_49; 
    TensorOutput_sign_6_T sign_6 = sub_75.sign(); 
    TensorOutput_sub_76_T sub_76 =                 clone_71 - clone_73; 
    TensorOutput_mul_50_T mul_50 =                     sign_6 * sub_76; 
    TensorOutput_neg_6_T neg_6 =                     -sign_6; 
    TensorOutput_sub_77_T sub_77 =                 clone_70 - clone_72; 
    TensorOutput_mul_51_T mul_51 =                     neg_6 * sub_77; 
    TensorOutput_neg_7_T neg_7 =                     -mul_50; 
  for (int i = 0; i < 1; i++) 
  { 
    bsx_ptr[thread_coord.y * 1 + i] = neg_7.values[i]; 
  } 
  for (int i = 0; i < 1; i++) 
  { 
    bsx.values[i] = neg_7.values[i];
  } 
    TensorOutput_neg_8_T neg_8 =                     -mul_51; 
  for (int i = 0; i < 1; i++) 
  { 
    bsy_ptr[thread_coord.y * 1 + i] = neg_8.values[i]; 
  } 
  for (int i = 0; i < 1; i++) 
  { 
    bsy.values[i] = neg_8.values[i];
  } 


      // output for aggregator
      

      // output for map
      
    }
  }
  // carry-out fix-up for aggregators
  
}
