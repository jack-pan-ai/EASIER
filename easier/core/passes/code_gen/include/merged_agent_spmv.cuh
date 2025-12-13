// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * @file merged_agent_flex_spmv.cuh
 * @brief Extension of CUB's AgentSpmv
 */

#pragma once

#include <iterator>

#include <cub/agent/agent_spmv_orig.cuh>
#include <cub/util_type.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_exchange.cuh>
#include <cub/thread/thread_search.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/tex_ref_input_iterator.cuh>
#include <cub/util_namespace.cuh>

// Add this include to get FlexParams and dimension macros
#include "merged_utils.cuh"
#include "merged_spmv_kernels.cuh"

/// CUB namespace
namespace merged
{
    // Import CUB namespace to avoid having to prefix every CUB function
    using namespace cub;

    // Reduce tensor by key op for tensor type
    template <typename TensorT>
    struct ReduceTensorByKeyOp
    {

        /// Constructor
        __host__ __device__ __forceinline__ ReduceTensorByKeyOp() {}

        /// Scan operator
        __host__ __device__ __forceinline__ TensorT operator()(
            const TensorT &first,  ///< First partial reduction
            const TensorT &second) ///< Second partial reduction
        {
            TensorT retval = second;

            if (first.key == second.key)
            {
                retval = first + retval;
            }

            return retval;
        }
    };

    /**
     * @brief AgentFlexSpmv implements SpMV using a matrix A and vector x
     */
    template <
        typename AgentSpmvPolicyT,   ///< Parameterized AgentSpmvPolicy tuning policy type
        typename ValueT,             ///< Matrix and vector value type
        typename OffsetT,            ///< Signed integer type for sequence offsets
        int PTX_ARCH = CUB_PTX_ARCH> ///< PTX compute capability
    struct AgentFlexSpmv
    {
        //---------------------------------------------------------------------
        // Types and constants
        //---------------------------------------------------------------------

        /// Constants
        enum
        {
            BLOCK_THREADS = AgentSpmvPolicyT::BLOCK_THREADS,
            ITEMS_PER_THREAD = AgentSpmvPolicyT::ITEMS_PER_THREAD,
            TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
        };

        /// 2D merge path coordinate type
        typedef typename cub::CubVector<OffsetT, 2>::Type CoordinateT;

        /// Input iterator wrapper types (for applying cache modifiers)
        typedef cub::CacheModifiedInputIterator<
            AgentSpmvPolicyT::ROW_OFFSETS_SEARCH_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
            RowOffsetsSearchIteratorT;

        typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::ROW_OFFSETS_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
            RowOffsetsIteratorT;

        typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::COLUMN_INDICES_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
            ColumnIndicesIteratorT;

        typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
            SpmValueIteratorT;

        typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::VECTOR_VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
            VectorValueIteratorT;

        // Smem for intermediate results and scan
        template <int DIM_REDUCER, typename BlockScanT>
        union SmemReuseReducer
        {
            typedef Tensor<ValueT, DIM_REDUCER> TensorT;
            // Smem for intermediate results
            TensorT s_tile_value_reducer[TILE_ITEMS]; 
            // Smem needed for tile scanning
            typename BlockScanT::TempStorage scan;
        };

        // Tensor and TensorKey for input vector x
          typedef Tensor<ValueT, 1> TensorInput_scatter_6_T; 
  typedef Tensor<ValueT, 1> TensorInput_area_T; 
  typedef Tensor<ValueT, 1> TensorInput_h_T; 


        // Tensor and TensorKey for reducers 
        
          typedef Tensor<ValueT, 1> TensorOutput_neg_6_T; 
  typedef Tensor<ValueT, 1> TensorOutput_truediv_12_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_63_T; 
  typedef Tensor<ValueT, 1> TensorOutput_add_36_T; 


        /// Shared memory type required by this thread block
        struct _TempStorage
        {
            // tile coordinates for blocks
            CoordinateT tile_coords[2];
            // smem for intermediate results and scan
            
        };

        /// Temporary storage type (unionable)
        struct TempStorage : Uninitialized<_TempStorage>
        {
        };

        //---------------------------------------------------------------------
        // Per-thread fields
        //---------------------------------------------------------------------

        _TempStorage &temp_storage; /// Reference to temp_storage

        FlexParams<ValueT, OffsetT> &spmv_params;
        RowOffsetsIteratorT wd_row_end_offsets;

        // [code generation] wrapper pointers for loading the data
          VectorValueIteratorT scatter_6_ptr; 
  VectorValueIteratorT area_ptr; 
  VectorValueIteratorT h_ptr; 


        //---------------------------------------------------------------------
        // Constructor
        //---------------------------------------------------------------------

        /**
         * Constructor // [code generation]
         */
        __device__ __forceinline__
        AgentFlexSpmv(
            TempStorage &temp_storage,                ///< Reference to temp_storage
            FlexParams<ValueT, OffsetT> &spmv_params) ///< SpMV input parameter bundle
            : temp_storage(temp_storage.Alias()),
                wd_row_end_offsets(spmv_params.d_row_end_offsets),
                  scatter_6_ptr(spmv_params.scatter_6_ptr), 
    area_ptr(spmv_params.area_ptr), 
    h_ptr(spmv_params.h_ptr), 

              spmv_params(spmv_params)
        {
        }

        //---------------------------------------------------------------------
        // Tile processing
        //---------------------------------------------------------------------

        __device__ __forceinline__ void loading_offsets(
            int tile_num_rows, 
            CoordinateT tile_start_coord){
            // Gather the row end-offsets for the merge tile into shared memory
            #pragma unroll 1
            for (int item = threadIdx.x; item < tile_num_rows + ITEMS_PER_THREAD; item += BLOCK_THREADS)
            {
                const OffsetT offset =
                    (cub::min)(static_cast<OffsetT>(tile_start_coord.x + item),
                            static_cast<OffsetT>(spmv_params.num_rows - 1));
                temp_storage.s_tile_row_end_offsets[item] = wd_row_end_offsets[offset];
            }

            CTA_SYNC();
        }

        __device__ __forceinline__ void search_thread_start_coord(
            OffsetT *s_tile_row_end_offsets, // [in] Shared memory array of row end offsets for the merge tile
            CoordinateT tile_start_coord, // [in] Starting coordinate of the merge tile
            int tile_num_rows, // [in] Number of rows in the merge tile
            int tile_num_nonzeros, // [in] Number of non-zeros in the merge tile
            CoordinateT &thread_start_coord // [out] Starting coordinate of the thread
        )
        {
            CountingInputIterator<OffsetT> tile_nonzero_indices(tile_start_coord.y);

            MergePathSearch(
                OffsetT(threadIdx.x * ITEMS_PER_THREAD), // Diagonal
                s_tile_row_end_offsets,                  // List A
                tile_nonzero_indices,                    // List B
                tile_num_rows,
                tile_num_nonzeros,
                thread_start_coord);

            CTA_SYNC(); // Perf-sync
        }


        template <int DimReducer, typename BlockScanT, typename TensorT, typename ReduceBySegmentOpT>
        __device__ __forceinline__ void reduce(
            TensorT *s_tile_value_nonzeros,      ///< [in, code gen] Shared memory array of non-zero values for the merge tile
            OffsetT *s_tile_row_end_offsets,    ///< [in, code gen] Shared memory array of row end offsets for the merge tile
            CoordinateT tile_start_coord,       ///< [in] Starting coordinate of the merge tile
            CoordinateT tile_end_coord,         ///< [in] Ending coordinate of the merge tile
            CoordinateT thread_start_coord,     ///< [in] Starting coordinate of the thread
            int tile_num_rows,                  ///< [in] Number of rows in the merge tile
            int tile_num_nonzeros,               ///< [in] Number of non-zeros in the merge tile
            ValueT *output_vector_y,             ///< [out] Output vector y
            typename BlockScanT::TempStorage &scan_storage ///< [in] Scan storage for BlockScanT operations
        )
        {
            typedef TensorKey<OffsetT, ValueT, DimReducer> TensorKeyT;
            // Compute the thread's merge path segment
            CoordinateT thread_current_coord = thread_start_coord;
            TensorKeyT scan_segment[ITEMS_PER_THREAD];
            TensorT running_total;
            CountingInputIterator<OffsetT> tile_nonzero_indices(tile_start_coord.y);

            OffsetT row_end_offset = s_tile_row_end_offsets[thread_current_coord.x];
            TensorT nonzero = s_tile_value_nonzeros[thread_current_coord.y];

// Reduce
#pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                if (tile_nonzero_indices[thread_current_coord.y] < row_end_offset)
                {
// Move down (accumulate)
                    scan_segment[ITEM].set(nonzero.values);
                    running_total = running_total + nonzero;
                    ++thread_current_coord.y;
                    nonzero = s_tile_value_nonzeros[thread_current_coord.y];
                }
                else
                {
// Move right (reset)
                    scan_segment[ITEM].set(0.0);
                    running_total.set(0.0);
                    ++thread_current_coord.x;
                    row_end_offset = s_tile_row_end_offsets[thread_current_coord.x];
                }
                scan_segment[ITEM].key = thread_current_coord.x;
            }

            CTA_SYNC();

            // Block-wide reduce-value-by-segment
            TensorKeyT tile_carry;
            ReduceBySegmentOpT scan_op;
            TensorKeyT scan_item(running_total.values);
            scan_item.key = thread_current_coord.x;

            BlockScanT(scan_storage).ExclusiveScan(scan_item, scan_item, scan_op, tile_carry);

            if (threadIdx.x == 0)
            {
                scan_item.key = thread_start_coord.x;
                scan_item.set(0.0);
            }

            if (tile_num_rows > 0)
            {

                CTA_SYNC();
                // Scan downsweep and scatter
                // memory reuse for the partial results 
                // TILE_ITEMS is used to avoid bank conflict
                TensorT *s_partials = s_tile_value_nonzeros;

                if (scan_item.key != scan_segment[0].key)
                {
                    s_partials[scan_item.key].set(scan_item.values);
                }
                else
                {
                    scan_segment[0] = scan_segment[0] + scan_item;
                }

#pragma unroll
                for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ++ITEM)
                {
                    if (scan_segment[ITEM - 1].key != scan_segment[ITEM].key)
                    {
                        s_partials[scan_segment[ITEM - 1].key].set(scan_segment[ITEM - 1].values);
                    }
                    else
                    {
                        scan_segment[ITEM] = scan_segment[ITEM] + scan_segment[ITEM - 1];
                    }
                }

                CTA_SYNC();

// memory coalescing for writing the output vector y
#pragma unroll 1
                for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS)
                {
                    #pragma unroll
                    for (int i = 0; i < DimReducer; i++)
                    {
                        atomicAdd(
                            &output_vector_y[(tile_start_coord.x + item) * DimReducer + i],
                            s_partials[item].values[i]
                        );
                    }
                }
            }

            CTA_SYNC();

            // atomic add the residual sum, the tile's carry-out, to the Global memory
            if (threadIdx.x == 0)
            {
                tile_carry.key += tile_start_coord.x;
                if (tile_carry.key < spmv_params.num_rows)
                {
                    #pragma unroll
                    for (int i = 0; i < DimReducer; i++)
                    {
                        atomicAdd(
                            &output_vector_y[tile_carry.key * DimReducer + i],
                            tile_carry.values[i]);
                    }
                };
            }
        }

        /**
         * Consume a merge tile, specialized for direct load of nonzeros
         */
        __device__ __forceinline__ void ConsumeTile(
            int tile_idx,
            CoordinateT tile_start_coord,
            CoordinateT tile_end_coord,
            Int2Type<true> is_direct_load) ///< Marker type indicating whether to load nonzeros directly during path-discovery or beforehand in batch
        {
            int tile_num_rows = tile_end_coord.x - tile_start_coord.x;
            int tile_num_nonzeros = tile_end_coord.y - tile_start_coord.y;

            

// Select
// Gather the nonzeros for the merge tile into shared memory
#pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                int nonzero_idx = threadIdx.x + (ITEM * BLOCK_THREADS);

                if (nonzero_idx < tile_num_nonzeros)
                {
                    // [code generation]
                        VectorValueIteratorT scatter_6_ptr_current = scatter_6_ptr +                     (tile_start_coord.y + nonzero_idx) * 1; 
    TensorInput_scatter_6_T scatter_6(scatter_6_ptr_current); 
    VectorValueIteratorT area_ptr_current = area_ptr +                     (tile_start_coord.y + nonzero_idx) * 1; 
    TensorInput_area_T area(area_ptr_current); 
    VectorValueIteratorT h_ptr_current = h_ptr +                     (tile_start_coord.y + nonzero_idx) * 1; 
    TensorInput_h_T h(h_ptr_current); 


                    // map
                        TensorOutput_neg_6_T neg_6 = -scatter_6; 
    TensorOutput_truediv_12_T truediv_12 = neg_6 /                     area; 
    TensorOutput_mul_63_T mul_63 = 0.00028571 *                     truediv_12; 
    TensorOutput_add_36_T add_36 = h +                     mul_63; 


                    //output for map
                      #pragma unroll 
  for (int i = 0; i < 1; i++) 
  { 
    spmv_params.output_y_truediv_12_ptr[(tile_start_coord.y + nonzero_idx)                     * 1 + i] = truediv_12.values[i]; 
  } 
  #pragma unroll 
  for (int i = 0; i < 1; i++) 
  { 
    spmv_params.output_y_add_36_ptr[(tile_start_coord.y + nonzero_idx)                     * 1 + i] = add_36.values[i]; 
  } 

                }
            }

            CTA_SYNC();

            // [code generation]
            
        }



        /**
         * Process a merge tile
         */
        __device__ __forceinline__ void ConsumeTile(
            CoordinateT *d_tile_coordinates, ///< [in] Pointer to the temporary array of tile starting coordinates
            int num_merge_tiles             ///< [in] Total number of merge tiles
        )
        {
            int tile_idx = (blockIdx.y * gridDim.x) + blockIdx.x;

            if (tile_idx >= num_merge_tiles)
                return;

            // just padding parameters here 
            CoordinateT tile_start_coord = {-1, tile_idx * TILE_ITEMS};
            CoordinateT tile_end_coord = {-1, min(tile_idx * TILE_ITEMS + TILE_ITEMS, spmv_params.num_nonzeros)};

            ConsumeTile(
                tile_idx,
                tile_start_coord,
                tile_end_coord,
                Int2Type<AgentSpmvPolicyT::DIRECT_LOAD_NONZEROS>()); // PTX >=520 use the indirect load of nonzeros
        }
    };

} // namespace merged