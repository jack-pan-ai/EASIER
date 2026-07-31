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
          typedef Tensor<ValueT, 4> TensorInput_particle_values_T; 
  typedef Tensor<ValueT, 2> TensorInput_receiver_centers_T; 
  typedef Tensor<ValueT, 2> TensorInput_positions_T; 


        // Tensor and TensorKey for map 
          typedef Tensor<ValueT, 1> TensorOutput_getitem_T; 
  typedef Tensor<ValueT, 1> TensorOutput_clone_T; 
  typedef Tensor<ValueT, 1> TensorOutput_getitem_1_T; 
  typedef Tensor<ValueT, 1> TensorOutput_clone_1_T; 
  typedef Tensor<ValueT, 1> TensorOutput_getitem_2_T; 
  typedef Tensor<ValueT, 1> TensorOutput_clone_2_T; 
  typedef Tensor<ValueT, 1> TensorOutput_getitem_3_T; 
  typedef Tensor<ValueT, 1> TensorOutput_clone_3_T; 
  typedef Tensor<ValueT, 1> TensorOutput_sub_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_T; 
  typedef Tensor<ValueT, 1> TensorOutput_sub_1_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_1_T; 
  typedef Tensor<ValueT, 1> TensorOutput_abs_1_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_2_T; 
  typedef Tensor<ValueT, 1> TensorOutput_sub_2_T; 
  typedef Tensor<ValueT, 1> TensorOutput_sub_3_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_3_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_4_T; 
  typedef Tensor<ValueT, 1> TensorOutput_lt_T; 
  typedef Tensor<ValueT, 1> TensorOutput_where_T; 
  typedef Tensor<ValueT, 1> TensorOutput_abs_2_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_5_T; 
  typedef Tensor<ValueT, 1> TensorOutput_sub_4_T; 
  typedef Tensor<ValueT, 1> TensorOutput_sub_5_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_6_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_7_T; 
  typedef Tensor<ValueT, 1> TensorOutput_lt_1_T; 
  typedef Tensor<ValueT, 1> TensorOutput_where_1_T; 
  typedef Tensor<ValueT, 1> TensorOutput_mul_8_T; 
  typedef Tensor<ValueT, 1> TensorOutput_getitem_4_T; 
  typedef Tensor<ValueT, 1> TensorOutput_clone_4_T; 
  typedef Tensor<ValueT, 4> TensorOutput_gt_T; 
  typedef Tensor<ValueT, 4> TensorOutput_where_2_T; 
  typedef Tensor<ValueT, 4> TensorOutput_mul_9_T; 

        // Tensor and TensorKey for reducers 
          // Tensor and TensorKey for reducers 
  typedef TensorKey<OffsetT, ValueT, 4>                     TensorKeyOutput_reduce_receiver_T; 
  typedef Tensor<ValueT, 4> TensorOutput_reduce_receiver_T; 
  // Reduce-value-by-segment scan operator 
  typedef ReduceTensorByKeyOp<TensorKeyOutput_reduce_receiver_T>                        ReduceBySegmentOp_reduce_receiver_T; 
  typedef BlockScan< 
            TensorKeyOutput_reduce_receiver_T, 
            BLOCK_THREADS, 
            AgentSpmvPolicyT::SCAN_ALGORITHM> 
            BlockScan_reduce_receiver_T; 


        /// Shared memory type required by this thread block
        struct _TempStorage
        {
            // tile coordinates for blocks
            CoordinateT tile_coords[2];
            // smem for intermediate results and scan
                           SmemReuseReducer<4,                     BlockScan_reduce_receiver_T> smem_reduce_receiver; 

            OffsetT s_tile_row_end_offsets[TILE_ITEMS];
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
          VectorValueIteratorT particle_values_ptr; 
  VectorValueIteratorT receiver_centers_ptr; 
  VectorValueIteratorT positions_ptr; 
  ColumnIndicesIteratorT select_source_ptr; 
  ColumnIndicesIteratorT select_receiver_center_ptr; 


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
                  particle_values_ptr(spmv_params.particle_values_ptr), 
    receiver_centers_ptr(spmv_params.receiver_centers_ptr), 
    positions_ptr(spmv_params.positions_ptr), 
    select_source_ptr(spmv_params.select_source_ptr), 
    select_receiver_center_ptr(spmv_params.select_receiver_center_ptr), 

              spmv_params(spmv_params)
        {
        }

        __device__ __forceinline__
        TensorOutput_reduce_receiver_T evaluate_reducer_nonzero(
            OffsetT nonzero_idx, OffsetT row_idx)
        {
    ColumnIndicesIteratorT select_source_ptr_current =                         select_source_ptr + nonzero_idx; 
    TensorInput_positions_T                         select_source(positions_ptr + *select_source_ptr_current * 2); 
    const OffsetT select_receiver_center_ptr_current = row_idx; 
    TensorInput_receiver_centers_T select_receiver_center(receiver_centers_ptr + static_cast<size_t>(select_receiver_center_ptr_current) * 2); 
    ColumnIndicesIteratorT select_source_1_ptr_current =                         select_source_ptr + nonzero_idx; 
    TensorInput_particle_values_T                         select_source_1(particle_values_ptr + *select_source_1_ptr_current * 4); 
    TensorOutput_getitem_T                         getitem(select_receiver_center.values[0]); 
    TensorOutput_clone_T clone(getitem); 
    TensorOutput_getitem_1_T                         getitem_1(select_receiver_center.values[1]); 
    TensorOutput_clone_1_T clone_1(getitem_1); 
    TensorOutput_getitem_2_T                         getitem_2(select_source.values[0]); 
    TensorOutput_clone_2_T clone_2(getitem_2); 
    TensorOutput_getitem_3_T                         getitem_3(select_source.values[1]); 
    TensorOutput_clone_3_T clone_3(getitem_3); 
    TensorOutput_sub_T sub = clone -                     clone_2; 
    TensorOutput_mul_T mul = sub *                     4608000.0; 
    TensorOutput_sub_1_T sub_1 = clone_1 -                     clone_3; 
    TensorOutput_mul_1_T mul_1 = sub_1 *                     4608000.0; 
    TensorOutput_abs_1_T abs_1 = mul.abs(); 
    TensorOutput_mul_2_T mul_2 = abs_1 *                     abs_1; 
    TensorOutput_sub_2_T sub_2 = 0.75 -                     mul_2; 
    TensorOutput_sub_3_T sub_3 = 1.5 -                     abs_1; 
    TensorOutput_mul_3_T mul_3 = 0.5 *                     sub_3; 
    TensorOutput_mul_4_T mul_4 = mul_3 *                     sub_3; 
    TensorOutput_lt_T lt = abs_1 < 0.5; 
    TensorOutput_where_T where =                     _where(lt, sub_2, mul_4); 
    TensorOutput_abs_2_T abs_2 = mul_1.abs(); 
    TensorOutput_mul_5_T mul_5 = abs_2 *                     abs_2; 
    TensorOutput_sub_4_T sub_4 = 0.75 -                     mul_5; 
    TensorOutput_sub_5_T sub_5 = 1.5 -                     abs_2; 
    TensorOutput_mul_6_T mul_6 = 0.5 *                     sub_5; 
    TensorOutput_mul_7_T mul_7 = mul_6 *                     sub_5; 
    TensorOutput_lt_1_T lt_1 = abs_2 < 0.5; 
    TensorOutput_where_1_T where_1 =                     _where(lt_1, sub_4, mul_7); 
    TensorOutput_mul_8_T mul_8 = where *                     where_1; 
    TensorOutput_getitem_4_T getitem_4(mul_8); 
    TensorOutput_clone_4_T clone_4(getitem_4); 
    TensorOutput_gt_T gt = select_source_1 > 0.0; 
    TensorOutput_where_2_T where_2 =                     _where(gt, clone_4, clone_4); 
    TensorOutput_mul_9_T mul_9 = select_source_1 *                     where_2; 
    return mul_9; 
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
                static_cast<OffsetT>(tile_num_rows),
                static_cast<OffsetT>(tile_num_nonzeros),
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
            typename BlockScanT::TempStorage &scan_storage, ///< [in] Scan storage for BlockScanT operations
            int tile_idx,                        ///< [in] Merge tile index
            int num_merge_tiles,                 ///< [in] Carry-buffer stride
            int carry_lane_offset                ///< [in] First lane for this reducer
        )
        {
            typedef TensorKey<OffsetT, ValueT, DimReducer> TensorKeyT;
            // Compute the thread's merge path segment
            CoordinateT thread_current_coord = thread_start_coord;
            TensorKeyT scan_segment[ITEMS_PER_THREAD];
            TensorT running_total;
            CountingInputIterator<OffsetT> tile_nonzero_indices(tile_start_coord.y);

            OffsetT row_end_offset = s_tile_row_end_offsets[thread_current_coord.x];


// Reduce
#pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                if (tile_nonzero_indices[thread_current_coord.y] < row_end_offset)
                {
// Move down (accumulate)
                    TensorT nonzero = evaluate_reducer_nonzero(
                        tile_start_coord.y + thread_current_coord.y,
                        tile_start_coord.x + thread_current_coord.x);

                    scan_segment[ITEM].set(nonzero.values);
                    running_total = running_total + nonzero;
                    ++thread_current_coord.y;

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

// Every row boundary is owned by exactly one merge tile.  Initialize that
// completed-row suffix directly; a later carry-only kernel adds prefixes from
// preceding tiles for rows that cross tile boundaries.
#pragma unroll 1
                for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS)
                {
                    #pragma unroll
                    for (int i = 0; i < DimReducer; i++)
                    {
                        output_vector_y[
                            (tile_start_coord.x + item) * DimReducer + i
                        ] = s_partials[item].values[i];
                    }
                }
            }

            CTA_SYNC();

            // Save one residual vector per tile.  Keeping carries separate
            // removes the full reducer-output zero pass and confines atomics
            // to the compact tile-carry stream instead of every completed row.
            if (threadIdx.x == 0)
            {
                tile_carry.key += tile_start_coord.x;
                spmv_params.d_tile_carry_keys[tile_idx] = tile_carry.key;
                #pragma unroll
                for (int i = 0; i < DimReducer; i++)
                {
                    spmv_params.d_tile_carry_values[
                        (carry_lane_offset + i) * num_merge_tiles + tile_idx
                    ] = tile_carry.values[i];
                }
            }
        }

        /**
         * Consume a merge tile, specialized for direct load of nonzeros
         */
        __device__ __forceinline__ void ConsumeTile(
            int tile_idx,
            int num_merge_tiles,
            CoordinateT tile_start_coord,
            CoordinateT tile_end_coord,
            Int2Type<true> is_direct_load) ///< Marker type indicating whether to load nonzeros directly during path-discovery or beforehand in batch
        {
            int tile_num_rows = tile_end_coord.x - tile_start_coord.x;
            int tile_num_nonzeros = tile_end_coord.y - tile_start_coord.y;

            loading_offsets(tile_num_rows, tile_start_coord);



            // reduce the intermeidate computations 
            // all reducers share the same row end offsets 
            // Search for the thread's starting coordinate within the merge tile 
            CoordinateT thread_start_coord; 
            search_thread_start_coord( 
                temp_storage.s_tile_row_end_offsets, 
                tile_start_coord, 
                tile_num_rows, 
                tile_num_nonzeros, 
                thread_start_coord); 
            // [code generation]
               reduce<4, BlockScan_reduce_receiver_T, TensorOutput_reduce_receiver_T,                 ReduceBySegmentOp_reduce_receiver_T>( 
                temp_storage.smem_reduce_receiver.s_tile_value_reducer,                          ///< [in, code gen] Shared memory array of non-zero values for the merge tile 
                temp_storage.s_tile_row_end_offsets,                         ///< [in, code gen] Shared memory array of row end offsets for the merge tile 
                tile_start_coord,                               ///< [in] Starting coordinate of the merge tile 
                tile_end_coord,                                 ///< [in] Ending coordinate of the merge tile 
                thread_start_coord,                             ///< [in] Starting coordinate of the thread 
                tile_num_rows,                                  ///< [in] Number of rows in the merge tile 
                tile_num_nonzeros,                               ///< [in] Number of non-zeros in the merge tile 
                spmv_params.output_y_reduce_receiver_ptr,                       ///< [out] Output vector y 
                temp_storage.smem_reduce_receiver.scan,                        ///< [in] Scan storage for BlockScanT 
                tile_idx, num_merge_tiles,                             ///< [in] Tile identity and carry-buffer stride 
                0                             ///< [in] First carry lane for this reducer 
            ); 
   CTA_SYNC(); 

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
                
            // Read our starting coordinates 
            if (threadIdx.x < 2) 
            { 
                if (d_tile_coordinates == NULL) 
                {
                    // Search our starting coordinates 
                    OffsetT diagonal = (tile_idx + threadIdx.x) * TILE_ITEMS; 
                    CoordinateT tile_coord; 
                    CountingInputIterator<OffsetT> nonzero_indices(0); 
    
                    // Search the merge path 
                    MergePathSearch( 
                        diagonal, 
                        RowOffsetsSearchIteratorT(spmv_params.d_row_end_offsets), 
                        nonzero_indices, 
                        spmv_params.num_rows, 
                        spmv_params.num_nonzeros, 
                        tile_coord); 
                    temp_storage.tile_coords[threadIdx.x] = tile_coord; 
                } 
                else 
                { 
                    temp_storage.tile_coords[threadIdx.x] = d_tile_coordinates[tile_idx + threadIdx.x]; 
                } 
            } 
            CTA_SYNC(); 
            CoordinateT tile_start_coord = temp_storage.tile_coords[0]; 
            CoordinateT tile_end_coord = temp_storage.tile_coords[1]; 

            ConsumeTile(
                tile_idx,
                num_merge_tiles,
                tile_start_coord,
                tile_end_coord,
                Int2Type<AgentSpmvPolicyT::DIRECT_LOAD_NONZEROS>()); // PTX >=520 use the indirect load of nonzeros
        }
    };

} // namespace merged
