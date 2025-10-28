#pragma once
#include <cub/cub.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

#include "merged_utils.cuh"
#include "merged_agent_spmv.cuh"

/**
 * Computes the begin offsets into A and B for the specific diagonal
 */
namespace merged
{
    template <
        typename AIteratorT,
        typename BIteratorT,
        typename OffsetT,
        typename CoordinateT>
    __host__ __device__ __forceinline__ void MergePathSearch(
        OffsetT diagonal,
        AIteratorT a,
        BIteratorT b,
        OffsetT a_len,
        OffsetT b_len,
        CoordinateT &path_coordinate)
    {
        /// The value type of the input iterator
        typedef typename std::iterator_traits<AIteratorT>::value_type T;

        OffsetT split_min = CUB_MAX(diagonal - b_len, 0);
        OffsetT split_max = CUB_MIN(diagonal, a_len);

        while (split_min < split_max)
        {
            OffsetT split_pivot = (split_min + split_max) >> 1;
            if (a[split_pivot] <= b[diagonal - split_pivot - 1])
            {
                // Move candidate split range up A, down B
                split_min = split_pivot + 1;
            }
            else
            {
                // Move candidate split range up B, down A
                split_max = split_pivot;
            }
        }

        path_coordinate.x = CUB_MIN(split_min, a_len);
        path_coordinate.y = diagonal - split_min;
    }

    /**
     * Spmv search kernel. Identifies merge path starting coordinates for each tile.
     */
    template <
        typename SpmvPolicyT, ///< Parameterized SpmvPolicy tuning policy type
        typename OffsetT,     ///< Signed integer type for sequence offsets
        typename CoordinateT, ///< Merge path coordinate type
        typename SpmvParamsT> ///< FlexParams type
    __global__ void SpmvSearchKernel(
        int num_merge_tiles,             ///< [in] Number of SpMV merge tiles (spmv grid size)
        CoordinateT *d_tile_coordinates, ///< [out] Pointer to the temporary array of tile starting coordinates
        SpmvParamsT spmv_params)         ///< [in] SpMV input parameter bundle
    {
        /// Constants
        enum
        {
            BLOCK_THREADS = SpmvPolicyT::BLOCK_THREADS,
            ITEMS_PER_THREAD = SpmvPolicyT::ITEMS_PER_THREAD,
            TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
        };

        typedef cub::CacheModifiedInputIterator<
            SpmvPolicyT::ROW_OFFSETS_SEARCH_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
            RowOffsetsSearchIteratorT;

        // Find the starting coordinate for all tiles (plus the end coordinate of the last one)
        int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (tile_idx < num_merge_tiles + 1)
        {
            OffsetT diagonal = (tile_idx * TILE_ITEMS);
            CoordinateT tile_coordinate;
            cub::CountingInputIterator<OffsetT> nonzero_indices(0);

            // Search the merge path
            merged::MergePathSearch(
                diagonal,
                RowOffsetsSearchIteratorT(spmv_params.d_row_end_offsets),
                nonzero_indices,
                spmv_params.num_rows,
                spmv_params.num_nonzeros,
                tile_coordinate);

            // Output starting offset
            d_tile_coordinates[tile_idx] = tile_coordinate;
        }
    }

    /**
     * Flexible Spmv kernel
     */
    template <
        typename SpmvPolicyT, ///< Parameterized SpmvPolicy tuning policy type
        typename ValueT,      ///< Matrix and vector value type
        typename OffsetT,     ///< Signed integer type for sequence offsets
        typename CoordinateT, ///< Merge path coordinate type
        typename SpmvParamsT> ///< SpmvParams type
    __launch_bounds__(int(SpmvPolicyT::BLOCK_THREADS))
        __global__ void SpmvKernel(
            SpmvParamsT spmv_params,         ///< [in] Flexible Spmv input parameter bundle
            CoordinateT *d_tile_coordinates, ///< [in] Pointer to the temporary array of tile starting coordinates
            int num_tiles                   ///< [in] Number of merge tiles
        )     
    {
        // Flexible Spmv agent type specialization
        typedef AgentFlexSpmv<
            SpmvPolicyT,
            ValueT,
            OffsetT>
            AgentFlexSpmvT;

        // Shared memory for AgentFlexSpmv
        __shared__ typename AgentFlexSpmvT::TempStorage temp_storage;

        AgentFlexSpmvT(temp_storage, spmv_params).ConsumeTile(d_tile_coordinates, num_tiles);
    }
}