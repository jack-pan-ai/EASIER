// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "merged_utils.cuh"
#include "merged_policy.cuh"
#include "merged_spmv_kernels.cuh"

/**
 * Launch kernel configuration. <cub>
 */
namespace merged
{
    // Import CUB namespace to avoid having to prefix every CUB function
    using namespace cub;

    template <int DimReducer, typename ValueT, typename OffsetT>
    __global__ void ReducerCarryFixupKernel(
        const OffsetT *carry_keys,
        const ValueT *carry_values,
        ValueT *output_vector_y,
        int num_merge_tiles,
        OffsetT num_rows)
    {
        const int tile_idx =
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (tile_idx >= num_merge_tiles)
            return;

        const OffsetT row = carry_keys[tile_idx];
        if (row >= num_rows)
            return;

        #pragma unroll
        for (int lane = 0; lane < DimReducer; ++lane)
        {
            atomicAdd(
                &output_vector_y[row * DimReducer + lane],
                carry_values[lane * num_merge_tiles + tile_idx]);
        }
    }

    template <
        typename ValueT,
        typename OffsetT,
        typename SpmvSearchKernelT,
        typename SpmvKernelT>
    __host__ __forceinline__ static cudaError_t merged_spmv_dispatch(
        FlexParams<ValueT, OffsetT> spmv_params,  ///< SpMV input parameter bundle
        void *d_temp_storage,                     ///< [in] Pointer to the device-accessible allocation of temporary storage
        size_t &temp_storage_bytes,               ///< [in,out] Reference to size in bytes of d_temp_storage allocations
        SpmvSearchKernelT spmv_search_kernel,     ///< [in] Kernel function pointer to parameterization of AgentSpmvSearchKernel
        SpmvKernelT spmv_kernel,                  ///< [in] Kernel function pointer to parameterization of AgentSpmvKernel
        LaunchKernelConfig spmv_config,           ///< [in] Dispatch parameters that match the policy that \p spmv_kernel was compiled for
        bool initialize_tile_coordinates,         ///< [in] Whether the immutable merge-path coordinate cache must be populated
        bool debug_synchronous = false,           ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        cudaStream_t stream = 0)                  ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    {
        cudaError error = cudaSuccess;
        do
        {
            using CoordinateT = typename cub::CubVector<OffsetT, 2>::Type;
            using SpmvParamsT = FlexParams<ValueT, OffsetT>;

            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal)))
                break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal)))
                break;

            // Get max x-dimension of grid
            int max_dim_x;
            if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal)))
                break;

            // Total number of spmv work items
            OffsetT num_merge_items =
                spmv_params.num_rows + spmv_params.num_nonzeros;

            // Tile sizes of kernels
            int merge_tile_size = spmv_config.block_threads * spmv_config.items_per_thread;

            // Number of tiles for kernels
            int num_merge_tiles = static_cast<int>(
                cub::DivideAndRoundUp(num_merge_items,
                                      static_cast<OffsetT>(merge_tile_size)));

            // Get SM occupancy for kernels
            int spmv_sm_occupancy;
            if (CubDebug(error = cub::MaxSmOccupancy(
                             spmv_sm_occupancy,
                             spmv_kernel,
                             spmv_config.block_threads)))
                break;

            // Get grid dimensions
            dim3 spmv_grid_size(
                CUB_MIN(num_merge_tiles, max_dim_x),
                cub::DivideAndRoundUp(num_merge_tiles, max_dim_x),
                1);

            static_assert(
                4 > 0,
                "reducer dispatch requires at least one output lane");
            size_t allocation_sizes[3];
            allocation_sizes[0] = (num_merge_tiles + 1) * sizeof(CoordinateT); // bytes needed for tile starting coordinates
            allocation_sizes[1] =
                static_cast<size_t>(num_merge_tiles) * sizeof(OffsetT);
            allocation_sizes[2] =
                static_cast<size_t>(num_merge_tiles) *
                4 * sizeof(ValueT);

            // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
            void *allocations[3] = {};
            if (CubDebug(error = cub::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
                break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                break;
            }

            // Alias the other allocations
           CoordinateT *d_tile_coordinates = (CoordinateT *)allocations[0]; // Agent starting coordinates
            spmv_params.d_tile_carry_keys =
                reinterpret_cast<OffsetT *>(allocations[1]);
            spmv_params.d_tile_carry_values =
                reinterpret_cast<ValueT *>(allocations[2]);


            // Get search/init grid dims
            int search_block_size = INIT_KERNEL_THREADS;
            int search_grid_size = cub::DivideAndRoundUp(num_merge_tiles + 1, search_block_size);

            // diagonal code
            if (search_grid_size < sm_count) 
            { 
                d_tile_coordinates = NULL; 
            } 
            else 
            { 
                // Row offsets and the merge policy are immutable for a
                // compiled selector. Populate their coordinate table once,
                // then reuse it for every subsequent reducer evaluation.
                if (initialize_tile_coordinates)
                {
                    // Log spmv_search_kernel configuration
                    if (debug_synchronous)
                    {
                        _CubLog("Invoking spmv_search_kernel<<<%d, %d, 0, %lld>>>()\n",
                                search_grid_size, search_block_size, (long long)stream);
                    }
                    // Invoke spmv_search_kernel
                    spmv_search_kernel<<<search_grid_size, search_block_size, 0, stream>>>(
                        num_merge_tiles, d_tile_coordinates, spmv_params);
                    // Check for failure to launch
                    if (CubDebug(error = cudaPeekAtLastError()))
                        break;
                    // Sync the stream if specified to flush runtime errors
                    if (debug_synchronous && (CubDebug(error = cub::SyncStream(stream))))
                        break;
                }
            } 

            // Log spmv_kernel configuration
            if (debug_synchronous)
                _CubLog("Invoking spmv_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                        spmv_grid_size.x, spmv_grid_size.y, spmv_grid_size.z, spmv_config.block_threads, (long long)stream, spmv_config.items_per_thread, spmv_sm_occupancy);

            // Invoke spmv_kernel
            spmv_kernel<<<spmv_grid_size, spmv_config.block_threads, 0, stream>>>(
                spmv_params, d_tile_coordinates, 
                num_merge_tiles
            );

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError()))
                break;

            const int carry_block_size = 256;
            const int carry_grid_size = cub::DivideAndRoundUp(
                num_merge_tiles, carry_block_size);
            ReducerCarryFixupKernel<4><<<carry_grid_size, carry_block_size, 0, stream>>>(
                spmv_params.d_tile_carry_keys,
                spmv_params.d_tile_carry_values + static_cast<size_t>(0) * num_merge_tiles,
                spmv_params.output_y_reduce_receiver_ptr,
                num_merge_tiles, spmv_params.num_rows);
            if (CubDebug(error = cudaPeekAtLastError()))
                break;


            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = cub::SyncStream(stream))))
                break;

       } while (0);

        // Return error
        return error;
    }

    template <
        typename ValueT,
        typename OffsetT>
    __host__ __forceinline__ static cudaError_t merged_spmv_launch(
        FlexParams<ValueT, OffsetT> spmv_params, ///< SpMV input parameter bundle
        void *d_temp_storage,                    ///< [in] Pointer to the device-accessible allocation of temporary storage
        size_t &temp_storage_bytes,              ///< [in,out] Reference to size in bytes of d_temp_storage allocations
        bool initialize_tile_coordinates,        ///< [in] Whether the immutable merge-path coordinate cache must be populated
        bool debug_synchronous = false,          ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        cudaStream_t stream = 0)
    {
        // kernel for PTX
        cudaError error = cudaSuccess;
        do
        {
            // policy for PTX and init the config
            using PtxSpmvPolicyT = PtxSpmvPolicyT<ValueT>;
            LaunchKernelConfig spmv_config;
            spmv_config.template Init<PtxSpmvPolicyT>();

            using CoordinateT = typename cub::CubVector<OffsetT, 2>::Type;
            using SpmvParamsT = FlexParams<ValueT, OffsetT>;

            // [INFO] the row_end_offsets is shifted by 1, 
            spmv_params.d_row_end_offsets = spmv_params.d_row_end_offsets + 1; 

            // fused fixup kernel with spmv
            error = merged_spmv_dispatch(spmv_params, d_temp_storage, temp_storage_bytes,
                                         SpmvSearchKernel<PtxSpmvPolicyT, OffsetT, CoordinateT, SpmvParamsT>,
                                         SpmvKernel<PtxSpmvPolicyT, ValueT, OffsetT, CoordinateT, SpmvParamsT>,
                                         spmv_config,
                                         initialize_tile_coordinates,
                                         debug_synchronous, stream);

            if (CubDebug(error))
                break;
        } while (0);
        return error;
    }

} // namespace merged
