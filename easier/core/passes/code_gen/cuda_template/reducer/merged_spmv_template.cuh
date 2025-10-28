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
            int num_merge_items = spmv_params.num_rows + spmv_params.num_nonzeros;

            // Tile sizes of kernels
            int merge_tile_size = spmv_config.block_threads * spmv_config.items_per_thread;

            // Number of tiles for kernels
            int num_merge_tiles = cub::DivideAndRoundUp(num_merge_items, merge_tile_size);

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

            size_t allocation_sizes[1];
            allocation_sizes[0] = (num_merge_tiles + 1) * sizeof(CoordinateT); // bytes needed for tile starting coordinates

            // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
            void *allocations[1] = {};
            if (CubDebug(error = cub::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
                break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                break;
            }

            // Alias the other allocations
           CoordinateT *d_tile_coordinates = (CoordinateT *)allocations[0]; // Agent starting coordinates


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
                // Use separate search kernel if we have enough spmv tiles to saturate the device 
                // Log spmv_search_kernel configuration 
                if (debug_synchronous) 
                { 
                    _CubLog("Invoking spmv_search_kernel<<<%d, %d, 0, %lld>>>()\n", 
                            search_grid_size, search_block_size, (long long)stream); 
                } 
                // Invoke spmv_search_kernel 
                spmv_search_kernel<<<search_grid_size, search_block_size, 0, stream>>>(num_merge_tiles, d_tile_coordinates, spmv_params); 
                // Check for failure to launch 
                if (CubDebug(error = cudaPeekAtLastError())) 
                    break; 
                // Sync the stream if specified to flush runtime errors 
                if (debug_synchronous && (CubDebug(error = cub::SyncStream(stream)))) 
                    break; 
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
                                         debug_synchronous, stream);

            if (CubDebug(error))
                break;
        } while (0);
        return error;
    }

} // namespace merged