// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "merged_utils.cuh"

namespace merged
{

constexpr int DIRECT_MAP_BLOCK_THREADS = 256;

/**
 * Direct selector -> pointwise map -> full-store kernel.
 *
 * A selector entry identifies one independent mapped item.  Since this
 * specialization has no reducer or global aggregator, it does not need CSR
 * row-offset staging, merge-path coordinates, shared memory, CTA barriers, or
 * carry handling.
 */
template <typename ValueT, typename OffsetT>
__global__ void DirectGatherMapStoreKernel(
    FlexParams<ValueT, OffsetT> spmv_params)
{
    constexpr OffsetT LANE_WIDTH =
        static_cast<OffsetT>(${direct_map_lane_width});
    const uint64_t flat_idx =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t num_values =
        static_cast<uint64_t>(spmv_params.num_nonzeros) * LANE_WIDTH;
    if (flat_idx >= num_values)
        return;
    const OffsetT item_idx =
        static_cast<OffsetT>(flat_idx / LANE_WIDTH);
    const OffsetT lane_idx =
        static_cast<OffsetT>(flat_idx % LANE_WIDTH);

    ${input_agent_tenosrs_code}
    ${output_agent_tenosrs_code}
    ${map_agent_tenosrs_code}

    ${selector_code}
    ${map_code}
    ${output_agent_forloop_code}
}

template <typename ValueT, typename OffsetT>
__host__ __forceinline__ static cudaError_t merged_spmv_launch(
    FlexParams<ValueT, OffsetT> spmv_params,
    void *d_temp_storage,
    size_t &temp_storage_bytes,
    bool debug_synchronous = false,
    cudaStream_t stream = 0)
{
    (void)d_temp_storage;
    temp_storage_bytes = 0;

    constexpr uint64_t LANE_WIDTH = ${direct_map_lane_width};
    const uint64_t num_values =
        static_cast<uint64_t>(spmv_params.num_nonzeros) * LANE_WIDTH;
    const uint64_t num_blocks =
        (num_values + DIRECT_MAP_BLOCK_THREADS - 1) /
        DIRECT_MAP_BLOCK_THREADS;
    if (num_blocks == 0 || num_blocks > 0x7fffffffULL)
        return cudaErrorInvalidConfiguration;

    DirectGatherMapStoreKernel<ValueT, OffsetT>
        <<<static_cast<unsigned int>(num_blocks),
           DIRECT_MAP_BLOCK_THREADS, 0, stream>>>(spmv_params);

    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess)
        return error;
    if (debug_synchronous)
        return cudaStreamSynchronize(stream);
    return cudaSuccess;
}

} // namespace merged
