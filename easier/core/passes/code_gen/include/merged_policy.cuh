#pragma once
#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_load.cuh>

/**
 * Parameterizable tuning policy type for AgentSegmentFixup
 */
namespace merged
{

    template <
        int _BLOCK_THREADS,                      ///< Threads per thread block
        int _ITEMS_PER_THREAD,                   ///< Items per thread (per tile of input)
        cub::BlockLoadAlgorithm _LOAD_ALGORITHM, ///< The BlockLoad algorithm to use
        cub::CacheLoadModifier _LOAD_MODIFIER,   ///< Cache load modifier for reading input elements
        cub::BlockScanAlgorithm _SCAN_ALGORITHM> ///< The BlockScan algorithm to use
    struct AgentSegmentFixupPolicy
    {
        enum
        {
            BLOCK_THREADS = _BLOCK_THREADS,       ///< Threads per thread block
            ITEMS_PER_THREAD = _ITEMS_PER_THREAD, ///< Items per thread (per tile of input)
        };

        static const cub::BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM; ///< The BlockLoad algorithm to use
        static const cub::CacheLoadModifier LOAD_MODIFIER = _LOAD_MODIFIER;    ///< Cache load modifier for reading input elements
        static const cub::BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM; ///< The BlockScan algorithm to use
    };

    /**
     * Parameterizable tuning policy type for AgentSpmv
     */
    template <
        int _BLOCK_THREADS,                                       ///< Threads per thread block
        int _ITEMS_PER_THREAD,                                    ///< Items per thread (per tile of input)
        cub::CacheLoadModifier _ROW_OFFSETS_SEARCH_LOAD_MODIFIER, ///< Cache load modifier for reading CSR row-offsets during search
        cub::CacheLoadModifier _ROW_OFFSETS_LOAD_MODIFIER,        ///< Cache load modifier for reading CSR row-offsets
        cub::CacheLoadModifier _COLUMN_INDICES_LOAD_MODIFIER,     ///< Cache load modifier for reading CSR column-indices
        cub::CacheLoadModifier _VALUES_LOAD_MODIFIER,             ///< Cache load modifier for reading CSR values
        cub::CacheLoadModifier _VECTOR_VALUES_LOAD_MODIFIER,      ///< Cache load modifier for reading vector values
        bool _DIRECT_LOAD_NONZEROS,                               ///< Whether to load nonzeros directly from global during sequential merging (vs. pre-staged through shared memory)
        cub::BlockScanAlgorithm _SCAN_ALGORITHM>                  ///< The BlockScan algorithm to use
    struct AgentSpmvPolicy
    {
        enum
        {
            BLOCK_THREADS = _BLOCK_THREADS,               ///< Threads per thread block
            ITEMS_PER_THREAD = _ITEMS_PER_THREAD,         ///< Items per thread (per tile of input)
            DIRECT_LOAD_NONZEROS = _DIRECT_LOAD_NONZEROS, ///< Whether to load nonzeros directly from global during sequential merging (pre-staged through shared memory)
        };

        static const cub::CacheLoadModifier ROW_OFFSETS_SEARCH_LOAD_MODIFIER = _ROW_OFFSETS_SEARCH_LOAD_MODIFIER; ///< Cache load modifier for reading CSR row-offsets
        static const cub::CacheLoadModifier ROW_OFFSETS_LOAD_MODIFIER = _ROW_OFFSETS_LOAD_MODIFIER;               ///< Cache load modifier for reading CSR row-offsets
        static const cub::CacheLoadModifier COLUMN_INDICES_LOAD_MODIFIER = _COLUMN_INDICES_LOAD_MODIFIER;         ///< Cache load modifier for reading CSR column-indices
        static const cub::CacheLoadModifier VALUES_LOAD_MODIFIER = _VALUES_LOAD_MODIFIER;                         ///< Cache load modifier for reading CSR values
        static const cub::CacheLoadModifier VECTOR_VALUES_LOAD_MODIFIER = _VECTOR_VALUES_LOAD_MODIFIER;           ///< Cache load modifier for reading vector values
        static const cub::BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;                                    ///< The BlockScan algorithm to use
    };

    /// SM60
    template <typename ValueT>
    struct Policy600
    {
        typedef AgentSpmvPolicy<
            (sizeof(ValueT) > 4) ? 64 : 128,
            (sizeof(ValueT) > 4) ? 5 : 7,
            cub::LOAD_DEFAULT,
            cub::LOAD_DEFAULT,
            cub::LOAD_DEFAULT,
            cub::LOAD_DEFAULT,
            cub::LOAD_DEFAULT,
            // false, // INDIRECT_LOAD_NONZEROS
            true, // DIRECT_LOAD_NONZEROS
            cub::BLOCK_SCAN_WARP_SCANS>
            SpmvPolicyT;

        typedef AgentSegmentFixupPolicy<
            128,
            3,
            cub::BLOCK_LOAD_DIRECT,
            cub::LOAD_LDG,
            cub::BLOCK_SCAN_WARP_SCANS>
            SegmentFixupPolicyT;
    };

} // namespace merged

// Define the policy to use based on architecture
#if (CUB_PTX_ARCH >= 600)
    template <typename ValueT>
    using PtxPolicy = merged::Policy600<ValueT>;
#else
    // Define a fallback policy if needed
    template <typename ValueT>
    using PtxPolicy = merged::Policy600<ValueT>;
#endif

// policy for PTX
template <typename ValueT>
struct PtxSpmvPolicyT : PtxPolicy<ValueT>::SpmvPolicyT {};

template <typename ValueT>
struct PtxSegmentFixupPolicy : PtxPolicy<ValueT>::SegmentFixupPolicyT {};