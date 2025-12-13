// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cstdio>
#include <type_traits>  // if you want to static_assert trivial copyability

#include "data_struct_shared.cuh"

#define INIT_KERNEL_THREADS 128 // INFO: this is from cub config

template <typename OffsetT, typename ValueT, int Dim>
struct TensorKey
{
    using Value = ValueT;

    OffsetT key;
    ValueT  values[Dim];

    // =====================================================================
    // Constructors & assignment – kept trivial / defaulted
    // =====================================================================

    // Default ctor: do NOT zero-initialize; user should call set() or
    // a value ctor if they need zeroing. This keeps codegen lighter.
    __host__ __device__ __forceinline__
    TensorKey() = default;

    // Trivial copy / assign – important for "trivially copyable"
    __host__ __device__ __forceinline__
    TensorKey(const TensorKey&) = default;

    __host__ __device__ __forceinline__
    TensorKey& operator=(const TensorKey&) = default;

    // Trivial destructor
    ~TensorKey() = default;

    // Construct from value array (values only)
    __host__ __device__ __forceinline__
    explicit TensorKey(const ValueT* v)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] = v[i];
    }

    // Construct from key + value array
    __host__ __device__ __forceinline__
    TensorKey(OffsetT k, const ValueT* v)
        : key(k)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] = v[i];
    }

    // Construct from iterator (values only)
    template <typename IteratorT>
    __host__ __device__ __forceinline__
    explicit TensorKey(IteratorT iter)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] = static_cast<ValueT>(iter[i]);
    }

    // =====================================================================
    // Indexing
    // =====================================================================

    __host__ __device__ __forceinline__
    ValueT& operator[](int idx)       { return values[idx]; }

    __host__ __device__ __forceinline__
    const ValueT& operator[](int idx) const { return values[idx]; }

    // =====================================================================
    // Compound assignment (TensorKey–TensorKey)
    // =====================================================================

    __host__ __device__ __forceinline__
    TensorKey& operator+=(const TensorKey& other)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] += other.values[i];
        return *this;
    }

    __host__ __device__ __forceinline__
    TensorKey& operator-=(const TensorKey& other)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] -= other.values[i];
        return *this;
    }

    __host__ __device__ __forceinline__
    TensorKey& operator*=(const TensorKey& other)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] *= other.values[i];
        return *this;
    }

    __host__ __device__ __forceinline__
    TensorKey& operator/=(const TensorKey& other)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] = (other.values[i] != ValueT(0)) ?
                        (values[i] / other.values[i]) : ValueT(0);
        return *this;
    }

    // =====================================================================
    // Compound assignment (TensorKey–scalar)
    // =====================================================================

    __host__ __device__ __forceinline__
    TensorKey& operator+=(ValueT scalar)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] += scalar;
        return *this;
    }

    __host__ __device__ __forceinline__
    TensorKey& operator-=(ValueT scalar)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] -= scalar;
        return *this;
    }

    __host__ __device__ __forceinline__
    TensorKey& operator*=(ValueT scalar)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] *= scalar;
        return *this;
    }

    __host__ __device__ __forceinline__
    TensorKey& operator/=(ValueT scalar)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] /= scalar;
        return *this;
    }

    // =====================================================================
    // Binary operators – built from compound ops (keeps code compact)
    // =====================================================================

    __host__ __device__ __forceinline__
    TensorKey operator+(const TensorKey& other) const
    {
        TensorKey result = *this;
        result += other;
        return result;
    }

    __host__ __device__ __forceinline__
    TensorKey operator-(const TensorKey& other) const
    {
        TensorKey result = *this;
        result -= other;
        return result;
    }

    __host__ __device__ __forceinline__
    TensorKey operator*(const TensorKey& other) const
    {
        TensorKey result = *this;
        result *= other;
        return result;
    }

    __host__ __device__ __forceinline__
    TensorKey operator/(const TensorKey& other) const
    {
        TensorKey result = *this;
        result /= other;
        return result;
    }

    __host__ __device__ __forceinline__
    TensorKey operator+(ValueT scalar) const
    {
        TensorKey result = *this;
        result += scalar;
        return result;
    }

    __host__ __device__ __forceinline__
    TensorKey operator-(ValueT scalar) const
    {
        TensorKey result = *this;
        result -= scalar;
        return result;
    }

    __host__ __device__ __forceinline__
    TensorKey operator*(ValueT scalar) const
    {
        TensorKey result = *this;
        result *= scalar;
        return result;
    }

    __host__ __device__ __forceinline__
    TensorKey operator/(ValueT scalar) const
    {
        TensorKey result = *this;
        result /= scalar;
        return result;
    }

    // =====================================================================
    // Norm and utility
    // =====================================================================

    __host__ __device__ __forceinline__
    ValueT l2Norm() const
    {
        ValueT sum = ValueT(0);
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            sum += values[i] * values[i];
        return sqrt(sum);
    }

    __host__ __device__ __forceinline__
    void set(const ValueT* array)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] = array[i];
    }

    __host__ __device__ __forceinline__
    void set(ValueT val)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] = val;
    }

    __host__ void print() const
    {
        printf("Key: %d, Values: [", (int)key);
        for (int i = 0; i < Dim; ++i)
        {
            printf("%.2f", static_cast<double>(values[i]));
            if (i < Dim - 1)
                printf(", ");
        }
        printf("]\n");
    }
};

// Optional: sanity check in a .cu or .cpp
static_assert(std::is_trivially_copyable<TensorKey<int, float, 1>>::value,
              "TensorKey should be trivially copyable");

// FlexParams
template <
    typename ValueT,  ///< Matrix and vector value type
    typename OffsetT> ///< Signed integer type for sequence offsets
struct FlexParams
{
    // [code generation]
    ${input_declarations_utils_code}
    int num_rows;                ///< Number of rows of matrix <b>A</b>.
    int num_cols;                ///< Number of columns of matrix <b>A</b>.
    int num_nonzeros;            ///< Number of nonzero elements of matrix <b>A</b>.
    OffsetT *d_row_end_offsets;  /// only used for compilation for search kernel
};

struct LaunchKernelConfig
{
    int block_threads;
    int items_per_thread;
    int tile_items;

    template <typename PolicyT>
    __host__ __forceinline__ void Init()
    {
        block_threads = PolicyT::BLOCK_THREADS;
        items_per_thread = PolicyT::ITEMS_PER_THREAD;
        tile_items = block_threads * items_per_thread;
    }
};

// Non-member operator to enable scalar * TensorKey
template <typename OffsetT, typename ValueT, int Dim>
__host__ __device__ __forceinline__ TensorKey<OffsetT, ValueT, Dim> operator*(ValueT scalar, const TensorKey<OffsetT, ValueT, Dim>& tensor)
{
    return tensor * scalar;  // Reuse the existing TensorKey * scalar operator
}

// Non-member operator to enable scalar / TensorKey
template <typename OffsetT, typename ValueT, int Dim>
__host__ __device__ __forceinline__ TensorKey<OffsetT, ValueT, Dim> operator/(ValueT scalar, const TensorKey<OffsetT, ValueT, Dim>& tensor)
{
    TensorKey<OffsetT, ValueT, Dim> result;
    result.key = tensor.key;
    #pragma unroll
    for (int i = 0; i < Dim; ++i)
        result.values[i] = (tensor.values[i] != 0.0) ? (scalar / tensor.values[i]) : 0.0;
    return result;
}
