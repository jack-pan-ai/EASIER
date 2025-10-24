// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cstdio>

#include "data_struct_shared.cuh"

#define INIT_KERNEL_THREADS 128 // INFO: this is from cub config

template <typename OffsetT, typename ValueT, int Dim>
struct TensorKey
{
    typedef ValueT Value;
    OffsetT key;
    ValueT values[Dim];

    // Constructor
    __host__ __device__ __forceinline__ TensorKey()
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] = 0.0f;
    }

    // Constructor with values
    __host__ __device__ __forceinline__ TensorKey(const ValueT *v)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] = v[i];
    }

    // Constructor with key and values
    __host__ __device__ __forceinline__ TensorKey(const OffsetT &k, const ValueT *v) : key(k)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] = v[i];
    }

    // Binary operators - optimized to avoid temporary variables
    __host__ __device__ __forceinline__ TensorKey operator+(const TensorKey &other) const
    {
        TensorKey result = *this;
        result += other;
        return result;
    }

    __host__ __device__ __forceinline__ TensorKey operator-(const TensorKey &other) const
    {
        TensorKey result = *this;
        result -= other;
        return result;
    }

    __host__ __device__ __forceinline__ TensorKey operator*(const TensorKey &other) const
    {
        TensorKey result = *this;
        result *= other;
        return result;
    }

    __host__ __device__ __forceinline__ TensorKey operator/(const TensorKey &other) const
    {
        TensorKey result = *this;
        result /= other;
        return result;
    }

    __host__ __device__ __forceinline__ TensorKey operator+(ValueT scalar) const
    {
        TensorKey result = *this;
        result += scalar;
        return result;
    }

    __host__ __device__ __forceinline__ TensorKey operator-(ValueT scalar) const
    {
        TensorKey result = *this;
        result -= scalar;
        return result;
    }

    __host__ __device__ __forceinline__ TensorKey operator*(ValueT scalar) const
    {
        TensorKey result = *this;
        result *= scalar;
        return result;
    }

    __host__ __device__ __forceinline__ TensorKey operator/(ValueT scalar) const
    {
        TensorKey result = *this;
        result /= scalar;
        return result;
    }

    // Compound assignment operators (more efficient, no temporaries)
    __host__ __device__ __forceinline__ TensorKey& operator+=(const TensorKey &other)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] += other.values[i];
        return *this;
    }

    __host__ __device__ __forceinline__ TensorKey& operator-=(const TensorKey &other)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] -= other.values[i];
        return *this;
    }

    __host__ __device__ __forceinline__ TensorKey& operator*=(const TensorKey &other)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] *= other.values[i];
        return *this;
    }

    __host__ __device__ __forceinline__ TensorKey& operator/=(const TensorKey &other)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] = (other.values[i] != 0.0) ? (values[i] / other.values[i]) : 0.0;
        return *this;
    }

    __host__ __device__ __forceinline__ TensorKey& operator+=(ValueT scalar)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] += scalar;
        return *this;
    }

    
    // Constructor with iterator
    template<typename IteratorT>
    __host__ __device__ __forceinline__ TensorKey(IteratorT iter)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] = static_cast<ValueT>(iter[i]);
    }

    __host__ __device__ __forceinline__ TensorKey& operator-=(ValueT scalar)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] -= scalar;
        return *this;
    }

    __host__ __device__ __forceinline__ TensorKey& operator*=(ValueT scalar)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] *= scalar;
        return *this;
    }

    __host__ __device__ __forceinline__ TensorKey& operator/=(ValueT scalar)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] /= scalar;
        return *this;
    }

    __host__ __device__ __forceinline__ TensorKey& operator=(const TensorKey& other)
    {
        key = other.key;
        if (this != &other) {  // Self-assignment check
            #pragma unroll
            for (int i = 0; i < Dim; ++i)
                values[i] = other.values[i];
        }
        return *this;
    }

    __host__ __device__ __forceinline__ TensorKey(const TensorKey& other)
    {
        key = other.key;
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] = other.values[i];
    }

    // Indexing operator for value access
    __host__ __device__ __forceinline__ ValueT &operator[](int idx)
    {
        return values[idx];
    }

    __host__ __device__ __forceinline__ const ValueT &operator[](int idx) const
    {
        return values[idx];
    }
    
    // L2 norm (Euclidean norm)
    __host__ __device__ __forceinline__ ValueT l2Norm() const
    {
        ValueT sum = 0.0;
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            sum += values[i] * values[i];
        return sqrt(sum);
    }

    // Function overloading: Set values from array
    __host__ __device__ __forceinline__ void set(const ValueT* array)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] = array[i];
    }

    __host__ __device__ __forceinline__ void set(const ValueT val)
    {
        #pragma unroll
        for (int i = 0; i < Dim; ++i)
            values[i] = val;
    }

    // Host-only print function
    __host__ void print() const
    {
        printf("Key: %d, Values: [", (int)key);
        for (int i = 0; i < Dim; ++i)
        {
            printf("%.2f", values[i]);
            if (i < Dim - 1)
                printf(", ");
        }
        printf("]\n");
    }
};


// FlexParams
template <
    typename ValueT,  ///< Matrix and vector value type
    typename OffsetT> ///< Signed integer type for sequence offsets
struct FlexParams
{
    // [code generation]
      ValueT *v_ptr; 
  ValueT *clone_1_ptr; 
  ValueT *x_ptr; 

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
