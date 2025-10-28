#pragma once
// =============================================================================
// Tensor Data Structure - Cross-Platform Implementation
// =============================================================================
// High-performance tensor class supporting:
// - Element-wise operations with broadcasting
// - Scalar operations with type safety  
// - CUDA and host compatibility
// - Template-based dimension support
// =============================================================================

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cmath>
#include <type_traits>
#include <algorithm>
#include <cstdio>
#else
#include <cmath>
#include <type_traits>
#include <algorithm>
#include <cstdio>
#endif

// =============================================================================
// Cross-Platform Compatibility Macros
// =============================================================================
#ifdef __CUDACC__
#define TENSOR_INLINE __host__ __device__ __forceinline__
#define TENSOR_HOST_ONLY __host__
#define TENSOR_PRAGMA_UNROLL _Pragma("unroll")
#else
#define TENSOR_INLINE inline
#define TENSOR_HOST_ONLY
#define TENSOR_PRAGMA_UNROLL
#endif

// =============================================================================
// Helper Functions
// =============================================================================
template<typename T>
TENSOR_INLINE T tensor_pow(T base, T exp) {
#ifdef __CUDA_ARCH__
    return pow(base, exp);
#else
    return std::pow(base, exp);
#endif
}

template<typename T>
TENSOR_INLINE T tensor_exp(T x) {
#ifdef __CUDA_ARCH__
    return exp(x);
#else
    return std::exp(x);
#endif
}

template<typename T>
TENSOR_INLINE T tensor_abs(T x) {
#ifdef __CUDA_ARCH__
    return fabs(x);
#else
    using std::abs; // enable ADL and select correct overload
    return abs(x);
#endif
}

template<typename T>
TENSOR_INLINE T tensor_sign(T x) {
    // Treat NaN as 0 by default since (x > 0) and (x < 0) are both false for NaN
    return (x > static_cast<T>(0)) ? static_cast<T>(1)
         : (x < static_cast<T>(0)) ? static_cast<T>(-1)
         : static_cast<T>(0);
}

// =============================================================================
// Main Tensor Class
// =============================================================================
template <typename ValueT, int Dim>
struct Tensor
{
    // Type aliases
    typedef ValueT Value;
    
    // Data storage
    ValueT values[Dim];

    // =============================================================================
    // Constructors and Assignment
    // =============================================================================
    
    TENSOR_INLINE Tensor()
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = 0.0f;
    }

    TENSOR_INLINE Tensor(const ValueT val)
    {
        static_assert(Dim == 1, "Tensor must be 1D");
        values[0] = val;
    }

    TENSOR_INLINE Tensor(const ValueT *v)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = v[i];
    }

    template<typename IteratorT>
    TENSOR_INLINE Tensor(IteratorT iter)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = static_cast<ValueT>(iter[i]);
    }

    TENSOR_INLINE Tensor(const Tensor& other)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = other.values[i];
    }

    TENSOR_INLINE Tensor& operator=(const Tensor& other)
    {
        if (this != &other) {
            TENSOR_PRAGMA_UNROLL
            for (int i = 0; i < Dim; ++i)
                values[i] = other.values[i];
        }
        return *this;
    }

    TENSOR_INLINE Tensor& operator=(const ValueT val)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = val;
        return *this;
    }

    // =============================================================================
    // Element Access
    // =============================================================================
    
    TENSOR_INLINE ValueT &operator[](int idx) { return values[idx]; }
    TENSOR_INLINE const ValueT &operator[](int idx) const { return values[idx]; }

    // =============================================================================
    // Unary Operators
    // =============================================================================
    
    TENSOR_INLINE Tensor operator-() const
    {
        Tensor result;
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            result.values[i] = -values[i];
        return result;
    }

    // =============================================================================
    // Compound Assignment Operators (Tensor-Tensor)
    // =============================================================================
    
    TENSOR_INLINE Tensor& operator+=(const Tensor &other)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] += other.values[i];
        return *this;
    }

    TENSOR_INLINE Tensor& operator-=(const Tensor &other)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] -= other.values[i];
        return *this;
    }

    TENSOR_INLINE Tensor& operator*=(const Tensor &other)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] *= other.values[i];
        return *this;
    }

    TENSOR_INLINE Tensor& operator/=(const Tensor &other)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = (other.values[i] != 0.0) ? (values[i] / other.values[i]) : 0.0;
        return *this;
    }

    TENSOR_INLINE Tensor& operator^=(const Tensor &other)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = tensor_pow(values[i], other.values[i]);
        return *this;
    }

    // =============================================================================
    // Compound Assignment Operators (Tensor-Scalar)
    // =============================================================================
    
    TENSOR_INLINE Tensor& operator+=(ValueT scalar)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] += scalar;
        return *this;
    }

    TENSOR_INLINE Tensor& operator-=(ValueT scalar)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] -= scalar;
        return *this;
    }

    TENSOR_INLINE Tensor& operator*=(ValueT scalar)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] *= scalar;
        return *this;
    }

    TENSOR_INLINE Tensor& operator/=(ValueT scalar)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] /= scalar;
        return *this;
    }

    TENSOR_INLINE Tensor& operator^=(ValueT scalar)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = tensor_pow(values[i], scalar);
        return *this;
    }

    // =============================================================================
    // Binary Operators - Auto-generated from compound assignments
    // =============================================================================
    
    // Tensor-Tensor operations
    TENSOR_INLINE Tensor operator+(const Tensor &other) const { Tensor result = *this; result += other; return result; }
    TENSOR_INLINE Tensor operator-(const Tensor &other) const { Tensor result = *this; result -= other; return result; }
    TENSOR_INLINE Tensor operator*(const Tensor &other) const { Tensor result = *this; result *= other; return result; }
    TENSOR_INLINE Tensor operator/(const Tensor &other) const { Tensor result = *this; result /= other; return result; }
    TENSOR_INLINE Tensor operator^(const Tensor &other) const { Tensor result = *this; result ^= other; return result; }

    // Tensor-Scalar operations  
    TENSOR_INLINE Tensor operator+(ValueT scalar) const { Tensor result = *this; result += scalar; return result; }
    TENSOR_INLINE Tensor operator-(ValueT scalar) const { Tensor result = *this; result -= scalar; return result; }
    TENSOR_INLINE Tensor operator*(ValueT scalar) const { Tensor result = *this; result *= scalar; return result; }
    TENSOR_INLINE Tensor operator/(ValueT scalar) const { Tensor result = *this; result /= scalar; return result; }
    TENSOR_INLINE Tensor operator^(ValueT scalar) const { Tensor result = *this; result ^= scalar; return result; }

    // =============================================================================
    // Utility Methods
    // =============================================================================
    
    TENSOR_INLINE Tensor pow(ValueT exponent) const
    {
        Tensor result;
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            result.values[i] = tensor_pow(values[i], exponent);
        return result;
    }

    TENSOR_INLINE Tensor pow(const Tensor &exponents) const
    {
        Tensor result;
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            result.values[i] = tensor_pow(values[i], exponents.values[i]);
        return result;
    }

    TENSOR_INLINE Tensor exp() const
    {
        Tensor result;
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            result.values[i] = tensor_exp(values[i]);
        return result;
    }

    TENSOR_INLINE Tensor abs() const
    {
        Tensor result;
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            result.values[i] = tensor_abs(values[i]);
        return result;
    }

    TENSOR_INLINE Tensor sign() const
    {
        Tensor result;
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            result.values[i] = tensor_sign(values[i]);
        return result;
    }

    // =============================================================================
    // Comparison with scalar (produces 0/1 mask in ValueT)
    // =============================================================================
    
    TENSOR_INLINE Tensor lt(ValueT scalar) const
    {
        Tensor result;
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            result.values[i] = (values[i] < scalar) ? static_cast<ValueT>(1) : static_cast<ValueT>(0);
        return result;
    }

    TENSOR_INLINE Tensor gt(ValueT scalar) const
    {
        Tensor result;
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            result.values[i] = (values[i] > scalar) ? static_cast<ValueT>(1) : static_cast<ValueT>(0);
        return result;
    }

    TENSOR_INLINE Tensor operator<(ValueT scalar) const { return this->lt(scalar); }
    TENSOR_INLINE Tensor operator>(ValueT scalar) const { return this->gt(scalar); }

    TENSOR_INLINE ValueT l2Norm() const
    {
        ValueT sum = 0.0;
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            sum += values[i] * values[i];
        return sqrt(sum);
    }

    TENSOR_INLINE void set(const ValueT* array)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = array[i];
    }

    TENSOR_INLINE void set(const ValueT val)
    {
        TENSOR_PRAGMA_UNROLL
        for (int i = 0; i < Dim; ++i)
            values[i] = val;
    }

    TENSOR_HOST_ONLY void print() const
    {
        printf("Values: [");
        for (int i = 0; i < Dim; ++i)
        {
            printf("%.2f", values[i]);
            if (i < Dim - 1)
                printf(", ");
        }
        printf("]\n");
    }
};

// =============================================================================
// Scalar-Tensor Binary Operators
// =============================================================================

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> operator*(ValueT scalar, const Tensor<ValueT, Dim>& tensor) { return tensor * scalar; }

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> operator+(ValueT scalar, const Tensor<ValueT, Dim>& tensor) { return tensor + scalar; }

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> operator-(ValueT scalar, const Tensor<ValueT, Dim>& tensor)
{
    Tensor<ValueT, Dim> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < Dim; ++i) result.values[i] = scalar - tensor.values[i];
    return result;
}

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> operator/(ValueT scalar, const Tensor<ValueT, Dim>& tensor)
{
    Tensor<ValueT, Dim> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < Dim; ++i) result.values[i] = (tensor.values[i] != 0.0) ? (scalar / tensor.values[i]) : 0.0;
    return result;
}

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> operator^(ValueT scalar, const Tensor<ValueT, Dim>& tensor)
{
    Tensor<ValueT, Dim> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < Dim; ++i) result.values[i] = tensor_pow(scalar, tensor.values[i]);
    return result;
}

// =============================================================================
// Mixed-Type Operators (arithmetic scalar with any tensor ValueT)
// =============================================================================

// For cases like: int - Tensor<double, Dim>
// We enable these only when ScalarT != ValueT to avoid ambiguity with exact-match overloads
#define DEFINE_MIXED_SCALAR_TENSOR_LEFT(op) \
template <typename ScalarT, typename ValueT, int Dim, \
          typename std::enable_if<std::is_arithmetic<ScalarT>::value && !std::is_same<ScalarT, ValueT>::value, int>::type = 0> \
TENSOR_INLINE Tensor<ValueT, Dim> operator op(ScalarT scalar, const Tensor<ValueT, Dim>& tensor) { \
    return static_cast<ValueT>(scalar) op tensor; \
}

#define DEFINE_MIXED_SCALAR_TENSOR_RIGHT(op) \
template <typename ScalarT, typename ValueT, int Dim, \
          typename std::enable_if<std::is_arithmetic<ScalarT>::value && !std::is_same<ScalarT, ValueT>::value, int>::type = 0> \
TENSOR_INLINE Tensor<ValueT, Dim> operator op(const Tensor<ValueT, Dim>& tensor, ScalarT scalar) { \
    return tensor op static_cast<ValueT>(scalar); \
}

DEFINE_MIXED_SCALAR_TENSOR_LEFT(+)
DEFINE_MIXED_SCALAR_TENSOR_LEFT(-)
DEFINE_MIXED_SCALAR_TENSOR_LEFT(*)
DEFINE_MIXED_SCALAR_TENSOR_LEFT(/)
DEFINE_MIXED_SCALAR_TENSOR_LEFT(^)

DEFINE_MIXED_SCALAR_TENSOR_RIGHT(+)
DEFINE_MIXED_SCALAR_TENSOR_RIGHT(-)
DEFINE_MIXED_SCALAR_TENSOR_RIGHT(*)
DEFINE_MIXED_SCALAR_TENSOR_RIGHT(/)
DEFINE_MIXED_SCALAR_TENSOR_RIGHT(^)

#undef DEFINE_MIXED_SCALAR_TENSOR_LEFT
#undef DEFINE_MIXED_SCALAR_TENSOR_RIGHT

// =============================================================================
// Utility Functions
// =============================================================================

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> pow(const Tensor<ValueT, Dim>& base, ValueT exponent) { return base.pow(exponent); }

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> pow(const Tensor<ValueT, Dim>& base, const Tensor<ValueT, Dim>& exponents) { return base.pow(exponents); }

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> pow(ValueT base, const Tensor<ValueT, Dim>& exponents)
{
    Tensor<ValueT, Dim> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < Dim; ++i) result.values[i] = tensor_pow(base, exponents.values[i]);
    return result;
}

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> exp(const Tensor<ValueT, Dim>& t) { return t.exp(); }

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> abs(const Tensor<ValueT, Dim>& t) { return t.abs(); }

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> sign(const Tensor<ValueT, Dim>& t) { return t.sign(); }

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> lt(const Tensor<ValueT, Dim>& t, ValueT scalar) { return t.lt(scalar); }

template <typename ValueT, int Dim>
TENSOR_INLINE Tensor<ValueT, Dim> gt(const Tensor<ValueT, Dim>& t, ValueT scalar) { return t.gt(scalar); }

// =============================================================================
// Broadcasting Operations (Different Dimensions)
// =============================================================================

// =============================================================================
// _where: element-wise select like torch.where
// Supports broadcasting among condition, x, y tensors; scalar overloads included
// =============================================================================

// Tensor-Tensor-Tensor with broadcasting (cond, x, y)
template <typename ValueT, int CN, int XN, int YN>
TENSOR_INLINE Tensor<ValueT, (CN > XN ? (CN > YN ? CN : YN) : (XN > YN ? XN : YN))>
_where(const Tensor<ValueT, CN>& cond,
      const Tensor<ValueT, XN>& x,
      const Tensor<ValueT, YN>& y)
{
    constexpr int OUT_DIM = (CN > XN ? (CN > YN ? CN : YN) : (XN > YN ? XN : YN));
    Tensor<ValueT, OUT_DIM> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < OUT_DIM; ++i) {
        const ValueT c = cond[i % CN];
        result[i] = (c != static_cast<ValueT>(0)) ? x[i % XN] : y[i % YN];
    }
    return result;
}

// Tensor-Scalar-Scalar: OUT_DIM = CN
template <typename ValueT, int CN>
TENSOR_INLINE Tensor<ValueT, CN>
_where(const Tensor<ValueT, CN>& cond, ValueT x, ValueT y)
{
    Tensor<ValueT, CN> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < CN; ++i) {
        const ValueT c = cond[i];
        result[i] = (c != static_cast<ValueT>(0)) ? x : y;
    }
    return result;
}

// Tensor-Tensor-Scalar: OUT_DIM = max(CN, XN)
template <typename ValueT, int CN, int XN>
TENSOR_INLINE Tensor<ValueT, (CN > XN ? CN : XN)>
_where(const Tensor<ValueT, CN>& cond, const Tensor<ValueT, XN>& x, ValueT y)
{
    constexpr int OUT_DIM = (CN > XN ? CN : XN);
    Tensor<ValueT, OUT_DIM> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < OUT_DIM; ++i) {
        const ValueT c = cond[i % CN];
        result[i] = (c != static_cast<ValueT>(0)) ? x[i % XN] : y;
    }
    return result;
}

// Tensor-Scalar-Tensor: OUT_DIM = max(CN, YN)
template <typename ValueT, int CN, int YN>
TENSOR_INLINE Tensor<ValueT, (CN > YN ? CN : YN)>
_where(const Tensor<ValueT, CN>& cond, ValueT x, const Tensor<ValueT, YN>& y)
{
    constexpr int OUT_DIM = (CN > YN ? CN : YN);
    Tensor<ValueT, OUT_DIM> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < OUT_DIM; ++i) {
        const ValueT c = cond[i % CN];
        result[i] = (c != static_cast<ValueT>(0)) ? x : y[i % YN];
    }
    return result;
}

// --------- Generic scalar overloads (accept double literals like 0.0) ---------

// Tensor-Tensor-Scalar(any arithmetic): OUT_DIM = max(CN, XN)
template <typename ValueT, int CN, int XN, typename ScalarY,
          typename std::enable_if<std::is_arithmetic<ScalarY>::value, int>::type = 0>
TENSOR_INLINE Tensor<ValueT, (CN > XN ? CN : XN)>
_where(const Tensor<ValueT, CN>& cond, const Tensor<ValueT, XN>& x, ScalarY y)
{
    constexpr int OUT_DIM = (CN > XN ? CN : XN);
    Tensor<ValueT, OUT_DIM> result;
    const ValueT y_cast = static_cast<ValueT>(y);
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < OUT_DIM; ++i) {
        const ValueT c = cond[i % CN];
        result[i] = (c != static_cast<ValueT>(0)) ? x[i % XN] : y_cast;
    }
    return result;
}

// Tensor-Scalar(any arithmetic)-Tensor: OUT_DIM = max(CN, YN)
template <typename ValueT, int CN, int YN, typename ScalarX,
          typename std::enable_if<std::is_arithmetic<ScalarX>::value, int>::type = 0>
TENSOR_INLINE Tensor<ValueT, (CN > YN ? CN : YN)>
_where(const Tensor<ValueT, CN>& cond, ScalarX x, const Tensor<ValueT, YN>& y)
{
    constexpr int OUT_DIM = (CN > YN ? CN : YN);
    Tensor<ValueT, OUT_DIM> result;
    const ValueT x_cast = static_cast<ValueT>(x);
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < OUT_DIM; ++i) {
        const ValueT c = cond[i % CN];
        result[i] = (c != static_cast<ValueT>(0)) ? x_cast : y[i % YN];
    }
    return result;
}

// Tensor-Scalar(any)-Scalar(any): OUT_DIM = CN
template <typename ValueT, int CN, typename ScalarX, typename ScalarY,
          typename std::enable_if<std::is_arithmetic<ScalarX>::value && std::is_arithmetic<ScalarY>::value, int>::type = 0>
TENSOR_INLINE Tensor<ValueT, CN>
_where(const Tensor<ValueT, CN>& cond, ScalarX x, ScalarY y)
{
    Tensor<ValueT, CN> result;
    const ValueT x_cast = static_cast<ValueT>(x);
    const ValueT y_cast = static_cast<ValueT>(y);
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < CN; ++i) {
        const ValueT c = cond[i];
        result[i] = (c != static_cast<ValueT>(0)) ? x_cast : y_cast;
    }
    return result;
}

#define DEFINE_BROADCASTING_OP(op) \
template <typename ValueT, int N, int M> \
TENSOR_INLINE Tensor<ValueT, (N > M ? N : M)> operator op(const Tensor<ValueT, N>& a, const Tensor<ValueT, M>& b) \
{ \
    constexpr int OUT_DIM = (N > M ? N : M); \
    Tensor<ValueT, OUT_DIM> result; \
    TENSOR_PRAGMA_UNROLL \
    for (int i = 0; i < OUT_DIM; ++i) result[i] = a[i % N] op b[i % M]; \
    return result; \
}

DEFINE_BROADCASTING_OP(+)
DEFINE_BROADCASTING_OP(-)
DEFINE_BROADCASTING_OP(*)
DEFINE_BROADCASTING_OP(/)

// Power operator requires special handling
template <typename ValueT, int N, int M>
TENSOR_INLINE Tensor<ValueT, (N > M ? N : M)> operator^(const Tensor<ValueT, N>& a, const Tensor<ValueT, M>& b)
{
    constexpr int OUT_DIM = (N > M ? N : M);
    Tensor<ValueT, OUT_DIM> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < OUT_DIM; ++i) result[i] = tensor_pow(a[i % N], b[i % M]);
    return result;
}

// Comparison operators with broadcasting
template <typename ValueT, int N, int M>
TENSOR_INLINE Tensor<ValueT, (N > M ? N : M)> operator<(const Tensor<ValueT, N>& a, const Tensor<ValueT, M>& b)
{
    constexpr int OUT_DIM = (N > M ? N : M);
    Tensor<ValueT, OUT_DIM> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < OUT_DIM; ++i)
        result[i] = (a[i % N] < b[i % M]) ? static_cast<ValueT>(1) : static_cast<ValueT>(0);
    return result;
}

template <typename ValueT, int N, int M>
TENSOR_INLINE Tensor<ValueT, (N > M ? N : M)> operator>(const Tensor<ValueT, N>& a, const Tensor<ValueT, M>& b)
{
    constexpr int OUT_DIM = (N > M ? N : M);
    Tensor<ValueT, OUT_DIM> result;
    TENSOR_PRAGMA_UNROLL
    for (int i = 0; i < OUT_DIM; ++i)
        result[i] = (a[i % N] > b[i % M]) ? static_cast<ValueT>(1) : static_cast<ValueT>(0);
    return result;
}

#undef DEFINE_BROADCASTING_OP
