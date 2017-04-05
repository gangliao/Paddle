/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef MATHFUNCTIONS_H_
#define MATHFUNCTIONS_H_

#ifdef PADDLE_USE_MKL
#include <mkl.h>
#include <mkl_lapacke.h>
#else
extern "C" {
#include <cblas.h>
}
#ifdef PADDLE_USE_ATLAS
extern "C" {
#include <clapack.h>
}
#else
extern "C" {
#include "blaswrap.h"
#include "f2c.h"
#include "clapack.h"
}

// FORTRAN <=> C convertor macros
#define FORTRAN_DOUBLE_ORDER(m, n, a) {\
    double* b = new double[m * n];\
    for (int i = 0; i < m; i++)\
        for (int j = 0; j < n; j++)\
            b[(i * n + j) % m * m + (i * n + j) / m] = a[i * n + j];\
    memmove(a, b, m * n * sizeof(double));\
    delete[] b;
}

#define FORTRAN_SINGLE_ORDER(m, n, a) {\
    float* b = new float[m * n];\
    for (int i = 0; i < m; i++)\
        for (int j = 0; j < n; j++)\
            b[(i * n + j) % m * m + (i * n + j) / m] = a[i * n + j];\
    memmove(a, b, m * n * sizeof(float));\
    delete[] b;
}

#define IPIV_FORTRAN(n, ipiv)\
    for (int i = 0; i < n; i++)\
        ipiv[i] = ipiv[i] + 1;

#define IPIV_C(n, ipiv)\
    for (int i = 0; i < n; i++)\
        ipiv[i] = ipiv[i] - 1;\

#endif
#endif

namespace paddle {

template <class T>
void gemm(const CBLAS_TRANSPOSE transA,
          const CBLAS_TRANSPOSE transB,
          const int M,
          const int N,
          const int K,
          const T alpha,
          const T* A,
          const int lda,
          const T* B,
          const int ldb,
          const T beta,
          T* C,
          const int ldc);

template <class T>
int getrf(const CBLAS_ORDER Order,
          int M,
          int N,
          T* A,
          int lda,
          int* ipiv);

template <class T>
int getri(
    const CBLAS_ORDER Order, int N, T* A, int lda, int* ipiv);

template <class T>
void axpy(const int n, const T alpha, const T* x, T* y);

template <class T>
T dotProduct(const int n, const T* x, const T* y);

template <class T>
void vExp(const int n, const T* a, T* r);

template <class T>
void vPow(const int n, const T* a, const T b, T* r);

template <class T>
void vLog(const int n, const T* a, T* r);

template <class T>
void vAdd(const int n, const T* a, const T* b, T* r);

template <class T>
void vInvSqrt(const int n, const T* a, T* r);

template <class T>
void vLog1p(const int n, const T* a, T* r);

template <class T>
void vTanh(const int n, const T* a, T* r);

}  // namespace paddle

#endif  // MATHFUNCTIONS_H_
