#ifndef CALS_CALS_BLAS_H
#define CALS_CALS_BLAS_H

#include "omp.h"

#if CALS_MKL

#include "mkl.h"
#define CALS_BACKEND "MKL"

#elif CALS_FLAME

#include "blis.h"
#include "mkl_lapack.h"
#define CALS_BACKEND "BLIS"

#elif CALS_OPENBLAS

#include "cblas.h"

#define CALS_BACKEND "OPENBLAS"

extern "C" {
void dposv_(const char *,
            const long int *,
            const long int *,
            double *,
            const long int *,
            double *,
            const long int *,
            long int *);
void dpotrf_(const char *, const long int *, double *, const long int *, long int *);
};

inline void dposv(const char *uplo,
                  int const *N,
                  int const *Rhs,
                  double *A,
                  int const *Lda,
                  double *B,
                  int const *Ldb,
                  int const *Info) {
  long int const n = *N;
  long int const rhs = *Rhs;
  long int const lda = *Lda;
  long int const ldb = *Ldb;
  long int info = *Info;
  dposv_(uplo, &n, &rhs, A, &lda, B, &ldb, &info);
}

inline void dpotrf(const char *uplo, int const *N, double *A, int const *Lda, int const *Info) {
  long int const n = *N;
  long int const lda = *Lda;
  long int info = *Info;
  dpotrf_(uplo, &n, A, &lda, &info);
}

#elif CALS_MATLAB

#include "blas.h"
#include "lapack.h"
#include "mex.h"
#include <cassert>

#define CALS_BACKEND "MATLAB"

enum CBLAS_ORDER { CblasColMajor = 0, CblasRowMajor = 1 };

enum CBLAS_TRANSPOSE { CblasTrans = 0, CblasNoTrans = 1 };

enum CBLAS_DIAG { CblasNonUnit = 0, CblasUnit = 1 };

enum CBLAS_UPLO { CblasLower = 0, CblasUpper = 1 };

enum CBLAS_SIDE { CblasRight = 0, CblasLeft = 1 };

inline double cblas_dnrm2(ptrdiff_t n_elements, double *data, ptrdiff_t stride) {
  return dnrm2(&n_elements, data, &stride);
}

inline double cblas_dasum(ptrdiff_t n_elements, double *data, ptrdiff_t stride) {
  return dasum(&n_elements, data, &stride);
}

inline ptrdiff_t cblas_idamax(ptrdiff_t n_elements, double *data, ptrdiff_t stride) {
  return idamax(&n_elements, data, &stride) - 1;
}

inline void cblas_dcopy(
    ptrdiff_t n_elements, double const *src_data, ptrdiff_t src_stride, double *dst_data, ptrdiff_t dst_stride) {
  dcopy(&n_elements, src_data, &src_stride, dst_data, &dst_stride);
}

inline void cblas_dscal(ptrdiff_t n_elements, double scalar, double *data, ptrdiff_t stride) {
  dscal(&n_elements, &scalar, data, &stride);
}

inline void cblas_daxpy(ptrdiff_t n_elements, double scalar, double *x, ptrdiff_t ldx, double *y, ptrdiff_t ldy) {
  daxpy(&n_elements, &scalar, x, &ldx, y, &ldy);
}

inline void cblas_dgemm([[maybe_unused]] CBLAS_ORDER Order,
                        CBLAS_TRANSPOSE TransA,
                        CBLAS_TRANSPOSE TransB,
                        ptrdiff_t M,
                        ptrdiff_t N,
                        ptrdiff_t K,
                        double alpha,
                        const double *A,
                        ptrdiff_t lda,
                        const double *B,
                        ptrdiff_t ldb,
                        double beta,
                        double *C,
                        ptrdiff_t ldc) {
  assert(Order == CblasColMajor);
  const char *transa = TransA == CblasTrans ? "T" : "N";
  const char *transb = TransB == CblasTrans ? "T" : "N";
  dgemm(transa, transb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

inline void cblas_dgemv([[maybe_unused]] CBLAS_ORDER order,
                        CBLAS_TRANSPOSE TransA,
                        ptrdiff_t M,
                        ptrdiff_t N,
                        double alpha,
                        const double *A,
                        ptrdiff_t lda,
                        const double *X,
                        ptrdiff_t incX,
                        double beta,
                        double *Y,
                        ptrdiff_t incY) {
  assert(order == CblasColMajor);
  const char *transa = TransA == CblasTrans ? "T" : "N";
  dgemv(transa, &M, &N, &alpha, A, &lda, X, &incX, &beta, Y, &incY);
}

inline void cblas_dtrsm([[maybe_unused]] CBLAS_ORDER Order,
                        CBLAS_SIDE Side,
                        CBLAS_UPLO Uplo,
                        CBLAS_TRANSPOSE TransA,
                        CBLAS_DIAG Diag,
                        ptrdiff_t M,
                        ptrdiff_t N,
                        double alpha,
                        const double *A,
                        ptrdiff_t lda,
                        double *B,
                        ptrdiff_t ldb) {
  assert(Order == CblasColMajor);
  const char *transa = TransA == CblasTrans ? "T" : "N";
  const char *side = Side == CblasLeft ? "L" : "R";
  const char *uplo = Uplo == CblasUpper ? "U" : "L";
  const char *diag = Diag == CblasNonUnit ? "N" : "U";
  dtrsm(side, uplo, transa, diag, &M, &N, &alpha, A, &lda, B, &ldb);
}

inline void dposv(const char *uplo,
                  int const *N,
                  int const *Rhs,
                  double *A,
                  int const *Lda,
                  double *B,
                  int const *Ldb,
                  int const *Info) {
  ptrdiff_t const n = *N;
  ptrdiff_t const rhs = *Rhs;
  ptrdiff_t const lda = *Lda;
  ptrdiff_t const ldb = *Ldb;
  ptrdiff_t info = *Info;
  dposv(uplo, &n, &rhs, A, &lda, B, &ldb, &info);
}

inline void dpotrf(const char *uplo, int const *N, double *A, int const *Lda, int const *Info) {
  ptrdiff_t const n = *N;
  ptrdiff_t const lda = *Lda;
  ptrdiff_t info = *Info;
  dpotrf(uplo, &n, A, &lda, &info);
}

#endif

void set_threads(int threads);

int get_threads();

#endif // CALS_CALS_BLAS_H
