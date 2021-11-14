using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Diagnostics;
using Cudafy.Types;
using Cudafy.Host;
using Cudafy.Maths.BLAS.Types;
using GASS.CUDA.BLAS;
using GASS.CUDA.BLAS.Types;
using GASS.CUDA.Types;
using GASS.CUDA;

namespace Cudafy.Maths.BLAS
{
    internal interface ICUBLASDriverv2Ex
    {
        #region BLAS Level 1 (Extended)
        CUBLASStatusv2 cublasCaxpy(cublasHandle handle, int n, ref ComplexF alpha, IntPtr x, int incx, IntPtr y, int incy);
        CUBLASStatusv2 cublasZaxpy(cublasHandle handle, int n, ref ComplexD alpha, IntPtr x, int incx, IntPtr y, int incy);
        CUBLASStatusv2 cublasCdotu(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result);
        CUBLASStatusv2 cublasCdotc(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result);
        CUBLASStatusv2 cublasZdotu(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result);
        CUBLASStatusv2 cublasZdotc(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result);
        CUBLASStatusv2 cublasCrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref ComplexF s);
        CUBLASStatusv2 cublasZrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref ComplexD s);
        CUBLASStatusv2 cublasCrotg(cublasHandle handle, ref ComplexF a, ref ComplexF b, ref float c, ref ComplexF s);
        CUBLASStatusv2 cublasZrotg(cublasHandle handle, ref ComplexD a, ref ComplexD b, ref double c, ref ComplexD s);
        #endregion

        #region BLAS Level 2
        #region GBMV
        CUBLASStatusv2 cublasSgbmv(cublasHandle handle, cublasOperation trans, int m, int n, int kl, int ku, ref float alpha, IntPtr A, int lda, IntPtr x, int incx, ref float beta, IntPtr y, int incy);
        CUBLASStatusv2 cublasDgbmv(cublasHandle handle, cublasOperation trans, int m, int n, int kl, int ku, ref double alpha, IntPtr A, int lda, IntPtr x, int incx, ref double beta, IntPtr y, int incy);
        #endregion

        #region GEMV
        CUBLASStatusv2 cublasSgemv(cublasHandle handle, cublasOperation trans, int m, int n, ref float alpha, IntPtr A, int lda, IntPtr x, int incx, ref float beta, IntPtr y, int incy);
        CUBLASStatusv2 cublasDgemv(cublasHandle handle, cublasOperation trans, int m, int n, ref double alpha, IntPtr A, int lda, IntPtr x, int incx, ref double beta, IntPtr y, int incy);
        #endregion

        #region GER
        CUBLASStatusv2 cublasSger(cublasHandle handle, int m, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        CUBLASStatusv2 cublasDger(cublasHandle handle, int m, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        #endregion

        #region SBMV
        CUBLASStatusv2 cublasSsbmv(cublasHandle handle, cublasFillMode uplo, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr x, int incx, ref float beta, IntPtr y, int incy);
        CUBLASStatusv2 cublasDsbmv(cublasHandle handle, cublasFillMode uplo, int n, int k, ref double alpha, IntPtr A, int lda, IntPtr x, int incx, ref double beta, IntPtr y, int incy);
        #endregion

        #region SPMV
        CUBLASStatusv2 cublasSspmv(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr ap, IntPtr x, int incx, ref float beta, IntPtr y, int incy);
        CUBLASStatusv2 cublasDspmv(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr ap, IntPtr x, int incx, ref double beta, IntPtr y, int incy);
        #endregion

        #region SPR
        CUBLASStatusv2 cublasSspr(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr x, int incx, IntPtr ap);
        CUBLASStatusv2 cublasDspr(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr x, int incx, IntPtr ap);
        #endregion

        #region SPR2
        CUBLASStatusv2 cublasSspr2(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr ap);
        CUBLASStatusv2 cublasDspr2(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr ap);
        #endregion

        #region SYMV
        CUBLASStatusv2 cublasSsymv(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr A, int lda, IntPtr x, int incx, ref float beta, IntPtr y, int incy);
        CUBLASStatusv2 cublasDsymv(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr A, int lda, IntPtr x, int incx, ref double beta, IntPtr y, int incy);
        #endregion

        #region SYR
        CUBLASStatusv2 cublasSsyr(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr x, int incx, IntPtr A, int lda);
        CUBLASStatusv2 cublasDsyr(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr x, int incx, IntPtr A, int lda);
        #endregion

        #region SYR2
        CUBLASStatusv2 cublasSsyr2(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        CUBLASStatusv2 cublasDsyr2(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        #endregion

        #region TBMV
        CUBLASStatusv2 cublasStbmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx);
        CUBLASStatusv2 cublasDtbmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx);
        #endregion

        #region TBSV
        CUBLASStatusv2 cublasStbsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx);
        CUBLASStatusv2 cublasDtbsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx);
        #endregion

        #region TPMV
        CUBLASStatusv2 cublasStpmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr AP, IntPtr x, int incx);
        CUBLASStatusv2 cublasDtpmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr AP, IntPtr x, int incx);
        #endregion

        #region TPSV
        CUBLASStatusv2 cublasStpsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr AP, IntPtr x, int incx);
        CUBLASStatusv2 cublasDtpsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr AP, IntPtr x, int incx);
        #endregion

        #region TRMV
        CUBLASStatusv2 cublasStrmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr A, int lda, IntPtr x, int incx);
        CUBLASStatusv2 cublasDtrmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr A, int lda, IntPtr x, int incx);
        #endregion

        #region TRSV
        CUBLASStatusv2 cublasStrsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr A, int lda, IntPtr x, int incx);
        CUBLASStatusv2 cublasDtrsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr A, int lda, IntPtr x, int incx);
        #endregion
        #endregion

        #region BLAS Level 3
        #region GEMM
        CUBLASStatusv2 cublasSgemm(cublasHandle handle, cublasOperation transa, cublasOperation transb, int m, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc);
        CUBLASStatusv2 cublasDgemm(cublasHandle handle, cublasOperation transa, cublasOperation transb, int m, int n, int k, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc);
        #endregion

        #region SYMM
        CUBLASStatusv2 cublasSsymm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, int m, int n, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc);
        CUBLASStatusv2 cublasDsymm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, int m, int n, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc);
        #endregion

        #region SYRK
        CUBLASStatusv2 cublasSsyrk(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, int n, int k, ref float alpha, IntPtr A, int lda, ref float beta, IntPtr C, int ldc);
        CUBLASStatusv2 cublasDsyrk(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, int n, int k, ref double alpha, IntPtr A, int lda, ref double beta, IntPtr C, int ldc);
        #endregion

        #region SYR2K
        CUBLASStatusv2 cublasSsyr2k(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc);
        CUBLASStatusv2 cublasDsyr2k(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, int n, int k, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc);
        #endregion

        #region TRMM
        CUBLASStatusv2 cublasStrmm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc);
        CUBLASStatusv2 cublasDtrmm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc);
        #endregion

        #region TRSM
        CUBLASStatusv2 cublasStrsm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb);
        CUBLASStatusv2 cublasDtrsm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb);
        #endregion
        #endregion
    }
}
