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
    internal class CUBLASDriver64Ex : ICUBLASDriverv2Ex
    {
        #region BLAS Level 1 Native (Extended)
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCaxpy_v2(cublasHandle handle, int n, ref ComplexF alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZaxpy_v2(cublasHandle handle, int n, ref ComplexD alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCdotu_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCdotc_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdotu_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdotc_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref ComplexF s);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref ComplexD s);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCrotg_v2(cublasHandle handle, ref ComplexF a, ref ComplexF b, ref float c, ref ComplexF s);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZrotg_v2(cublasHandle handle, ref ComplexD a, ref ComplexD b, ref double c, ref ComplexD s);
        #endregion

        #region BLAS Level 2 Native
        #region GBMV
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSgbmv_v2(cublasHandle handle, cublasOperation trans, int m, int n, int kl, int ku, ref float alpha, IntPtr A, int lda, IntPtr x, int incx, ref float beta, IntPtr y, int incy);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDgbmv_v2(cublasHandle handle, cublasOperation trans, int m, int n, int kl, int ku, ref double alpha, IntPtr A, int lda, IntPtr x, int incx, ref double beta, IntPtr y, int incy);
        #endregion

        #region GEMV
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSgemv_v2(cublasHandle handle, cublasOperation trans, int m, int n, ref float alpha, IntPtr A, int lda, IntPtr x, int incx, ref float beta, IntPtr y, int incy);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDgemv_v2(cublasHandle handle, cublasOperation trans, int m, int n, ref double alpha, IntPtr A, int lda, IntPtr x, int incx, ref double beta, IntPtr y, int incy);
        #endregion

        #region GER
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSger_v2(cublasHandle handle, int m, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDger_v2(cublasHandle handle, int m, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        #endregion

        #region SBMV
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSsbmv_v2(cublasHandle handle, cublasFillMode uplo, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr x, int incx, ref float beta, IntPtr y, int incy);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDsbmv_v2(cublasHandle handle, cublasFillMode uplo, int n, int k, ref double alpha, IntPtr A, int lda, IntPtr x, int incx, ref double beta, IntPtr y, int incy);
        #endregion

        #region SPMV
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSspmv_v2(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr ap, IntPtr x, int incx, ref float beta, IntPtr y, int incy);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDspmv_v2(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr ap, IntPtr x, int incx, ref double beta, IntPtr y, int incy);
        #endregion

        #region SPR
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSspr_v2(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr x, int incx, IntPtr ap);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDspr_v2(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr x, int incx, IntPtr ap);
        #endregion

        #region SPR2
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSspr2_v2(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr ap);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDspr2_v2(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr ap);
        #endregion

        #region SYMV
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSsymv_v2(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr A, int lda, IntPtr x, int incx, ref float beta, IntPtr y, int incy);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDsymv_v2(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr A, int lda, IntPtr x, int incx, ref double beta, IntPtr y, int incy);
        #endregion

        #region SYR
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSsyr_v2(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr x, int incx, IntPtr A, int lda);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDsyr_v2(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr x, int incx, IntPtr A, int lda);
        #endregion

        #region SYR2
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSsyr2_v2(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDsyr2_v2(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        #endregion

        #region TBMV
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasStbmv_v2(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDtbmv_v2(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx);
        #endregion

        #region TBMV
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasStbsv_v2(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDtbsv_v2(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx);
        #endregion

        #region TPMV
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasStpmv_v2(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr AP, IntPtr x, int incx);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDtpmv_v2(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr AP, IntPtr x, int incx);
        #endregion

        #region TPSV
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasStpsv_v2(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr AP, IntPtr x, int incx);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDtpsv_v2(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr AP, IntPtr x, int incx);
        #endregion

        #region TRMV
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasStrmv_v2(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr A, int lda, IntPtr x, int incx);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDtrmv_v2(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr A, int lda, IntPtr x, int incx);
        #endregion

        #region TRSV
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasStrsv_v2(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr A, int lda, IntPtr x, int incx);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDtrsv_v2(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr A, int lda, IntPtr x, int incx);
        #endregion
        #endregion

        #region BLAS Level 3 Native
        #region GEMM
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSgemm_v2(cublasHandle handle, cublasOperation transa, cublasOperation transb, int m, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDgemm_v2(cublasHandle handle, cublasOperation transa, cublasOperation transb, int m, int n, int k, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc);
        #endregion

        #region SYMM
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSsymm_v2(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, int m, int n, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDsymm_v2(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, int m, int n, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc);
        #endregion

        #region SYRK
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSsyrk_v2(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, int n, int k, ref float alpha, IntPtr A, int lda, ref float beta, IntPtr C, int ldc);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDsyrk_v2(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, int n, int k, ref double alpha, IntPtr A, int lda, ref double beta, IntPtr C, int ldc);
        #endregion

        #region SYR2K
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSsyr2k_v2(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDsyr2k_v2(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, int n, int k, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc);
        #endregion

        #region TRMM
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasStrmm_v2(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDtrmm_v2(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc);
        #endregion

        #region TRSM
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasStrsm_v2(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDtrsm_v2(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb);
        #endregion
        #endregion

        #region BLAS Level 1
        public CUBLASStatusv2 cublasCaxpy(cublasHandle handle, int n, ref ComplexF alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasCaxpy_v2(handle, n, ref alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasZaxpy(cublasHandle handle, int n, ref ComplexD alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasZaxpy_v2(handle, n, ref alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasCdotu(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result)
        {
            return cublasCdotu_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasCdotc(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result)
        {
            return cublasCdotc_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasZdotu(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result)
        {
            return cublasZdotu_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasZdotc(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result)
        {
            return cublasZdotc_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasCrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref ComplexF s)
        {
            return cublasCrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasZrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref ComplexD s)
        {
            return cublasZrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasCrotg(cublasHandle handle, ref ComplexF a, ref ComplexF b, ref float c, ref ComplexF s)
        {
            return cublasCrotg_v2(handle, ref a, ref b, ref c, ref s);
        }

        public CUBLASStatusv2 cublasZrotg(cublasHandle handle, ref ComplexD a, ref ComplexD b, ref double c, ref ComplexD s)
        {
            return cublasZrotg_v2(handle, ref a, ref b, ref c, ref s);
        }
        #endregion

        #region BLAS Level 2
        #region GBMV
        public CUBLASStatusv2 cublasSgbmv(cublasHandle handle, cublasOperation trans, int m, int n, int kl, int ku, ref float alpha, IntPtr A, int lda, IntPtr x, int incx, ref float beta, IntPtr y, int incy)
        {
            return cublasSgbmv_v2(handle, trans, m, n, kl, ku, ref alpha, A, lda, x, incx, ref beta, y, incy);
        }
        public CUBLASStatusv2 cublasDgbmv(cublasHandle handle, cublasOperation trans, int m, int n, int kl, int ku, ref double alpha, IntPtr A, int lda, IntPtr x, int incx, ref double beta, IntPtr y, int incy)
        {
            return cublasDgbmv_v2(handle, trans, m, n, kl, ku, ref alpha, A, lda, x, incx, ref beta, y, incy);
        }
        #endregion

        #region GEMV
        public CUBLASStatusv2 cublasSgemv(cublasHandle handle, cublasOperation trans, int m, int n, ref float alpha, IntPtr A, int lda, IntPtr x, int incx, ref float beta, IntPtr y, int incy)
        {
            return cublasSgemv_v2(handle, trans, m, n, ref alpha, A, lda, x, incx, ref beta, y, incy);
        }
        public CUBLASStatusv2 cublasDgemv(cublasHandle handle, cublasOperation trans, int m, int n, ref double alpha, IntPtr A, int lda, IntPtr x, int incx, ref double beta, IntPtr y, int incy)
        {
            return cublasDgemv_v2(handle, trans, m, n, ref alpha, A, lda, x, incx, ref beta, y, incy);
        }
        #endregion

        #region GER
        public CUBLASStatusv2 cublasSger(cublasHandle handle, int m, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda)
        {
            return cublasSger_v2(handle, m, n, ref alpha, x, incx, y, incy, A, lda);
        }
        public CUBLASStatusv2 cublasDger(cublasHandle handle, int m, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda)
        {
            return cublasDger_v2(handle, m, n, ref alpha, x, incx, y, incy, A, lda);
        }
        #endregion

        #region SBMV
        public CUBLASStatusv2 cublasSsbmv(cublasHandle handle, cublasFillMode uplo, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr x, int incx, ref float beta, IntPtr y, int incy)
        {
            return cublasSsbmv_v2(handle, uplo, n, k, ref alpha, A, lda, x, incx, ref beta, y, incy);
        }
        public CUBLASStatusv2 cublasDsbmv(cublasHandle handle, cublasFillMode uplo, int n, int k, ref double alpha, IntPtr A, int lda, IntPtr x, int incx, ref double beta, IntPtr y, int incy)
        {
            return cublasDsbmv_v2(handle, uplo, n, k, ref alpha, A, lda, x, incx, ref beta, y, incy);
        }
        #endregion

        #region SPMV
        public CUBLASStatusv2 cublasSspmv(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr ap, IntPtr x, int incx, ref float beta, IntPtr y, int incy)
        {
            return cublasSspmv_v2(handle, uplo, n, ref alpha, ap, x, incx, ref beta, y, incy);
        }
        public CUBLASStatusv2 cublasDspmv(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr ap, IntPtr x, int incx, ref double beta, IntPtr y, int incy)
        {
            return cublasDspmv_v2(handle, uplo, n, ref alpha, ap, x, incx, ref beta, y, incy);
        }
        #endregion

        #region SPR
        public CUBLASStatusv2 cublasSspr(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr x, int incx, IntPtr ap)
        {
            return cublasSspr_v2(handle, uplo, n, ref alpha, x, incx, ap);
        }
        public CUBLASStatusv2 cublasDspr(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr x, int incx, IntPtr ap)
        {
            return cublasDspr_v2(handle, uplo, n, ref alpha, x, incx, ap);
        }
        
        #endregion

        #region SPR2
        public CUBLASStatusv2 cublasSspr2(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr ap)
        {
            return cublasSspr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, ap);
        }
        public CUBLASStatusv2 cublasDspr2(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr ap)
        {
            return cublasDspr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, ap);
        }
        #endregion

        #region SYMV
        public CUBLASStatusv2 cublasSsymv(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr A, int lda, IntPtr x, int incx, ref float beta, IntPtr y, int incy)
        {
            return cublasSsymv_v2(handle, uplo, n, ref alpha, A, lda, x, incx, ref beta, y, incy);
        }
        public CUBLASStatusv2 cublasDsymv(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr A, int lda, IntPtr x, int incx, ref double beta, IntPtr y, int incy)
        {
            return cublasDsymv_v2(handle, uplo, n, ref alpha, A, lda, x, incx, ref beta, y, incy);
        }
        #endregion

        #region SYR
        public CUBLASStatusv2 cublasSsyr(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr x, int incx, IntPtr A, int lda)
        {
            return cublasSsyr_v2(handle, uplo, n, ref alpha, x, incx, A, lda);
        }
        public CUBLASStatusv2 cublasDsyr(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr x, int incx, IntPtr A, int lda)
        {
            return cublasDsyr_v2(handle, uplo, n, ref alpha, x, incx, A, lda);
        }
        #endregion

        #region SYR2
        public CUBLASStatusv2 cublasSsyr2(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda)
        {
            return cublasSsyr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, A, lda);
        }
        public CUBLASStatusv2 cublasDsyr2(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda)
        {
            return cublasDsyr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, A, lda);
        }
        #endregion

        #region TBMV
        public CUBLASStatusv2 cublasStbmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx)
        {
            return cublasStbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
        }
        public CUBLASStatusv2 cublasDtbmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx)
        {
            return cublasDtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
        }
        #endregion

        #region TBSV
        public CUBLASStatusv2 cublasStbsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx)
        {
            return cublasStbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
        }
        public CUBLASStatusv2 cublasDtbsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx)
        {
            return cublasDtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
        }
        #endregion

        #region TPMV
        public CUBLASStatusv2 cublasStpmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr AP, IntPtr x, int incx)
        {
            return cublasStpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
        }
        public CUBLASStatusv2 cublasDtpmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr AP, IntPtr x, int incx)
        {
            return cublasDtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
        }
        #endregion

        #region TPSV
        public CUBLASStatusv2 cublasStpsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr AP, IntPtr x, int incx)
        {
            return cublasStpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
        }
        public CUBLASStatusv2 cublasDtpsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr AP, IntPtr x, int incx)
        {
            return cublasDtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
        }
        #endregion

        #region TRMV
        public CUBLASStatusv2 cublasStrmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr A, int lda, IntPtr x, int incx)
        {
            return cublasStrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
        }
        public CUBLASStatusv2 cublasDtrmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr A, int lda, IntPtr x, int incx)
        {
            return cublasDtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
        }
        #endregion

        #region TRSV
        public CUBLASStatusv2 cublasStrsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr A, int lda, IntPtr x, int incx)
        {
            return cublasStrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
        }
        public CUBLASStatusv2 cublasDtrsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr A, int lda, IntPtr x, int incx)
        {
            return cublasDtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
        }
        #endregion
        #endregion

        #region BLAS Level 3
        #region GEMM
        public CUBLASStatusv2 cublasSgemm(cublasHandle handle, cublasOperation transa, cublasOperation transb, int m, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc)
        {
            return cublasSgemm_v2(handle, transa, transb, m, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc);
        }
        public CUBLASStatusv2 cublasDgemm(cublasHandle handle, cublasOperation transa, cublasOperation transb, int m, int n, int k, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc)
        {
            return cublasDgemm_v2(handle, transa, transb, m, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc);
        }
        #endregion

        #region SYMM
        public CUBLASStatusv2 cublasSsymm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, int m, int n, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc)
        {
            return cublasSsymm_v2(handle, side, uplo, m, n, ref alpha, A, lda, B, ldb, ref beta, C, ldc);
        }
        public CUBLASStatusv2 cublasDsymm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, int m, int n, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc)
        {
            return cublasDsymm_v2(handle, side, uplo, m, n, ref alpha, A, lda, B, ldb, ref beta, C, ldc);
        }
        #endregion

        #region SYRK
        public CUBLASStatusv2 cublasSsyrk(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, int n, int k, ref float alpha, IntPtr A, int lda, ref float beta, IntPtr C, int ldc)
        {
            return cublasSsyrk_v2(handle, uplo, trans, n, k, ref alpha, A, lda, ref beta, C, ldc);
        }
        public CUBLASStatusv2 cublasDsyrk(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, int n, int k, ref double alpha, IntPtr A, int lda, ref double beta, IntPtr C, int ldc)
        {
            return cublasDsyrk_v2(handle, uplo, trans, n, k, ref alpha, A, lda, ref beta, C, ldc);
        }
        #endregion

        #region SYR2K
        public CUBLASStatusv2 cublasSsyr2k(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc)
        {
            return cublasSsyr2k_v2(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc);
        }
        public CUBLASStatusv2 cublasDsyr2k(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, int n, int k, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc)
        {
            return cublasDsyr2k_v2(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc);
        }
        #endregion

        #region TRMM
        public CUBLASStatusv2 cublasStrmm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc)
        {
            return cublasStrmm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb, C, ldc);
        }
        public CUBLASStatusv2 cublasDtrmm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc)
        {
            return cublasDtrmm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb, C, ldc);
        }
        #endregion

        #region TRSM
        public CUBLASStatusv2 cublasStrsm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb)
        {
            return cublasStrsm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb);
        }

        public CUBLASStatusv2 cublasDtrsm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb)
        {
            return cublasDtrsm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb);
        }
        #endregion
        #endregion
    }
}
