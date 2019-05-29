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
    internal class CUBLASDriver32Ex : ICUBLASDriverv2Ex
    {
        #region BLAS Level 1 Native (Extended)
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCaxpy_v2(cublasHandle handle, int n, ref ComplexF alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZaxpy_v2(cublasHandle handle, int n, ref ComplexD alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCdotu_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCdotc_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdotu_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdotc_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref ComplexF s);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref ComplexD s);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCrotg_v2(cublasHandle handle, ref ComplexF a, ref ComplexF b, ref float c, ref ComplexF s);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZrotg_v2(cublasHandle handle, ref ComplexD a, ref ComplexD b, ref double c, ref ComplexD s);
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


        public CUBLASStatusv2 cublasSgbmv(cublasHandle handle, cublasOperation trans, int m, int n, int kl, int ku, ref float alpha, IntPtr A, int lda, IntPtr x, int incx, ref float beta, IntPtr y, int incy)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDgbmv(cublasHandle handle, cublasOperation trans, int m, int n, int kl, int ku, ref double alpha, IntPtr A, int lda, IntPtr x, int incx, ref double beta, IntPtr y, int incy)
        {
            throw new NotImplementedException();
        }
        #endregion


        public CUBLASStatusv2 cublasSgemv(cublasHandle handle, cublasOperation trans, int m, int n, ref float alpha, IntPtr A, int lda, IntPtr x, int incx, ref float beta, IntPtr y, int incy)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDgemv(cublasHandle handle, cublasOperation trans, int m, int n, ref double alpha, IntPtr A, int lda, IntPtr x, int incx, ref double beta, IntPtr y, int incy)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasSger(cublasHandle handle, int m, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDger(cublasHandle handle, int m, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasSsbmv(cublasHandle handle, cublasFillMode uplo, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr x, int incx, ref float beta, IntPtr y, int incy)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDsbmv(cublasHandle handle, cublasFillMode uplo, int n, int k, ref double alpha, IntPtr A, int lda, IntPtr x, int incx, ref double beta, IntPtr y, int incy)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasSspmv(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr ap, IntPtr x, int incx, ref float beta, IntPtr y, int incy)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDspmv(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr ap, IntPtr x, int incx, ref double beta, IntPtr y, int incy)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasSspr(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr x, int incx, IntPtr ap)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDspr(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr x, int incx, IntPtr ap)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasSspr2(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr ap)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDspr2(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr ap)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasSsymv(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr A, int lda, IntPtr x, int incx, ref float beta, IntPtr y, int incy)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDsymv(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr A, int lda, IntPtr x, int incx, ref double beta, IntPtr y, int incy)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasSsyr(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr x, int incx, IntPtr A, int lda)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDsyr(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr x, int incx, IntPtr A, int lda)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasSsyr2(cublasHandle handle, cublasFillMode uplo, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDsyr2(cublasHandle handle, cublasFillMode uplo, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasStbmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDtbmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasStbsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDtbsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasStpmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr AP, IntPtr x, int incx)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDtpmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr AP, IntPtr x, int incx)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasStpsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr AP, IntPtr x, int incx)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDtpsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr AP, IntPtr x, int incx)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasStrmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr A, int lda, IntPtr x, int incx)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDtrmv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr A, int lda, IntPtr x, int incx)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasStrsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr A, int lda, IntPtr x, int incx)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDtrsv(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int n, IntPtr A, int lda, IntPtr x, int incx)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasSgemm(cublasHandle handle, cublasOperation transa, cublasOperation transb, int m, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDgemm(cublasHandle handle, cublasOperation transa, cublasOperation transb, int m, int n, int k, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasSsymm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, int m, int n, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDsymm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, int m, int n, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasSsyrk(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, int n, int k, ref float alpha, IntPtr A, int lda, ref float beta, IntPtr C, int ldc)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDsyrk(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, int n, int k, ref double alpha, IntPtr A, int lda, ref double beta, IntPtr C, int ldc)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasSsyr2k(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDsyr2k(cublasHandle handle, cublasFillMode uplo, cublasOperation trans, int n, int k, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasStrmm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDtrmm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc)
        {
            throw new NotImplementedException();
        }


        public CUBLASStatusv2 cublasStrsm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb)
        {
            throw new NotImplementedException();
        }

        public CUBLASStatusv2 cublasDtrsm(cublasHandle handle, cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb)
        {
            throw new NotImplementedException();
        }
    }
}
