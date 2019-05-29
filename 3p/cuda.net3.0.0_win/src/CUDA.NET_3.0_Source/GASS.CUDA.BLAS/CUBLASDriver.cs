namespace GASS.CUDA.BLAS
{
    using GASS.CUDA.Types;
    using System;
    using System.Runtime.InteropServices;

    public class CUBLASDriver
    {
        internal const string CUBLAS_DLL_NAME = "cublas";

        //[DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        //public static extern IntPtr LoadLibrary(string lpFileName);//[In, MarshalAs(UnmanagedType.LPStr)] 


        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasAlloc(int n, int elemSize, ref CUdeviceptr devicePtr);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasCaxpy(int n, cuFloatComplex alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasCcopy(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern cuFloatComplex cublasCdotc(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern cuFloatComplex cublasCdotu(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCgbmv(char trans, int m, int n, int kl, int ku, cuFloatComplex alpha, CUdeviceptr A, int lda, CUdeviceptr x, int incx, cuFloatComplex beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCgemm(char transa, char transb, int m, int n, int k, cuFloatComplex alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, cuFloatComplex beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCgemv(char trans, int m, int n, cuFloatComplex alpha, CUdeviceptr A, int lda, CUdeviceptr x, int incx, cuFloatComplex beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCgerc(int m, int n, cuFloatComplex alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy, CUdeviceptr A, int lda);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCgeru(int m, int n, cuFloatComplex alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy, CUdeviceptr A, int lda);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasChbmv(char uplo, int n, int k, cuFloatComplex alpha, CUdeviceptr A, int lda, CUdeviceptr x, int incx, cuFloatComplex beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasChemm(char side, char uplo, int m, int n, cuFloatComplex alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, cuFloatComplex beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasChemv(char uplo, int n, cuFloatComplex alpha, CUdeviceptr A, int lda, CUdeviceptr x, int incx, cuFloatComplex beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCher(char uplo, int n, cuFloatComplex alpha, CUdeviceptr x, int incx, CUdeviceptr A, int lda);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCher2(char uplo, int n, cuFloatComplex alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy, CUdeviceptr A, int lda);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCher2k(char uplo, char trans, int n, int k, cuFloatComplex alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, cuFloatComplex beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCherk(char uplo, char trans, int n, int k, cuFloatComplex alpha, CUdeviceptr A, int lda, cuFloatComplex beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasChpmv(char uplo, int n, cuFloatComplex alpha, CUdeviceptr AP, CUdeviceptr x, int incx, cuFloatComplex beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasChpr(char uplo, int n, cuFloatComplex alpha, CUdeviceptr x, int incx, CUdeviceptr AP);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasChpr2(char uplo, int n, cuFloatComplex alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy, CUdeviceptr AP);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasCrot(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy, float c, cuFloatComplex s);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasCrotg(CUdeviceptr pca, cuFloatComplex cb, CUdeviceptr psc, CUdeviceptr pcs);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasCscal(int n, cuFloatComplex alpha, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasCsrot(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy, float c, float s);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasCsscal(int n, float alpha, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasCswap(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCsymm(char side, char uplo, int m, int n, cuFloatComplex alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, cuFloatComplex beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCsyr2k(char uplo, char trans, int n, int k, cuFloatComplex alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, cuFloatComplex beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCsyrk(char uplo, char trans, int n, int k, cuFloatComplex alpha, CUdeviceptr A, int lda, cuFloatComplex beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCtbmv(char uplo, char trans, char diag, int n, int k, CUdeviceptr A, int lda, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCtbsv(char uplo, char trans, char diag, int n, int k, CUdeviceptr A, int lda, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCtpmv(char uplo, char trans, char diag, int n, CUdeviceptr AP, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCtpsv(char uplo, char trans, char diag, int n, CUdeviceptr AP, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCtrmm(char side, char uplo, char transa, char diag, int m, int n, cuFloatComplex alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCtrmv(char uplo, char trans, char diag, int n, CUdeviceptr A, int lda, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCtrsm(char side, char uplo, char transa, char diag, int m, int n, cuFloatComplex alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasCtrsv(char uplo, char trans, char diag, int n, CUdeviceptr A, int lda, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern double cublasDasum(int n, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasDaxpy(int n, double alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasDcopy(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern double cublasDdot(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDgbmv(char trans, int m, int n, int kl, int ku, double alpha, CUdeviceptr A, int lda, CUdeviceptr x, int incx, double beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDgemm(char transa, char transb, int m, int n, int k, double alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, double beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDgemv(char trans, int m, int n, double alpha, CUdeviceptr A, int lda, CUdeviceptr x, int incx, double beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDger(int m, int n, double alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy, CUdeviceptr A, int lda);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern double cublasDnrm2(int n, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasDrot(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy, double sc, double ss);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasDrotg(CUdeviceptr sa, CUdeviceptr sb, CUdeviceptr sc, CUdeviceptr ss);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasDrotm(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy, CUdeviceptr sparam);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasDrotmg(CUdeviceptr sd1, CUdeviceptr sd2, CUdeviceptr sx1, CUdeviceptr sy1, CUdeviceptr sparam);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDsbmv(char uplo, int n, int k, double alpha, CUdeviceptr A, int lda, CUdeviceptr x, int incx, double beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasDscal(int n, double alpha, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDspmv(char uplo, int n, double alpha, CUdeviceptr AP, CUdeviceptr x, int incx, double beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDspr(char uplo, int n, double alpha, CUdeviceptr x, int incx, CUdeviceptr AP);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDspr2(char uplo, int n, double alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy, CUdeviceptr AP);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasDswap(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDsymm(char side, char uplo, int m, int n, double alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, double beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDsymv(char uplo, int n, double alpha, CUdeviceptr A, int lda, CUdeviceptr x, int incx, double beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDsyr(char uplo, int n, double alpha, CUdeviceptr x, int incx, CUdeviceptr A, int lda);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDsyr2(char uplo, int n, double alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy, CUdeviceptr A, int lda);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDsyr2k(char uplo, char trans, int n, int k, double alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, double beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDsyrk(char uplo, char trans, int n, int k, double alpha, CUdeviceptr A, int lda, double beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDtbmv(char uplo, char trans, char diag, int n, int k, CUdeviceptr A, int lda, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDtbsv(char uplo, char trans, char diag, int n, int k, CUdeviceptr A, int lda, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDtpmv(char uplo, char trans, char diag, int n, CUdeviceptr AP, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDtpsv(char uplo, char trans, char diag, int n, CUdeviceptr AP, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDtrmm(char side, char uplo, char transa, char diag, int m, int n, double alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDtrmv(char uplo, char trans, char diag, int n, CUdeviceptr A, int lda, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDtrsm(char side, char uplo, char transa, char diag, int m, int n, double alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasDtrsv(char uplo, char trans, char diag, int n, CUdeviceptr A, int lda, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern double cublasDzasum(int n, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern double cublasDznrm2(int n, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasFree(CUdeviceptr devicePtr);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetError();
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Char1[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Char2[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Char3[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Char4[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] cuDoubleComplex[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] cuFloatComplex[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Double1[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Double2[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Float1[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Float2[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Float3[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Float4[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Int1[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Int2[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Int3[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Int4[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Long1[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Long2[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Long3[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Long4[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Short1[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Short2[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Short3[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] Short4[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] UChar1[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] UChar2[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] UChar3[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] UChar4[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] UInt1[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] UInt2[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] UInt3[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] UInt4[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] ULong1[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] ULong2[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] ULong3[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] ULong4[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] UShort1[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] UShort2[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] UShort3[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] UShort4[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] byte[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] double[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] short[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] int[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] long[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] sbyte[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] float[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] ushort[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, IntPtr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] uint[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetMatrix(int rows, int cols, int elemSize, CUdeviceptr A, int lda, [Out] ulong[] B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Char1[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Char2[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Char3[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Char4[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] cuDoubleComplex[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] cuFloatComplex[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Double1[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Double2[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Float1[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Float2[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Float3[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Float4[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Int1[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Int2[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Int3[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Int4[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Long1[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Long2[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Long3[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Long4[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Short1[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Short2[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Short3[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] Short4[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] UChar1[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] UChar2[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] UChar3[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] UChar4[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] UInt1[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] UInt2[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] UInt3[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] UInt4[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] ULong1[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] ULong2[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] ULong3[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] ULong4[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] UShort1[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] UShort2[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] UShort3[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] UShort4[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] byte[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] double[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] short[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] int[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] long[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] sbyte[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] float[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] ushort[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] uint[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, [Out] ulong[] y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasGetVector(int n, int elemSize, CUdeviceptr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern int cublasIcamax(int n, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern int cublasIcamin(int n, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern int cublasIdamax(int n, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern int cublasIdamin(int n, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasInit();
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern int cublasIsamax(int n, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern int cublasIsamin(int n, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern int cublasIzamax(int n, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern int cublasIzamin(int n, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern float cublasSasum(int n, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasSaxpy(int n, float alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern float cublasScasum(int n, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern float cublasScnrm2(int n, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasScopy(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern float cublasSdot(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Char1[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Char2[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Char3[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Char4[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] cuDoubleComplex[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] cuFloatComplex[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Double1[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Double2[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Float1[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Float2[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Float3[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Float4[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Int1[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Int2[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Int3[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Int4[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Long1[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Long2[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Long3[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Long4[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Short1[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Short2[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Short3[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] Short4[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] UChar1[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] UChar2[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, IntPtr A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] UChar3[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] UChar4[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] UInt1[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] UInt2[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] UInt3[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] UInt4[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] ULong1[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] ULong2[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] ULong3[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] ULong4[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] UShort1[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] UShort2[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] UShort3[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] UShort4[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] byte[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] double[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] short[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] int[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] long[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] sbyte[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] float[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] ushort[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] uint[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] ulong[] A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Char1[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Char2[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Char3[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Char4[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] cuDoubleComplex[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] cuFloatComplex[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Double1[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Double2[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Float1[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Float2[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Float3[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Float4[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Int1[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Int2[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Int3[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Int4[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Long1[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Long2[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Long3[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Long4[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Short1[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Short2[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Short3[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] Short4[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] UChar1[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] UChar2[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, IntPtr x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] UChar3[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] UChar4[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] UInt1[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] UInt2[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] UInt3[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] UInt4[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] ULong1[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] ULong2[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] ULong3[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] ULong4[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] UShort1[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] UShort2[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] UShort3[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] UShort4[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] byte[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] double[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] short[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] int[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] long[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] sbyte[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] float[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] ushort[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] uint[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasSetVector(int n, int elemSize, [In] ulong[] x, int incx, CUdeviceptr devicePtr, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasSgbmv(char trans, int m, int n, int kl, int ku, float alpha, CUdeviceptr A, int lda, CUdeviceptr x, int incx, float beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasSgemm(char transa, char transb, int m, int n, int k, float alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, float beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasSgemv(char trans, int m, int n, float alpha, CUdeviceptr A, int lda, CUdeviceptr x, int incx, float beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasSger(int m, int n, float alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy, CUdeviceptr A, int lda);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern CUBLASStatus cublasShutdown();
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern float cublasSnrm2(int n, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasSrot(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy, float sc, float ss);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasSrotg(CUdeviceptr sa, CUdeviceptr sb, CUdeviceptr sc, CUdeviceptr ss);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasSrotm(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy, CUdeviceptr sparam);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasSrotmg(CUdeviceptr sd1, CUdeviceptr sd2, CUdeviceptr sx1, CUdeviceptr sy1, CUdeviceptr sparam);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasSsbmv(char uplo, int n, int k, float alpha, CUdeviceptr A, int lda, CUdeviceptr x, int incx, float beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasSscal(int n, float alpha, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasSspmv(char uplo, int n, float alpha, CUdeviceptr AP, CUdeviceptr x, int incx, float beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasSspr(char uplo, int n, float alpha, CUdeviceptr x, int incx, CUdeviceptr AP);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasSspr2(char uplo, int n, float alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy, CUdeviceptr AP);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasSswap(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasSsymm(char side, char uplo, int m, int n, float alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, float beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasSsymv(char uplo, int n, float alpha, CUdeviceptr A, int lda, CUdeviceptr x, int incx, float beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasSsyr(char uplo, int n, float alpha, CUdeviceptr x, int incx, CUdeviceptr A, int lda);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasSsyr2(char uplo, int n, float alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy, CUdeviceptr A, int lda);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasSsyr2k(char uplo, char trans, int n, int k, float alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, float beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasSsyrk(char uplo, char trans, int n, int k, float alpha, CUdeviceptr A, int lda, float beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasStbmv(char uplo, char trans, char diag, int n, int k, CUdeviceptr A, int lda, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasStbsv(char uplo, char trans, char diag, int n, int k, CUdeviceptr A, int lda, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasStpmv(char uplo, char trans, char diag, int n, CUdeviceptr AP, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasStpsv(char uplo, char trans, char diag, int n, CUdeviceptr AP, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasStrmm(char side, char uplo, char transa, char diag, int m, int n, float alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasStrmv(char uplo, char trans, char diag, int n, CUdeviceptr A, int lda, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasStrsm(char side, char uplo, char transa, char diag, int m, int n, float alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasStrsv(char uplo, char trans, char diag, int n, CUdeviceptr A, int lda, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasXerbla(string srName, int info);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasZaxpy(int n, cuDoubleComplex alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasZcopy(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern cuDoubleComplex cublasZdotc(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern cuDoubleComplex cublasZdotu(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasZdrot(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy, double c, double s);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasZdscal(int n, double alpha, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZgbmv(char trans, int m, int n, int kl, int ku, cuDoubleComplex alpha, CUdeviceptr A, int lda, CUdeviceptr x, int incx, cuDoubleComplex beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZgemm(char transa, char transb, int m, int n, int k, cuDoubleComplex alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, cuDoubleComplex beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZgemv(char trans, int m, int n, cuDoubleComplex alpha, CUdeviceptr A, int lda, CUdeviceptr x, int incx, cuDoubleComplex beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZgerc(int m, int n, cuDoubleComplex alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy, CUdeviceptr A, int lda);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZgeru(int m, int n, cuDoubleComplex alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy, CUdeviceptr A, int lda);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZhbmv(char uplo, int n, int k, cuDoubleComplex alpha, CUdeviceptr A, int lda, CUdeviceptr x, int incx, cuDoubleComplex beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZhemm(char side, char uplo, int m, int n, cuDoubleComplex alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, cuDoubleComplex beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZhemv(char uplo, int n, cuDoubleComplex alpha, CUdeviceptr A, int lda, CUdeviceptr x, int incx, cuDoubleComplex beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZher(char uplo, int n, double alpha, CUdeviceptr x, int incx, CUdeviceptr A, int lda);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZher2(char uplo, int n, cuDoubleComplex alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy, CUdeviceptr A, int lda);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZher2k(char uplo, char trans, int n, int k, cuDoubleComplex alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, double beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZherk(char uplo, char trans, int n, int k, double alpha, CUdeviceptr A, int lda, double beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZhpmv(char uplo, int n, cuDoubleComplex alpha, CUdeviceptr AP, CUdeviceptr x, int incx, cuDoubleComplex beta, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZhpr(char uplo, int n, double alpha, CUdeviceptr x, int incx, CUdeviceptr AP);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZhpr2(char uplo, int n, cuDoubleComplex alpha, CUdeviceptr x, int incx, CUdeviceptr y, int incy, CUdeviceptr AP);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasZrot(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy, double sc, cuDoubleComplex cs);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasZrotg(ref cuDoubleComplex host_ca, cuDoubleComplex cb, ref double host_sc, ref cuDoubleComplex host_cs);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasZscal(int n, cuDoubleComplex alpha, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        public static extern void cublasZswap(int n, CUdeviceptr x, int incx, CUdeviceptr y, int incy);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZsymm(char side, char uplo, int m, int n, cuDoubleComplex alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, cuDoubleComplex beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZsyr2k(char uplo, char trans, int n, int k, cuDoubleComplex alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, cuDoubleComplex beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZsyrk(char uplo, char trans, int n, int k, cuDoubleComplex alpha, CUdeviceptr A, int lda, cuDoubleComplex beta, CUdeviceptr C, int ldc);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZtbmv(char uplo, char trans, char diag, int n, int k, CUdeviceptr A, int lda, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZtbsv(char uplo, char trans, char diag, int n, int k, CUdeviceptr A, int lda, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZtpmv(char uplo, char trans, char diag, int n, CUdeviceptr AP, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZtpsv(char uplo, char trans, char diag, int n, CUdeviceptr AP, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZtrmm(char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZtrmv(char uplo, char trans, char diag, int n, CUdeviceptr A, int lda, CUdeviceptr x, int incx);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZtrsm(char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb);
        [DllImport(CUBLAS_DLL_NAME, CharSet = CharSet.Ansi)]
        public static extern void cublasZtrsv(char uplo, char trans, char diag, int n, CUdeviceptr A, int lda, CUdeviceptr x, int incx);
    }
}

