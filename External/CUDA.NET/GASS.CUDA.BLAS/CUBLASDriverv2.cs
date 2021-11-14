namespace GASS.CUDA.BLAS
{
    using GASS.CUDA.Types;
    using GASS.CUDA.BLAS.Types;
    using System;
    using System.Runtime.InteropServices;

    public class CUBLASDriver32 : ICUBLASDriverv2
    {
#if LINUX
        public const string CUBLAS_DLL_NAME = "libcublas";
#else
        public const string CUBLAS_DLL_NAME = "cublas64_70";
#endif

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCreate_v2(ref cublasHandle handle);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDestroy_v2(cublasHandle handle);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasGetVersion_v2(cublasHandle handle, ref int version);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSetStream_v2(cublasHandle handle, cudaStream streamId);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasGetStream_v2(cublasHandle handle, ref cudaStream streamId);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasGetPointerMode_v2(cublasHandle handle, ref CUBLASPointerMode mode);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSetPointerMode_v2(cublasHandle handle, CUBLASPointerMode mode);

        #region Level 1
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasIcamax_v2(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasIdamax_v2(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasIsamax_v2(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasIzamax_v2(cublasHandle handle, int n, IntPtr x, int incx, ref int result);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasIcamin_v2(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasIdamin_v2(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasIsamin_v2(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasIzamin_v2(cublasHandle handle, int n, IntPtr x, int incx, ref int result);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSasum_v2(cublasHandle handle, int n, IntPtr x, int incx, ref float result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDasum_v2(cublasHandle handle, int n, IntPtr x, int incx, ref double result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasScasum_v2(cublasHandle handle, int n, IntPtr x, int incx, ref float result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDzasum_v2(cublasHandle handle, int n, IntPtr x, int incx, ref double result);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSaxpy_v2(cublasHandle handle, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDaxpy_v2(cublasHandle handle, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCaxpy_v2(cublasHandle handle, int n, ref cuFloatComplex alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZaxpy_v2(cublasHandle handle, int n, ref cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSaxpy_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDaxpy_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCaxpy_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZaxpy_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasScopy_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDcopy_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCcopy_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZcopy_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSdot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDdot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCdotu_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuFloatComplex result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCdotc_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuFloatComplex result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdotu_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuDoubleComplex result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdotc_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuDoubleComplex result);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSnrm2_v2(cublasHandle handle, int n, IntPtr x, int incx, ref float result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDnrm2_v2(cublasHandle handle, int n, IntPtr x, int incx, ref double result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasScnrm2_v2(cublasHandle handle, int n, IntPtr x, int incx, ref float result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDznrm2_v2(cublasHandle handle, int n, IntPtr x, int incx, ref double result);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref float s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref double s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref cuFloatComplex s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCsrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCsrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref float s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref cuDoubleComplex s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref double s);


        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSrotg_v2(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSrotg_v2(cublasHandle handle, ref float a, ref float b, ref float c, ref float s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDrotg_v2(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDrotg_v2(cublasHandle handle, ref double a, ref double b, ref double c, ref double s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCrotg_v2(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCrotg_v2(cublasHandle handle, ref cuFloatComplex a, ref cuFloatComplex b, ref float c, ref cuFloatComplex s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZrotg_v2(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZrotg_v2(cublasHandle handle, ref cuDoubleComplex a, ref cuDoubleComplex b, ref double c, ref cuDoubleComplex s);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSrotm_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr param);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDrotm_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr param);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSrotmg_v2(cublasHandle handle, IntPtr d1, IntPtr d2, IntPtr x1, IntPtr y1, IntPtr param);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDrotmg_v2(cublasHandle handle, IntPtr d1, IntPtr d2, IntPtr x1, IntPtr y1, IntPtr param);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSrotmg_v2(cublasHandle handle, ref float d1, ref float d2, ref float x1, ref float y1, IntPtr param);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDrotmg_v2(cublasHandle handle, ref double d1, ref double d2, ref double x1, ref double y1, IntPtr param);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSscal_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDscal_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCscal_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCsscal_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZscal_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdscal_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSswap_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDswap_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCswap_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZswap_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);

        #endregion


        public CUBLASStatusv2 cublasCreate(ref cublasHandle handle)
        {
            return cublasCreate_v2(ref handle);
        }

        public CUBLASStatusv2 cublasDestroy(cublasHandle handle)
        {
            return cublasDestroy_v2(handle);
        }

        public CUBLASStatusv2 cublasGetVersion(cublasHandle handle, ref int version)
        {
            return cublasGetVersion_v2(handle, ref version);
        }

        public CUBLASStatusv2 cublasSetStream(cublasHandle handle, cudaStream streamId)
        {
            return cublasSetStream_v2(handle, streamId);
        }

        public CUBLASStatusv2 cublasGetStream(cublasHandle handle, ref cudaStream streamId)
        {
            return cublasGetStream_v2(handle, ref streamId);
        }

        public CUBLASStatusv2 cublasGetPointerMode(cublasHandle handle, ref CUBLASPointerMode mode)
        {
            return cublasGetPointerMode_v2(handle, ref mode);
        }

        public CUBLASStatusv2 cublasSetPointerMode(cublasHandle handle, CUBLASPointerMode mode)
        {
            return cublasSetPointerMode_v2(handle, mode);
        }

        public CUBLASStatusv2 cublasIcamax(cublasHandle handle, int n, IntPtr x, int incx, ref int result)
        {
            return cublasIcamax_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasIdamax(cublasHandle handle, int n, IntPtr x, int incx, ref int result)
        {
            return cublasIdamax_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasIsamax(cublasHandle handle, int n, IntPtr x, int incx, ref int result)
        {
            return cublasIsamax_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasIzamax(cublasHandle handle, int n, IntPtr x, int incx, ref int result)
        {
            return cublasIzamax_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasIcamin(cublasHandle handle, int n, IntPtr x, int incx, ref int result)
        {
            return cublasIcamin_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasIdamin(cublasHandle handle, int n, IntPtr x, int incx, ref int result)
        {
            return cublasIdamin_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasIsamin(cublasHandle handle, int n, IntPtr x, int incx, ref int result)
        {
            return cublasIsamin_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasIzamin(cublasHandle handle, int n, IntPtr x, int incx, ref int result)
        {
            return cublasIzamin_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasSasum(cublasHandle handle, int n, IntPtr x, int incx, ref float result)
        {
            return cublasSasum_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasDasum(cublasHandle handle, int n, IntPtr x, int incx, ref double result)
        {
            return cublasDasum_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasScasum(cublasHandle handle, int n, IntPtr x, int incx, ref float result)
        {
            return cublasScasum_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasDzasum(cublasHandle handle, int n, IntPtr x, int incx, ref double result)
        {
            return cublasDzasum_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasSaxpy(cublasHandle handle, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasSaxpy_v2(handle, n, ref alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasDaxpy(cublasHandle handle, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasDaxpy_v2(handle, n, ref alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasCaxpy(cublasHandle handle, int n, ref cuFloatComplex alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasCaxpy_v2(handle, n, ref alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasZaxpy(cublasHandle handle, int n, ref cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasZaxpy_v2(handle, n, ref alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasSaxpy(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasDaxpy(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasDaxpy_v2(handle, n, alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasCaxpy(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasCaxpy_v2(handle, n, alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasZaxpy(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasZaxpy_v2(handle, n, alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasScopy(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasScopy_v2(handle, n, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasDcopy(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasDcopy_v2(handle, n, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasCcopy(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasCcopy_v2(handle, n, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasZcopy(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasZcopy_v2(handle, n, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasSdot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float result)
        {
            return cublasSdot_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasDdot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double result)
        {
            return cublasDdot_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasCdotu(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuFloatComplex result)
        {
            return cublasCdotu_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasCdotc(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuFloatComplex result)
        {
            return cublasCdotc_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasZdotu(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuDoubleComplex result)
        {
            return cublasZdotu_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasZdotc(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuDoubleComplex result)
        {
            return cublasZdotc_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasSnrm2(cublasHandle handle, int n, IntPtr x, int incx, ref float result)
        {
            return cublasSnrm2_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasDnrm2(cublasHandle handle, int n, IntPtr x, int incx, ref double result)
        {
            return cublasDnrm2_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasScnrm2(cublasHandle handle, int n, IntPtr x, int incx, ref float result)
        {
            return cublasScnrm2_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasDznrm2(cublasHandle handle, int n, IntPtr x, int incx, ref double result)
        {
            return cublasDznrm2_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasSrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s)
        {
            return cublasSrot_v2(handle, n, x, incx, y, incy, c, s);
        }

        public CUBLASStatusv2 cublasSrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref float s)
        {
            return cublasSrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasDrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s)
        {
            return cublasDrot_v2(handle, n, x, incx, y, incy, c, s);
        }

        public CUBLASStatusv2 cublasDrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref double s)
        {
            return cublasDrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasCrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s)
        {
            return cublasCrot_v2(handle, n, x, incx, y, incy, c, s);
        }

        public CUBLASStatusv2 cublasCrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref cuFloatComplex s)
        {
            return cublasCrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasCsrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s)
        {
            return cublasCsrot_v2(handle, n, x, incx, y, incy, c, s);
        }

        public CUBLASStatusv2 cublasCsrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref float s)
        {
            return cublasCsrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasZrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s)
        {
            return cublasZrot_v2(handle, n, x, incx, y, incy, c, s);
        }

        public CUBLASStatusv2 cublasZrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref cuDoubleComplex s)
        {
            return cublasZrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasZdrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s)
        {
            return cublasZdrot_v2(handle, n, x, incx, y, incy, c, s);
        }

        public CUBLASStatusv2 cublasZdrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref double s)
        {
            return cublasZdrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasSrotg(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s)
        {
            return cublasSrotg_v2(handle, a, b, c, s);
        }

        public CUBLASStatusv2 cublasSrotg(cublasHandle handle, ref float a, ref float b, ref float c, ref float s)
        {
            return cublasSrotg_v2(handle, ref a, ref b, ref c, ref s);
        }

        public CUBLASStatusv2 cublasDrotg(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s)
        {
            return cublasDrotg_v2(handle, a, b, c, s);
        }

        public CUBLASStatusv2 cublasDrotg(cublasHandle handle, ref double a, ref double b, ref double c, ref double s)
        {
            return cublasDrotg_v2(handle, ref a, ref b, ref c, ref s);
        }

        public CUBLASStatusv2 cublasCrotg(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s)
        {
            return cublasCrotg_v2(handle, a, b, c, s);
        }

        public CUBLASStatusv2 cublasCrotg(cublasHandle handle, ref cuFloatComplex a, ref cuFloatComplex b, ref float c, ref cuFloatComplex s)
        {
            return cublasCrotg_v2(handle, ref a, ref b, ref c, ref s);
        }

        public CUBLASStatusv2 cublasZrotg(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s)
        {
            return cublasZrotg_v2(handle, a, b, c, s);
        }

        public CUBLASStatusv2 cublasZrotg(cublasHandle handle, ref cuDoubleComplex a, ref cuDoubleComplex b, ref double c, ref cuDoubleComplex s)
        {
            return cublasZrotg_v2(handle, ref a, ref b, ref c, ref s);
        }

        public CUBLASStatusv2 cublasSrotm(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr param)
        {
            return cublasSrotm_v2(handle, n, x, incx, y, incy, param);
        }

        public CUBLASStatusv2 cublasDrotm(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr param)
        {
            return cublasDrotm_v2(handle, n, x, incx, y, incy, param);
        }

        public CUBLASStatusv2 cublasSrotmg(cublasHandle handle, IntPtr d1, IntPtr d2, IntPtr x1, IntPtr y1, IntPtr param)
        {
            return cublasSrotmg_v2(handle, d1, d2, x1, y1, param);
        }

        public CUBLASStatusv2 cublasDrotmg(cublasHandle handle, IntPtr d1, IntPtr d2, IntPtr x1, IntPtr y1, IntPtr param)
        {
            return cublasDrotmg_v2(handle, d1, d2, x1, y1, param);
        }

        public CUBLASStatusv2 cublasSrotmg(cublasHandle handle, ref float d1, ref float d2, ref float x1, ref float y1, IntPtr param)
        {
            return cublasSrotmg_v2(handle, ref d1, ref d2, ref x1, ref y1, param);
        }

        public CUBLASStatusv2 cublasDrotmg(cublasHandle handle, ref double d1, ref double d2, ref double x1, ref double y1, IntPtr param)
        {
            return cublasDrotmg_v2(handle, ref d1, ref d2, ref x1, ref y1, param);
        }

        public CUBLASStatusv2 cublasSscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx)
        {
            return cublasSscal_v2(handle, n, alpha, x, incx);
        }

        public CUBLASStatusv2 cublasDscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx)
        {
            return cublasDscal_v2(handle, n, alpha, x, incx);
        }

        public CUBLASStatusv2 cublasCscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx)
        {
            return cublasCscal_v2(handle, n, alpha, x, incx);
        }

        public CUBLASStatusv2 cublasCsscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx)
        {
            return cublasCsscal_v2(handle, n, alpha, x, incx);
        }

        public CUBLASStatusv2 cublasZscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx)
        {
            return cublasZscal_v2(handle, n, alpha, x, incx);
        }

        public CUBLASStatusv2 cublasZdscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx)
        {
            return cublasZdscal_v2(handle, n, alpha, x, incx);
        }

        public CUBLASStatusv2 cublasSswap(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasSswap_v2(handle, n, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasDswap(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasDswap_v2(handle, n, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasCswap(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasCswap_v2(handle, n, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasZswap(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasZswap_v2(handle, n, x, incx, y, incy);
        }

        public string GetDllName()
        {
            return CUBLAS_DLL_NAME;
        }
    }

    public class CUBLASDriver64 : ICUBLASDriverv2
    {
#if LINUX
        public const string CUBLAS_DLL_NAME = "libcublas";
#else
        public const string CUBLAS_DLL_NAME = "cublas64_70";
#endif

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCreate_v2(ref cublasHandle handle);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDestroy_v2(cublasHandle handle);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasGetVersion_v2(cublasHandle handle, ref int version);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSetStream_v2(cublasHandle handle, cudaStream streamId);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasGetStream_v2(cublasHandle handle, ref cudaStream streamId);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasGetPointerMode_v2(cublasHandle handle, ref CUBLASPointerMode mode);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSetPointerMode_v2(cublasHandle handle, CUBLASPointerMode mode);

        #region Level 1
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasIcamax_v2(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasIdamax_v2(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasIsamax_v2(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasIzamax_v2(cublasHandle handle, int n, IntPtr x, int incx, ref int result);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasIcamin_v2(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasIdamin_v2(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasIsamin_v2(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasIzamin_v2(cublasHandle handle, int n, IntPtr x, int incx, ref int result);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSasum_v2(cublasHandle handle, int n, IntPtr x, int incx, ref float result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDasum_v2(cublasHandle handle, int n, IntPtr x, int incx, ref double result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasScasum_v2(cublasHandle handle, int n, IntPtr x, int incx, ref float result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDzasum_v2(cublasHandle handle, int n, IntPtr x, int incx, ref double result);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSaxpy_v2(cublasHandle handle, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDaxpy_v2(cublasHandle handle, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCaxpy_v2(cublasHandle handle, int n, ref cuFloatComplex alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZaxpy_v2(cublasHandle handle, int n, ref cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSaxpy_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDaxpy_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCaxpy_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZaxpy_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasScopy_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDcopy_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCcopy_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZcopy_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSdot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDdot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCdotu_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuFloatComplex result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCdotc_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuFloatComplex result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdotu_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuDoubleComplex result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdotc_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuDoubleComplex result);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSnrm2_v2(cublasHandle handle, int n, IntPtr x, int incx, ref float result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDnrm2_v2(cublasHandle handle, int n, IntPtr x, int incx, ref double result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasScnrm2_v2(cublasHandle handle, int n, IntPtr x, int incx, ref float result);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDznrm2_v2(cublasHandle handle, int n, IntPtr x, int incx, ref double result);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSrot_v2(cublasHandle handle,  int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSrot_v2(cublasHandle handle,  int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref float s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDrot_v2(cublasHandle handle,  int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDrot_v2(cublasHandle handle,  int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref double s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCrot_v2(cublasHandle handle,  int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCrot_v2(cublasHandle handle,  int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref cuFloatComplex s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCsrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCsrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref float s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZrot_v2(cublasHandle handle,  int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZrot_v2(cublasHandle handle,  int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref cuDoubleComplex s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref double s);


        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSrotg_v2(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSrotg_v2(cublasHandle handle, ref float a, ref float b, ref float c, ref float s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDrotg_v2(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDrotg_v2(cublasHandle handle, ref double a, ref double b, ref double c, ref double s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCrotg_v2(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCrotg_v2(cublasHandle handle, ref cuFloatComplex a, ref cuFloatComplex b, ref float c, ref cuFloatComplex s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZrotg_v2(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZrotg_v2(cublasHandle handle, ref cuDoubleComplex a, ref cuDoubleComplex b, ref double c, ref cuDoubleComplex s);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSrotm_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr param);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDrotm_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr param);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSrotmg_v2(cublasHandle handle, IntPtr d1, IntPtr d2, IntPtr x1, IntPtr y1, IntPtr param);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDrotmg_v2(cublasHandle handle, IntPtr d1, IntPtr d2, IntPtr x1, IntPtr y1, IntPtr param);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSrotmg_v2(cublasHandle handle, ref float d1, ref float d2, ref float x1, ref float y1, IntPtr param);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDrotmg_v2(cublasHandle handle, ref double d1, ref double d2, ref double x1, ref double y1, IntPtr param);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSscal_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDscal_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCscal_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCsscal_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZscal_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdscal_v2(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);

        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasSswap_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasDswap_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCswap_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZswap_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);

        #endregion


        public CUBLASStatusv2 cublasCreate(ref cublasHandle handle)
        {
            return cublasCreate_v2(ref handle);
        }

        public CUBLASStatusv2 cublasDestroy(cublasHandle handle)
        {
            return cublasDestroy_v2(handle);
        }

        public CUBLASStatusv2 cublasGetVersion(cublasHandle handle, ref int version)
        {
            return cublasGetVersion_v2(handle, ref version);
        }

        public CUBLASStatusv2 cublasSetStream(cublasHandle handle, cudaStream streamId)
        {
            return cublasSetStream_v2(handle, streamId);
        }

        public CUBLASStatusv2 cublasGetStream(cublasHandle handle, ref cudaStream streamId)
        {
            return cublasGetStream_v2(handle, ref streamId);
        }

        public CUBLASStatusv2 cublasGetPointerMode(cublasHandle handle, ref CUBLASPointerMode mode)
        {
            return cublasGetPointerMode_v2(handle, ref mode);
        }

        public CUBLASStatusv2 cublasSetPointerMode(cublasHandle handle, CUBLASPointerMode mode)
        {
            return cublasSetPointerMode_v2(handle, mode);
        }

        public CUBLASStatusv2 cublasIcamax(cublasHandle handle, int n, IntPtr x, int incx, ref int result)
        {
            return cublasIcamax_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasIdamax(cublasHandle handle, int n, IntPtr x, int incx, ref int result)
        {
            return cublasIdamax_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasIsamax(cublasHandle handle, int n, IntPtr x, int incx, ref int result)
        {
            return cublasIsamax_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasIzamax(cublasHandle handle, int n, IntPtr x, int incx, ref int result)
        {
            return cublasIzamax_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasIcamin(cublasHandle handle, int n, IntPtr x, int incx, ref int result)
        {
            return cublasIcamin_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasIdamin(cublasHandle handle, int n, IntPtr x, int incx, ref int result)
        {
            return cublasIdamin_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasIsamin(cublasHandle handle, int n, IntPtr x, int incx, ref int result)
        {
            return cublasIsamin_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasIzamin(cublasHandle handle, int n, IntPtr x, int incx, ref int result)
        {
            return cublasIzamin_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasSasum(cublasHandle handle, int n, IntPtr x, int incx, ref float result)
        {
            return cublasSasum_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasDasum(cublasHandle handle, int n, IntPtr x, int incx, ref double result)
        {
            return cublasDasum_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasScasum(cublasHandle handle, int n, IntPtr x, int incx, ref float result)
        {
            return cublasScasum_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasDzasum(cublasHandle handle, int n, IntPtr x, int incx, ref double result)
        {
            return cublasDzasum_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasSaxpy(cublasHandle handle, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasSaxpy_v2(handle, n, ref alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasDaxpy(cublasHandle handle, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasDaxpy_v2(handle, n, ref alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasCaxpy(cublasHandle handle, int n, ref cuFloatComplex alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasCaxpy_v2(handle, n, ref alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasZaxpy(cublasHandle handle, int n, ref cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasZaxpy_v2(handle, n, ref alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasSaxpy(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasDaxpy(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasDaxpy_v2(handle, n, alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasCaxpy(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasCaxpy_v2(handle, n, alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasZaxpy(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasZaxpy_v2(handle, n, alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasScopy(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasScopy_v2(handle, n, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasDcopy(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasDcopy_v2(handle, n, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasCcopy(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasCcopy_v2(handle, n, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasZcopy(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasZcopy_v2(handle, n, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasSdot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float result)
        {
            return cublasSdot_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasDdot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double result)
        {
            return cublasDdot_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasCdotu(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuFloatComplex result)
        {
            return cublasCdotu_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasCdotc(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuFloatComplex result)
        {
            return cublasCdotc_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasZdotu(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuDoubleComplex result)
        {
            return cublasZdotu_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasZdotc(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuDoubleComplex result)
        {
            return cublasZdotc_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasSnrm2(cublasHandle handle, int n, IntPtr x, int incx, ref float result)
        {
            return cublasSnrm2_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasDnrm2(cublasHandle handle, int n, IntPtr x, int incx, ref double result)
        {
            return cublasDnrm2_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasScnrm2(cublasHandle handle, int n, IntPtr x, int incx, ref float result)
        {
            return cublasScnrm2_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasDznrm2(cublasHandle handle, int n, IntPtr x, int incx, ref double result)
        {
            return cublasDznrm2_v2(handle, n, x, incx, ref result);
        }

        public CUBLASStatusv2 cublasSrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s)
        {
            return cublasSrot_v2(handle, n, x, incx, y, incy, c, s);
        }

        public CUBLASStatusv2 cublasSrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref float s)
        {
            return cublasSrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasDrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s)
        {
            return cublasDrot_v2(handle, n, x, incx, y, incy, c, s);
        }

        public CUBLASStatusv2 cublasDrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref double s)
        {
            return cublasDrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasCrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s)
        {
            return cublasCrot_v2(handle, n, x, incx, y, incy, c, s);
        }

        public CUBLASStatusv2 cublasCrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref cuFloatComplex s)
        {
            return cublasCrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasCsrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s)
        {
            return cublasCsrot_v2(handle, n, x, incx, y, incy, c, s);
        }

        public CUBLASStatusv2 cublasCsrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref float s)
        {
            return cublasCsrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasZrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s)
        {
            return cublasZrot_v2(handle, n, x, incx, y, incy, c, s);
        }

        public CUBLASStatusv2 cublasZrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref cuDoubleComplex s)
        {
            return cublasZrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasZdrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s)
        {
            return cublasZdrot_v2(handle, n, x, incx, y, incy, c, s);
        }

        public CUBLASStatusv2 cublasZdrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref double s)
        {
            return cublasZdrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasSrotg(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s)
        {
            return cublasSrotg_v2(handle, a, b, c, s);
        }

        public CUBLASStatusv2 cublasSrotg(cublasHandle handle, ref float a, ref float b, ref float c, ref float s)
        {
            return cublasSrotg_v2(handle, ref a, ref b, ref c, ref s);
        }

        public CUBLASStatusv2 cublasDrotg(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s)
        {
            return cublasDrotg_v2(handle, a, b, c, s);
        }

        public CUBLASStatusv2 cublasDrotg(cublasHandle handle, ref double a, ref double b, ref double c, ref double s)
        {
            return cublasDrotg_v2(handle, ref a, ref b, ref c, ref s);
        }

        public CUBLASStatusv2 cublasCrotg(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s)
        {
            return cublasCrotg_v2(handle, a, b, c, s);
        }

        public CUBLASStatusv2 cublasCrotg(cublasHandle handle, ref cuFloatComplex a, ref cuFloatComplex b, ref float c, ref cuFloatComplex s)
        {
            return cublasCrotg_v2(handle, ref a, ref b, ref c, ref s);
        }

        public CUBLASStatusv2 cublasZrotg(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s)
        {
            return cublasZrotg_v2(handle, a, b, c, s);
        }

        public CUBLASStatusv2 cublasZrotg(cublasHandle handle, ref cuDoubleComplex a, ref cuDoubleComplex b, ref double c, ref cuDoubleComplex s)
        {
            return cublasZrotg_v2(handle, ref a, ref b, ref c, ref s);
        }

        public CUBLASStatusv2 cublasSrotm(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr param)
        {
            return cublasSrotm_v2(handle, n, x, incx, y, incy, param);
        }

        public CUBLASStatusv2 cublasDrotm(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr param)
        {
            return cublasDrotm_v2(handle, n, x, incx, y, incy, param);
        }

        public CUBLASStatusv2 cublasSrotmg(cublasHandle handle, IntPtr d1, IntPtr d2, IntPtr x1, IntPtr y1, IntPtr param)
        {
            return cublasSrotmg_v2(handle, d1, d2, x1, y1, param);
        }

        public CUBLASStatusv2 cublasDrotmg(cublasHandle handle, IntPtr d1, IntPtr d2, IntPtr x1, IntPtr y1, IntPtr param)
        {
            return cublasDrotmg_v2(handle, d1, d2, x1, y1, param);
        }

        public CUBLASStatusv2 cublasSrotmg(cublasHandle handle, ref float d1, ref float d2, ref float x1, ref float y1, IntPtr param)
        {
            return cublasSrotmg_v2(handle, ref d1, ref d2, ref x1, ref y1, param);
        }

        public CUBLASStatusv2 cublasDrotmg(cublasHandle handle, ref double d1, ref double d2, ref double x1, ref double y1, IntPtr param)
        {
            return cublasDrotmg_v2(handle, ref d1, ref d2, ref x1, ref y1, param);
        }

        public CUBLASStatusv2 cublasSscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx)
        {
            return cublasSscal_v2(handle, n, alpha, x, incx);
        }

        public CUBLASStatusv2 cublasDscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx)
        {
            return cublasDscal_v2(handle, n, alpha, x, incx);
        }

        public CUBLASStatusv2 cublasCscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx)
        {
            return cublasCscal_v2(handle, n, alpha, x, incx);
        }

        public CUBLASStatusv2 cublasCsscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx)
        {
            return cublasCsscal_v2(handle, n, alpha, x, incx);
        }

        public CUBLASStatusv2 cublasZscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx)
        {
            return cublasZscal_v2(handle, n, alpha, x, incx);
        }

        public CUBLASStatusv2 cublasZdscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx)
        {
            return cublasZdscal_v2(handle, n, alpha, x, incx);
        }

        public CUBLASStatusv2 cublasSswap(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasSswap_v2(handle, n, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasDswap(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasDswap_v2(handle, n, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasCswap(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasCswap_v2(handle, n, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasZswap(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasZswap_v2(handle, n, x, incx, y, incy);
        }

        public string GetDllName()
        {
            return CUBLAS_DLL_NAME;
        }
    }
}
