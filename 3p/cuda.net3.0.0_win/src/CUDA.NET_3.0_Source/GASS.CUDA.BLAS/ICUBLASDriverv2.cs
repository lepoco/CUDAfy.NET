namespace GASS.CUDA.BLAS
{
    using GASS.CUDA.Types;
    using GASS.CUDA.BLAS.Types;
    using System;
    using System.Runtime.InteropServices;    
    public interface ICUBLASDriverv2
    {
        string GetDllName();

        CUBLASStatusv2 cublasCreate(ref cublasHandle handle);

        CUBLASStatusv2 cublasDestroy(cublasHandle handle);
        
        CUBLASStatusv2 cublasGetVersion(cublasHandle handle, ref int version);
        
        CUBLASStatusv2 cublasSetStream(cublasHandle handle, cudaStream streamId);
        
        CUBLASStatusv2 cublasGetStream(cublasHandle handle, ref cudaStream streamId);
        
        CUBLASStatusv2 cublasGetPointerMode(cublasHandle handle, ref CUBLASPointerMode mode);
        
        CUBLASStatusv2 cublasSetPointerMode(cublasHandle handle, CUBLASPointerMode mode);
        
        CUBLASStatusv2 cublasIcamax(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        
        CUBLASStatusv2 cublasIdamax(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        
        CUBLASStatusv2 cublasIsamax(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        
        CUBLASStatusv2 cublasIzamax(cublasHandle handle, int n, IntPtr x, int incx, ref int result);

        
        CUBLASStatusv2 cublasIcamin(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        
        CUBLASStatusv2 cublasIdamin(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        
        CUBLASStatusv2 cublasIsamin(cublasHandle handle, int n, IntPtr x, int incx, ref int result);
        
        CUBLASStatusv2 cublasIzamin(cublasHandle handle, int n, IntPtr x, int incx, ref int result);

        
        CUBLASStatusv2 cublasSasum(cublasHandle handle, int n, IntPtr x, int incx, ref float result);
        
        CUBLASStatusv2 cublasDasum(cublasHandle handle, int n, IntPtr x, int incx, ref double result);
        
        CUBLASStatusv2 cublasScasum(cublasHandle handle, int n, IntPtr x, int incx, ref float result);
        
        CUBLASStatusv2 cublasDzasum(cublasHandle handle, int n, IntPtr x, int incx, ref double result);

        
        CUBLASStatusv2 cublasSaxpy(cublasHandle handle, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy);
        
        CUBLASStatusv2 cublasDaxpy(cublasHandle handle, int n, ref double alpha, IntPtr x, int incx, IntPtr y, int incy);
        
        CUBLASStatusv2 cublasCaxpy(cublasHandle handle, int n, ref cuFloatComplex alpha, IntPtr x, int incx, IntPtr y, int incy);
        
        CUBLASStatusv2 cublasZaxpy(cublasHandle handle, int n, ref cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy);

        
        CUBLASStatusv2 cublasSaxpy(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
        
        CUBLASStatusv2 cublasDaxpy(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
        
        CUBLASStatusv2 cublasCaxpy(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
        
        CUBLASStatusv2 cublasZaxpy(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);

        
        CUBLASStatusv2 cublasScopy(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        
        CUBLASStatusv2 cublasDcopy(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        
        CUBLASStatusv2 cublasCcopy(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        
        CUBLASStatusv2 cublasZcopy(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);

        
        CUBLASStatusv2 cublasSdot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float result);
        
        CUBLASStatusv2 cublasDdot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double result);
        
        CUBLASStatusv2 cublasCdotu(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuFloatComplex result);
        
        CUBLASStatusv2 cublasCdotc(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuFloatComplex result);
        
        CUBLASStatusv2 cublasZdotu(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuDoubleComplex result);
        
        CUBLASStatusv2 cublasZdotc(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref cuDoubleComplex result);

        
        CUBLASStatusv2 cublasSnrm2(cublasHandle handle, int n, IntPtr x, int incx, ref float result);
        
        CUBLASStatusv2 cublasDnrm2(cublasHandle handle, int n, IntPtr x, int incx, ref double result);
        
        CUBLASStatusv2 cublasScnrm2(cublasHandle handle, int n, IntPtr x, int incx, ref float result);
        
        CUBLASStatusv2 cublasDznrm2(cublasHandle handle, int n, IntPtr x, int incx, ref double result);

        
        CUBLASStatusv2 cublasSrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        
        CUBLASStatusv2 cublasSrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref float s);
        
        CUBLASStatusv2 cublasDrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        
        CUBLASStatusv2 cublasDrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref double s);
        
        CUBLASStatusv2 cublasCrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        
        CUBLASStatusv2 cublasCrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref cuFloatComplex s);
        
        CUBLASStatusv2 cublasCsrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        
        CUBLASStatusv2 cublasCsrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref float s);
        
        CUBLASStatusv2 cublasZrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        
        CUBLASStatusv2 cublasZrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref cuDoubleComplex s);
        
        CUBLASStatusv2 cublasZdrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        
        CUBLASStatusv2 cublasZdrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref double s);


        
        CUBLASStatusv2 cublasSrotg(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);
        
        CUBLASStatusv2 cublasSrotg(cublasHandle handle, ref float a, ref float b, ref float c, ref float s);
        
        CUBLASStatusv2 cublasDrotg(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);
        
        CUBLASStatusv2 cublasDrotg(cublasHandle handle, ref double a, ref double b, ref double c, ref double s);
        
        CUBLASStatusv2 cublasCrotg(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);
        
        CUBLASStatusv2 cublasCrotg(cublasHandle handle, ref cuFloatComplex a, ref cuFloatComplex b, ref float c, ref cuFloatComplex s);
        
        CUBLASStatusv2 cublasZrotg(cublasHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);
        
        CUBLASStatusv2 cublasZrotg(cublasHandle handle, ref cuDoubleComplex a, ref cuDoubleComplex b, ref double c, ref cuDoubleComplex s);

        
        CUBLASStatusv2 cublasSrotm(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr param);
        
        CUBLASStatusv2 cublasDrotm(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr param);

        
        CUBLASStatusv2 cublasSrotmg(cublasHandle handle, IntPtr d1, IntPtr d2, IntPtr x1, IntPtr y1, IntPtr param);
        
        CUBLASStatusv2 cublasDrotmg(cublasHandle handle, IntPtr d1, IntPtr d2, IntPtr x1, IntPtr y1, IntPtr param);
        
        CUBLASStatusv2 cublasSrotmg(cublasHandle handle, ref float d1, ref float d2, ref float x1, ref float y1, IntPtr param);
        
        CUBLASStatusv2 cublasDrotmg(cublasHandle handle, ref double d1, ref double d2, ref double x1, ref double y1, IntPtr param);

        
        CUBLASStatusv2 cublasSscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
        
        CUBLASStatusv2 cublasDscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
        
        CUBLASStatusv2 cublasCscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
        
        CUBLASStatusv2 cublasCsscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
        
        CUBLASStatusv2 cublasZscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
        
        CUBLASStatusv2 cublasZdscal(cublasHandle handle, int n, IntPtr alpha, IntPtr x, int incx);

        
        CUBLASStatusv2 cublasSswap(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        
        CUBLASStatusv2 cublasDswap(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        
        CUBLASStatusv2 cublasCswap(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        
        CUBLASStatusv2 cublasZswap(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);

    }
}
