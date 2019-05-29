namespace GASS.CUDA.FFT
{
    using GASS.CUDA;
    using GASS.CUDA.FFT.Types;
    using GASS.CUDA.Types;
    using System;
    using System.Runtime.InteropServices;
    
    public interface ICUFFTDriver
    {
        string GetDllName();
        CUFFTResult cufftDestroy(cufftHandle plan);
        CUFFTResult cufftExecC2C(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata, CUFFTDirection direction);
        CUFFTResult cufftExecC2R(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata);
        CUFFTResult cufftExecD2Z(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata);
        CUFFTResult cufftExecR2C(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata);
        CUFFTResult cufftExecZ2D(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata);
        CUFFTResult cufftExecZ2Z(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata, CUFFTDirection direction);
        CUFFTResult cufftPlan1d(ref cufftHandle plan, int nx, CUFFTType type, int batch);
        CUFFTResult cufftPlan2d(ref cufftHandle plan, int nx, int ny, CUFFTType type);
        CUFFTResult cufftPlan3d(ref cufftHandle plan, int nx, int ny, int nz, CUFFTType type);
        CUFFTResult cufftPlanMany(ref cufftHandle plan, int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, CUFFTType type, int batch);
        CUFFTResult cufftPlanMany(ref cufftHandle plan, int rank, [In, Out] int[] n, [In, Out] int[] inembed, int istride, int idist, [In, Out] int[] onembed, int ostride, int odist, CUFFTType type, int batch);
        CUFFTResult cufftSetStream(cufftHandle p, cudaStream stream);
        CUFFTResult cufftGetVersion(ref int version);
        CUFFTResult cufftSetCompatibilityMode(cufftHandle plan, CUFFTCompatibility mode);
    }
}
