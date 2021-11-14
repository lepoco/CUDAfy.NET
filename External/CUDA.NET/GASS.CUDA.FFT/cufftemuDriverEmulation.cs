namespace GASS.CUDA.FFT
{
    using GASS.CUDA;
    using GASS.CUDA.FFT.Types;
    using GASS.CUDA.Types;
    using System;
    using System.Runtime.InteropServices;

    public class cufftemuDriverEmulation
    {
        internal const string CUFFT_DLL_NAME = "cufftemu";

        [DllImport("cufftemu")]
        public static extern CUFFTResult cufftDestroy(cufftHandle plan);
        [DllImport("cufftemu")]
        public static extern CUFFTResult cufftExecC2C(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata, CUFFTDirection direction);
        [DllImport("cufftemu")]
        public static extern CUFFTResult cufftExecC2R(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata);
        [DllImport("cufftemu")]
        public static extern CUFFTResult cufftExecD2Z(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata);
        [DllImport("cufftemu")]
        public static extern CUFFTResult cufftExecR2C(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata);
        [DllImport("cufftemu")]
        public static extern CUFFTResult cufftExecZ2D(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata);
        [DllImport("cufftemu")]
        public static extern CUFFTResult cufftExecZ2Z(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata, CUFFTDirection direction);
        [DllImport("cufftemu")]
        public static extern CUFFTResult cufftPlan1d(ref cufftHandle plan, int nx, CUFFTType type, int batch);
        [DllImport("cufftemu")]
        public static extern CUFFTResult cufftPlan2d(ref cufftHandle plan, int nx, int ny, CUFFTType type);
        [DllImport("cufftemu")]
        public static extern CUFFTResult cufftPlan3d(ref cufftHandle plan, int nx, int ny, int nz, CUFFTType type);
        [DllImport("cufftemu")]
        public static extern CUFFTResult cufftPlanMany(ref cufftHandle plan, int rank, [In, Out] int[] n, [In, Out] int[] inembed, int istride, int idist, [In, Out] int[] onembed, int ostride, int odist, CUFFTType type, int batch);
        [DllImport("cufftemu")]
        public static extern CUFFTResult cufftSetStream(cufftHandle p, cudaStream stream);
    }
}

