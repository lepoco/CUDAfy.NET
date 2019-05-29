namespace GASS.CUDA.FFT
{
    using GASS.CUDA;
    using GASS.CUDA.FFT.Types;
    using GASS.CUDA.Types;
    using System;
    using System.Runtime.InteropServices;

    public class CUFFTDriver32 : ICUFFTDriver
    {
#if LINUX
        internal const string CUFFT_DLL_NAME = "libcufft";
#else
        internal const string CUFFT_DLL_NAME = "cufft32_70";
#endif
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftDestroy")]
        private static extern CUFFTResult cufftDestroy_ext(cufftHandle plan);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftExecC2C")]
        private static extern CUFFTResult cufftExecC2C_ext(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata, CUFFTDirection direction);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftExecC2R")]
        private static extern CUFFTResult cufftExecC2R_ext(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftExecD2Z")]
        private static extern CUFFTResult cufftExecD2Z_ext(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftExecR2C")]
        private static extern CUFFTResult cufftExecR2C_ext(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftExecZ2D")]
        private static extern CUFFTResult cufftExecZ2D_ext(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftExecZ2Z")]
        private static extern CUFFTResult cufftExecZ2Z_ext(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata, CUFFTDirection direction);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftPlan1d")]
        private static extern CUFFTResult cufftPlan1d_ext(ref cufftHandle plan, int nx, CUFFTType type, int batch);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftPlan2d")]
        private static extern CUFFTResult cufftPlan2d_ext(ref cufftHandle plan, int nx, int ny, CUFFTType type);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftPlan3d")]
        private static extern CUFFTResult cufftPlan3d_ext(ref cufftHandle plan, int nx, int ny, int nz, CUFFTType type);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftPlanMany")]
        private static extern CUFFTResult cufftPlanMany_ext(ref cufftHandle plan, int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, CUFFTType type, int batch);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftPlanMany")]
        private static extern CUFFTResult cufftPlanMany_ext(ref cufftHandle plan, int rank, [In, Out] int[] n, [In, Out] int[] inembed, int istride, int idist, [In, Out] int[] onembed, int ostride, int odist, CUFFTType type, int batch);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftSetStream")]
        private static extern CUFFTResult cufftSetStream_ext(cufftHandle p, cudaStream stream);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftGetVersion")]
        private static extern CUFFTResult cufftGetVersion_ext(ref int version);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftSetCompatibilityMode")]
        private static extern CUFFTResult cufftSetCompatibilityMode_ext(cufftHandle plan, CUFFTCompatibility mode);

        public CUFFTResult cufftDestroy(cufftHandle plan)
        {
            return cufftDestroy_ext(plan);
        }

        public CUFFTResult cufftExecC2C(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata, CUFFTDirection direction)
        {
            return cufftExecC2C_ext(plan, idata, odata, direction);
        }

        public CUFFTResult cufftExecC2R(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata)
        {
            return cufftExecC2R_ext(plan, idata, odata);
        }

        public CUFFTResult cufftExecD2Z(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata)
        {
            return cufftExecD2Z_ext(plan, idata, odata);
        }

        public CUFFTResult cufftExecR2C(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata)
        {
            return cufftExecR2C_ext(plan, idata, odata);
        }

        public CUFFTResult cufftExecZ2D(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata)
        {
            return cufftExecZ2D_ext(plan, idata, odata);
        }

        public CUFFTResult cufftExecZ2Z(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata, CUFFTDirection direction)
        {
            return cufftExecZ2Z_ext(plan, idata, odata, direction);
        }

        public CUFFTResult cufftPlan1d(ref cufftHandle plan, int nx, CUFFTType type, int batch)
        {
            return cufftPlan1d_ext(ref plan, nx, type, batch);
        }

        public CUFFTResult cufftPlan2d(ref cufftHandle plan, int nx, int ny, CUFFTType type)
        {
            return cufftPlan2d_ext(ref plan, nx, ny, type);
        }

        public CUFFTResult cufftPlan3d(ref cufftHandle plan, int nx, int ny, int nz, CUFFTType type)
        {
            return cufftPlan3d_ext(ref plan, nx, ny, nz, type);
        }

        public CUFFTResult cufftPlanMany(ref cufftHandle plan, int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, CUFFTType type, int batch)
        {
            return cufftPlanMany_ext(ref plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
        }

        public CUFFTResult cufftPlanMany(ref cufftHandle plan, int rank, int[] n, int[] inembed, int istride, int idist, int[] onembed, int ostride, int odist, CUFFTType type, int batch)
        {
            return cufftPlanMany_ext(ref plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
        }

        public CUFFTResult cufftSetStream(cufftHandle plan, cudaStream stream)
        {
            return cufftSetStream_ext(plan, stream);
        }

        public CUFFTResult cufftGetVersion(ref int version)
        {
            return cufftGetVersion_ext(ref version);
        }

        public CUFFTResult cufftSetCompatibilityMode(cufftHandle plan, CUFFTCompatibility mode)
        {
            return cufftSetCompatibilityMode_ext(plan, mode);
        }

        public string GetDllName()
        {
            return CUFFT_DLL_NAME;
        }
    }

    public class CUFFTDriver64 : ICUFFTDriver
    {
#if LINUX
        internal const string CUFFT_DLL_NAME = "libcufft";
#else
        internal const string CUFFT_DLL_NAME = "cufft64_70";
#endif

        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftDestroy")]
        private static extern CUFFTResult cufftDestroy_ext(cufftHandle plan);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftExecC2C")]
        private static extern CUFFTResult cufftExecC2C_ext(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata, CUFFTDirection direction);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftExecC2R")]
        private static extern CUFFTResult cufftExecC2R_ext(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftExecD2Z")]
        private static extern CUFFTResult cufftExecD2Z_ext(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftExecR2C")]
        private static extern CUFFTResult cufftExecR2C_ext(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftExecZ2D")]
        private static extern CUFFTResult cufftExecZ2D_ext(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftExecZ2Z")]
        private static extern CUFFTResult cufftExecZ2Z_ext(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata, CUFFTDirection direction);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftPlan1d")]
        private static extern CUFFTResult cufftPlan1d_ext(ref cufftHandle plan, int nx, CUFFTType type, int batch);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftPlan2d")]
        private static extern CUFFTResult cufftPlan2d_ext(ref cufftHandle plan, int nx, int ny, CUFFTType type);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftPlan3d")]
        private static extern CUFFTResult cufftPlan3d_ext(ref cufftHandle plan, int nx, int ny, int nz, CUFFTType type);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftPlanMany")]
        private static extern CUFFTResult cufftPlanMany_ext(ref cufftHandle plan, int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, CUFFTType type, int batch);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftPlanMany")]
        private static extern CUFFTResult cufftPlanMany_ext(ref cufftHandle plan, int rank, [In, Out] int[] n, [In, Out] int[] inembed, int istride, int idist, [In, Out] int[] onembed, int ostride, int odist, CUFFTType type, int batch);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftSetStream")]
        private static extern CUFFTResult cufftSetStream_ext(cufftHandle p, cudaStream stream);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftGetVersion")]
        private static extern CUFFTResult cufftGetVersion_ext(ref int version);
        [DllImport(CUFFT_DLL_NAME, EntryPoint = "cufftSetCompatibilityMode")]
        private static extern CUFFTResult cufftSetCompatibilityMode_ext(cufftHandle plan, CUFFTCompatibility mode);

        public CUFFTResult cufftDestroy(cufftHandle plan)
        {
            return cufftDestroy_ext(plan);
        }

        public CUFFTResult cufftExecC2C(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata, CUFFTDirection direction)
        {
            return cufftExecC2C_ext(plan, idata, odata, direction);
        }

        public CUFFTResult cufftExecC2R(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata)
        {
            return cufftExecC2R_ext(plan, idata, odata);
        }

        public CUFFTResult cufftExecD2Z(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata)
        {
            return cufftExecD2Z_ext(plan, idata, odata);
        }

        public CUFFTResult cufftExecR2C(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata)
        {
            return cufftExecR2C_ext(plan, idata, odata);
        }

        public CUFFTResult cufftExecZ2D(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata)
        {
            return cufftExecZ2D_ext(plan, idata, odata);
        }

        public CUFFTResult cufftExecZ2Z(cufftHandle plan, CUdeviceptr idata, CUdeviceptr odata, CUFFTDirection direction)
        {
            return cufftExecZ2Z_ext(plan, idata, odata, direction);
        }

        public CUFFTResult cufftPlan1d(ref cufftHandle plan, int nx, CUFFTType type, int batch)
        {
            return cufftPlan1d_ext(ref plan, nx, type, batch);
        }

        public CUFFTResult cufftPlan2d(ref cufftHandle plan, int nx, int ny, CUFFTType type)
        {
            return cufftPlan2d_ext(ref plan, nx, ny, type);
        }

        public CUFFTResult cufftPlan3d(ref cufftHandle plan, int nx, int ny, int nz, CUFFTType type)
        {
            return cufftPlan3d_ext(ref plan, nx, ny, nz, type);
        }

        public CUFFTResult cufftPlanMany(ref cufftHandle plan, int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, CUFFTType type, int batch)
        {
            return cufftPlanMany_ext(ref plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
        }

        public CUFFTResult cufftPlanMany(ref cufftHandle plan, int rank, int[] n, int[] inembed, int istride, int idist, int[] onembed, int ostride, int odist, CUFFTType type, int batch)
        {
            return cufftPlanMany_ext(ref plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
        }

        public CUFFTResult cufftSetStream(cufftHandle plan, cudaStream stream)
        {
            return cufftSetStream_ext(plan, stream);
        }

        public CUFFTResult cufftGetVersion(ref int version)
        {
            return cufftGetVersion_ext(ref version);
        }

        public CUFFTResult cufftSetCompatibilityMode(cufftHandle plan, CUFFTCompatibility mode)
        {
            return cufftSetCompatibilityMode_ext(plan, mode);
        }

        public string GetDllName()
        {
            return CUFFT_DLL_NAME;
        }
    }
}

