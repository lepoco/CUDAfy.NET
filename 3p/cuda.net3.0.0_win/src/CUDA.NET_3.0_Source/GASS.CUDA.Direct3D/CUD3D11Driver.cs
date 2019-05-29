namespace GASS.CUDA.Direct3D
{
    using GASS.CUDA;
    using GASS.CUDA.Types;
    using System;
    using System.Runtime.InteropServices;

    public class CUD3D11Driver
    {
#if LINUX
        internal const string CUDA_DLL_NAME = "libcuda";
#else
        internal const string CUDA_DLL_NAME = "nvcuda";
#endif
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D11CtxCreate(ref CUcontext pCtx, ref CUdevice pCudaDevice, uint Flags, IntPtr pD3DDevice);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D11GetDevice(ref CUdevice pCudaDevice, IntPtr pAdapter);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuGraphicsD3D11RegisterResource(ref CUgraphicsResource pCudaResource, IntPtr pD3DResource, uint Flags);
    }
}

