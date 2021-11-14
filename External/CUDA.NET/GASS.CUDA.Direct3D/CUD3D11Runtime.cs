namespace GASS.CUDA.Direct3D
{
    using GASS.CUDA;
    using System;
    using System.Runtime.InteropServices;

    public class CUD3D11Runtime
    {
#if LINUX
        internal const string CUDA_DLL_NAME = "libcuda";
#else
        internal const string CUDA_DLL_NAME = "nvcuda";
#endif
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cudaD3D11GetDevice(ref int device, IntPtr pAdapter);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cudaD3D11SetDirect3DDevice(IntPtr pD3DDevice);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cudaGraphicsD3D11RegisterResource(ref cudaGraphicsResource resource, IntPtr pD3DResource, uint flags);
    }
}

