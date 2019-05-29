namespace GASS.CUDA.Direct3D
{
    using GASS.CUDA;
    using GASS.CUDA.Types;
    using System;
    using System.Runtime.InteropServices;

    public class CUD3D10Driver
    {
        internal const string CUDA3_DEPRECATED_MSG = "Function deprecated for use with CUDA 3.0, use new API";
#if LINUX
        internal const string CUDA_DLL_NAME = "libcuda";
#else
        internal const string CUDA_DLL_NAME = "nvcuda";
#endif
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D10CtxCreate(ref CUcontext pCtx, ref CUdevice pCuDevice, uint Flags, IntPtr pDxDevice);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D10GetDevice(ref CUdevice pDevice, IntPtr pAdapter);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D10MapResources(uint count, [In] IntPtr[] ppResources);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D10RegisterResource(IntPtr pResource, CUD3D10RegisterFlags Flags);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D10ResourceGetMappedArray(ref CUarray pArray, IntPtr pResource, uint SubResource);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D10ResourceGetMappedPitch(ref uint pPitch, ref uint pPitchSlice, IntPtr pResource, uint SubResource);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D10ResourceGetMappedPointer(ref CUdeviceptr pDevPtr, IntPtr pResource, uint SubResource);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D10ResourceGetMappedSize(ref uint pSize, IntPtr pResource, uint SubResource);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D10ResourceGetSurfaceDimensions(ref uint pWidth, ref uint pHeight, ref uint pDepth, IntPtr pResource, uint SubResource);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D10ResourceSetMapFlags(IntPtr pResource, CUD3D10MapFlags Flags);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D10UnmapResources(uint count, [In] IntPtr[] ppResources);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D10UnregisterResource(IntPtr pResource);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuGraphicsD3D10RegisterResource(ref CUgraphicsResource pCudaResource, IntPtr pD3DResource, uint Flags);
    }
}

