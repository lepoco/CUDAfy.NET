namespace GASS.CUDA.Direct3D
{
    using GASS.CUDA;
    using GASS.CUDA.Types;
    using System;
    using System.Runtime.InteropServices;

    public class CUD3D9Driver
    {
        internal const string CUDA2_DEPRECATED_MSG = "Function deprecated for use with CUDA 2.0";
        internal const string CUDA3_DEPRECATED_MSG = "Function deprecated for use with CUDA 3.0, use new API";
#if LINUX
        internal const string CUDA_DLL_NAME = "libcuda";
#else
        internal const string CUDA_DLL_NAME = "nvcuda";
#endif
        [Obsolete("Function deprecated for use with CUDA 2.0"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9Begin(IntPtr pDevice);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9CtxCreate(ref CUcontext pCtx, ref CUdevice pCuDevice, CUCtxFlags Flags, IntPtr pDxDevice);
        [Obsolete("Function deprecated for use with CUDA 2.0"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9End();
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9GetDevice(ref CUdevice pDevice, string pszAdapterName);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9GetDirect3DDevice(ref IntPtr ppDxDevice);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9MapResources(uint count, [In] IntPtr[] ppResource);
        [Obsolete("Function deprecated for use with CUDA 2.0"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9MapVertexBuffer(ref CUdeviceptr pDevPtr, ref uint pSize, IntPtr pVB);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9RegisterResource(IntPtr pResource, CUD3D9RegisterFlags Flags);
        [Obsolete("Function deprecated for use with CUDA 2.0"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9RegisterVertexBuffer(IntPtr pVB);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9ResourceGetMappedArray(ref CUarray pArray, IntPtr pResource, uint Face, uint Level);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9ResourceGetMappedPitch(ref uint pPitch, ref uint pPitchSlice, IntPtr pResource, uint Face, uint Level);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9ResourceGetMappedPointer(ref CUdeviceptr pDevPtr, IntPtr pResource, uint Face, uint Level);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9ResourceGetMappedSize(ref uint pSize, IntPtr pResource, uint Face, uint Level);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9ResourceGetSurfaceDimensions(ref uint pWidth, ref uint pHeight, ref uint pDepth, IntPtr pResource, uint Face, uint Level);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9ResourceSetMapFlags(IntPtr pResource, CUD3D9MapFlags Flags);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9UnmapResources(uint count, [In] IntPtr[] ppResource);
        [Obsolete("Function deprecated for use with CUDA 2.0"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9UnmapVertexBuffer(IntPtr pVB);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9UnregisterResource(IntPtr pResource);
        [Obsolete("Function deprecated for use with CUDA 2.0"), DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuD3D9UnregisterVertexBuffer(IntPtr pVB);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuGraphicsD3D9RegisterResource(ref CUgraphicsResource pCudaResource, IntPtr pD3DResource, uint Flags);
    }
}

