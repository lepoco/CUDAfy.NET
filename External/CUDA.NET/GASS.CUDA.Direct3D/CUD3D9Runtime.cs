namespace GASS.CUDA.Direct3D
{
    using GASS.CUDA;
    using GASS.Types;
    using System;
    using System.Runtime.InteropServices;

    public class CUD3D9Runtime
    {
        [Obsolete("Function deprecated for use with CUDA 2.0"), DllImport("cudart")]
        public static extern cudaError cudaD3D9Begin(IntPtr pDevice);
        [Obsolete("Function deprecated for use with CUDA 2.0"), DllImport("cudart")]
        public static extern cudaError cudaD3D9End();
        [DllImport("cudart")]
        public static extern cudaError cudaD3D9GetDevice(ref int device, string pszAdapterName);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9GetDirect3DDevice(ref IntPtr ppDxDevice);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9MapResources(uint count, [In] IntPtr[] ppResources);
        [Obsolete("Function deprecated for use with CUDA 2.0"), DllImport("cudart")]
        public static extern cudaError cudaD3D9MapVertexBuffer(ref IntPtr dptr, IntPtr pVB);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9RegisterResource(IntPtr pResource, cudaD3D9RegisterFlags flags);
        [Obsolete("Function deprecated for use with CUDA 2.0"), DllImport("cudart")]
        public static extern cudaError cudaD3D9RegisterVertexBuffer(IntPtr pVB);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9ResourceGetMappedArray(ref cudaArray ppArray, IntPtr pResource, uint face, uint level);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9ResourceGetMappedPitch(ref SizeT pPitch, ref SizeT pSlicePitch, IntPtr pResource, uint face, uint level);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9ResourceGetMappedPitch(ref uint pPitch, ref uint pSlicePitch, IntPtr pResource, uint face, uint level);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9ResourceGetMappedPitch(ref ulong pPitch, ref ulong pSlicePitch, IntPtr pResource, uint face, uint level);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9ResourceGetMappedPointer(ref IntPtr pPointer, IntPtr pResource, uint face, uint level);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9ResourceGetMappedSize(ref SizeT pSize, IntPtr pResource, uint face, uint level);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9ResourceGetMappedSize(ref uint pSize, IntPtr pResource, uint face, uint level);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9ResourceGetMappedSize(ref ulong pSize, IntPtr pResource, uint face, uint level);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9ResourceGetSurfaceDimensions(ref SizeT pWidth, ref SizeT pHeight, ref SizeT pDepth, IntPtr pResource, uint face, uint level);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9ResourceGetSurfaceDimensions(ref uint pWidth, ref uint pHeight, ref uint pDepth, IntPtr pResource, uint face, uint level);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9ResourceGetSurfaceDimensions(ref ulong pWidth, ref ulong pHeight, ref ulong pDepth, IntPtr pResource, uint face, uint level);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9ResourceSetMapFlags(IntPtr pResource, cudaD3D9MapFlags flags);
        [DllImport("cudart")]
        public static extern cudaError cudaD3D9SetDirect3DDevice(IntPtr pDxDevice);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9UnmapResources(uint count, [In] IntPtr[] ppResources);
        [Obsolete("Function deprecated for use with CUDA 2.0"), DllImport("cudart")]
        public static extern cudaError cudaD3D9UnmapVertexBuffer(IntPtr pVB);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D9UnregisterResource(IntPtr pResource);
        [Obsolete("Function deprecated for use with CUDA 2.0"), DllImport("cudart")]
        public static extern cudaError cudaD3D9UnregisterVertexBuffer(IntPtr pVB);
        [DllImport("cudart")]
        public static extern cudaError cudaGraphicsD3D9RegisterResource(ref cudaGraphicsResource resource, IntPtr pD3DResource, uint flags);
    }
}

