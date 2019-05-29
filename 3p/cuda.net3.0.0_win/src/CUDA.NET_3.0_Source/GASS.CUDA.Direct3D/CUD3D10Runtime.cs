namespace GASS.CUDA.Direct3D
{
    using GASS.CUDA;
    using GASS.Types;
    using System;
    using System.Runtime.InteropServices;

    public class CUD3D10Runtime
    {
        [DllImport("cudart")]
        public static extern cudaError cudaD3D10GetDevice(ref int device, IntPtr pAdapter);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D10MapResources(int count, [In] IntPtr[] ppResources);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D10RegisterResource(IntPtr pResource, cudaD3D10RegisterFlags flags);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D10ResourceGetMappedArray(ref cudaArray ppArray, IntPtr pResource, uint subResource);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D10ResourceGetMappedPitch(ref SizeT pPitch, ref SizeT pSlicePitch, IntPtr pResource, uint subResource);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D10ResourceGetMappedPitch(ref uint pPitch, ref uint pSlicePitch, IntPtr pResource, uint subResource);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D10ResourceGetMappedPitch(ref ulong pPitch, ref ulong pSlicePitch, IntPtr pResource, uint subResource);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D10ResourceGetMappedPointer(ref IntPtr pPointer, IntPtr pResource, uint subResource);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D10ResourceGetMappedSize(ref SizeT pSize, IntPtr pResource, uint subResource);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D10ResourceGetMappedSize(ref uint pSize, IntPtr pResource, uint subResource);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D10ResourceGetMappedSize(ref ulong pSize, IntPtr pResource, uint subResource);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D10ResourceGetSurfaceDimensions(ref SizeT pWidth, ref SizeT pHeight, ref SizeT pDepth, IntPtr pResource, uint subResource);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D10ResourceGetSurfaceDimensions(ref uint pWidth, ref uint pHeight, ref uint pDepth, IntPtr pResource, uint subResource);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D10ResourceGetSurfaceDimensions(ref ulong pWidth, ref ulong pHeight, ref ulong pDepth, IntPtr pResource, uint subResource);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D10ResourceSetMapFlags(IntPtr pResource, cudaD3D10MapFlags flags);
        [DllImport("cudart")]
        public static extern cudaError cudaD3D10SetDirect3DDevice(IntPtr pDxDevice);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D10UnmapResources(int count, [In] IntPtr[] ppResources);
        [Obsolete("Function deprecated for use with CUDA 3.0, use new API"), DllImport("cudart")]
        public static extern cudaError cudaD3D10UnregisterResource(IntPtr pResource);
        [DllImport("cudart")]
        public static extern cudaError cudaGraphicsD3D10RegisterResource(ref cudaGraphicsResource resource, IntPtr pD3DResource, uint flags);
    }
}

