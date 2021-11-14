namespace GASS.CUDA
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct cudaMemcpy3DParms
    {
        public cudaArray srcArray;
        public cudaPos srcPos;
        public cudaPitchedPtr srcPtr;
        public cudaArray dstArray;
        public cudaPos dstPos;
        public cudaPitchedPtr dstPtr;
        public cudaExtent extent;
        public cudaMemcpyKind kind;
    }
}

