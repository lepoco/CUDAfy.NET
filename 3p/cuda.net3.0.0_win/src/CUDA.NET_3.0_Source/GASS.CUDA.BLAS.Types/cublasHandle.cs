namespace GASS.CUDA.BLAS.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct cublasHandle
    {
        public ulong handle;
    }
}

