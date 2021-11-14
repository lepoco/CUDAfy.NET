namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct CUgraphicsResource
    {
        public IntPtr Pointer;
    }
}

