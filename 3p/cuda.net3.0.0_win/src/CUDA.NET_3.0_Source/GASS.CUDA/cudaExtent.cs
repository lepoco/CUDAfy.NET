namespace GASS.CUDA
{
    using GASS.Types;
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct cudaExtent
    {
        public SizeT width;
        public SizeT height;
        public SizeT depth;
    }
}

