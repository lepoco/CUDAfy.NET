namespace GASS.CUDA
{
    using GASS.Types;
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct cudaPos
    {
        public SizeT x;
        public SizeT y;
        public SizeT z;
    }
}

