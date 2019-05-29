namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct Long3
    {
        public long x;
        public long y;
        public long z;
    }
}

