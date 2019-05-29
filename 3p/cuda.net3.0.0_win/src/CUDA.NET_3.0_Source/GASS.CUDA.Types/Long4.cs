namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct Long4
    {
        public long x;
        public long y;
        public long z;
        public long w;
    }
}

