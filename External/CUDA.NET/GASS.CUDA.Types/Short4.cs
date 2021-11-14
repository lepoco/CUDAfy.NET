namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct Short4
    {
        public short x;
        public short y;
        public short z;
        public short w;
    }
}

