namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct Short3
    {
        public short x;
        public short y;
        public short z;
    }
}

