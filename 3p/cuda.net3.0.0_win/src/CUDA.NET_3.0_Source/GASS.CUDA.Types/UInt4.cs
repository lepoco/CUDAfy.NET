namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct UInt4
    {
        public uint x;
        public uint y;
        public uint z;
        public uint w;
    }
}

