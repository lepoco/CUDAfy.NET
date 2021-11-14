namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct UChar4
    {
        public byte x;
        public byte y;
        public byte z;
        public byte w;
    }
}

