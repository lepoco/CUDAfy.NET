namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct Char3
    {
        public sbyte x;
        public sbyte y;
        public sbyte z;
    }
}

