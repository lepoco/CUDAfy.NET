namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct ULong3
    {
        public ulong x;
        public ulong y;
        public ulong z;
    }
}

