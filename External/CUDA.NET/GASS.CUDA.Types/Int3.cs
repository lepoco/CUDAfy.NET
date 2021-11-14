namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct Int3
    {
        public int x;
        public int y;
        public int z;
    }
}

