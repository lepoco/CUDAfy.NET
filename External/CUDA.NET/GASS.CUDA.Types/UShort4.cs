namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct UShort4
    {
        public ushort x;
        public ushort y;
        public ushort z;
        public ushort w;
    }
}

