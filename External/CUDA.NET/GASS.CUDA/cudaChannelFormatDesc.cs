namespace GASS.CUDA
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct cudaChannelFormatDesc
    {
        public int x;
        public int y;
        public int z;
        public int w;
        public cudaChannelFormatKind f;
    }
}

