namespace GASS.CUDA
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDAArray3DDescriptor
    {
        public uint Width;
        public uint Height;
        public uint Depth;
        public CUArrayFormat Format;
        public uint NumChannels;
        public uint Flags;
    }
}

