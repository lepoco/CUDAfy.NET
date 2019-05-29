namespace GASS.CUDA
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDAArrayDescriptor
    {
        public uint Width;
        public uint Height;
        public CUArrayFormat Format;
        public uint NumChannels;
    }
}

