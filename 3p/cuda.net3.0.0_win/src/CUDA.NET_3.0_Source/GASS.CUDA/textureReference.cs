namespace GASS.CUDA
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct textureReference
    {
        public int normalized;
        public cudaTextureFilterMode filterMode;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=3)]
        public cudaTextureAddressMode[] addressMode;
        public cudaChannelFormatDesc channelDesc;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=0x10)]
        public int[] __cudaReserved;
    }
}

