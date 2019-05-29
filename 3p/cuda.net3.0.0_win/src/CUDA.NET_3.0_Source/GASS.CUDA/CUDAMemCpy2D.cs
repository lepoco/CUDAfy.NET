namespace GASS.CUDA
{
    using GASS.CUDA.Types;
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDAMemCpy2D
    {
        public uint srcXInBytes;
        public uint srcY;
        public CUMemoryType srcMemoryType;
        public IntPtr srcHost;
        public CUdeviceptr srcDevice;
        public CUarray srcArray;
        public uint srcPitch;
        public uint dstXInBytes;
        public uint dstY;
        public CUMemoryType dstMemoryType;
        public IntPtr dstHost;
        public CUdeviceptr dstDevice;
        public CUarray dstArray;
        public uint dstPitch;
        public uint WidthInBytes;
        public uint Height;
    }
}

