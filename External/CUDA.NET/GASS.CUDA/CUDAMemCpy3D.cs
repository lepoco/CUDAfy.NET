namespace GASS.CUDA
{
    using GASS.CUDA.Types;
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDAMemCpy3D
    {
        public uint srcXInBytes;
        public uint srcY;
        public uint srcZ;
        public uint srcLOD;
        public CUMemoryType srcMemoryType;
        public IntPtr srcHost;
        public CUdeviceptr srcDevice;
        public CUarray srcArray;
        public IntPtr reserved0;
        public uint srcPitch;
        public uint srcHeight;
        public uint dstXInBytes;
        public uint dstY;
        public uint dstZ;
        public uint dstLOD;
        public CUMemoryType dstMemoryType;
        public IntPtr dstHost;
        public CUdeviceptr dstDevice;
        public CUarray dstArray;
        public IntPtr reserved1;
        public uint dstPitch;
        public uint dstHeight;
        public uint WidthInBytes;
        public uint Height;
        public uint Depth;
    }
}

