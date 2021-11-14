namespace GASS.CUDA
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    [Obsolete]
    public struct CUDeviceProperties
    {
        public int maxThreadsPerBlock;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=3, ArraySubType=UnmanagedType.I4)]
        public int[] maxThreadsDim;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=3, ArraySubType=UnmanagedType.I4)]
        public int[] maxGridSize;
        public int sharedMemPerBlock;
        public int totalConstantMemory;
        public int SIMDWidth;
        public int memPitch;
        public int regsPerBlock;
        public int clockRate;
        public int textureAlign;
    }
}

