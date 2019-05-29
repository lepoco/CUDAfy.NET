namespace GASS.CUDA
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct cudaFuncAttributes
    {
        public int sharedSizeBytes;
        public int constSizeBytes;
        public int localSizeBytes;
        public int maxThreadsPerBlock;
        public int numRegs;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=8)]
        public int[] __cudaReserved;
    }
}

