namespace GASS.CUDA.FFT.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct cufftHandle
    {
        public uint handle;
    }
}

