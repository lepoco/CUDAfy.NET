namespace GASS.CUDA
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct cudaEvent
    {
        public int Value;
    }
}

