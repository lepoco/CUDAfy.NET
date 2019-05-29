namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct cuDoubleReal
    {
        public double real;
    }
}

