namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct cuDoubleComplex
    {
        public double real;
        public double imag;
    }
}

