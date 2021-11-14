namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct cuFloatComplex
    {
        public float real;
        public float imag;
    }
}

