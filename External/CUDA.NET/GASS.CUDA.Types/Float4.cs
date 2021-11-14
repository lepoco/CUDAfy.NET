namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct Float4
    {
        public float x;
        public float y;
        public float z;
        public float w;
    }
}

