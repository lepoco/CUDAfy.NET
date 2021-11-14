namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct CUcontext
    {
        public IntPtr Pointer;
        public override string ToString()
        {
            return Pointer.ToString();
        }
    }
}

