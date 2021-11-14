namespace GASS.CUDA
{
    using GASS.Types;
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct cudaPitchedPtr
    {
        public IntPtr ptr;
        public SizeT pitch;
        public SizeT xsize;
        public SizeT ysize;
    }
}

