namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public class CUuuid
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=0x10)]
        public byte[] Bytes = new byte[0x10];
    }
}

