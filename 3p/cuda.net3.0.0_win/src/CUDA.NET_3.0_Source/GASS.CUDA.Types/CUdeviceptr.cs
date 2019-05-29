namespace GASS.CUDA.Types
{
    using System;
    using System.Runtime.InteropServices;

    //[StructLayout(LayoutKind.Sequential)]
    //public struct CUdeviceptr
    //{
    //    public uint Pointer;
    //    public static CUdeviceptr operator +(CUdeviceptr src, uint value)
    //    {
    //        return new CUdeviceptr { Pointer = src.Pointer + value };
    //    }

    //    public static CUdeviceptr operator -(CUdeviceptr src, uint value)
    //    {
    //        return new CUdeviceptr { Pointer = src.Pointer - value };
    //    }

    //    public static implicit operator uint(CUdeviceptr src)
    //    {
    //        return src.Pointer;
    //    }

    //    public static explicit operator CUdeviceptr(uint src)
    //    {
    //        return new CUdeviceptr { Pointer = src };
    //    }
    //}

    //[StructLayout(LayoutKind.Sequential)]
    //public struct CUdeviceptr
    //{
    //    public ulong Pointer;
    //    public static CUdeviceptr operator +(CUdeviceptr src, ulong value)
    //    {
    //        return new CUdeviceptr { Pointer = src.Pointer + value };
    //    }

    //    public static CUdeviceptr operator -(CUdeviceptr src, ulong value)
    //    {
    //        return new CUdeviceptr { Pointer = src.Pointer - value };
    //    }

    //    public static implicit operator ulong(CUdeviceptr src)
    //    {
    //        return src.Pointer;
    //    }

    //    public static explicit operator CUdeviceptr(ulong src)
    //    {
    //        return new CUdeviceptr { Pointer = src };
    //    }
    //}

    [StructLayout(LayoutKind.Sequential)]
    public struct CUdeviceptr
    {
        public IntPtr Pointer;
        public static CUdeviceptr operator +(CUdeviceptr src, long value)
        {
            return new CUdeviceptr { Pointer = new IntPtr(src.Pointer.ToInt64() + value) };
        }

        public static CUdeviceptr operator -(CUdeviceptr src, long value)
        {
            return new CUdeviceptr { Pointer = new IntPtr(src.Pointer.ToInt64() - value) };
        }

        public static implicit operator long(CUdeviceptr src)
        {
            return src.Pointer.ToInt64();
        }

        public static explicit operator CUdeviceptr(long src)
        {
            return new CUdeviceptr { Pointer = new IntPtr(src) };
        }

        public static int Size
        {
            get { return IntPtr.Size; }
        }
    }
}

