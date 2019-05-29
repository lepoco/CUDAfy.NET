namespace GASS.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct SizeT
    {
        private IntPtr value;
        public SizeT(int value)
        {
            this.value = new IntPtr(value);
        }

        public SizeT(uint value)
        {
            this.value = new IntPtr((int) value);
        }

        public SizeT(long value)
        {
            this.value = new IntPtr(value);
        }

        public SizeT(ulong value)
        {
            this.value = new IntPtr((long) value);
        }

        public static implicit operator int(SizeT t)
        {
            return t.value.ToInt32();
        }

        public static implicit operator uint(SizeT t)
        {
            return (uint) ((int) t.value);
        }

        public static implicit operator long(SizeT t)
        {
            return t.value.ToInt64();
        }

        public static implicit operator ulong(SizeT t)
        {
            return (ulong) ((long) t.value);
        }

        public static implicit operator SizeT(int value)
        {
            return new SizeT(value);
        }

        public static implicit operator SizeT(uint value)
        {
            return new SizeT(value);
        }

        public static implicit operator SizeT(long value)
        {
            return new SizeT(value);
        }

        public static implicit operator SizeT(ulong value)
        {
            return new SizeT(value);
        }

        public static bool operator !=(SizeT val1, SizeT val2)
        {
            return (val1.value != val2.value);
        }

        public static bool operator ==(SizeT val1, SizeT val2)
        {
            return (val1.value == val2.value);
        }

        public override bool Equals(object obj)
        {
            return this.value.Equals(obj);
        }

        public override string ToString()
        {
            return this.value.ToString();
        }

        public override int GetHashCode()
        {
            return this.value.GetHashCode();
        }
    }
}

