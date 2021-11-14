/*
CUDAfy.NET - LGPL 2.1 License
Please consider purchasing a commerical license - it helps development, frees you from LGPL restrictions
and provides you with support.  Thank you!
Copyright (C) 2011 Hybrid DSP Systems
http://www.hybriddsp.com

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace Cudafy.Host
{
    /// <summary>
    /// Extension methods for IntPtr to allow easy access to values. Typically used with HostAllocated memory.
    /// </summary>
    public static class IntPtrEx
    {
        /// <summary>
        /// Allows for x86 AND x64 pointer arithmetic
        /// </summary>
        /// <typeparam name="T">type pointed to</typeparam>
        /// <param name="pt"></param>
        /// <param name="offset">Offsets the prt by a number of bytes equal to offset+sizeof(T)</param>
        /// <returns></returns>
        public static IntPtr AddOffset<T>(this IntPtr pt, long offset) where T : struct
        {
            IntPtr hostArrOffset = IntPtr.Zero;
            if (IntPtr.Size == 8)
                hostArrOffset = new IntPtr(pt.ToInt64() + offset * (long)Marshal.SizeOf(typeof(T)));
            else
#if NET35
                hostArrOffset = new IntPtr(pt.ToInt32() + offset * (int)Marshal.SizeOf(typeof(T)));
#else
            hostArrOffset = IntPtr.Add(pt, (int)offset * Marshal.SizeOf(typeof(T)));// eventual truncation is of the user's responsability
#endif
            return hostArrOffset;
        }


        /// <summary>
        /// Sets the specified value.
        /// </summary>
        /// <param name="ptr">The host allocated memory.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="value">The value.</param>
        public unsafe static void Set(this IntPtr ptr, int offset, int value)
        {
            int* src = (int*)ptr;
            src[offset] = value;
        }

        /// <summary>
        /// Sets the specified value.
        /// </summary>
        /// <param name="ptr">The host allocated memory.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="value">The value.</param>
        public unsafe static void Set(this IntPtr ptr, int offset, uint value)
        {
            uint* src = (uint*)ptr;
            src[offset] = value;
        }

        /// <summary>
        /// Sets the specified host allocated memory.
        /// </summary>
        /// <param name="ptr">The host allocated memory.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="value">The value.</param>
        public unsafe static void Set(this IntPtr ptr, int offset, long value)
        {
            long* src = (long*)ptr;
            src[offset] = value;
        }

        /// <summary>
        /// Sets the specified value.
        /// </summary>
        /// <param name="ptr">The host allocated memory.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="value">The value.</param>
        public unsafe static void Set(this IntPtr ptr, int offset, ulong value)
        {
            ulong* src = (ulong*)ptr;
            src[offset] = value;
        }

        /// <summary>
        /// Sets the specified value.
        /// </summary>
        /// <param name="ptr">The host allocated memory.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="value">The value.</param>
        public unsafe static void Set(this IntPtr ptr, int offset, float value)
        {
            float* src = (float*)ptr;
            src[offset] = value;
        }

        /// <summary>
        /// Sets the specified value.
        /// </summary>
        /// <param name="ptr">The host allocated memory.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="value">The value.</param>
        public unsafe static void Set(this IntPtr ptr, int offset, double value)
        {
            double* src = (double*)ptr;
            src[offset] = value;
        }


        /// <summary>
        /// Writes the specified data array to the IntPtr.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ptr">The host allocated memory.</param>
        /// <param name="srcData">The source data.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements (set to zero for automatic).</param>
        public static void Write<T>(this IntPtr ptr, T[] srcData, int srcOffset = 0, int dstOffset = 0, int count = 0)
        {
            int cnt = count == 0 ? srcData.Length : count;
            GPGPU.CopyOnHost(srcData, srcOffset, ptr, dstOffset, cnt);
        }

        /// <summary>
        /// Writes the specified data array to the IntPtr.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ptr">The host allocated memory.</param>
        /// <param name="srcData">The source data.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements (set to zero for automatic).</param>
        public static void Write<T>(this IntPtr ptr, T[,] srcData, int srcOffset = 0, int dstOffset = 0, int count = 0)
        {
            int cnt = count == 0 ? srcData.Length : count;
            GPGPU.CopyOnHost(srcData, srcOffset, ptr, dstOffset, cnt);
        }

        /// <summary>
        /// Writes the specified data array to the IntPtr.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ptr">The host allocated memory.</param>
        /// <param name="srcData">The source data.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements (set to zero for automatic).</param>
        public static void Write<T>(this IntPtr ptr, T[,,] srcData, int srcOffset = 0, int dstOffset = 0, int count = 0)
        {
            int cnt = count == 0 ? srcData.Length : count;
            GPGPU.CopyOnHost(srcData, srcOffset, ptr, dstOffset, cnt);
        }

        /// <summary>
        /// Reads from the IntPtr to the specified data array.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ptr">The host allocated memory.</param>
        /// <param name="dstData">The destination data.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements (set to zero for automatic).</param>
        public static void Read<T>(this IntPtr ptr, T[] dstData, int srcOffset = 0, int dstOffset = 0, int count = 0)
        {
            int cnt = count == 0 ? dstData.Length : count;
            GPGPU.CopyOnHost(ptr, srcOffset, dstData, dstOffset, cnt);
        }

        /// <summary>
        /// Reads from the IntPtr to the specified data array.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ptr">The host allocated memory.</param>
        /// <param name="dstData">The destination data.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements (set to zero for automatic).</param>
        public static void Read<T>(this IntPtr ptr, T[,] dstData, int srcOffset = 0, int dstOffset = 0, int count = 0)
        {
            int cnt = count == 0 ? dstData.Length : count;
            GPGPU.CopyOnHost(ptr, srcOffset, dstData, dstOffset, cnt);
        }

        /// <summary>
        /// Reads from the IntPtr to the specified data array.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ptr">The host allocated memory.</param>
        /// <param name="dstData">The destination data.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements (set to zero for automatic).</param>
        public static void Read<T>(this IntPtr ptr, T[,,] dstData, int srcOffset = 0, int dstOffset = 0, int count = 0)
        {
            int cnt = count == 0 ? dstData.Length : count;
            GPGPU.CopyOnHost(ptr, srcOffset, dstData, dstOffset, cnt);
        }
    }
}
