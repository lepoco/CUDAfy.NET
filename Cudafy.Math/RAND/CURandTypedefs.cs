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

namespace Cudafy.Maths.RAND
{
    /// <summary>
    /// Array of 32 * 32-bit direction vectors.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct RandDirectionVectors32
    {
        /// <summary>
        /// Fixed size array of 32 direction vectors.
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 32)]
        public uint[] direction_vectors;
    };

    /// <summary>
    /// Array of 64 * 64-bit direction vectors.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct RandDirectionVectors64
    {
        /// <summary>
        /// Fixed size array of 64 direction vectors.
        /// </summary>        
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 64)]
        public ulong[] direction_vectors;
    };

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct RandGenerator
    {
        public ulong handle;
    }

    //public class RandGeneratorDevice : RandGenerator
    //{

    //}

    //public class RandGeneratorHost : RandGenerator
    //{

    //}

}
