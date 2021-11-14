/* Copyright (C) Leszek Pomianowski
 * https://github.com/lepoco/CUDAfy.NET
 * This file is released under the MIT License
 */

using System.Runtime.InteropServices;

namespace Cudafy.Host.Sys
{
    public class Kernel32
    {
        [DllImport("kernel32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool GetPhysicallyInstalledSystemMemory(out long TotalMemoryInKilobytes);
    }
}
