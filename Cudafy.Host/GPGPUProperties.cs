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
using System.Diagnostics;

namespace Cudafy.Host
{
    /// <summary>
    /// Represents the generic properties of a GPGPU device. Not all properties will be relevant
    /// to a particular GPGPU device.
    /// </summary>
    public class GPGPUProperties
    {
        internal GPGPUProperties(bool simulate = false, bool useAdvanced = true)
        {
            IsSimulated = simulate;
            Message = string.Empty;
            UseAdvanced = useAdvanced;
            MultiProcessorCount = 0;
            HighPerformanceDriver = false;
            SupportsDoublePrecision = true;
            AsynchEngineCount = 1;
            if (simulate)
            {
                Capability = new Version(0, 0);
                Name = "Simulator";
                DeviceId = 0;
                ulong freeMem = Int32.MaxValue;
                try
                {
                    PerformanceCounter pc = new PerformanceCounter("Memory", "Available Bytes");
                    freeMem = Convert.ToUInt64(pc.NextValue());
                }
                catch (Exception ex)
                {
                    Debug.WriteLine(ex.Message);
#if DEBUG
                    throw;
#endif 
                }
                TotalMemory = freeMem;
                MaxGridSize = new dim3(65536, 65536);
                MaxThreadsSize = new dim3(1024, 1024);
                MaxThreadsPerBlock = 1024;
            }
        }

        /// <summary>
        /// Gets a value indicating whether device supports code containing double precision.
        /// Although early CUDA devices do not support double, it is still possible to write code containing doubles.
        /// For many AMD GPUs this is not the case.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if supports double precision; otherwise, <c>false</c>.
        /// </value>
        public bool SupportsDoublePrecision { get; internal set; }

        /// <summary>
        /// Gets a value indicating whether this instance is simulated or emulated.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if this instance is simulated or emulated; otherwise, <c>false</c>.
        /// </value>
        public bool IsSimulated { get; internal set; }

        /// <summary>
        /// Gets the capability.
        /// </summary>
        /// <value>The capability.</value>
        public Version Capability { get; internal set; }

        /// <summary>
        /// Gets the name.
        /// </summary>
        /// <value>The name.</value>
        public string Name { get; internal set; }

        /// <summary>
        /// Gets the name of the platform.
        /// </summary>
        /// <value>
        /// The name of the platform.
        /// </value>
        public string PlatformName { get; internal set; }

        /// <summary>
        /// Gets the device id.
        /// </summary>
        /// <value>The device id.</value>
        public int DeviceId { get; internal set; }

        /// <summary>
        /// Gets the total memory.
        /// </summary>
        /// <value>The total memory.</value>
        public ulong TotalMemory { get; internal set; }

        /// <summary>
        /// Gets the clock rate.
        /// </summary>
        /// <value>The clock rate.</value>
        public int ClockRate { get; internal set; }

        /// <summary>
        /// Gets the max size of the grid.
        /// </summary>
        /// <value>The max size of the grid.</value>
        public dim3 MaxGridSize { get; internal set; }

        /// <summary>
        /// Gets the max number of threads.
        /// </summary>
        /// <value>The max number of threads.</value>
        public dim3 MaxThreadsSize { get; internal set; }

        /// <summary>
        /// Gets the max number of threads per block.
        /// </summary>
        /// <value>The max number of threads per block.</value>
        public int MaxThreadsPerBlock { get; internal set; }

        /// <summary>
        /// Gets the memory pitch.
        /// </summary>
        /// <value>The memory pitch.</value>
        public int MemoryPitch { get; internal set; }
        /// <summary>
        /// Gets the registers per block.
        /// </summary>
        /// <value>The registers per block.</value>
        public int RegistersPerBlock { get; internal set; }

        /// <summary>
        /// Gets the shared memory per block.
        /// </summary>
        /// <value>The shared memory per block.</value>
        public int SharedMemoryPerBlock { get; internal set; }

        /// <summary>
        /// Gets the size of the warp.
        /// </summary>
        /// <value>The size of the warp.</value>
        public int WarpSize  { get; internal set; }

        /// <summary>
        /// Gets the total constant memory.
        /// </summary>
        /// <value>The total constant memory.</value>
        public int TotalConstantMemory { get; internal set; }

        /// <summary>
        /// Gets the texture alignment.
        /// </summary>
        /// <value>The texture alignment.</value>
        public int TextureAlignment { get; internal set; }


        /// <summary>
        /// Gets a value indicating whether advanced was used.
        /// </summary>
        /// <value>
        ///   <c>true</c> if advanced used; otherwise, <c>false</c>.
        /// </value>
        public bool UseAdvanced { get; internal set; }

        /// <summary>
        /// Gets the multi processor count. UseAdvanced must be set to true.
        /// </summary>
        public int MultiProcessorCount { get; internal set; }

        /// <summary>
        /// Gets the max number of threads per multi processor. UseAdvanced must be set to true.
        /// </summary>
        public int MaxThreadsPerMultiProcessor { get; internal set; } 

        internal string Message { get; set; }

        /// <summary>
        /// Gets a value indicating whether this instance can map host memory.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if this instance can map host memory; otherwise, <c>false</c>.
        /// </value>
        public bool CanMapHostMemory { get; internal set; }
        /// <summary>
        /// Gets the number of concurrent kernels.
        /// </summary>
        /// <value>The concurrent kernels.</value>
        public int ConcurrentKernels { get; internal set; }
        /// <summary>
        /// Gets the compute mode.
        /// </summary>
        /// <value>The compute mode.</value>
        public int ComputeMode { get; internal set; }
        /// <summary>
        /// Gets a value indicating whether device overlap supported.
        /// </summary>
        /// <value><c>true</c> if device overlap supported; otherwise, <c>false</c>.</value>
        public bool DeviceOverlap { get; internal set; }
        /// <summary>
        /// Gets a value indicating whether ECC enabled.
        /// </summary>
        /// <value><c>true</c> if ECC enabled; otherwise, <c>false</c>.</value>
        public bool ECCEnabled { get; internal set; }
        /// <summary>
        /// Gets a value indicating whether GPU is integrated.
        /// </summary>
        /// <value><c>true</c> if integrated; otherwise, <c>false</c>.</value>
        public bool Integrated { get; internal set; }
        /// <summary>
        /// Gets a value indicating whether kernel execution timeout enabled.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if kernel execution timeout enabled; otherwise, <c>false</c>.
        /// </value>
        public bool KernelExecTimeoutEnabled { get; internal set; }
        /// <summary>
        /// Gets the pci bus ID.
        /// </summary>
        /// <value>The pci bus ID.</value>
        public int PciBusID { get; internal set; }
        /// <summary>
        /// Gets the pci device ID.
        /// </summary>
        /// <value>The pci device ID.</value>
        public int PciDeviceID { get; internal set; }
        /// <summary>
        /// Gets the total global memory.
        /// </summary>
        /// <value>The total global memory.</value>
        public long TotalGlobalMem { get; internal set; }
        /// <summary>
        /// Gets a value indicating whether device is using HighPerformanceDriver driver (tcc in Windows).
        /// </summary>
        /// <value>
        ///   <c>true</c> if performance driver; otherwise, <c>false</c>.
        /// </value>
        public bool HighPerformanceDriver { get; internal set; }
        /// <summary>
        /// Gets the number of asynchronous engines.
        /// </summary>
        /// <value>The number of asynchronous engines.</value>
        public int AsynchEngineCount { get; internal set; }
    }
}
