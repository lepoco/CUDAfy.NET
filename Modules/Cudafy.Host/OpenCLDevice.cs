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
using System.IO;
using System.Collections.ObjectModel;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Threading;
using Cloo;
using Cloo.Bindings;
using System.Security;
namespace Cudafy.Host
{
    [SuppressUnmanagedCodeSecurity]
    public class CL11_ex : CL11
    {
        /// <summary>
        /// See the OpenCL specification.
        /// </summary>
        [DllImport(libName, EntryPoint = "clEnqueueFillBuffer")]
        public extern static ComputeErrorCode EnqueueFillBuffer(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle buffer,
            IntPtr pattern,
            IntPtr pattern_size,
            IntPtr offset,
            IntPtr size,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst = 1)] CLEventHandle[] new_event
            );

        [DllImport(libName, EntryPoint = "clEnqueueCopyBuffer")]
        public extern static ComputeErrorCode EnqueueCopyBuffer(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle src_buffer,
            CLMemoryHandle dst_buffer,
            IntPtr src_offset,
            IntPtr dst_offset,
            IntPtr cb,
            Int32 num_events_in_wait_list,
            [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
            [Out, MarshalAs(UnmanagedType.LPArray, SizeConst = 1)] CLEventHandle[] new_event);

    }

    public class OpenCLDevice : GPGPU
    {
        #region constructors

        /// <summary>
        /// Initializes a new instance of the <see cref="CudaGPU"/> class.
        /// </summary>
        /// <param name="deviceId">The device id.</param>
        public OpenCLDevice(int deviceId = 0)
            : base(deviceId)
        {
            try
            {
                _computeDevice = GetComputeDevice(deviceId);
                OpenCLVersion = _computeDevice.OpenCLCVersion;
                _memsetArray = new byte[OpenCLVersion.Minor < 2 ? 128 * 1024 : 0];
                _kernels = new List<ComputeKernel>();
                ComputeContextPropertyList properties = new ComputeContextPropertyList(_computeDevice.Platform);
                _context = new ComputeContext(new[] { _computeDevice }, properties, null, IntPtr.Zero);
                _defaultSynchronousQueue = new ComputeCommandQueue(_context, _computeDevice, ComputeCommandQueueFlags.None); // default synchronous stream
                _streams.Add(0, _defaultSynchronousQueue);
            }
            catch (IndexOutOfRangeException)
            {
                throw new CudafyHostException(CudafyHostException.csDEVICE_ID_OUT_OF_RANGE);
            }

        }

        static OpenCLDevice()
        {
            var tempComputeDevices = new List<ComputeDevice>();
            foreach (var platform in ComputePlatform.Platforms)
                foreach (var device in platform.Devices)
                    tempComputeDevices.Add(device);
            ComputeDevices = new ReadOnlyCollection<ComputeDevice>(tempComputeDevices);
        }

        #endregion constructors

        #region fields & props

        internal static ReadOnlyCollection<ComputeDevice> ComputeDevices;

        private ComputeContext _context;

        private List<ComputeKernel> _kernels;

        private ComputeDevice _computeDevice;

        private ComputeCommandQueue _defaultSynchronousQueue;

        private readonly Version OpenCLVersion;

        /// <summary>
        /// Gets or sets a value indicating whether device supports smart copy.
        /// </summary>
        /// <value>
        ///   <c>true</c> if supports smart copy; otherwise, <c>false</c>.
        /// </value>
        public override bool SupportsSmartCopy
        {
            get { return false; }
        }

        #endregion fields & props

        private ComputeDevice GetComputeDevice(int id)
        {
            if (id < 0 || id > ComputeDevices.Count - 1)
                throw new ArgumentOutOfRangeException("id");
            return ComputeDevices[id];
        }

        public override eArchitecture GetArchitecture()
        {
            //if (!(this is CudaGPU) && !(this is EmulatedGPU))
            //    throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, this.GetType());

            var capability = this.GetDeviceProperties(false).Capability;

            switch (capability.Major)
            {

                case 1:

                    switch (capability.Minor)
                    {
                        case 0: return eArchitecture.OpenCL;


                        case 1: return eArchitecture.OpenCL11;

                        case 2: return eArchitecture.OpenCL12;

                        default:
                            throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, capability.ToString());

                    }

  

                default:
                    throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, capability.ToString());
            }

        }

        #region GetDeviceProperties

        public override GPGPUProperties GetDeviceProperties(bool useAdvanced = true)
        {
            return GetDeviceProperties(_computeDevice, DeviceId, useAdvanced);
        }

#warning TODO implement "useAdvanced"
        internal static GPGPUProperties GetDeviceProperties(ComputeDevice computeDevice, int deviceId, bool useAdvanced = true)
        {
            GPGPUProperties props = new GPGPUProperties();
            props.Capability = computeDevice.Version;
            props.ClockRate = (int)computeDevice.MaxClockFrequency;
            props.DeviceId = deviceId;
            props.DeviceOverlap = true;
            props.ECCEnabled = computeDevice.ErrorCorrectionSupport;
            props.HighPerformanceDriver = false; 
            props.Integrated = computeDevice.Type != ComputeDeviceTypes.Cpu;
            props.IsSimulated = false;
            props.KernelExecTimeoutEnabled = true;// TODO 
            
            props.MaxThreadsPerBlock = (int)computeDevice.MaxWorkGroupSize;//CHECK
            props.MaxThreadsPerMultiProcessor = 1; //TODO
            props.MaxThreadsSize = new dim3(computeDevice.MaxWorkItemSizes.ToArray());
            props.MaxGridSize = new dim3(0x80000000 / props.MaxThreadsSize.x, 0x80000000 / props.MaxThreadsSize.y, 0x80000000 / props.MaxThreadsSize.z);
            props.MemoryPitch = Int32.MaxValue;//TODO
            props.MultiProcessorCount = (int)computeDevice.MaxComputeUnits;
            props.Name = computeDevice.Name;
            props.PlatformName = computeDevice.Platform.Name;
            props.PciBusID = 0;//TODO
            props.PciDeviceID = 0;//TODO
            props.RegistersPerBlock = 64 * 1024; // TODO
            props.SharedMemoryPerBlock = (int)computeDevice.LocalMemorySize;//CHECK
            props.TextureAlignment = 0;//TODO
            props.TotalConstantMemory = (int)computeDevice.MaxConstantBufferSize;
            props.TotalGlobalMem = computeDevice.GlobalMemorySize;
            props.TotalMemory = (ulong)props.TotalConstantMemory + (ulong)props.TotalGlobalMem;
            props.UseAdvanced = true;
            props.WarpSize = 32;// TODO
            props.SupportsDoublePrecision = computeDevice.NativeVectorWidthDouble > 0;
            return props;
        }

        /// <summary>
        /// Gets the device count.
        /// </summary>
        /// <returns>Number of Cuda devices in system.</returns>
        public new static int GetDeviceCount()
        {
            int cnt = 0;
            try
            {
                cnt = CudafyHost.GetDeviceCount(eGPUType.OpenCL);               
            }
            catch (Exception ex)
            {
#if DEBUG
                throw;
#endif
                Debug.WriteLine(ex.Message);
            }
            return cnt;
        }

        #endregion GetDeviceProperties

        public override bool CanAccessPeer(GPGPU peer)
        {
            OpenCLDevice OCL_peer = peer as OpenCLDevice;
            if (OCL_peer == null)
                return false;
            return (OCL_peer._context.Handle.Value == _context.Handle.Value); // self is the only allowed peer
        }

        private void HandleOpenCLException(ComputeException ex)
        {
            string addInfo = string.Empty;
#warning TODO - all cases
            //switch (ex.ComputeErrorCode)
            //{
            //    case ComputeErrorCode.OutOfHostMemory:
            //        break;
            //    default:
            //        break;
            //}
            if (string.IsNullOrEmpty(addInfo))
                throw new CudafyHostException(ex, CudafyHostException.csOPENCL_EXCEPTION_X, ex.ComputeErrorCode);
            else
                throw new CudafyHostException(ex, CudafyHostException.csOPENCL_EXCEPTION_X_X, ex.ComputeErrorCode, addInfo);
        }

        #region GetStream

        /// <summary>
        /// Gets the ComputeCommandQueue object identified by streamId.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        /// <param name="flags">Extra flags for queue creation.</param>
        /// <returns>ComputeCommandQueue object.</returns>
        public virtual object GetStream(int streamId, ComputeCommandQueueFlags flags)
        {
            lock (_lock)
            {
                ComputeCommandQueue clStr = null;
                if (streamId > 0)
                {
                    if (!_streams.ContainsKey(streamId))
                    {
                        clStr = CreateStream(streamId, flags);
                    }
                    else
                    {
                        clStr = (ComputeCommandQueue)_streams[streamId];
                    }
                }
                else if (!_streams.ContainsKey(0))
                    throw new CudafyHostException(CudafyHostException.csOPENCL_EXCEPTION_X_X, "Default queue not initialized");
                else
                    clStr = (ComputeCommandQueue)_streams[0];

                return clStr;
            }
        }

        /// <summary>
        /// Gets the ComputeCommandQueue object identified by streamId.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        /// <returns>ComputeCommandQueue object.</returns>
        public override object GetStream(int streamId)
        {
            return GetStream(streamId, ComputeCommandQueueFlags.None);
        }

        #endregion GetStream

        #region CreateStream

        /// <summary>
        /// Explicitly creates a stream.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        public override void CreateStream(int streamId)
        {
            CreateStream(streamId, ComputeCommandQueueFlags.None);
        }

        /// <summary>
        /// Explicitly creates a stream.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        /// <param name="flags">Extra flags for queue creation.</param>
        /// <returns>Command queue.</returns>
        public ComputeCommandQueue CreateStream(int streamId, ComputeCommandQueueFlags flags)
        {
            ComputeCommandQueue clStr = null;
            lock (_lock)
            {
                if (streamId < 0)
                    throw new ArgumentOutOfRangeException("streamId must be greater than or equal to zero");
                if (_streams.ContainsKey(streamId))
                    throw new CudafyHostException(CudafyHostException.csSTREAM_X_ALREADY_SET, streamId);
        
                try
                {
                    clStr = new ComputeCommandQueue(_context, _computeDevice, flags);
                }
                catch (ComputeException ex)
                {
                    HandleOpenCLException(ex);
                }
               // var props = GetDeviceProperties();
               // Console.WriteLine(string.Format("Platform {0}, Device {1}, CreateStream {2}", props.PlatformName, props.Name, streamId));
                _streams.Add(streamId, clStr);
            }
            return clStr;
        }

        #endregion CreateStream

        #region DestroyStream

        public override void DestroyStream(int streamId)
        {
            lock (_lock)
            {
                if (!_streams.ContainsKey(streamId))
                    throw new CudafyHostException(CudafyHostException.csSTREAM_X_NOT_SET, streamId);

                ComputeCommandQueue clStr = (ComputeCommandQueue)_streams[streamId];
                //clStr.Finish();
                //var props = GetDeviceProperties();
                //Console.WriteLine(string.Format("Platform {0}, Device {1}, DestroyStream {2}", props.PlatformName, props.Name, streamId));
                _streams.Remove(streamId);
                clStr.Dispose(); // release unmanaged resources
            }
        }

        public override void DestroyStreams()
        {
            foreach (int streamId in _streams.Keys.ToList())
                DestroyStream(streamId);
        }

        #endregion DestroyStream

        #region DoCopyDeviceToDevice

        protected override void DoCopyDeviceToDevice<T>(Array srcDevArray, int srcOffset, GPGPU peer, Array dstDevArray, int dstOffet, int count)
        {
            DoCopyDeviceToDevice<T>(srcDevArray, (long)srcOffset, peer, dstDevArray, (long)dstOffet, (long)count);
        }

        protected void DoCopyDeviceToDevice<T>(Array srcDevArray, long srcOffset, GPGPU peer, Array dstDevArray, long dstOffet, long count) where T : struct
        {
            if (!CanAccessPeer(peer))
                throw new NotSupportedException("Device to Device copy not supported between different contexts.");
            CLDevicePtrEx<T> ptr_source = GetDeviceMemory(srcDevArray) as CLDevicePtrEx<T>;
            CLDevicePtrEx<T> ptr_dest = peer.GetDeviceMemory(dstDevArray) as CLDevicePtrEx<T>;
            int sizeofT = HDSPUtils.SizeOf(typeof(T));
            ComputeErrorCode error = CL11_ex.EnqueueCopyBuffer(
                _defaultSynchronousQueue.Handle, ptr_source.Handle, ptr_dest.Handle, new IntPtr(srcOffset), new IntPtr(dstOffet), new IntPtr(count * sizeofT), 0, null, null);
#warning TODO Throw a CUDAfy exception instead
            ComputeException.ThrowOnError(error);
            _defaultSynchronousQueue.Finish();
        }

        #endregion DoCopyDeviceToDevice

        #region DoCopyDeviceToDeviceAsync

        protected override void DoCopyDeviceToDeviceAsync<T>(Array srcDevArray, int srcOffset, GPGPU peer, Array dstDevArray, int dstOffet, int count, int stream)
        {
            DoCopyDeviceToDeviceAsync<T>(srcDevArray, (long)srcOffset, peer, dstDevArray, (long)dstOffet, (long)count, stream);
        }

        protected void DoCopyDeviceToDeviceAsync<T>(Array srcDevArray, long srcOffset, GPGPU peer, Array dstDevArray, long dstOffet, long count, int stream) where T : struct
        {
            if (!CanAccessPeer(peer))
                throw new NotSupportedException("Device to Device copy not supported between different contexts.");
            ComputeCommandQueue queue = (ComputeCommandQueue)GetStream(stream);
            CLDevicePtrEx<T> ptr_source = GetDeviceMemory(srcDevArray) as CLDevicePtrEx<T>;
            CLDevicePtrEx<T> ptr_dest = peer.GetDeviceMemory(dstDevArray) as CLDevicePtrEx<T>;
            int sizeofT = HDSPUtils.SizeOf(typeof(T));
            ComputeErrorCode error = CL11_ex.EnqueueCopyBuffer(
                queue.Handle, ptr_source.Handle, ptr_dest.Handle, new IntPtr(srcOffset), new IntPtr(dstOffet), new IntPtr(count * sizeofT), 0, null, null);
#warning TODO Throw a CUDAfy exception instead
            ComputeException.ThrowOnError(error);
            if (stream <= 0)
                queue.Finish();
        }

        #endregion DoCopyDeviceToDeviceAsync

        /// <summary>
        /// Gets the free memory available on device. Note that this is very approximate and does not
        /// take into account any other applications including OS graphics that may be using the device.
        /// It merely subtracts all allocated memory from the TotalMemory.
        /// </summary>
        /// <value>
        /// The free memory.
        /// </value>
        public override ulong FreeMemory
        {
            get 
            { 
                long totalMemory = (long)_computeDevice.GlobalMemorySize; 
                // Add up all memory currently used and subtract
                foreach (CLDevicePtrExInter mem in GetDeviceMemories())
                {
                    long size = mem.TotalSize * mem.ElementSize;
                    totalMemory -= size;
                }
                return (ulong)totalMemory;
            }
        }

        public override ulong TotalMemory
        {
            get { return (ulong)_computeDevice.GlobalMemorySize; }
        }

        public override void Synchronize()
        {
            foreach (int streamId in _streams.Keys)
                SynchronizeStream(streamId);
        }

        public override void SynchronizeStream(int streamId = 0)
        {
            ComputeCommandQueue queue = (ComputeCommandQueue)GetStream(streamId);
            queue.Finish();
        }

        public void FlushStream(int streamId = 0)
        {
            ComputeCommandQueue queue = (ComputeCommandQueue)GetStream(streamId);
            queue.Flush();
        }

        public override void EnableSmartCopy()
        {
            throw new NotSupportedException("smart copy");
        }

        public override void DisableSmartCopy()
        {
            throw new NotSupportedException("smart copy");
        }

        


        private string clTestProgramSource = @"
kernel void VectorAdd(
    global  read_only int* a,
    global  read_only int* b,
    global write_only int* c )
{
    int index = get_global_id(0);
    c[index] = a[index] + b[index];
}
";
        public override void LoadModule(CudafyModule module, bool unload = true)
        {
            // Create and build the opencl program.//
           // module.CudaSourceCode = module.CudaSourceCode.Replace("\r\n", "\n");
            Debug.WriteLine(module.SourceCode);
            ComputeProgram program = new ComputeProgram(_context, module.SourceCode);
            try
            {
                program.Build(null, null, null, IntPtr.Zero);
            }
            catch (Exception ex)
            {                
                module.CompilerOutput = program.GetBuildLog(_computeDevice);
                throw new CudafyCompileException(ex, CudafyCompileException.csCOMPILATION_ERROR_X, module.CompilerOutput); ;
            }
            finally
            {
                module.CompilerOutput = program.GetBuildLog(_computeDevice);
                Debug.WriteLine(module.CompilerOutput);
            }

            if (unload)
                UnloadModules();
            else
                CheckForDuplicateMembers(module);

            // Create the kernel function and set its arguments.
            foreach (ComputeKernel kernel in program.CreateAllKernels())
                _kernels.Add(kernel);

            // Load constants
            foreach (var kvp in module.Constants)
            {
                if (!kvp.Value.IsDummy)
                {
                    int elemSize = MSizeOf(kvp.Value.Information.FieldType.GetElementType());
                    int totalLength = kvp.Value.GetTotalLength();
                    ComputeBuffer<byte> a = new ComputeBuffer<byte>(_context, ComputeMemoryFlags.ReadOnly, totalLength * elemSize);
                    module.Constants[kvp.Key].Handle = a;
                    module.Constants[kvp.Key].CudaPointer = a.Handle;
                }
            }

            _modules.Add(module);
        }

        public override void UnloadModule(CudafyModule module)
        {
            //throw new NotImplementedException();
        }

        public override void UnloadModules()
        {
            _kernels.Clear();
            base.UnloadModules();
        }

        protected override void DoLaunch(dim3 gridSize, dim3 blockSize, int streamId, KernelMethodInfo gpuMI, params object[] arguments)
        {
            ComputeKernel kernel = _kernels.Where(k => k.FunctionName == gpuMI.Name).FirstOrDefault();//
            int totalArgs = arguments.Length;
            int actualArgCtr = 0;
            int i = 0;
            for (i = 0; i < totalArgs; i++, actualArgCtr++)
            {
                object arg = arguments[actualArgCtr];
                if (arg is Array)
                {
                    var ptrEx = GetDeviceMemory(arg) as CLDevicePtrExInter;
                    kernel.SetMemoryArgument(i, ptrEx.Handle);
                    int[] dims = ptrEx.GetDimensions();
                    for (int d = 0; d < ptrEx.Dimensions; d++, totalArgs++)
                        kernel.SetValueArgument(++i, dims[d]);

                }
                else if (arg is Char)
                {
                    byte[] ba = Encoding.Unicode.GetBytes(new char[] { (char)arg });
                    ushort shrt = BitConverter.ToUInt16(ba, 0);
                    kernel.SetValueArgument(i, 2, shrt);
                }
                else
                {
                    kernel.SetValueArgument(i, MSizeOf(arg.GetType()), arg);
                }
            }

            // Add constants
            foreach (KeyValuePair<string, KernelConstantInfo> kvp in gpuMI.ParentModule.Constants)
            {
                CLMemoryHandle clm = (CLMemoryHandle)kvp.Value.CudaPointer;
                Debug.Assert(clm.IsValid);
                kernel.SetMemoryArgument(i++, clm);
            }

            bool isSynch = (streamId <= 0);
            ComputeCommandQueue queue = isSynch ?
                _defaultSynchronousQueue :
                (ComputeCommandQueue)GetStream(streamId);

            // Convert from CUDA grid and block size to OpenCL grid size
            int gridDims = gridSize.ToArray().Length;
            int blockDims = blockSize.ToArray().Length;
            int maxDims = Math.Max(gridDims, blockDims);

            long[] blockSizeArray = blockSize.ToFixedSizeArray(maxDims);
            long[] gridSizeArray = gridSize.ToFixedSizeArray(maxDims);
            for (i = 0; i < maxDims; i++)
                gridSizeArray[i] *= blockSizeArray[i];
            queue.Execute(kernel, null, gridSizeArray, blockSizeArray, null);

            if (isSynch)
                queue.Finish();
        }

        #region DoCopyToConstantMemory

        protected override void DoCopyToConstantMemory<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, KernelConstantInfo ci)
        {
            DoCopyToConstantMemory<T>(hostArray, (long)hostOffset, devArray, (long)devOffset, (long)count, ci);
        }

        protected void DoCopyToConstantMemory<T>(Array hostArray, long hostOffset, Array devArray, long devOffset, long count, KernelConstantInfo ci) where T : struct
        {
            _defaultSynchronousQueue.WriteToBufferEx<T>(hostArray, (CLMemoryHandle)ci.CudaPointer, true, hostOffset, devOffset, count, null);
            _defaultSynchronousQueue.Finish();
        }

        #endregion DoCopyToConstantMemory

        #region DoCopyToConstantMemoryAsync

        protected override void DoCopyToConstantMemoryAsync<T>(IntPtr hostArray, int hostOffset, Array devArray, int devOffset, int count, KernelConstantInfo ci, int streamId)
        {
            DoCopyToConstantMemoryAsync<T>(hostArray, (long)hostOffset, devArray, (long)devOffset, (long)count, ci, streamId);
        }

        protected void DoCopyToConstantMemoryAsync<T>(IntPtr hostArray, long hostOffset, Array devArray, long devOffset, long count, KernelConstantInfo ci, int streamId) where T : struct
        {
            ComputeCommandQueue queue = (ComputeCommandQueue)GetStream(streamId);
            IntPtr hostArrOffset = hostArray.AddOffset<T>(hostOffset);
            queue.WriteEx<T>((CLMemoryHandle)ci.CudaPointer, false, devOffset, count, hostArrOffset, null);
            if (streamId <= 0)
                queue.Finish();
        }

        #endregion DoCopyToConstantMemoryAsync

        #region DoCopyToDevice

        protected override void DoCopyToDevice<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count)
        {
            DoCopyToDevice<T>(hostArray, (long)hostOffset, devArray, (long)devOffset, (long)count);
        }

        protected void DoCopyToDevice<T>(Array hostArray, long hostOffset, Array devArray, long devOffset, long count) where T : struct
        {
            CLDevicePtrEx<T> ptr = GetDeviceMemory(devArray) as CLDevicePtrEx<T>;
            _defaultSynchronousQueue.WriteToBufferEx(hostArray, ptr.DevPtr, true, hostOffset, devOffset, count, null);
            _defaultSynchronousQueue.Finish();
        }

        #endregion DoCopyToDevice

        #region DoCopyFromDevice

        protected override void DoCopyFromDevice<T>(Array devArray, Array hostArray)
        {
            DoCopyFromDevice<T>(devArray, 0, hostArray, 0, -1);
        }

        protected override void DoCopyFromDevice<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count)
        {
            DoCopyFromDevice<T>(devArray, (long)devOffset, hostArray, (long)hostOffset, (long)count);
        }
        protected void DoCopyFromDevice<T>(Array devArray, long devOffset, Array hostArray, long hostOffset, long count) where T : struct
        {
            CLDevicePtrEx<T> ptr = GetDeviceMemory(devArray) as CLDevicePtrEx<T>;
            _defaultSynchronousQueue.ReadFromBufferEx(ptr.DevPtr, ref hostArray, true, devOffset, hostOffset, count < 0 ? ptr.TotalSize : count, null);
            _defaultSynchronousQueue.Finish();
        }

        #endregion DoCopyFromDevice

        #region DoCopyToDeviceAsync
        protected override void DoCopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, Array devArray, int devOffset, int count, int streamId)
        {
            DoCopyToDeviceAsync<T>(hostArray, (long)hostOffset, devArray, (long)devOffset, (long)count, streamId);
        }

        protected override void DoCopyToDeviceAsync<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, int streamId)
        {
            DoCopyToDeviceAsync<T>(hostArray, (long)hostOffset, devArray, (long)devOffset, (long)count, streamId);
        }

        protected override void DoCopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, DevicePtrEx devArray, int devOffset, int count, int streamId)
        {
            DoCopyToDeviceAsync<T>(hostArray, (long)hostOffset, devArray, (long)devOffset, (long)count, streamId);
        }

        protected override void DoCopyToDeviceAsync<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, int streamId, IntPtr stagingPost, bool isConstantMemory = false)
        {
            DoCopyToDeviceAsync<T>(hostArray, (long)hostOffset, devArray, (long)devOffset, (long)count, streamId, stagingPost, isConstantMemory);
        }

        protected void DoCopyToDeviceAsync<T>(IntPtr hostArray, long hostOffset, Array devArray, long devOffset, long count, int streamId) where T : struct
        {
            ComputeCommandQueue queue = (ComputeCommandQueue)GetStream(streamId);
            IntPtr hostArrPlusOffset = hostArray.AddOffset<T>(hostOffset);
            CLDevicePtrEx<T> ptr = GetDeviceMemory(devArray) as CLDevicePtrEx<T>;
            queue.WriteEx<T>(ptr.DevPtr.Handle, false, devOffset, count, hostArrPlusOffset, null);
            if (streamId <= 0)
                queue.Finish();
        }

        protected void DoCopyToDeviceAsync<T>(Array hostArray, long hostOffset, Array devArray, long devOffset, long count, int streamId) where T : struct
        {
            ComputeCommandQueue queue = (ComputeCommandQueue)GetStream(streamId);
            CLDevicePtrEx<T> ptr = GetDeviceMemory(devArray) as CLDevicePtrEx<T>;
            queue.WriteToBufferEx(hostArray, ptr.DevPtr, false, hostOffset, devOffset, count, null);
            if (streamId <= 0)
                queue.Finish();
        }

        protected void DoCopyToDeviceAsync<T>(IntPtr hostArray, long hostOffset, DevicePtrEx devArray, long devOffset, long count, int streamId) where T : struct
        {
            CLDevicePtrEx<T> ptr = devArray as CLDevicePtrEx<T>;
            if (ptr == null)
                throw new CudafyHostException(CudafyHostException.csCUDA_EXCEPTION_X, "Invalid device Array. Must be of type CLDevicePtrEx<T>.");
            ComputeCommandQueue queue = (ComputeCommandQueue)GetStream(streamId);
            IntPtr hostArrPlusOffset = hostArray.AddOffset<T>(hostOffset);
            queue.Write<T>(ptr.DevPtr, false, devOffset, count, hostArrPlusOffset, null);
            if (streamId <= 0)
                queue.Finish();
        }

        protected void DoCopyToDeviceAsync<T>(Array hostArray, long hostOffset, Array devArray, long devOffset, long count, int streamId, IntPtr stagingPost, bool isConstantMemory = false) where T : struct
        {
            throw new NotSupportedException();
        }

        #endregion DoCopyToDeviceAsync

        #region DoCopyFromDeviceAsync

        protected override void DoCopyFromDeviceAsync<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count, int streamId)
        {
            DoCopyFromDeviceAsync<T>(devArray, (long)devOffset, hostArray, (long)hostOffset, (long)count, streamId);
        }

        protected override void DoCopyFromDeviceAsync<T>(Array devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId)
        {
            DoCopyFromDeviceAsync<T>(devArray, (long)devOffset, hostArray, (long)hostOffset, (long)count, streamId);
        }

        protected override void DoCopyFromDeviceAsync<T>(DevicePtrEx devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId)
        {
            DoCopyFromDeviceAsync<T>(devArray, (long)devOffset, hostArray, (long)hostOffset, (long)count, streamId);
        }

        protected override void DoCopyFromDeviceAsync<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count, int streamId, IntPtr stagingPost)
        {
            throw new NotSupportedException();
        }

        protected void DoCopyFromDeviceAsync<T>(Array devArray, long devOffset, Array hostArray, long hostOffset, long count, int streamId) where T : struct
        {
            ComputeCommandQueue queue = (ComputeCommandQueue)GetStream(streamId);
            CLDevicePtrEx<T> ptr = GetDeviceMemory(devArray) as CLDevicePtrEx<T>;
            queue.ReadFromBufferEx(ptr.DevPtr, ref hostArray, false, devOffset, hostOffset, count, null);
            if (streamId <= 0)
                queue.Finish();
        }

        protected void DoCopyFromDeviceAsync<T>(Array devArray, long devOffset, IntPtr hostArray, long hostOffset, long count, int streamId) where T : struct
        {
            ComputeCommandQueue queue = (ComputeCommandQueue)GetStream(streamId);
            IntPtr hostArrPlusOffset = hostArray.AddOffset<T>(hostOffset);
            CLDevicePtrEx<T> ptr = GetDeviceMemory(devArray) as CLDevicePtrEx<T>;
            queue.Read(ptr.DevPtr, false, devOffset, count, hostArrPlusOffset, null);
            if (streamId <= 0)
                queue.Finish();
        }

        protected void DoCopyFromDeviceAsync<T>(DevicePtrEx devArray, long devOffset, IntPtr hostArray, long hostOffset, long count, int streamId) where T : struct
        {
            CLDevicePtrEx<T> ptr = devArray as CLDevicePtrEx<T>;
            if (ptr == null)
                throw new CudafyHostException(CudafyHostException.csCUDA_EXCEPTION_X, "Invalid device Array. Must be of type CLDevicePtrEx<T>.");
            ComputeCommandQueue queue = (ComputeCommandQueue)GetStream(streamId);
            IntPtr hostArrPlusOffset = hostArray.AddOffset<T>(hostOffset);
            queue.Read(ptr.DevPtr, false, devOffset, count, hostArrPlusOffset, null);
            if (streamId <= 0)
                queue.Finish();
        }

        #endregion DoCopyFromDeviceAsync

        // OpenCL doesn't provide a mechanism for merely allocating properly-alligned pinned memory on the host. Such buffers need to be created by the developer.
        // While it may be tempting to implement here that mechanism (outside OpenCL) using the windows api, the fact is that different gpu implementations
        // will probably require different memory allignments on the host to be efficient, and may or may not allow write-combined memory, and who knows 
        // which other details are hardware dependent?
        #region Host mem Allocation

        //public override IntPtr HostAllocate<T>(int x)
        //{
        //    throw new NotSupportedException();
        //}

        //public override void HostFree(IntPtr ptr)
        //{
        //    throw new NotSupportedException();
        //}

        //public override void HostFreeAll()
        //{
        //    //throw new NotSupportedException();
        //}

        #endregion Host mem Allocation

        // in OpenCL, a handle to device memory isn't necessarily a pointer. It could be anything. Host-side pointer arithmetic on device objects should be discouraged.
        #region DoCast

        protected override Array DoCast<T, U>(int offset, Array devArray, int n)
        {
            throw new NotSupportedException();
        }

        protected override Array DoCast<T, U>(int offset, Array devArray, int x, int y)
        {
            throw new NotImplementedException();
        }

        protected override Array DoCast<T, U>(int offset, Array devArray, int x, int y, int z)
        {
            throw new NotImplementedException();
        }

        #endregion DoCast

        #region CopyToDevice
        public override T[] CopyToDevice<T>(T[] hostArray)
        {
            T[] devMem = new T[0];
            ComputeBuffer<T> a = new ComputeBuffer<T>(_context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, hostArray);
            AddToDeviceMemory(devMem, new CLDevicePtrEx<T>(a, (int)hostArray.LongLength, _context));
            return devMem;
        }

        public override T[,] CopyToDevice<T>(T[,] hostArray)
        {
            // We're using Marshal.SizeOf since the host buffer will be pinned and marshalled onto the OpenCL API. Is this correct?
#warning test it thoroughly, since the pinned mem layout may include unused bytes for array edge padding
            long hostBuffLen = hostArray.GetLongLength(0) * hostArray.GetLongLength(1);//MSizeOf(typeof(T)) * 
            T[,] devMem = new T[0, 0];
            ComputeBuffer<T> a;
            GCHandle dataPtr = GCHandle.Alloc(hostArray, GCHandleType.Pinned);
            try
            {
                a = new ComputeBuffer<T>(_context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, hostBuffLen, dataPtr.AddrOfPinnedObject());
            }
            finally
            {
                dataPtr.Free();
            }
            AddToDeviceMemory(devMem, new CLDevicePtrEx<T>(a, (int)hostArray.GetLongLength(0), (int)hostArray.GetLongLength(1), _context));
            return devMem;
        }

        public override T[, ,] CopyToDevice<T>(T[, ,] hostArray)
        {
            // We're using Marshal.SizeOf since the host buffer will be pinned and marshalled onto the OpenCL API. Is this correct?
#warning test it thoroughly, since the pinned mem layout may include unused bytes for array edge padding
            long hostBuffLen = hostArray.GetLongLength(0) * hostArray.GetLongLength(1) * hostArray.GetLongLength(2);
            T[, ,] devMem = new T[0, 0, 0];//MSizeOf(typeof(T)) * 
            ComputeBuffer<T> a;
            GCHandle dataPtr = GCHandle.Alloc(hostArray, GCHandleType.Pinned);
            try
            {
                a = new ComputeBuffer<T>(_context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, hostBuffLen, dataPtr.AddrOfPinnedObject());
            }
            finally
            {
                dataPtr.Free();
            }
            AddToDeviceMemory(devMem, new CLDevicePtrEx<T>(a, (int)hostArray.GetLongLength(0), (int)hostArray.GetLongLength(1), (int)hostArray.GetLongLength(2), _context));
            return devMem;
        }

        #endregion CopyToDevice

        #region CopyOnDevice

        public override void CopyOnDevice<T>(T[] srcDevArray, T[] dstDevArray)
        {
            CLDevicePtrEx<T> ptr_src = GetDeviceMemory(srcDevArray) as CLDevicePtrEx<T>;
            CLDevicePtrEx<T> ptr_dst = GetDeviceMemory(dstDevArray) as CLDevicePtrEx<T>;
            if (ptr_src.XSize != ptr_dst.XSize)
                throw new CudafyHostException(CudafyHostException.csCUDA_EXCEPTION_X, "array sizes differ");
            _defaultSynchronousQueue.Copy<T>(ptr_src.DevPtr, ptr_dst.DevPtr, 0, 0, ptr_src.XSize, null);
            _defaultSynchronousQueue.Finish();
        }

        protected override void DoCopyOnDevice<T>(Array srcDevArray, int srcOffset, Array dstDevArray, int dstOffet, int count)
        {
            DoCopyOnDevice<T>(srcDevArray, (long)srcOffset, dstDevArray, (long)dstOffet, (long)count);
        }
        protected void DoCopyOnDevice<T>(Array srcDevArray, long srcOffset, Array dstDevArray, long dstOffet, long count) where T : struct
        {
            CLDevicePtrEx<T> ptr_src = GetDeviceMemory(srcDevArray) as CLDevicePtrEx<T>;
            CLDevicePtrEx<T> ptr_dst = GetDeviceMemory(dstDevArray) as CLDevicePtrEx<T>;
            _defaultSynchronousQueue.Copy<T>(ptr_src.DevPtr, ptr_dst.DevPtr, srcOffset, dstOffet, count, null);
            _defaultSynchronousQueue.Finish();
        }

        protected override void DoCopyOnDevice<T>(DevicePtrEx srcDevArray, int srcOffset, DevicePtrEx dstDevArray, int dstOffet, int count)
        {
            DoCopyOnDevice<T>(srcDevArray, (long)srcOffset, dstDevArray, (long)dstOffet, (long)count);
        }
        protected void DoCopyOnDevice<T>(DevicePtrEx srcDevArray, long srcOffset, DevicePtrEx dstDevArray, long dstOffet, long count) where T : struct
        {
            CLDevicePtrEx<T> ptr_src = srcDevArray as CLDevicePtrEx<T>;
            CLDevicePtrEx<T> ptr_dst = dstDevArray as CLDevicePtrEx<T>;
            if (ptr_dst == null || ptr_src == null)
                throw new CudafyHostException(CudafyHostException.csCUDA_EXCEPTION_X, "Invalid device Array. Must be of type CLDevicePtrEx<T>.");
            _defaultSynchronousQueue.Copy<T>(ptr_src.DevPtr, ptr_dst.DevPtr, srcOffset, dstOffet, count, null);
            _defaultSynchronousQueue.Finish();
        }

        #endregion CopyOnDevice

        #region CopyOnDeviceAsync

        protected override void DoCopyOnDeviceAsync<T>(Array srcDevArray, int srcOffset, Array dstDevArray, int dstOffet, int count, int streamId)
        {
            DoCopyOnDeviceAsync<T>(srcDevArray, (long)srcOffset, dstDevArray, (long)dstOffet, (long)count, streamId);
        }

        protected void DoCopyOnDeviceAsync<T>(Array srcDevArray, long srcOffset, Array dstDevArray, long dstOffet, long count, int streamId) where T : struct
        {
            ComputeCommandQueue queue = (ComputeCommandQueue)GetStream(streamId);
            CLDevicePtrEx<T> ptr_src = GetDeviceMemory(srcDevArray) as CLDevicePtrEx<T>;
            CLDevicePtrEx<T> ptr_dst = GetDeviceMemory(dstDevArray) as CLDevicePtrEx<T>;
            queue.Copy<T>(ptr_src.DevPtr, ptr_dst.DevPtr, srcOffset, dstOffet, count, null);
            if (streamId <= 0)
                queue.Finish();
        }

        protected override void DoCopyOnDeviceAsync<T>(DevicePtrEx srcDevArray, int srcOffset, DevicePtrEx dstDevArray, int dstOffet, int count, int streamId)
        {
            DoCopyOnDeviceAsync<T>(srcDevArray, (long)srcOffset, dstDevArray, (long)dstOffet, (long)count, streamId);
        }

        protected void DoCopyOnDeviceAsync<T>(DevicePtrEx srcDevArray, long srcOffset, DevicePtrEx dstDevArray, long dstOffet, long count, int streamId) where T : struct
        {
            ComputeCommandQueue queue = (ComputeCommandQueue)GetStream(streamId);
            CLDevicePtrEx<T> ptr_src = srcDevArray as CLDevicePtrEx<T>;
            CLDevicePtrEx<T> ptr_dst = dstDevArray as CLDevicePtrEx<T>;
            if (ptr_dst == null || ptr_src == null)
                throw new CudafyHostException(CudafyHostException.csCUDA_EXCEPTION_X, "Invalid device Array. Must be of type CLDevicePtrEx<T>.");
            queue.Copy<T>(ptr_src.DevPtr, ptr_dst.DevPtr, srcOffset, dstOffet, count, null);
            if (streamId <= 0)
                queue.Finish();
        }

        #endregion CopyOnDeviceAsync

        public override T[] Allocate<T>(int x)
        {
            return Allocate<T>((long)x);
        }
        public T[] Allocate<T>(long x, ComputeMemoryFlags flags = ComputeMemoryFlags.ReadWrite) where T : struct
        {
            T[] devMem = new T[0];
            ComputeBuffer<T> a = new ComputeBuffer<T>(_context, flags, x);
            AddToDeviceMemory(devMem, new CLDevicePtrEx<T>(a, (int)x, _context));
            return devMem;
        }

        public override T[,] Allocate<T>(int x, int y)
        {
            return Allocate<T>((long)x, (long)y);
        }
        public T[,] Allocate<T>(long x, long y, ComputeMemoryFlags flags = ComputeMemoryFlags.ReadWrite) where T : struct
        {
            T[,] devMem = new T[0, 0];
            ComputeBuffer<T> a = new ComputeBuffer<T>(_context, flags, x * y);
            AddToDeviceMemory(devMem, new CLDevicePtrEx<T>(a, (int)x, (int)y, _context));
            return devMem;
        }
        public override T[, ,] Allocate<T>(int x, int y, int z)
        {
            return Allocate<T>((long)x, (long)y, (long)z);
        }
        public T[, ,] Allocate<T>(long x, long y, long z, ComputeMemoryFlags flags = ComputeMemoryFlags.ReadWrite) where T : struct
        {
            T[,,] devMem = new T[0, 0, 0];
            ComputeBuffer<T> a = new ComputeBuffer<T>(_context, flags, x * y * z);
            AddToDeviceMemory(devMem, new CLDevicePtrEx<T>(a, (int)x, (int)y, (int)z, _context));
            return devMem;
        }

        public override T[] Allocate<T>(T[] hostArray)
        {
            return Allocate<T>(hostArray.LongLength);
        }

        public override T[,] Allocate<T>(T[,] hostArray)
        {
            return Allocate<T>(hostArray.GetLongLength(0), hostArray.GetLongLength(1));
        }

        public override T[, ,] Allocate<T>(T[, ,] hostArray)
        {
            return Allocate<T>(hostArray.GetLongLength(0), hostArray.GetLongLength(1), hostArray.GetLongLength(2));
        }

        /// <summary>
        /// Performs an aligned host memory allocation.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">The x size.</param>
        /// <returns>
        /// Pointer to allocated memory.
        /// </returns>
        public override IntPtr HostAllocate<T>(int x)
        {
            int bytes = MSizeOf(typeof(T)) * x;
            int align = 4096;
            byte[] buffer = new byte[bytes + align];
            GCHandle handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            IntPtr intPtr;// = handle.AddrOfPinnedObject();

            long ptr = handle.AddrOfPinnedObject().ToInt64();
            // round up ptr to nearest 'byteAlignment' boundary
            ptr = align > 0 ? (ptr + align - 1) & ~(align - 1) : ptr;
            intPtr = new IntPtr(ptr);

            _hostHandles.Add(intPtr, handle);
            return intPtr;
        }

        #region DoSet


        /// <summary>
        /// Gets or sets the size of the array used for Set operations in OpenCL 1.0 and 1.1 devices.
        /// </summary>
        /// <value>
        /// The size of the memset array.
        /// </value>
        public int MemsetArraySize
        {
            get { return _memsetArray.Length; }
            set 
            {
                if (value != _memsetArray.Length)
                    _memsetArray = new byte[value];
            }
        }

        private byte[] _memsetArray;

        protected override void DoSet<T>(Array devArray, int offset = 0, int count = 0)
        {
            DoSet<T>(devArray, (long)offset, (long)count);
        }

        protected void DoSet<T>(Array devArray, long offset = 0, long count = 0) where T : struct
        {
            CLDevicePtrEx<T> ptr = GetDeviceMemory(devArray) as CLDevicePtrEx<T>;
            long totalSize = ptr.TotalSize;
            long elemSize = MSizeOf(typeof(T));
            
            if (count == 0)
                count = totalSize;
            long countBytes = count * elemSize;
            if (offset < 0 || offset >= totalSize)
                throw new IndexOutOfRangeException("offset");
            long offsetBytes = offset * elemSize;
            if (OpenCLVersion.Minor > 1)
            {
                int[] fillBuff = new int[] { 0 };
                GCHandle handle_fillBuff = GCHandle.Alloc(fillBuff, GCHandleType.Pinned);
                ComputeErrorCode error;
                try
                {
                    error = CL11_ex.EnqueueFillBuffer(
                        _defaultSynchronousQueue.Handle, ptr.Handle, handle_fillBuff.AddrOfPinnedObject(), new IntPtr(1), new IntPtr(offsetBytes), new IntPtr(countBytes), 0, null, null);
                }
                finally
                {
                    handle_fillBuff.Free();
                }
#warning TODO Throw a CUDAfy exception instead
                ComputeException.ThrowOnError(error);
            }
            else
            {
                GCHandle sourceGCHandle = GCHandle.Alloc(_memsetArray, GCHandleType.Pinned);
                try
                {
                    IntPtr _ptrHostBuff = sourceGCHandle.AddrOfPinnedObject();
                    long sizeofT = Marshal.SizeOf(typeof(T));
                    long len0hostBuffT = _memsetArray.Length / sizeofT;
                    long maxScanIx = Math.Min(totalSize, offset + count);
                    for (long k = offset; k < maxScanIx; k += len0hostBuffT)
                    {
                        long patchSize = Math.Min(len0hostBuffT, maxScanIx - k);
                        _defaultSynchronousQueue.WriteEx<T>(ptr.DevPtr.Handle, false, k, patchSize, _ptrHostBuff, null);
                    }
                }
                finally
                {
                    sourceGCHandle.Free();
                }
            }
            _defaultSynchronousQueue.Finish();
        }

        #endregion DoSet

        public override void Free(object devArray)
        {
            if (devArray == null)
                throw new ArgumentNullException("devArray is null");
            CLDevicePtrExInter ptr_src = GetDeviceMemory(devArray) as CLDevicePtrExInter;
            ptr_src.DevPtr_base.Dispose();
            RemoveFromDeviceMemoryEx(ptr_src);
        }

        public override void FreeAll()
        {
            lock (_lock)
            {
                foreach (CLDevicePtrExInter ptrEx in GetDeviceMemories())
                {
                    ptrEx.DevPtr_base.Dispose();
                }
                ClearDeviceMemory();
            }
        }
    }

    public abstract class CLDevicePtrExInter : DevicePtrEx
    {
        public abstract CLMemoryHandle Handle { get; }

        /// <summary>
        /// stores the dev PTR in its base form without generic types.
        /// </summary>
        public ComputeMemory DevPtr_base { get; protected set; }

        /// <summary>
        /// Gets or sets the size of the element.
        /// </summary>
        /// <value>
        /// The size of the element.
        /// </value>
        public int ElementSize { get; protected set; }
    }

    /// <summary>
    /// Internal use.
    /// </summary>
    public class CLDevicePtrEx<T> : CLDevicePtrExInter where T : struct
    {
        //protected CUDevicePtrEx(CUcontext? context)
        //{
        //    Context = context;
        //}

        /// <summary>
        /// Initializes a new instance of the <see cref="CUDevicePtrEx"/> class.
        /// </summary>
        /// <param name="devPtr">The dev PTR.</param>
        /// <param name="context">The context.</param>       
        public CLDevicePtrEx(ComputeBuffer<T> devPtr, ComputeContext context)
            : this(devPtr, 1, 1, 1, context)
        {
            Dimensions = 0;
        }

#warning array lengths should be in long, everywhere.
        /// <summary>
        /// Initializes a new instance of the <see cref="CUDevicePtrEx"/> class.
        /// </summary>
        /// <param name="devPtr">The dev PTR.</param>
        /// <param name="xSize">Size of the x.</param>
        /// <param name="context">The context.</param>     
        public CLDevicePtrEx(ComputeBuffer<T> devPtr, int xSize, ComputeContext context)
            : this(devPtr, xSize, 1, 1, context)
        {
            Dimensions = 1;
        }

        ///// <summary>
        ///// Casts the specified pointer.
        ///// </summary>
        ///// <typeparam name="T"></typeparam>
        ///// <param name="ptrEx">The pointer.</param>
        ///// <param name="offset">The offset.</param>
        ///// <param name="xSize">Size of the x.</param>
        ///// <returns></returns>
        //public CLDevicePtrEx Cast<T>(CUDevicePtrEx ptrEx, int offset, int xSize)
        //{
        //    int size = GPGPU.MSizeOf(typeof(T));
        //    CUdeviceptr ptrOffset = ptrEx.DevPtr + (long)(offset * size);
        //    CUDevicePtrEx newPtrEx = new CUDevicePtrEx(ptrOffset, xSize, ptrEx.Context);
        //    newPtrEx.CreatedFromCast = true;
        //    ptrEx.AddChild(newPtrEx);
        //    return newPtrEx;
        //}

        /// <summary>
        /// Initializes a new instance of the <see cref="CUDevicePtrEx"/> class.
        /// </summary>
        /// <param name="devPtr">The dev PTR.</param>
        /// <param name="xSize">Size of the x.</param>
        /// <param name="ySize">Size of the y.</param>
        /// <param name="context">The context.</param>       
        public CLDevicePtrEx(ComputeBuffer<T> devPtr, int xSize, int ySize, ComputeContext context)
            : this(devPtr, xSize, ySize, 1, context)
        {
            Dimensions = 2;
        }

        ///// <summary>
        ///// Casts the specified pointer.
        ///// </summary>
        ///// <typeparam name="T"></typeparam>
        ///// <param name="ptrEx">The pointer.</param>
        ///// <param name="offset">The offset.</param>
        ///// <param name="xSize">Size of the x.</param>
        ///// <param name="ySize">Size of the y.</param>
        ///// <returns></returns>
        //public CLDevicePtrEx Cast<T>(CUDevicePtrEx ptrEx, int offset, int xSize, int ySize)
        //{
        //    int size = GPGPU.MSizeOf(typeof(T));
        //    CUdeviceptr ptrOffset = ptrEx.DevPtr + (long)(offset * size);
        //    CUDevicePtrEx newPtrEx = new CUDevicePtrEx(ptrOffset, xSize, ySize, ptrEx.Context);
        //    newPtrEx.CreatedFromCast = true;
        //    ptrEx.AddChild(newPtrEx);
        //    return newPtrEx;
        //}

        /// <summary>
        /// Initializes a new instance of the <see cref="CUDevicePtrEx"/> class.
        /// </summary>
        /// <param name="devPtr">The dev PTR.</param>
        /// <param name="xSize">Size of the x.</param>
        /// <param name="ySize">Size of the y.</param>
        /// <param name="zSize">Size of the z.</param>
        /// <param name="context">The context.</param>       
        public CLDevicePtrEx(ComputeBuffer<T> devPtr, int xSize, int ySize, int zSize, ComputeContext context)
        {
            CreatedFromCast = false;
            DevPtr = devPtr;
            XSize = xSize;
            YSize = ySize;
            ZSize = zSize;
            Dimensions = 3;
            Context = context;
            ElementSize = GPGPU.MSizeOf(typeof(T));

            //OriginalSize = originalSize < 0 ? TotalSize : originalSize;
        }

        ///// <summary>
        ///// Casts the specified pointer.
        ///// </summary>
        ///// <typeparam name="T"></typeparam>
        ///// <param name="ptrEx">The pointer.</param>
        ///// <param name="offset">The offset.</param>
        ///// <param name="xSize">Size of the x.</param>
        ///// <param name="ySize">Size of the y.</param>
        ///// <param name="zSize">Size of the z.</param>
        ///// <returns></returns>
        //public CLDevicePtrEx Cast<T>(CUDevicePtrEx ptrEx, int offset, int xSize, int ySize, int zSize)
        //{
        //    int size = GPGPU.MSizeOf(typeof(T));
        //    CUdeviceptr ptrOffset = ptrEx.DevPtr + (long)(offset * size);
        //    CUDevicePtrEx newPtrEx = new CUDevicePtrEx(ptrOffset, xSize, ySize, zSize, ptrEx.Context);
        //    newPtrEx.CreatedFromCast = true;
        //    ptrEx.AddChild(newPtrEx);
        //    return newPtrEx;
        //}

        /// <summary>
        /// Gets the dev PTR.
        /// </summary>
        public ComputeBuffer<T> DevPtr
        {
            get
            {
                return (ComputeBuffer<T>)DevPtr_base;
            }
            protected set 
            {
                DevPtr_base = value;
            }
        }

        public override CLMemoryHandle Handle
        {
            get
            {
                return DevPtr.Handle;
            }
        }

        /// <summary>
        /// Gets the IntPtr in DevPtr.
        /// </summary>
        public override IntPtr Pointer
        {
            get { return DevPtr.Handle.Value; }
        }


        /// <summary>
        /// Gets the context.
        /// </summary>
        public ComputeContext Context { get; private set; }


    }
}
