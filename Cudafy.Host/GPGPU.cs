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
using System.Runtime.InteropServices;
#if !NET35
using System.Dynamic;
#endif
using GASS.CUDA;
using GASS.CUDA.Types;

namespace Cudafy.Host
{    
    public enum ePointerAttribute
    {
        Context = 1,        /**< The ::CUcontext on which a pointer was allocated or registered */
        MemoryType = 2,     /**< The ::CUmemorytype describing the physical location of a pointer */
        DevicePointer = 3,  /**< The address at which a pointer's memory may be accessed on the device */
        HostPointer = 4,    /**< The address at which a pointer's memory may be accessed on the host */
        P2PTokens = 5       /**< A pair of tokens for use with the nv-p2p.h Linux kernel interface */
    }
    
    
    /// <summary>
    /// Abstract base class for General Purpose GPUs.
    /// </summary>
    public abstract class GPGPU : IDisposable
    {       
        /// <summary>
        /// Initializes a new instance of the <see cref="GPGPU"/> class.
        /// </summary>
        /// <param name="deviceId">The device id.</param>
        protected GPGPU(int deviceId = 0)
        {
            _hostHandles = new Dictionary<IntPtr, GCHandle>();
            _lock = new object();
            _stopWatch = new Stopwatch();
            _deviceMemory = new Dictionary<object, DevicePtrEx>();
            _streams = new Dictionary<int, object>();
#if !NET35
            _dynamicLauncher = new DynamicLauncher(this);
#endif
            DeviceId = deviceId;
        }

        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="GPGPU"/> is reclaimed by garbage collection.
        /// </summary>
        ~GPGPU()
        {
            Dispose(false);
        }

        #region Properties

        /// <summary>
        /// Gets the device id.
        /// </summary>
        public int DeviceId { get; private set; }

        #endregion

        // Track whether Dispose has been called.
        private bool _disposed = false;
#if !NET35
        private DynamicLauncher _dynamicLauncher;
#endif
        /// <summary>
        /// Gets a value indicating whether this instance is disposed.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if this instance is disposed; otherwise, <c>false</c>.
        /// </value>
        public bool IsDisposed
        {
            get { lock(_lock){ return _disposed; } }
        }

        /// <summary>
        /// Releases unmanaged and - optionally - managed resources
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            lock (_lock)
            {
                Debug.WriteLine(string.Format("GPGPU::Dispose({0})", disposing));
                // Check to see if Dispose has already been called.
                if (!this._disposed)
                {
                    Debug.WriteLine("Disposing");
                    // If disposing equals true, dispose all managed
                    // and unmanaged resources.
                    if (disposing)
                    {
                        // Dispose managed resources.
                    }
                    
                    // Call the appropriate methods to clean up
                    // unmanaged resources here.
                    // If disposing is false,
                    // only the following code is executed.
                    FreeAll();
                    HostFreeAll();
                    DestroyStreams();

                    // Note disposing has been done.
                    //_disposed = true;
                    UnloadModules();
                }
                else
                    Debug.WriteLine("Already disposed");
            }
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            _disposed = true;
            // This object will be cleaned up by the Dispose method.
            // Therefore, you should call GC.SupressFinalize to
            // take this object off the finalization queue
            // and prevent finalization code for this object
            // from executing a second time.
            GC.SuppressFinalize(this);
            
        }

        private Stopwatch _stopWatch;
        
        /// <summary>
        /// Internal use.
        /// </summary>
        protected object _lock;

        /// <summary>
        /// Stores pointers to data on the device.
        /// </summary>
        private Dictionary<object, DevicePtrEx> _deviceMemory;

        /// <summary>
        /// Locks this instance.
        /// </summary>
        public virtual void Lock()
        {
            if (!IsMultithreadingEnabled)
                throw new CudafyHostException(CudafyHostException.csMULTITHREADING_IS_NOT_ENABLED);
            IsLocked = true;
        }

        /// <summary>
        /// Unlocks this instance.
        /// </summary>
        public virtual void Unlock()
        {
            if (!IsMultithreadingEnabled)
                throw new CudafyHostException(CudafyHostException.csMULTITHREADING_IS_NOT_ENABLED);
            IsLocked = false;
        }

        /// <summary>
        /// Gets a value indicating whether this instance is locked.
        /// </summary>
        /// <value>
        ///   <c>true</c> if this instance is locked; otherwise, <c>false</c>.
        /// </value>
        public virtual bool IsLocked { get; protected set; }

        /// <summary>
        /// Allows multiple threads to access this GPU.
        /// </summary>
        public virtual void EnableMultithreading()
        {
            _isMultithreadedEnabled = true;
        }

        /// <summary>
        /// Called once multiple threads have completed work.
        /// </summary>
        public virtual void DisableMultithreading()
        {
            _isMultithreadedEnabled = false;
        }

        protected object _peerAccessLock = new object();

        protected List<GPGPU> _peerAccessList = new List<GPGPU>();

        /// <summary>
        /// Enables peer access from within a kernel. 
        /// </summary>
        /// <param name="peer">Peer to access. This is a one-way relationship.</param>
        public virtual void EnablePeerAccess(GPGPU peer)
        {
            lock (_peerAccessLock)
            {
                if (_peerAccessList.Contains(peer))
                    throw new CudafyHostException(CudafyHostException.csPEER_ACCESS_ALREADY_ENABLED);
                if (this == peer)
                    throw new CudafyHostException(CudafyHostException.csPEER_ACCESS_TO_SELF_NOT_ALLOWED);
                _peerAccessList.Add(peer);
            }
        }

        /// <summary>
        /// Disables peer access.
        /// </summary>
        /// <param name="peer">Accessible peer to disable access to.</param>
        public virtual void DisablePeerAccess(GPGPU peer)
        {
            lock (_peerAccessLock)
            {
                if (!_peerAccessList.Contains(peer))
                    throw new CudafyHostException(CudafyHostException.csPEER_ACCESS_WAS_NOT_ENABLED);
                _peerAccessList.Remove(peer);
            }
        }

        /// <summary>
        /// Use this to check if device supports direct access from kernel to another device.
        /// </summary>
        /// <param name="peer">Peer to access.</param>
        /// <returns>True if access is possible, else false.</returns>
        public abstract bool CanAccessPeer(GPGPU peer);
        //Context = 1,        /**< The ::CUcontext on which a pointer was allocated or registered */
        //MemoryType = 2,     /**< The ::CUmemorytype describing the physical location of a pointer */
        //DevicePointer = 3,  /**< The address at which a pointer's memory may be accessed on the device */
        //HostPointer = 4,    /**< The address at which a pointer's memory may be accessed on the host */
        //P2PTokens = 5       /**< A pair of tokens for use with the nv-p2p.h Linux kernel interface */
        //public IntPtr GetPointerAttribute<T>(ePointerAttribute attr, T[] data)
        //{
        //    return DoGetPointerAttribute<T>(attr, data);
        //}

        //public IntPtr GetPointerAttribute<T>(ePointerAttribute attr, T[,] data)
        //{
        //    return DoGetPointerAttribute<T>(attr, data);
        //}

        //public IntPtr GetPointerAttribute<T>(ePointerAttribute attr, T[,,] data)
        //{
        //    return DoGetPointerAttribute<T>(attr, data);
        //}
        //http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/group__CUDA__UNIFIED_g0c28ed0aff848042bc0533110e45820c.html

        //public enum eMemoryType
        //{
        //    Host,
        //    Device,
        //    Array,
        //    Unified
        //}

        //public eMemoryType GetPointerMemoryType<T>(T[] data)
        //{
        //    if(_deviceMemory.ContainsKey(data))
        //        return eMeory
        //}

        ////protected eMemoryType 

        //protected virtual IntPtr DoGetPointerAttribute<T>(ePointerAttribute attr, Array data)
        //{
        //    throw new NotSupportedException();
        //}

        public virtual eArchitecture GetArchitecture()
        {
            //if (!(this is CudaGPU) && !(this is EmulatedGPU))
            //    throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, this.GetType());
            
            var capability = this.GetDeviceProperties(false).Capability;

            switch (capability.Major)
            {
                case 0:
                    switch (capability.Minor)
                    {
                        case 1: return eArchitecture.Emulator;
                        default:
                            throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, capability.ToString());
                    }

                case 1:

                    switch (capability.Minor)
                    {
                        case 0:  return eArchitecture.sm_10;


                        case 1: return eArchitecture.sm_11;

                        case 2: return eArchitecture.sm_12;

                        case 3: return eArchitecture.sm_13;

                        default:
                            throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, capability.ToString());

                    }

                case 2:

                    switch (capability.Minor)
                    {

                        case 0: return eArchitecture.sm_20;

                        case 1: return eArchitecture.sm_21;

                        default:
                            throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, capability.ToString());

                    }

                case 3:

                    switch (capability.Minor)
                    {

                        case 0: return eArchitecture.sm_30;

                        case 5: return eArchitecture.sm_35;

                        case 7: return eArchitecture.sm_37;
                        default:
                            throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, capability.ToString());
                    }
                case 5:

                    switch (capability.Minor)
                    {

                        case 0: return eArchitecture.sm_50;

                        case 2: return eArchitecture.sm_52;

                        default:
                            throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, capability.ToString());
                    }
                default:
                    throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, capability.ToString());
            }

        }

        /// <summary>
        /// Copies from one device to another device. Depending on whether RDMA is supported the transfer may or may not be via CPU and system memory.
        /// </summary>
        /// <typeparam name="T">Data </typeparam>
        /// <param name="src"></param>
        /// <param name="srcOffset"></param>
        /// <param name="peer"></param>
        /// <param name="dst"></param>
        /// <param name="dstOffset"></param>
        /// <param name="count"></param>
        public virtual void CopyDeviceToDevice<T>(T[] src, int srcOffset, GPGPU peer, T[] dst, int dstOffset, int count) where T : struct
        {
            DoCopyDeviceToDevice<T>(src, srcOffset, peer, dst, dstOffset, count);
        }

        /// <summary>
        /// Copies from one device to another device. Depending on whether RDMA is supported the transfer may or may not be via CPU and system memory.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="src">Source data.</param>
        /// <param name="srcOffset">Source array.</param>
        /// <param name="peer">Target device.</param>
        /// <param name="dst">Destination array.</param>
        /// <param name="dstOffset">Destination offset.</param>
        /// <param name="count">Number of samples.</param>
        public virtual void CopyDeviceToDeviceAsync<T>(T[] src, int srcOffset, GPGPU peer, T[] dst, int dstOffset, int count, int stream) where T : struct
        {
            DoCopyDeviceToDeviceAsync<T>(src, srcOffset, peer, dst, dstOffset, count, stream);
        }


        /// <summary>
        /// Does copy to peer asynchronously.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The SRC dev array.</param>
        /// <param name="srcOffset">The SRC offset.</param>
        /// <param name="peer">The peer.</param>
        /// <param name="dstDevArray">The DST dev array.</param>
        /// <param name="dstOffet">The DST offet.</param>
        /// <param name="count">The count.</param>
        protected abstract void DoCopyDeviceToDevice<T>(Array srcDevArray, int srcOffset, GPGPU peer, Array dstDevArray, int dstOffet, int count) where T : struct;

        /// <summary>
        /// Does copy to peer asynchronously.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The SRC dev array.</param>
        /// <param name="srcOffset">The SRC offset.</param>
        /// <param name="peer">The peer.</param>
        /// <param name="dstDevArray">The DST dev array.</param>
        /// <param name="dstOffet">The DST offet.</param>
        /// <param name="count">The count.</param>
        /// <param name="stream">Stream id.</param>
        protected abstract void DoCopyDeviceToDeviceAsync<T>(Array srcDevArray, int srcOffset, GPGPU peer, Array dstDevArray, int dstOffet, int count, int stream) where T : struct;

        /// <summary>
        /// Gets a value indicating whether this instance has multithreading enabled.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if this instance is multithreading enabled; otherwise, <c>false</c>.
        /// </value>
        public virtual bool IsMultithreadingEnabled
        {
            get { return _isMultithreadedEnabled; }
        }

        private bool _isMultithreadedEnabled = false;

        /// <summary>
        /// Sets the current context to the context associated with this device when it was created.
        /// </summary>
        public virtual void SetCurrentContext()
        {

        }

        /// <summary>
        /// Gets a value indicating whether this instance is current context. You must ensure this is true before 
        /// attempting communication with device.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if this instance is current context; otherwise, <c>false</c>.
        /// </value>
        public virtual bool IsCurrentContext
        {
            get { return true; }
        }


        /// <summary>
        /// Explicitly creates a stream.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        public abstract void CreateStream(int streamId);

        #region Dynamic
#if !NET35
        /// <summary>
        /// Gets the dynamic launcher with grid and block sizes equal to 1.
        /// Allows GPU functions to be called using dynamic language run-time. For example:
        /// gpgpu.Launch().myGPUFunction(x, y, res)         
        /// </summary>
        /// <returns>Dynamic launcher</returns>
        public dynamic Launch()
        {
            return Launch(1, 1, -1);
        }

        ///// <summary>
        ///// Gets the dynamic launcher.
        ///// Allows GPU functions to be called using dynamic language run-time. For example:
        ///// gpgpu.Launch(16, new dim3(8,8)).myGPUFunction(x, y, res)   
        ///// </summary>
        ///// <param name="gridSize">Size of the grid.</param>
        ///// <param name="blockSize">Size of the block.</param>
        ///// <param name="streamId">The stream id or -1 for synchronous.</param>
        ///// <returns>Dynamic launcher</returns>
        //public dynamic Launch(int gridSize, dim3 blockSize, int streamId = -1)
        //{
        //    return Launch(new dim3(gridSize), blockSize, streamId);
        //}

        ///// <summary>
        ///// Gets the dynamic launcher.
        ///// Allows GPU functions to be called using dynamic language run-time. For example:
        ///// gpgpu.Launch(new dim3(8,8), 16).myGPUFunction(x, y, res)   
        ///// </summary>
        ///// <param name="gridSize">Size of the grid.</param>
        ///// <param name="blockSize">Size of the block.</param>
        ///// <param name="streamId">The stream id or -1 for synchronous.</param>
        ///// <returns>Dynamic launcher</returns>
        //public dynamic Launch(dim3 gridSize, int blockSize, int streamId = -1)
        //{
        //    return Launch(gridSize, new dim3(blockSize), streamId);
        //}

        ///// <summary>
        ///// Gets the dynamic launcher.
        ///// Allows GPU functions to be called using dynamic language run-time. For example:
        ///// gpgpu.Launch(16, 16).myGPUFunction(x, y, res)   
        ///// </summary>
        ///// <param name="gridSize">Size of the grid.</param>
        ///// <param name="blockSize">Size of the block.</param>
        ///// <param name="streamId">The stream id or -1 for synchronous.</param>
        ///// <returns>Dynamic launcher</returns>
        //public dynamic Launch(int gridSize, int blockSize, int streamId = -1)
        //{
        //    return Launch(new dim3(gridSize), new dim3(blockSize), streamId);
        //}

        /// <summary>
        /// Gets the dynamic launcher.
        /// Allows GPU functions to be called using dynamic language run-time. For example:
        /// gpgpu.Launch(new dim3(8,8), new dim3(8,8)).myGPUFunction(x, y, res)   
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">The stream id or -1 for synchronous.</param>
        /// <returns>Dynamic launcher</returns>
        public dynamic Launch(dim3 gridSize, dim3 blockSize, int streamId = -1)
        {
            _dynamicLauncher.BlockSize = blockSize;
            _dynamicLauncher.GridSize = gridSize;
            _dynamicLauncher.StreamId = streamId;
            return _dynamicLauncher;
        }


#endif
        #endregion

        /// <summary>
        /// Adds to device memory.
        /// </summary>
        /// <param name="key">The key.</param>
        /// <param name="value">The value.</param>
        protected void AddToDeviceMemory(object key, DevicePtrEx value)
        {
            lock (_lock)
            {
                _deviceMemory.Add(key, value);
            }
        }

        /// <summary>
        /// Gets the device memory pointers.
        /// </summary>
        /// <returns>All data pointers currently on device.</returns>
        public object[] GetDeviceMemoryPointers()
        {
            lock (_lock)
            {
                return _deviceMemory.Keys.ToArray();
            }
        }

        /// <summary>
        /// Gets the device memory pointer.
        /// </summary>
        /// <param name="ptrEx">The pointer.</param>
        /// <returns></returns>
        public object GetDeviceMemoryPointer(DevicePtrEx ptrEx)
        {
            lock (_lock)
            {
                return _deviceMemory.Values.Where(v => v == ptrEx).FirstOrDefault();
            }
        }

        /// <summary>
        /// Gets the device memory from IntPtr.
        /// </summary>
        /// <param name="ptr">The PTR.</param>
        /// <returns></returns>
        public object GetDeviceMemoryFromIntPtr(IntPtr ptr)
        {
            lock (_lock)
            {
                var kvps = _deviceMemory.Where(kvp => kvp.Value.Pointer == ptr);
                if (kvps.Count() == 0)
                    return null;
                var k = kvps.FirstOrDefault();
                return k.Key;                
            }
        }

        /// <summary>
        /// Gets the device memory for key specified.
        /// </summary>
        /// <param name="devArray">The dev array.</param>
        /// <returns>Device memory</returns>
        public DevicePtrEx GetDeviceMemory(object devArray)
        {
            DevicePtrEx ptr;
            lock (_lock)
            {
                if (devArray == null || !_deviceMemory.ContainsKey(devArray))
                    throw new CudafyHostException(CudafyHostException.csDATA_IS_NOT_ON_GPU);
                ptr = _deviceMemory[devArray] as DevicePtrEx;
            }
            return ptr;
        }

        /// <summary>
        /// Tries to get the device memory.
        /// </summary>
        /// <param name="devArray">The dev array.</param>
        /// <returns>Device memory or null if not found.</returns>
        public DevicePtrEx TryGetDeviceMemory(object devArray)
        {
            DevicePtrEx ptr;
            lock (_lock)
            {
                if (devArray == null || !_deviceMemory.ContainsKey(devArray))
                    ptr = null;
                else
                    ptr = _deviceMemory[devArray] as DevicePtrEx;
            }
            return ptr;
        }

        /// <summary>
        /// Checks if specified device memory value exists.
        /// </summary>
        /// <param name="val">The device memory instance.</param>
        /// <returns></returns>
        protected bool DeviceMemoryValueExists(object val)
        {
#if !NET35
                return _deviceMemory.Values.Contains(val);
#else
            return _deviceMemory.Values.Contains(val as DevicePtrEx); //foreach(var v in _deviceMemory.Values) 
#endif
        }

        /// <summary>
        /// Gets the device memories.
        /// </summary>
        /// <returns></returns>
        protected IEnumerable<object> GetDeviceMemories()
        {
            lock (_lock)
            {
#if !NET35                
                return _deviceMemory.Values;
#else
                foreach (var v in _deviceMemory.Values)
                    yield return v;
#endif
            }
        }

        /// <summary>
        /// Clears the device memory.
        /// </summary>
        protected void ClearDeviceMemory()
        {
            lock (_lock)
            {
                _deviceMemory.Clear();
            }
        }

        /// <summary>
        /// Removes from device memory.
        /// </summary>
        /// <param name="key">The key.</param>
        protected void RemoveFromDeviceMemory(object key)
        {
            lock (_lock)
            {
                _deviceMemory.Remove(key);
            }
        }

        /// <summary>
        /// Removes from device memory based on specified pointer.
        /// </summary>
        /// <param name="ptrEx">The PTR ex.</param>
        protected void RemoveFromDeviceMemoryEx(DevicePtrEx ptrEx)
        {
            lock (_lock)
            {
                var kvp = _deviceMemory.Where(k => k.Value == ptrEx).FirstOrDefault();
                foreach (var child in kvp.Value.GetAllChildren())
                {
                    var list = _deviceMemory.Where(ch => ch.Value == child).ToList();
                    foreach(var item in list)
                        _deviceMemory.Remove(item.Key);


                }
                _deviceMemory.Remove(kvp.Key);
                
            }
        }

        /// <summary>
        /// Stores streams.
        /// </summary>
        protected Dictionary<int, object> _streams;

        ///// <summary>
        ///// Currently loaded module.
        ///// </summary>
        //protected CudafyModule _module;

        /// <summary>
        /// Gets the device properties.
        /// </summary>
        /// <param name="useAdvanced">States whether to get advanced properties.</param>
        /// <returns>Device properties.</returns>
        public abstract GPGPUProperties GetDeviceProperties(bool useAdvanced = true);

        /// <summary>
        /// Gets the free memory.
        /// </summary>
        /// <value>The free memory.</value>
        public abstract ulong FreeMemory { get; }

        /// <summary>
        /// Gets the total memory.
        /// </summary>
        /// <value>The total memory.</value>
        public abstract ulong TotalMemory { get; }

        /// <summary>
        /// Gets the names of all global functions.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<string> GetFunctionNames()
        {
            foreach (var mod in _modules)
                foreach (var f in mod.Functions.Values.Where(f => f.MethodType == eKernelMethodType.Global))
                    yield return f.Name;
        }

        /// <summary>
        /// Gets the stream object.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        /// <returns>Stream object.</returns>
        public virtual object GetStream(int streamId)
        {
            lock (_lock)
            {
                if (streamId >= 0 && !_streams.ContainsKey(streamId))
                    _streams.Add(streamId, streamId);
                return _streams[streamId];
            }
        }

        /// <summary>
        /// Copies to constant memory.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="devArray">The device array.</param>
        public void CopyToConstantMemory<T>(T[] hostArray, T[] devArray) where T : struct
        {
            int count = hostArray.Length;
            KernelConstantInfo ci = InitializeCopyToConstantMemory(hostArray, 0, devArray, 0, ref count);
            DoCopyToConstantMemory<T>(hostArray, 0, devArray, 0, count, ci);
        }

        /// <summary>
        /// Copies to constant memory.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="devArray">The device array.</param>
        public void CopyToConstantMemory<T>(T[,] hostArray, T[,] devArray) where T : struct
        {
            int count = hostArray.Length;
            KernelConstantInfo ci = InitializeCopyToConstantMemory(hostArray, 0, devArray, 0, ref count);
            DoCopyToConstantMemory<T>(hostArray, 0, devArray, 0, count, ci);
        }

        /// <summary>
        /// Copies to constant memory.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="devArray">The device array.</param>
        public void CopyToConstantMemory<T>(T[, ,] hostArray, T[, ,] devArray) where T : struct
        {
            int count = hostArray.Length;
            KernelConstantInfo ci = InitializeCopyToConstantMemory(hostArray, 0, devArray, 0, ref count);
            DoCopyToConstantMemory<T>(hostArray, 0, devArray, 0, count, ci);
        }

        /// <summary>
        /// Copies to constant memory.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="count">The number of element to copy.</param>
        public void CopyToConstantMemory<T>(T[] hostArray, int hostOffset, T[] devArray, int devOffset, int count) where T : struct
        {
            KernelConstantInfo ci = InitializeCopyToConstantMemory(hostArray, hostOffset, devArray, devOffset, ref count);
            DoCopyToConstantMemory<T>(hostArray, hostOffset, devArray, devOffset, count, ci);
        }

        /// <summary>
        /// Copies to constant memory async.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyToConstantMemoryAsync<T>(IntPtr hostArray, int hostOffset, T[] devArray, int devOffset, int count, int streamId) where T : struct
        {
            KernelConstantInfo ci = InitializeCopyToConstantMemory(null, hostOffset, devArray, devOffset, ref count);
            DoCopyToConstantMemoryAsync<T>(hostArray, hostOffset, devArray, devOffset, count, ci, streamId);
        }

        /// <summary>
        /// Copies to constant memory asynchronously using smart copy.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="streamId">The stream id.</param>
        /// <param name="stagingPost">The staging post.</param>
        public void CopyToConstantMemoryAsync<T>(T[] hostArray, int hostOffset, T[] devArray, int devOffset, int count, int streamId, IntPtr stagingPost) where T : struct
        {
            if (!IsSmartCopyEnabled)
                throw new CudafyHostException(CudafyHostException.csSMART_COPY_IS_NOT_ENABLED);
            DoCopyToDeviceAsync<T>(hostArray, hostOffset, devArray, devOffset, count, streamId, stagingPost, true);
        }

        /// <summary>
        /// Copies to constant memory async.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyToConstantMemoryAsync<T>(IntPtr hostArray, int hostOffset, T[,] devArray, int devOffset, int count, int streamId) where T : struct
        {
            KernelConstantInfo ci = InitializeCopyToConstantMemory(null, hostOffset, devArray, devOffset, ref count);
            DoCopyToConstantMemoryAsync<T>(hostArray, hostOffset, devArray, devOffset, count, ci, streamId);
        }

        /// <summary>
        /// Copies to constant memory async.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyToConstantMemoryAsync<T>(IntPtr hostArray, int hostOffset, T[, ,] devArray, int devOffset, int count, int streamId) where T : struct
        {
            KernelConstantInfo ci = InitializeCopyToConstantMemory(null, hostOffset, devArray, devOffset, ref count);
            DoCopyToConstantMemoryAsync<T>(hostArray, hostOffset, devArray, devOffset, count, ci, streamId);
        }

        ///// <summary>
        ///// Copies to constant memory.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="hostArray">The host array.</param>
        ///// <param name="hostOffset">The host offset.</param>
        ///// <param name="devArray">The device array.</param>
        ///// <param name="devOffset">The device offset.</param>
        ///// <param name="count">The number of element to copy.</param>
        ///// <param name="streamId">Stream id.</param>
        //public void CopyToConstantMemoryAsync<T>(T[] hostArray, int hostOffset, T[] devArray, int devOffset, int count, int streamId)
        //{
        //    KernelConstantInfo ci = InitializeCopyToConstantMemory(hostArray, hostOffset, devArray, devOffset, count);
        //    DoCopyToConstantMemory<T>(hostArray, hostOffset, devArray, devOffset, count, ci, streamId);
        //}

        ///// <summary>
        ///// Copies to constant memory.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="hostArray">The host array.</param>
        ///// <param name="hostOffset">The host offset.</param>
        ///// <param name="devArray">The device array.</param>
        ///// <param name="devOffset">The device offset.</param>
        ///// <param name="count">The number of element to copy.</param>
        ///// <param name="streamId">Stream id.</param>
        //public void CopyToConstantMemoryAsync<T>(T[,] hostArray, int hostOffset, T[,] devArray, int devOffset, int count, int streamId)
        //{
        //    KernelConstantInfo ci = InitializeCopyToConstantMemory(hostArray, hostOffset, devArray, devOffset, count);
        //    DoCopyToConstantMemory<T>(hostArray, hostOffset, devArray, devOffset, count, ci, streamId);
        //}

        ///// <summary>
        ///// Copies to constant memory.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="hostArray">The host array.</param>
        ///// <param name="hostOffset">The host offset.</param>
        ///// <param name="devArray">The device array.</param>
        ///// <param name="devOffset">The device offset.</param>
        ///// <param name="count">The number of element to copy.</param>
        ///// <param name="streamId">Stream id.</param>
        //public void CopyToConstantMemoryAsync<T>(T[,,] hostArray, int hostOffset, T[,,] devArray, int devOffset, int count, int streamId)
        //{
        //    KernelConstantInfo ci = InitializeCopyToConstantMemory(hostArray, hostOffset, devArray, devOffset, count);
        //    DoCopyToConstantMemory<T>(hostArray, hostOffset, devArray, devOffset, count, ci, streamId);
        //}

        /// <summary>
        /// Initializes the copy to constant memory.
        /// </summary>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        /// <returns></returns>
        protected KernelConstantInfo InitializeCopyToConstantMemory(Array hostArray, int hostOffset, Array devArray, int devOffset, ref int count)
        {
            object o = null;
            KernelConstantInfo ci = null;
            foreach (var module in _modules)
            {
                foreach (var kvp in module.Constants)
                {
                    ci = kvp.Value;
                    o = ci.Information.GetValue(null);
                    if (o == devArray)
                        break;
                    o = null;
                }
                if (o != null)
                    break;
            }
            if (o == null)
                throw new CudafyHostException(CudafyHostException.csCONSTANT_MEMORY_NOT_FOUND);
            if (count == 0 && hostArray != null)
                count = hostArray.Length;
            if (count > devArray.Length - devOffset)
                throw new CudafyHostException(CudafyHostException.csINDEX_OUT_OF_RANGE);
            return ci;
        }

        /// <summary>
        /// Gets the device count.
        /// </summary>
        /// <returns>Number of devices of this type.</returns>
        public static int GetDeviceCount()
        {
            return 0;
        }

        /// <summary>
        /// Synchronizes context.
        /// </summary>
        public abstract void Synchronize();

        /// <summary>
        /// Starts the timer.
        /// </summary>
        public virtual void StartTimer()
        {
            _stopWatch.Start();
        }

        /// <summary>
        /// Stops the timer.
        /// </summary>
        /// <returns>Elapsed time.</returns>
        public virtual float StopTimer()
        {
            float time = 0;
            _stopWatch.Stop();
            time = (float)_stopWatch.ElapsedMilliseconds;
            _stopWatch.Reset();
            return time;
        }

        /// <summary>
        /// Loads module from file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        public void LoadModule(string filename)
        {
            CudafyModule km = string.IsNullOrEmpty(filename) ? null : CudafyModule.Deserialize(filename);
            LoadModule(km);
        }

        /// <summary>
        /// Internal use.
        /// </summary>
        protected List<CudafyModule> _modules = new List<CudafyModule>();

        /// <summary>
        /// Gets the modules.
        /// </summary>
        public IEnumerable<CudafyModule> Modules
        {
            get { return _modules; }
        }

        /// <summary>
        /// Gets the names of all members in all loaded modules.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<string> GetMemberNames()
        {
            foreach(var mod in Modules)
                foreach (var member in mod.GetMemberNames())
                {
                    yield return member;
                }
        }

        /// <summary>
        /// Determines whether a module is loaded with the specified name.
        /// </summary>
        /// <param name="moduleName">Name of the module.</param>
        /// <returns>
        ///   <c>true</c> if module loaded; otherwise, <c>false</c>.
        /// </returns>
        public bool IsModuleLoaded(string moduleName)
        {
            return Modules.Any(mod => mod.Name == moduleName);
        }

        /// <summary>
        /// Internal use.
        /// </summary>
        protected CudafyModule _module;

        /// <summary>
        /// Internal use. Checks for duplicate members.
        /// </summary>
        /// <param name="module">The module.</param>
        protected void CheckForDuplicateMembers(CudafyModule module)
        {
            if (_modules.Contains(module))
                throw new CudafyHostException(CudafyHostException.csMODULE_ALREADY_LOADED);
            bool duplicateFunc = _modules.Any(m => m.Functions.Any(f => module.Functions.ContainsKey(f.Key)));
            if (duplicateFunc)
                throw new CudafyHostException(CudafyHostException.csDUPLICATE_X_NAME, "function");
            bool duplicateConstant = _modules.Any(m => m.Constants.Any(c => module.Constants.ContainsKey(c.Key)));
            if (duplicateConstant)
                throw new CudafyHostException(CudafyHostException.csDUPLICATE_X_NAME, "constant");
            bool duplicateType = _modules.Any(m => m.Constants.Any(t => module.Types.ContainsKey(t.Key)));
            if (duplicateType)
                throw new CudafyHostException(CudafyHostException.csDUPLICATE_X_NAME, "type");
        }

        /// <summary>
        /// Loads module from module instance optionally unloading all already loaded modules. To load the same module to different GPUs you need
        /// to first Clone the module with cudafyModuleInstance.Clone().
        /// </summary>
        /// <param name="module">The module.</param>
        /// <param name="unload">If true then unload any currently loaded modules first.</param>
        public abstract void LoadModule(CudafyModule module, bool unload = true);

        /// <summary>
        /// Unloads the specified module.
        /// </summary>
        /// <param name="module">Module to unload.</param>
        public abstract void UnloadModule(CudafyModule module);

        /// <summary>
        /// Unloads the current module.
        /// </summary>
        public virtual void UnloadModule()
        {
            if (_module != null)
                UnloadModule(_module);
        }

        /// <summary>
        /// Unloads all modules.
        /// </summary>
        public virtual void UnloadModules()
        {
            UnloadModule();
            _modules.Clear();
        }

        /// <summary>
        /// Gets the current module.
        /// </summary>
        /// <value>The current module.</value>
        public CudafyModule CurrentModule
        {
            get { return _module; }
        }

        #region Strongly typed Launch
#if !NET35
        /// <summary>
        /// Safe launches the specified action.
        /// </summary>
        /// <typeparam name="T1">The type.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">First argument.</param>
        public void LaunchAsync<T1>(dim3 gridSize, dim3 blockSize, int streamId, Action<GThread, T1> action, T1 t1)
        {
            LaunchAsync(gridSize, blockSize, streamId, action.Method.Name, new object[] { t1 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        public void LaunchAsync<T1, T2>(dim3 gridSize, dim3 blockSize, int streamId, Action<GThread, T1, T2> action, T1 t1, T2 t2)
        {
            LaunchAsync(gridSize, blockSize, streamId, action.Method.Name,
                new object[] { t1, t2 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        public void LaunchAsync<T1, T2, T3>(dim3 gridSize, dim3 blockSize, int streamId, Action<GThread, T1, T2, T3> action, T1 t1, T2 t2, T3 t3)
        {
            LaunchAsync(gridSize, blockSize, streamId, action.Method.Name,
                new object[] { t1, t2, t3 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        public void LaunchAsync<T1, T2, T3, T4>
   (dim3 gridSize, dim3 blockSize, int streamId,
   Action<GThread, T1, T2, T3, T4> action,
   T1 t1, T2 t2, T3 t3, T4 t4)
        {
            LaunchAsync(gridSize, blockSize, streamId, action.Method.Name,
                new object[] { t1, t2, t3, t4 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        public void LaunchAsync<T1, T2, T3, T4, T5>
   (dim3 gridSize, dim3 blockSize, int streamId,
   Action<GThread, T1, T2, T3, T4, T5> action, T1 t1, T2 t2, T3 t3, T4 t4, T5 t5)
        {
            LaunchAsync(gridSize, blockSize, streamId, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        public void LaunchAsync<T1, T2, T3, T4, T5, T6>
   (dim3 gridSize, dim3 blockSize, int streamId,
   Action<GThread, T1, T2, T3, T4, T5, T6> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6)
        {
            LaunchAsync(gridSize, blockSize, streamId, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        public void LaunchAsync<T1, T2, T3, T4, T5, T6, T7>
   (dim3 gridSize, dim3 blockSize, int streamId,
   Action<GThread, T1, T2, T3, T4, T5, T6, T7> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7)
        {
            LaunchAsync(gridSize, blockSize, streamId, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <typeparam name="T8">The type of the 8.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        /// <param name="t8">The t8.</param>
        public void LaunchAsync<T1, T2, T3, T4, T5, T6, T7, T8>
   (dim3 gridSize, dim3 blockSize, int streamId,
   Action<GThread, T1, T2, T3, T4, T5, T6, T7, T8> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8)
        {
            LaunchAsync(gridSize, blockSize, streamId, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7, t8 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <typeparam name="T8">The type of the 8.</typeparam>
        /// <typeparam name="T9">The type of the 9.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        /// <param name="t8">The t8.</param>
        /// <param name="t9">The t9.</param>
        public void LaunchAsync<T1, T2, T3, T4, T5, T6, T7, T8, T9>
   (dim3 gridSize, dim3 blockSize, int streamId,
   Action<GThread, T1, T2, T3, T4, T5, T6, T7, T8, T9> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9)
        {
            LaunchAsync(gridSize, blockSize, streamId, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7, t8, t9 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <typeparam name="T8">The type of the 8.</typeparam>
        /// <typeparam name="T9">The type of the 9.</typeparam>
        /// <typeparam name="T10">The type of the 10.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        /// <param name="t8">The t8.</param>
        /// <param name="t9">The t9.</param>
        /// <param name="t10">The T10.</param>
        public void LaunchAsync<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>
   (dim3 gridSize, dim3 blockSize, int streamId,
   Action<GThread, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9, T10 t10)
        {
            LaunchAsync(gridSize, blockSize, streamId, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <typeparam name="T8">The type of the 8.</typeparam>
        /// <typeparam name="T9">The type of the 9.</typeparam>
        /// <typeparam name="T10">The type of the 10.</typeparam>
        /// <typeparam name="T11">The type of the 11.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        /// <param name="t8">The t8.</param>
        /// <param name="t9">The t9.</param>
        /// <param name="t10">The T10.</param>
        /// <param name="t11">The T11.</param>
        public void LaunchAsync<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>
   (dim3 gridSize, dim3 blockSize, int streamId,
   Action<GThread, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9, T10 t10, T11 t11)
        {
            LaunchAsync(gridSize, blockSize, -1, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <typeparam name="T8">The type of the 8.</typeparam>
        /// <typeparam name="T9">The type of the 9.</typeparam>
        /// <typeparam name="T10">The type of the 10.</typeparam>
        /// <typeparam name="T11">The type of the 11.</typeparam>
        /// <typeparam name="T12">The type of the 12.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">The stream id.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        /// <param name="t8">The t8.</param>
        /// <param name="t9">The t9.</param>
        /// <param name="t10">The T10.</param>
        /// <param name="t11">The T11.</param>
        /// <param name="t12">The T12.</param>
        public void LaunchAsync<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>
   (dim3 gridSize, dim3 blockSize, int streamId,
   Action<GThread, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9, T10 t10, T11 t11, T12 t12)
        {
            LaunchAsync(gridSize, blockSize, streamId, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <typeparam name="T8">The type of the 8.</typeparam>
        /// <typeparam name="T9">The type of the 9.</typeparam>
        /// <typeparam name="T10">The type of the 10.</typeparam>
        /// <typeparam name="T11">The type of the 11.</typeparam>
        /// <typeparam name="T12">The type of the 12.</typeparam>
        /// <typeparam name="T13">The type of the 13.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        /// <param name="t8">The t8.</param>
        /// <param name="t9">The t9.</param>
        /// <param name="t10">The T10.</param>
        /// <param name="t11">The T11.</param>
        /// <param name="t12">The T12.</param>
        /// <param name="t13">The T13.</param>
        public void LaunchAsync<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>
   (dim3 gridSize, dim3 blockSize, int streamId,
   Action<GThread, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9, T10 t10, T11 t11, T12 t12, T13 t13)
        {
            LaunchAsync(gridSize, blockSize, streamId, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <typeparam name="T8">The type of the 8.</typeparam>
        /// <typeparam name="T9">The type of the 9.</typeparam>
        /// <typeparam name="T10">The type of the 10.</typeparam>
        /// <typeparam name="T11">The type of the 11.</typeparam>
        /// <typeparam name="T12">The type of the 12.</typeparam>
        /// <typeparam name="T13">The type of the 13.</typeparam>
        /// <typeparam name="T14">The type of the 14.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        /// <param name="t8">The t8.</param>
        /// <param name="t9">The t9.</param>
        /// <param name="t10">The T10.</param>
        /// <param name="t11">The T11.</param>
        /// <param name="t12">The T12.</param>
        /// <param name="t13">The T13.</param>
        /// <param name="t14">The T14.</param>
        public void LaunchAsync<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14>
   (dim3 gridSize, dim3 blockSize, int streamId,
   Action<GThread, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9, T10 t10, T11 t11, T12 t12, T13 t13, T14 t14)
        {
            LaunchAsync(gridSize, blockSize, streamId, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <typeparam name="T8">The type of the 8.</typeparam>
        /// <typeparam name="T9">The type of the 9.</typeparam>
        /// <typeparam name="T10">The type of the 10.</typeparam>
        /// <typeparam name="T11">The type of the 11.</typeparam>
        /// <typeparam name="T12">The type of the 12.</typeparam>
        /// <typeparam name="T13">The type of the 13.</typeparam>
        /// <typeparam name="T14">The type of the 14.</typeparam>
        /// <typeparam name="T15">The type of the 15.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id.</param>
        /// <param name="action">The action.</param>
        /// <param name="streamId">Stream number.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        /// <param name="t8">The t8.</param>
        /// <param name="t9">The t9.</param>
        /// <param name="t10">The T10.</param>
        /// <param name="t11">The T11.</param>
        /// <param name="t12">The T12.</param>
        /// <param name="t13">The T13.</param>
        /// <param name="t14">The T14.</param>
        /// <param name="t15">The T15.</param>
        public void LaunchAsync<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15>
           (dim3 gridSize, dim3 blockSize, int streamId,
           Action<GThread, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15> action,
           T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9, T10 t10, T11 t11, T12 t12, T13 t13, T14 t14, T15 t15)
        {
            LaunchAsync(gridSize, blockSize, streamId, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15 });
        }

        /// <summary>
        /// Safe launches the specified action.
        /// </summary>
        /// <typeparam name="T1">The type.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">First argument.</param>
        public void Launch<T1>(dim3 gridSize, dim3 blockSize, Action<GThread, T1> action, T1 t1)
        {
            LaunchAsync(gridSize, blockSize, -1, action.Method.Name, new object[] { t1 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        public void Launch<T1, T2>(dim3 gridSize, dim3 blockSize, Action<GThread, T1, T2> action, T1 t1, T2 t2)
        {
            LaunchAsync(gridSize, blockSize, -1, action.Method.Name,
                new object[] { t1, t2 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        public void Launch<T1, T2, T3>(dim3 gridSize, dim3 blockSize, Action<GThread, T1, T2, T3> action, T1 t1, T2 t2, T3 t3)
        {
            LaunchAsync(gridSize, blockSize, -1, action.Method.Name,
                new object[] { t1, t2, t3 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        public void Launch<T1, T2, T3, T4>
   (dim3 gridSize, dim3 blockSize,
   Action<GThread, T1, T2, T3, T4> action,
   T1 t1, T2 t2, T3 t3, T4 t4)
        {
            LaunchAsync(gridSize, blockSize, -1, action.Method.Name,
                new object[] { t1, t2, t3, t4 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        public void Launch<T1, T2, T3, T4, T5>
   (dim3 gridSize, dim3 blockSize,
   Action<GThread, T1, T2, T3, T4, T5> action, T1 t1, T2 t2, T3 t3, T4 t4, T5 t5)
        {
            LaunchAsync(gridSize, blockSize, -1, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        public void Launch<T1, T2, T3, T4, T5, T6>
   (dim3 gridSize, dim3 blockSize,
   Action<GThread, T1, T2, T3, T4, T5, T6> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6)
        {
            LaunchAsync(gridSize, blockSize, -1, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        public void Launch<T1, T2, T3, T4, T5, T6, T7>
   (dim3 gridSize, dim3 blockSize,
   Action<GThread, T1, T2, T3, T4, T5, T6, T7> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7)
        {
            LaunchAsync(gridSize, blockSize, -1, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <typeparam name="T8">The type of the 8.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        /// <param name="t8">The t8.</param>
        public void Launch<T1, T2, T3, T4, T5, T6, T7, T8>
   (dim3 gridSize, dim3 blockSize,
   Action<GThread, T1, T2, T3, T4, T5, T6, T7, T8> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8)
        {
            LaunchAsync(gridSize, blockSize, -1, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7, t8 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <typeparam name="T8">The type of the 8.</typeparam>
        /// <typeparam name="T9">The type of the 9.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        /// <param name="t8">The t8.</param>
        /// <param name="t9">The t9.</param>
        public void Launch<T1, T2, T3, T4, T5, T6, T7, T8, T9>
   (dim3 gridSize, dim3 blockSize,
   Action<GThread, T1, T2, T3, T4, T5, T6, T7, T8, T9> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9)
        {
            LaunchAsync(gridSize, blockSize, -1, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7, t8, t9 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <typeparam name="T8">The type of the 8.</typeparam>
        /// <typeparam name="T9">The type of the 9.</typeparam>
        /// <typeparam name="T10">The type of the 10.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        /// <param name="t8">The t8.</param>
        /// <param name="t9">The t9.</param>
        /// <param name="t10">The T10.</param>
        public void Launch<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>
   (dim3 gridSize, dim3 blockSize,
   Action<GThread, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9, T10 t10)
        {
            LaunchAsync(gridSize, blockSize, -1, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <typeparam name="T8">The type of the 8.</typeparam>
        /// <typeparam name="T9">The type of the 9.</typeparam>
        /// <typeparam name="T10">The type of the 10.</typeparam>
        /// <typeparam name="T11">The type of the 11.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        /// <param name="t8">The t8.</param>
        /// <param name="t9">The t9.</param>
        /// <param name="t10">The T10.</param>
        /// <param name="t11">The T11.</param>
        public void Launch<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>
   (dim3 gridSize, dim3 blockSize,
   Action<GThread, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9, T10 t10, T11 t11)
        {
            LaunchAsync(gridSize, blockSize, -1, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11 });
        }

        public void Launch<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>
   (dim3 gridSize, dim3 blockSize,
   Action<GThread, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9, T10 t10, T11 t11, T12 t12)
        {
            LaunchAsync(gridSize, blockSize, -1, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <typeparam name="T8">The type of the 8.</typeparam>
        /// <typeparam name="T9">The type of the 9.</typeparam>
        /// <typeparam name="T10">The type of the 10.</typeparam>
        /// <typeparam name="T11">The type of the 11.</typeparam>
        /// <typeparam name="T12">The type of the 12.</typeparam>
        /// <typeparam name="T13">The type of the 13.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        /// <param name="t8">The t8.</param>
        /// <param name="t9">The t9.</param>
        /// <param name="t10">The T10.</param>
        /// <param name="t11">The T11.</param>
        /// <param name="t12">The T12.</param>
        /// <param name="t13">The T13.</param>
        public void Launch<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>
   (dim3 gridSize, dim3 blockSize,
   Action<GThread, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9, T10 t10, T11 t11, T12 t12, T13 t13)
        {
            LaunchAsync(gridSize, blockSize, -1, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <typeparam name="T8">The type of the 8.</typeparam>
        /// <typeparam name="T9">The type of the 9.</typeparam>
        /// <typeparam name="T10">The type of the 10.</typeparam>
        /// <typeparam name="T11">The type of the 11.</typeparam>
        /// <typeparam name="T12">The type of the 12.</typeparam>
        /// <typeparam name="T13">The type of the 13.</typeparam>
        /// <typeparam name="T14">The type of the 14.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        /// <param name="t8">The t8.</param>
        /// <param name="t9">The t9.</param>
        /// <param name="t10">The T10.</param>
        /// <param name="t11">The T11.</param>
        /// <param name="t12">The T12.</param>
        /// <param name="t13">The T13.</param>
        /// <param name="t14">The T14.</param>
        public void Launch<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14>
   (dim3 gridSize, dim3 blockSize,
   Action<GThread, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14> action,
   T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9, T10 t10, T11 t11, T12 t12, T13 t13, T14 t14)
        {
            LaunchAsync(gridSize, blockSize, -1, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14 });
        }

        /// <summary>
        /// Launches the specified grid size.
        /// </summary>
        /// <typeparam name="T1">The type of the 1.</typeparam>
        /// <typeparam name="T2">The type of the 2.</typeparam>
        /// <typeparam name="T3">The type of the 3.</typeparam>
        /// <typeparam name="T4">The type of the 4.</typeparam>
        /// <typeparam name="T5">The type of the 5.</typeparam>
        /// <typeparam name="T6">The type of the 6.</typeparam>
        /// <typeparam name="T7">The type of the 7.</typeparam>
        /// <typeparam name="T8">The type of the 8.</typeparam>
        /// <typeparam name="T9">The type of the 9.</typeparam>
        /// <typeparam name="T10">The type of the 10.</typeparam>
        /// <typeparam name="T11">The type of the 11.</typeparam>
        /// <typeparam name="T12">The type of the 12.</typeparam>
        /// <typeparam name="T13">The type of the 13.</typeparam>
        /// <typeparam name="T14">The type of the 14.</typeparam>
        /// <typeparam name="T15">The type of the 15.</typeparam>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="action">The action.</param>
        /// <param name="t1">The t1.</param>
        /// <param name="t2">The t2.</param>
        /// <param name="t3">The t3.</param>
        /// <param name="t4">The t4.</param>
        /// <param name="t5">The t5.</param>
        /// <param name="t6">The t6.</param>
        /// <param name="t7">The t7.</param>
        /// <param name="t8">The t8.</param>
        /// <param name="t9">The t9.</param>
        /// <param name="t10">The T10.</param>
        /// <param name="t11">The T11.</param>
        /// <param name="t12">The T12.</param>
        /// <param name="t13">The T13.</param>
        /// <param name="t14">The T14.</param>
        /// <param name="t15">The T15.</param>
        public void Launch<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15>
           (dim3 gridSize, dim3 blockSize,
           Action<GThread, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15> action,
           T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9, T10 t10, T11 t11, T12 t12, T13 t13, T14 t14, T15 t15)
        {
            LaunchAsync(gridSize, blockSize, -1, action.Method.Name,
                new object[] { t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15 });
        }
#endif
#endregion

        ///// <summary>
        ///// Launches the specified kernel.
        ///// </summary>
        ///// <param name="gridSize">Size of the grid.</param>
        ///// <param name="blockSize">Size of the block.</param>
        ///// <param name="methodName">Name of the method.</param>
        ///// <param name="arguments">The arguments.</param>
        //public void Launch(int gridSize, int blockSize, string methodName, params object[] arguments)
        //{
        //    LaunchAsync(new dim3(gridSize), new dim3(blockSize), -1, methodName, arguments);
        //}


        ///// <summary>
        ///// Launches the specified kernel.
        ///// </summary>
        ///// <param name="gridSize">Size of the grid.</param>
        ///// <param name="blockSize">Size of the block.</param>
        ///// <param name="streamId">The stream id.</param>
        ///// <param name="methodName">Name of the method.</param>
        ///// <param name="arguments">The arguments.</param>
        //public void LaunchAsync(int gridSize, int blockSize, int streamId, string methodName, params object[] arguments)
        //{
        //    LaunchAsync(new dim3(gridSize), new dim3(blockSize), streamId, methodName, arguments);
        //}

        ///// <summary>
        ///// Launches the specified kernel.
        ///// </summary>
        ///// <param name="gridSize">Size of the grid.</param>
        ///// <param name="blockSize">Size of the block.</param>
        ///// <param name="methodName">Name of the method.</param>
        ///// <param name="arguments">The arguments.</param>
        //public void Launch(dim3 gridSize, int blockSize, string methodName, params object[] arguments)
        //{
        //    LaunchAsync(gridSize, new dim3(blockSize), -1, methodName, arguments);
        //}

        ///// <summary>
        ///// Launches the specified kernel.
        ///// </summary>
        ///// <param name="gridSize">Size of the grid.</param>
        ///// <param name="blockSize">Size of the block.</param>
        ///// <param name="streamId">The stream id.</param>
        ///// <param name="methodName">Name of the method.</param>
        ///// <param name="arguments">The arguments.</param>
        //public void LaunchAsync(dim3 gridSize, int blockSize, int streamId, string methodName, params object[] arguments)
        //{
        //    LaunchAsync(gridSize, new dim3(blockSize), streamId, methodName, arguments);
        //}

        ///// <summary>
        ///// Launches the specified kernel.
        ///// </summary>
        ///// <param name="gridSize">Size of the grid.</param>
        ///// <param name="blockSize">Size of the block.</param>
        ///// <param name="methodName">Name of the method.</param>
        ///// <param name="arguments">The arguments.</param>
        //public void Launch(int gridSize, dim3 blockSize, string methodName, params object[] arguments)
        //{
        //    LaunchAsync(new dim3(gridSize), blockSize, -1, methodName, arguments);
        //}

        /// <summary>
        /// Launches the specified kernel.
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="methodName">Name of the method.</param>
        /// <param name="arguments">The arguments.</param>
        public void Launch(dim3 gridSize, dim3 blockSize, string methodName, params object[] arguments)
        {
            LaunchAsync(gridSize, blockSize, -1, methodName, arguments);
        }

        ///// <summary>
        ///// Launches the specified kernel.
        ///// </summary>
        ///// <param name="gridSize">Size of the grid.</param>
        ///// <param name="blockSize">Size of the block.</param>
        ///// <param name="streamId">The stream id.</param>
        ///// <param name="methodName">Name of the method.</param>
        ///// <param name="arguments">The arguments.</param>
        //public void LaunchAsync(int gridSize, dim3 blockSize, int streamId, string methodName, params object[] arguments)
        //{
        //    LaunchAsync(new dim3(gridSize), blockSize, streamId, methodName, arguments);
        //}

        /// <summary>
        /// Launches the specified kernel.
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id.</param>
        /// <param name="methodName">Name of the method.</param>
        /// <param name="arguments">The arguments.</param>
        public void LaunchAsync(dim3 gridSize, dim3 blockSize, int streamId, string methodName, params object[] arguments)
        {
            if (_modules.Count == 0)
                throw new CudafyHostException(CudafyHostException.csNO_MODULE_LOADED);
            CudafyModule module = _modules.Where(mod => mod.Functions.ContainsKey(methodName)).FirstOrDefault();
            if(module == null)
                throw new CudafyHostException(CudafyHostException.csCOULD_NOT_FIND_FUNCTION_X, methodName);
            _module = module;
            VerifyMembersAreOnGPU(arguments);
            KernelMethodInfo gpuMI = module.Functions[methodName];
            if (gpuMI.MethodType != eKernelMethodType.Global)
                throw new CudafyHostException(CudafyHostException.csCAN_ONLY_LAUNCH_GLOBAL_METHODS);
            DoLaunch(gridSize, blockSize, streamId, gpuMI, arguments);
        }

        /// <summary>
        /// Does the launch.
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id, or -1 for non-async.</param>
        /// <param name="gpuMI">The gpu MI.</param>
        /// <param name="arguments">The arguments.</param>
        protected abstract void DoLaunch(dim3 gridSize, dim3 blockSize, int streamId, KernelMethodInfo gpuMI, params object[] arguments);


        /// <summary>
        /// Does the copy to constant memory.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="ci">The ci.</param>
        protected abstract void DoCopyToConstantMemory<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, KernelConstantInfo ci) where T : struct;

        /// <summary>
        /// Does the copy to constant memory async.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="ci">The ci.</param>
        /// <param name="streamId">The stream id.</param>
        protected abstract void DoCopyToConstantMemoryAsync<T>(IntPtr hostArray, int hostOffset, Array devArray, int devOffset, int count, KernelConstantInfo ci, int streamId) where T : struct;


        /// <summary>
        /// Does the copy to device.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        protected abstract void DoCopyToDevice<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count) where T : struct;

        /// <summary>
        /// Does the copy from device.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="hostArray">The host array.</param>
        protected abstract void DoCopyFromDevice<T>(Array devArray, Array hostArray) where T : struct;

        /// <summary>
        /// Does the copy from device.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The count.</param>
        protected abstract void DoCopyFromDevice<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count) where T : struct;

        protected abstract void DoCopyFromDeviceAsync<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count, int streamId) where T : struct;

        /// <summary>
        /// Does the copy to device async.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="streamId">The stream id.</param>
        protected abstract void DoCopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, Array devArray, int devOffset, int count, int streamId) where T : struct;

        protected abstract void DoCopyToDeviceAsync<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, int streamId) where T : struct;

        /// <summary>
        /// Does the copy to device async.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="streamId">The stream id.</param>
        protected abstract void DoCopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, DevicePtrEx devArray, int devOffset, int count, int streamId) where T : struct;

        /// <summary>
        /// Does the copy to device async.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="streamId">The stream id.</param>
        /// <param name="stagingPost">The staging post.</param>
        protected abstract void DoCopyToDeviceAsync<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, int streamId, IntPtr stagingPost, bool isConstantMemory = false) where T : struct;


        /// <summary>
        /// Performs an asynchronous data transfer.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="streamId">The stream id.</param>
        protected abstract void DoCopyFromDeviceAsync<T>(Array devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId) where T : struct;

        /// <summary>
        /// Performs an asynchronous data transfer.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="streamId">The stream id.</param>
        protected abstract void DoCopyFromDeviceAsync<T>(DevicePtrEx devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId) where T : struct;

        /// <summary>
        /// Performs an asynchronous data transfer.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="streamId">The stream id.</param>
        /// <param name="stagingPost">The staging post.</param>
        protected abstract void DoCopyFromDeviceAsync<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count, int streamId, IntPtr stagingPost) where T : struct;
        
        /// <summary>
        /// Copies to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="devArray">The device array.</param>
        public void CopyToDevice<T>(T[] hostArray, T[] devArray) where T : struct
        {
            DoCopyToDevice<T>(hostArray, 0, devArray, 0, hostArray.Length);
        }

        /// <summary>
        /// Copies to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="count">The number of elements.</param>
        public void CopyToDevice<T>(T[] hostArray, int hostOffset, T[] devArray, int devOffset, int count) where T : struct
        {
            DoCopyToDevice<T>(hostArray, hostOffset, devArray, devOffset, count);
        }

        //public void SmartCopyToDevice<T>(TextWriterTraceListener[])

        /// <summary>
        /// Copies to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="count">The number of elements.</param>
        public void CopyToDevice<T>(IntPtr hostArray, int hostOffset, T[] devArray, int devOffset, int count) where T : struct
        {
            DoCopyToDeviceAsync<T>(hostArray, hostOffset, devArray, devOffset, count, -1);
        }

        /// <summary>
        /// Copies to device asynchronously making use of the previously allocated staging post.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="count">Number of elements.</param>
        /// <param name="streamId">The stream id.</param>
        /// <param name="stagingPost">The staging post of equal or greater size to count. Use HostAllocate to create.</param>
        public void CopyToDeviceAsync<T>(T[] hostArray, int hostOffset, T[] devArray, int devOffset, int count, int streamId, IntPtr stagingPost) where T : struct
        {
            if (!IsSmartCopyEnabled)
                throw new CudafyHostException(CudafyHostException.csSMART_COPY_IS_NOT_ENABLED);
            DoCopyToDeviceAsync<T>(hostArray, hostOffset, devArray, devOffset, count, streamId, stagingPost);
        }

        /// <summary>
        /// Copies from device asynchronously making use of the previously allocated staging post.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">Number of elements.</param>
        /// <param name="streamId">The stream id.</param>
        /// <param name="stagingPost">The staging post of equal or greater size to count. Use HostAllocate to create.</param>
        public void CopyFromDeviceAsync<T>(T[] devArray, int devOffset, T[] hostArray, int hostOffset, int count, int streamId, IntPtr stagingPost) where T : struct
        {
            if (!IsSmartCopyEnabled)
                throw new CudafyHostException(CudafyHostException.csSMART_COPY_IS_NOT_ENABLED);
            DoCopyFromDeviceAsync<T>(devArray, devOffset, hostArray, hostOffset, count, streamId, stagingPost);
        }

        /// <summary>
        /// Copies asynchronously to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, T[] devArray, int devOffset, int count, int streamId = 0) where T : struct
        {
            DoCopyToDeviceAsync<T>(hostArray, hostOffset, devArray, devOffset, count, streamId);
        }

        ///// <summary>
        ///// Copies asynchronously to preallocated array on device.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="hostArray">The host array.</param>
        ///// <param name="hostOffset">The host offset.</param>
        ///// <param name="devArray">The device array.</param>
        ///// <param name="devOffset">The device offset.</param>
        ///// <param name="count">The number of elements.</param>
        ///// <param name="streamId">The stream id.</param>
        //public void CopyToDeviceAsync<T>(T[] hostArray, int hostOffset, T[] devArray, int devOffset, int count, int streamId)
        //{
        //    DoCopyToDeviceAsync<T>(hostArray, hostOffset, devArray, devOffset, count, streamId);
        //}

        ///// <summary>
        ///// Copies asynchronously to preallocated array on device.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="hostArray">The host array.</param>
        ///// <param name="hostOffset">The host offset.</param>
        ///// <param name="devArray">The device array.</param>
        ///// <param name="devOffset">The device offset.</param>
        ///// <param name="count">The number of elements.</param>
        ///// <param name="streamId">The stream id.</param>
        //public void CopyToDeviceAsync<T>(T[,] hostArray, int hostOffset, T[,] devArray, int devOffset, int count, int streamId)
        //{
        //    DoCopyToDeviceAsync<T>(hostArray, hostOffset, devArray, devOffset, count, streamId);
        //}

        ///// <summary>
        ///// Copies asynchronously to preallocated array on device.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="hostArray">The host array.</param>
        ///// <param name="hostOffset">The host offset.</param>
        ///// <param name="devArray">The device array.</param>
        ///// <param name="devOffset">The device offset.</param>
        ///// <param name="count">The number of elements.</param>
        ///// <param name="streamId">The stream id.</param>
        //public void CopyToDeviceAsync<T>(T[,,] hostArray, int hostOffset, T[,,] devArray, int devOffset, int count, int streamId)
        //{
        //    DoCopyToDeviceAsync<T>(hostArray, hostOffset, devArray, devOffset, count, streamId);
        //}

        /// <summary>
        /// Copies asynchronously to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, DevicePtrEx devArray, int devOffset, int count, int streamId = 0) where T : struct
        {
            DoCopyToDeviceAsync<T>(hostArray, hostOffset, devArray, devOffset, count, streamId);
        }

        /// <summary>
        /// Copies asynchronously to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, T[,] devArray, int devOffset, int count, int streamId = 0) where T : struct
        {
            DoCopyToDeviceAsync<T>(hostArray, hostOffset, devArray, devOffset, count, streamId);
        }

        /// <summary>
        /// Copies asynchronously to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, T[, ,] devArray, int devOffset, int count, int streamId = 0) where T : struct
        {
            DoCopyToDeviceAsync<T>(hostArray, hostOffset, devArray, devOffset, count, streamId);
        }

        //public void CopyToDeviceAsync()

        /// <summary>
        /// Copies from device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The number of elements.</param>
        public void CopyFromDevice<T>(T[] devArray, int devOffset, T[] hostArray, int hostOffset, int count) where T : struct
        {
            DoCopyFromDevice<T>(devArray, devOffset, hostArray, hostOffset, count);
        }

        /// <summary>
        /// Copies from device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The number of elements.</param>
        public void CopyFromDevice<T>(T[] devArray, int devOffset, IntPtr hostArray, int hostOffset, int count) where T : struct
        {
            DoCopyFromDeviceAsync<T>(devArray, devOffset, hostArray, hostOffset, count, -1);
        }

        /// <summary>
        /// Copies from device asynchronously.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyFromDeviceAsync<T>(T[] devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId = 0) where T : struct
        {
            DoCopyFromDeviceAsync<T>(devArray, devOffset, hostArray, hostOffset, count, streamId);
        }

        ///// <summary>
        ///// Copies from device asynchronously.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="devArray">The dev array.</param>
        ///// <param name="devOffset">The device offset.</param>
        ///// <param name="hostArray">The host array.</param>
        ///// <param name="hostOffset">The host offset.</param>
        ///// <param name="count">The number of elements.</param>
        ///// <param name="streamId">The stream id.</param>
        //public void CopyFromDeviceAsync<T>(T[] devArray, int devOffset, T[] hostArray, int hostOffset, int count, int streamId)
        //{
        //    DoCopyFromDeviceAsync<T>(devArray, devOffset, hostArray, hostOffset, count, streamId);
        //}

        ///// <summary>
        ///// Copies from device asynchronously.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="devArray">The dev array.</param>
        ///// <param name="devOffset">The device offset.</param>
        ///// <param name="hostArray">The host array.</param>
        ///// <param name="hostOffset">The host offset.</param>
        ///// <param name="count">The number of elements.</param>
        ///// <param name="streamId">The stream id.</param>
        //public void CopyFromDeviceAsync<T>(T[,] devArray, int devOffset, T[,] hostArray, int hostOffset, int count, int streamId)
        //{
        //    DoCopyFromDeviceAsync<T>(devArray, devOffset, hostArray, hostOffset, count, streamId);
        //}

        ///// <summary>
        ///// Copies from device asynchronously.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="devArray">The dev array.</param>
        ///// <param name="devOffset">The device offset.</param>
        ///// <param name="hostArray">The host array.</param>
        ///// <param name="hostOffset">The host offset.</param>
        ///// <param name="count">The number of elements.</param>
        ///// <param name="streamId">The stream id.</param>
        //public void CopyFromDeviceAsync<T>(T[,,] devArray, int devOffset, T[,,] hostArray, int hostOffset, int count, int streamId)
        //{
        //    DoCopyFromDeviceAsync<T>(devArray, devOffset, hostArray, hostOffset, count, streamId);
        //}

        /// <summary>
        /// Copies from device asynchronously.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyFromDeviceAsync<T>(DevicePtrEx devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId = 0) where T : struct
        {
            DoCopyFromDeviceAsync<T>(devArray, devOffset, hostArray, hostOffset, count, streamId);
        }

        /// <summary>
        /// Copies from device asynchronously.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyFromDeviceAsync<T>(T[,] devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId = 0) where T : struct
        {
            DoCopyFromDeviceAsync<T>(devArray, devOffset, hostArray, hostOffset, count, streamId);
        }

        /// <summary>
        /// Copies from device asynchronously.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyFromDeviceAsync<T>(T[, ,] devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId = 0) where T : struct
        {
            DoCopyFromDeviceAsync<T>(devArray, devOffset, hostArray, hostOffset, count, streamId);
        }

        /// <summary>
        /// Synchronizes the stream.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        public abstract void SynchronizeStream(int streamId = 0);

        ///// <summary>
        ///// Performs a default host memory allocation.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="x">The number of elements.</param>
        ///// <returns>Pointer to allocated memory.</returns>
        ///// <remarks>Remember to free this memory with HostFree.</remarks>
        //public abstract IntPtr HostAllocate<T>(int x);





        /// <summary>
        /// Performs a default host memory allocation.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="x">The x size.</param>
        /// <param name="y">The y size.</param>
        /// <param name="z">The z size.</param>
        /// <returns>Pointer to allocated memory.</returns>
        /// <remarks>Remember to free this memory with HostFree.</remarks>
        public IntPtr HostAllocate<T>(int x, int y, int z)
        {
            return HostAllocate<T>(x * y * z);
        }

        /// <summary>
        /// Performs a default host memory allocation.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">The x size.</param>
        /// <param name="y">The y size.</param>
        /// <returns>Pointer to allocated memory.</returns>
        /// <remarks>Remember to free this memory with HostFree.</remarks>
        public IntPtr HostAllocate<T>(int x, int y)
        {
            return HostAllocate<T>(x * y);
        }

        /// <summary>
        /// Performs a default host memory allocation.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">The x size.</param>
        /// <returns>
        /// Pointer to allocated memory.
        /// </returns>
        public virtual IntPtr HostAllocate<T>(int x)
        {
            int bytes = MSizeOf(typeof(T)) * x;
            byte[] buffer = new byte[bytes];
            GCHandle handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            IntPtr intPtr = handle.AddrOfPinnedObject();
            _hostHandles.Add(intPtr, handle);
            return intPtr;
        }

        protected Dictionary<IntPtr, GCHandle> _hostHandles;

        /// <summary>
        /// Frees memory allocated by HostAllocate.
        /// </summary>
        /// <param name="ptr">The pointer.</param>
        /// <exception cref="CudafyHostException">Pointer not found.</exception>
        public virtual void HostFree(IntPtr ptr)
        {
            lock (_lock)
            {
                if (_hostHandles.ContainsKey(ptr))
                {
                    GCHandle handle = _hostHandles[ptr];
                    handle.Free();
                    _hostHandles.Remove(ptr);
                }
                else
                    throw new CudafyHostException(CudafyHostException.csPOINTER_NOT_FOUND);
            }
        }

        /// <summary>
        /// Frees all memory allocated by HostAllocate.
        /// </summary>
        public virtual void HostFreeAll()
        {
            lock (_lock)
            {
                foreach (var v in _hostHandles)
                {
                    GCHandle handle = v.Value;
                    try
                    {
                        handle.Free();
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine(ex);
                    }
                }
                _hostHandles.Clear();
            }
        }

        ///// <summary>
        ///// Frees memory allocated by HostAllocate.
        ///// </summary>
        ///// <param name="ptr">The pointer to free.</param>
        ///// <exception cref="CudafyHostException">Pointer not found.</exception>
        //public abstract void HostFree(IntPtr ptr);

        ///// <summary>
        ///// Frees all memory allocated by HostAllocate.
        ///// </summary>
        //public abstract void HostFreeAll();


//
        unsafe struct copystruct2
        {
            fixed long l[2];
        }
        unsafe struct copystruct4
        {
            fixed long l[4];
        }
        unsafe struct copystruct16
        {
            fixed long l[16];
        }
        unsafe struct copystruct128
        {
            fixed long l[128];
        }
#if LINUX     
        public unsafe static void CopyMemory(IntPtr Destination, IntPtr Source, uint Length)
        {
            copystruct128* src = (copystruct128*)Source;
            copystruct128* dest = (copystruct128*)Destination;
            long blocks = Length / sizeof(copystruct128);
            for (int i = 0; i < blocks; i++)
            {
                dest[i] = src[i];
            }
            byte* srcb = (byte*)Source;
            byte* destb = (byte*)Destination;
            for (long i = blocks * sizeof(copystruct128); i < Length; i++)
            {
                destb[i] = srcb[i];
            }
        }
#else

        /// <summary>
        /// Copies memory on host using native CopyMemory function from kernel32.dll.
        /// </summary>
        /// <param name="Destination">The destination.</param>
        /// <param name="Source">The source.</param>
        /// <param name="Length">The length.</param>
        [DllImport("kernel32.dll", EntryPoint = "RtlMoveMemory")]
        public static extern void CopyMemory(IntPtr Destination, IntPtr Source, uint Length);
#endif
        /// <summary>
        /// Gets the value at specified index.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <returns>Value at index.</returns>
        public T GetValue<T>(T[] devArray, int x) where T : struct
        {
            T[] hostArray = new T[1];
            CopyFromDevice(devArray, x, hostArray, 0, 1);
            return hostArray[0];
        }

        /// <summary>
        /// Gets the value at specified index.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        public T GetValue<T>(T[,] devArray, int x, int y) where T : struct
        {
            T[] hostArray = new T[1];
            var ptrEx = GetDeviceMemory(devArray) as DevicePtrEx;
            DoCopyFromDevice<T>(devArray, ptrEx.GetOffset1D(x, y), hostArray, 0, 1);
            return hostArray[0];
        }

        /// <summary>
        /// Gets the value.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public T GetValue<T>(T[, ,] devArray, int x, int y, int z) where T : struct
        {
            T[] hostArray = new T[1];
            var ptrEx = GetDeviceMemory(devArray) as DevicePtrEx;
            DoCopyFromDevice<T>(devArray, ptrEx.GetOffset1D(x, y, z), hostArray, 0, 1);
            return hostArray[0];
        }

        /// <summary>
        /// Casts the specified dev array.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The number of samples.</param>
        /// <returns></returns>
        public U[] Cast<T,U>(T[] devArray, int n)
        {
            return (U[])DoCast<T, U>(0, devArray, n);
        }

        /// <summary>
        /// Casts the specified dev array to 1D.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The number of samples.</param>
        /// <returns>1D array.</returns>
        public T[] Cast<T>(T[,] devArray, int n)
        {
            return (T[])DoCast<T,T>(0, devArray, n);
        }

        /// <summary>
        /// Casts the specified dev array.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public U[] Cast<T,U>(T[,] devArray, int n)
        {
            return (U[])DoCast<T, U>(0, devArray, n);
        }

        /// <summary>
        /// Casts the specified dev array to 2D.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        public T[,] Cast<T>(T[] devArray, int x, int y)
        {
            return (T[,])DoCast<T,T>(0, devArray, x, y);
        }

        /// <summary>
        /// Casts the specified dev array.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        public U[,] Cast<T,U>(T[] devArray, int x, int y)
        {
            return (U[,])DoCast<T, U>(0, devArray, x, y);
        }

        /// <summary>
        /// Casts the specified dev array to 3D.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public T[,,] Cast<T>(T[] devArray, int x, int y, int z)
        {
            return (T[, ,])DoCast<T, T>(0, devArray, x, y, z);
        }

        /// <summary>
        /// Casts the specified dev array.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public U[, ,] Cast<T,U>(T[] devArray, int x, int y, int z)
        {
            return (U[, ,])DoCast<T, U>(0, devArray, x, y, z);
        }

        /// <summary>
        /// Casts the specified dev array to 1D.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public T[] Cast<T>(T[,,] devArray, int n)
        {
            return (T[])DoCast<T,T>(0, devArray, n);
        }

        /// <summary>
        /// Casts the specified dev array.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public U[] Cast<T,U>(T[, ,] devArray, int n)
        {
            return (U[])DoCast<T, U>(0, devArray, n);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public T[] Cast<T>(int offset, T[] devArray, int n)
        {
            return (T[])DoCast<T, T>(offset, devArray, n);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public U[] Cast<T,U>(int offset, T[] devArray, int n)
        {
            return (U[])DoCast<T, U>(offset, devArray, n);
        }

        /// <summary>
        /// Casts the specified dev array to 1D.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="offset">Offset into dev array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The number of samples.</param>
        /// <returns>1D array.</returns>
        public T[] Cast<T>(int offset, T[,] devArray, int n)
        {
            return (T[])DoCast<T, T>(offset, devArray, n);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public U[] Cast<T, U>(int offset, T[,] devArray, int n)
        {
            return (U[])DoCast<T, U>(offset, devArray, n);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        public T[,] Cast<T>(int offset, T[,] devArray, int x, int y)
        {
            return (T[,])DoCast<T, T>(offset, devArray, x, y);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        public U[,] Cast<T,U>(int offset, T[,] devArray, int x, int y)
        {
            return (U[,])DoCast<T, U>(offset, devArray, x, y);
        }

        /// <summary>
        /// Casts the specified dev array to 2D.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="offset">Offset into dev array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        public T[,] Cast<T>(int offset, T[] devArray, int x, int y)
        {
            return (T[,])DoCast<T, T>(offset, devArray, x, y);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        public U[,] Cast<T,U>(int offset, T[] devArray, int x, int y)
        {
            return (U[,])DoCast<T, U>(offset, devArray, x, y);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public T[,,] Cast<T>(int offset, T[,,] devArray, int x, int y, int z)
        {
            return (T[, ,])DoCast<T, T>(offset, devArray, x, y, z);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="U"></typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public U[, ,] Cast<T,U>(int offset, T[, ,] devArray, int x, int y, int z)
        {
            return (U[, ,])DoCast<T, U>(offset, devArray, x, y, z);
        }

        /// <summary>
        /// Casts the specified dev array to 3D.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="offset">Offset into dev array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public T[,,] Cast<T>(int offset, T[] devArray, int x, int y, int z)
        {
            return (T[, ,])DoCast<T, T>(offset, devArray, x, y, z);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public U[, ,] Cast<T,U>(int offset, T[] devArray, int x, int y, int z)
        {
            return (U[, ,])DoCast<T, U>(offset, devArray, x, y, z);
        }

        /// <summary>
        /// Casts the specified dev array to 1D.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <param name="offset">Offset into dev array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public T[] Cast<T>(int offset, T[, ,] devArray, int n)
        {
            return (T[])DoCast<T, T>(offset, devArray, n);
        }

        /// <summary>
        /// Casts the specified dev array to 1D.
        /// </summary>
        /// <typeparam name="T">Type of source array.</typeparam>
        /// <typeparam name="U">Type of destination array.</typeparam>
        /// <param name="offset">Offset into dev array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public U[] Cast<T,U>(int offset, T[, ,] devArray, int n)
        {
            return (U[])DoCast<T,U>(offset, devArray, n);
        }

        /// <summary>
        /// Does the cast.
        /// </summary>
        /// <typeparam name="T">Type of source array.</typeparam>
        /// <typeparam name="U">Type of result array.</typeparam>
        /// <param name="offset">Offset into dev array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        protected abstract Array DoCast<T, U>(int offset, Array devArray, int n);

        /// <summary>
        /// Does the cast.
        /// </summary>
        /// <typeparam name="T">Type of source array.</typeparam>
        /// <typeparam name="U">Type of result array.</typeparam>
        /// <param name="offset">Offset into dev array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        protected abstract Array DoCast<T, U>(int offset, Array devArray, int x, int y);

        /// <summary>
        /// Does the cast.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="U"></typeparam>
        /// <param name="offset">Offset into dev array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        protected abstract Array DoCast<T, U>(int offset, Array devArray, int x, int y, int z);

        /// <summary>
        /// Copies data on host.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="nativeHostArraySrc">The source native host array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="hostAllocatedMemory">The destination host allocated memory.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements.</param>
        public static void CopyOnHost<T>(T[] nativeHostArraySrc, int srcOffset, IntPtr hostAllocatedMemory, int dstOffset, int count)
        {
            DoCopyOnHost<T>(nativeHostArraySrc, srcOffset, hostAllocatedMemory, dstOffset, count);
        }

        /// <summary>
        /// Copies data on host.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="nativeHostArraySrc">The source native host array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="hostAllocatedMemory">The destination host allocated memory.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements.</param>
        public static void CopyOnHost<T>(T[,] nativeHostArraySrc, int srcOffset, IntPtr hostAllocatedMemory, int dstOffset, int count)
        {
            DoCopyOnHost<T>(nativeHostArraySrc, srcOffset, hostAllocatedMemory, dstOffset, count);
        }

        /// <summary>
        /// Copies data on host.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="nativeHostArraySrc">The source native host array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="hostAllocatedMemory">The destination host allocated memory.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements.</param>
        public static void CopyOnHost<T>(T[,,] nativeHostArraySrc, int srcOffset, IntPtr hostAllocatedMemory, int dstOffset, int count)
        {
            DoCopyOnHost<T>(nativeHostArraySrc, srcOffset, hostAllocatedMemory, dstOffset, count);
        }

        /// <summary>
        /// Does the copy on host.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="nativeHostArraySrc">The native host array SRC.</param>
        /// <param name="srcOffset">The SRC offset.</param>
        /// <param name="hostAllocatedMemory">The host allocated memory.</param>
        /// <param name="dstOffset">The DST offset.</param>
        /// <param name="count">The count.</param>
        protected unsafe static void DoCopyOnHost<T>(Array nativeHostArraySrc, int srcOffset, IntPtr hostAllocatedMemory, int dstOffset, int count)
        {
            //Type type = (typeof(T));

            GCHandle handle = GCHandle.Alloc(nativeHostArraySrc, GCHandleType.Pinned);
            try
            {
                int size = MSizeOf(typeof(T));
                IntPtr srcIntPtr = handle.AddrOfPinnedObject();
                IntPtr srcOffsetPtr = new IntPtr(srcIntPtr.ToInt64() + srcOffset * size);
                IntPtr dstIntPtrOffset = new IntPtr(hostAllocatedMemory.ToInt64() + dstOffset * size);
                CopyMemory(dstIntPtrOffset, srcOffsetPtr, (uint)(count * size));
            }
            finally
            {
                handle.Free();
            }

        }

        //private unsafe static IntPtr MarshalArray<T>(ref Array items, int srcOffset, IntPtr dstPtr, int dstOffset, int count = 0)
        //{
        //    int length = count <= 0 ? (items.Length - srcOffset) : count;
        //    int iSizeOfOneItemPos = Marshal.SizeOf(typeof(T));
        //    IntPtr dstIntPtrOffset = new IntPtr(dstPtr.ToInt64() + (dstOffset * length));
        //    byte* pbyUnmanagedItems = (byte*)(dstIntPtrOffset.ToPointer());
            
        //    for (int i = srcOffset; i < (srcOffset + length); i++, pbyUnmanagedItems += (iSizeOfOneItemPos))
        //    {
        //        IntPtr pOneItemPos = new IntPtr(pbyUnmanagedItems);
        //        GCHandle handle = GCHandle.Alloc(items.GetValue(i));//, GCHandleType.Pinned);
        //        CopyMemory(pOneItemPos, handle.AddrOfPinnedObject(), (uint)(iSizeOfOneItemPos));
        //        handle.Free();
        //        //Marshal.StructureToPtr(, pOneItemPos, false);
        //    }

        //    return dstPtr;
        //}

        //private unsafe static void UnMarshalArray<T>(IntPtr srcItems, int srcOffset, ref Array items, int dstOffset, int count = 0)
        //{
        //    int length = count <= 0 ? (items.Length - srcOffset) : count;
        //    int iSizeOfOneItemPos = Marshal.SizeOf(typeof(T));
        //    IntPtr srcIntPtrOffset = new IntPtr(srcItems.ToInt64() + (srcOffset * length));
        //    byte* pbyUnmanagedItems = (byte*)(srcIntPtrOffset.ToPointer());

        //    for (int i = dstOffset; i < (dstOffset + length); i++, pbyUnmanagedItems += (iSizeOfOneItemPos))
        //    {
        //        IntPtr pOneItemPos = new IntPtr(pbyUnmanagedItems);
        //        items.SetValue((T)(Marshal.PtrToStructure(pOneItemPos, typeof(T))), i);
        //    }
        //}

        /// <summary>
        /// Copies data on host.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostAllocatedMemory">The source host allocated memory.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="nativeHostArrayDst">The destination native host array.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements.</param>
        public static void CopyOnHost<T>(IntPtr hostAllocatedMemory, int srcOffset, T[] nativeHostArrayDst, int dstOffset, int count)
        {
            DoCopyOnHost<T>(hostAllocatedMemory, srcOffset, nativeHostArrayDst, dstOffset, count);
        }

        /// <summary>
        /// Copies data on host.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostAllocatedMemory">The source host allocated memory.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="nativeHostArrayDst">The destination native host array.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements.</param>
        public static void CopyOnHost<T>(IntPtr hostAllocatedMemory, int srcOffset, T[,] nativeHostArrayDst, int dstOffset, int count)
        {
            DoCopyOnHost<T>(hostAllocatedMemory, srcOffset, nativeHostArrayDst, dstOffset, count);
        }

        /// <summary>
        /// Copies data on host.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostAllocatedMemory">The source host allocated memory.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="nativeHostArrayDst">The destination native host array.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements.</param>
        public static void CopyOnHost<T>(IntPtr hostAllocatedMemory, int srcOffset, T[,,] nativeHostArrayDst, int dstOffset, int count)
        {
            DoCopyOnHost<T>(hostAllocatedMemory, srcOffset, nativeHostArrayDst, dstOffset, count);
        }

        /// <summary>
        /// Does the copy on host.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostAllocatedMemory">The host allocated memory.</param>
        /// <param name="srcOffset">The SRC offset.</param>
        /// <param name="nativeHostArrayDst">The native host array DST.</param>
        /// <param name="dstOffset">The DST offset.</param>
        /// <param name="count">The count.</param>
        protected unsafe static void DoCopyOnHost<T>(IntPtr hostAllocatedMemory, int srcOffset, Array nativeHostArrayDst, int dstOffset, int count)
        {
            //Type type = typeof(T);
            GCHandle handle = GCHandle.Alloc(nativeHostArrayDst, GCHandleType.Pinned);
            try
            {
                int size = MSizeOf(typeof(T));
                IntPtr srcIntPtrOffset = new IntPtr(hostAllocatedMemory.ToInt64() + srcOffset * size);
                IntPtr dstIntPtr = handle.AddrOfPinnedObject();
                IntPtr dstOffsetPtr = new IntPtr(dstIntPtr.ToInt64() + dstOffset * size);
                CopyMemory(dstOffsetPtr, srcIntPtrOffset, (uint)(count * size));
            }
            finally
            {
                handle.Free();
            }     
        }

        /// <summary>
        /// Does the copy.
        /// </summary>
        /// <param name="srcArray">The source array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstArray">The destination array.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="size">The size.</param>
        protected static void DoCopy(Array srcArray, int srcOffset, Array dstArray, int dstOffset, int count, int size)
        {
            unsafe
            {
                GCHandle dstHandle = GCHandle.Alloc(dstArray, GCHandleType.Pinned);
                GCHandle srcHandle = GCHandle.Alloc(srcArray, GCHandleType.Pinned);
                try
                {
                    IntPtr srcIntPtr = srcHandle.AddrOfPinnedObject();
                    IntPtr srcOffsetPtr = new IntPtr(srcIntPtr.ToInt64() + srcOffset * size);
                    IntPtr dstIntPtr = dstHandle.AddrOfPinnedObject();
                    IntPtr dstOffsetPtr = new IntPtr(dstIntPtr.ToInt64() + dstOffset * size);
                    CopyMemory(dstOffsetPtr, srcOffsetPtr, (uint)(count * size));
                }
                finally
                {
                    dstHandle.Free();
                    srcHandle.Free();
                }
            }
        }

        /// <summary>
        /// Does the copy.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="srcArray">The source array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstArray">The destination array.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The count.</param>
        protected static void DoCopy<T>(Array srcArray, int srcOffset, Array dstArray, int dstOffset, int count)
        {
            DoCopy(srcArray, srcOffset, dstArray, dstOffset, count, MSizeOf(typeof(T)));
        }

        /// <summary>
        /// Does the copy.
        /// </summary>
        /// <param name="srcArray">The source array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstArray">The destination array.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="type">The type.</param>
        protected static void DoCopy(Array srcArray, int srcOffset, Array dstArray, int dstOffset, int count, Type type)
        {
            DoCopy(srcArray, srcOffset, dstArray, dstOffset, count, MSizeOf(type));
        }

        /// <summary>
        /// Destroys the stream.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        public abstract void DestroyStream(int streamId);

        /// <summary>
        /// Destroys all streams.
        /// </summary>
        public abstract void DestroyStreams();

        /// <summary>
        /// Copies to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="devArray">The device array.</param>
        public void CopyToDevice<T>(T[,] hostArray, T[,] devArray) where T : struct
        {
            DoCopyToDevice<T>(hostArray, 0, devArray, 0, hostArray.Length);
        }


        /// <summary>
        /// Copies to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="devArray">The device array.</param>
        public void CopyToDevice<T>(T[, ,] hostArray, T[, ,] devArray) where T : struct
        {
            DoCopyToDevice<T>(hostArray, 0, devArray, 0, hostArray.Length);
        }

        /// <summary>
        /// Allocates Unicode character array on device, copies to device and returns pointer.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <returns>The device array.</returns>
        public char[] CopyToDevice(string text)
        {
            return CopyToDevice(text.ToCharArray());
        }

        /// <summary>
        /// Allocates array on device, copies to device and returns pointer.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>The device array.</returns>
        public abstract T[] CopyToDevice<T>(T[] hostArray) where T : struct;

        /// <summary>
        /// Allocates array on device, copies to device and returns pointer.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>The device array.</returns>
        public abstract T[,] CopyToDevice<T>(T[,] hostArray) where T : struct;

        /// <summary>
        /// Allocates array on device, copies to device and returns pointer.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>The device array.</returns>
        public abstract T[, ,] CopyToDevice<T>(T[, ,] hostArray) where T : struct;

        ///// <summary>
        ///// Copies from device.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="dev">The device array.</param>
        ///// <param name="host">The host array.</param>
        //public abstract void CopyFromDevice<T>(T dev, out T host);

        /// <summary>
        /// Copies from device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="hostData">The host data.</param>
        public void CopyFromDevice<T>(T[] devArray, out T hostData) where T : struct
        {
            T[] hostArray = new T[1];
            DoCopyFromDevice<T>(devArray, 0, hostArray, 0, 1);
            hostData = hostArray[0];
        }

        /// <summary>
        /// Copies the complete device array to the host array.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="hostArray">The host array.</param>
        public void CopyFromDevice<T>(T[] devArray, T[] hostArray) where T : struct
        {
            DoCopyFromDevice<T>(devArray, hostArray);
        }

        /// <summary>
        /// Copies the complete device array to the host array.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="hostArray">The host array.</param>
        public void CopyFromDevice<T>(T[,] devArray, T[,] hostArray) where T : struct
        {
            DoCopyFromDevice<T>(devArray, hostArray);
        }

        /// <summary>
        /// Copies from device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The number of elements.</param>
        public void CopyFromDevice<T>(T[,] devArray, int devOffset, T[] hostArray, int hostOffset, int count) where T : struct
        {
            DoCopyFromDevice<T>(devArray, devOffset, hostArray, hostOffset, count);
        }

        /// <summary>
        /// Copies the complete device array to the host array.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="hostArray">The host array.</param>
        public void CopyFromDevice<T>(T[, ,] devArray, T[, ,] hostArray) where T : struct
        {
            DoCopyFromDevice<T>(devArray, hostArray);
        }

        /// <summary>
        /// Copies between preallocated arrays on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The source device array.</param>
        /// <param name="dstDevArray">The destination device array.</param>
        public abstract void CopyOnDevice<T>(T[] srcDevArray, T[] dstDevArray) where T : struct;

        /// <summary>
        /// Copies between preallocated arrays on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The source device array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstDevArray">The destination device array.</param>
        /// <param name="dstOffset">The destination offet.</param>
        /// <param name="count">The number of element.</param>
        public void CopyOnDevice<T>(T[] srcDevArray, int srcOffset, T[] dstDevArray, int dstOffset, int count) where T : struct
        {
            DoCopyOnDevice<T>(srcDevArray, srcOffset, dstDevArray, dstOffset, count);
        }

        /// <summary>
        /// Copies between preallocated arrays on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The source device array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstDevArray">The destination device array.</param>
        /// <param name="dstOffset">The destination offet.</param>
        /// <param name="count">The number of element.</param>
        public void CopyOnDevice<T>(DevicePtrEx srcDevArray, int srcOffset, DevicePtrEx dstDevArray, int dstOffset, int count) where T : struct
        {
            DoCopyOnDevice<T>(srcDevArray, srcOffset, dstDevArray, dstOffset, count);
        }

        /// <summary>
        /// Copies between preallocated arrays on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The source device array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstDevArray">The destination device array.</param>
        /// <param name="dstOffset">The destination offet.</param>
        /// <param name="count">The number of element.</param>
        /// <param name="streamId">Stream id.</param>
        public void CopyOnDeviceAsync<T>(T[] srcDevArray, int srcOffset, T[] dstDevArray, int dstOffset, int count, int streamId) where T : struct
        {
            DoCopyOnDeviceAsync<T>(srcDevArray, srcOffset, dstDevArray, dstOffset, count, streamId);
        }

        /// <summary>
        /// Copies between preallocated arrays on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The source device array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstDevArray">The destination device array.</param>
        /// <param name="dstOffset">The destination offet.</param>
        /// <param name="count">The number of element.</param>
        /// <param name="streamId">Stream id.</param>
        public void CopyOnDeviceAsync<T>(DevicePtrEx srcDevArray, int srcOffset, DevicePtrEx dstDevArray, int dstOffset, int count, int streamId) where T : struct
        {
            DoCopyOnDeviceAsync<T>(srcDevArray, srcOffset, dstDevArray, dstOffset, count, streamId);
        }

        /// <summary>
        /// Copies between preallocated arrays on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The source device array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstDevArray">The destination device array.</param>
        /// <param name="dstOffset">The destination offet.</param>
        /// <param name="count">The number of element.</param>
        public void CopyOnDevice<T>(T[,] srcDevArray, int srcOffset, T[,] dstDevArray, int dstOffset, int count) where T : struct
        {
            DoCopyOnDevice<T>(srcDevArray, srcOffset, dstDevArray, dstOffset, count);
        }

        /// <summary>
        /// Copies between preallocated arrays on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The source device array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstDevArray">The destination device array.</param>
        /// <param name="dstOffset">The destination offet.</param>
        /// <param name="count">The number of element.</param>
        public void CopyOnDevice<T>(T[, ,] srcDevArray, int srcOffset, T[, ,] dstDevArray, int dstOffset, int count) where T : struct
        {
            DoCopyOnDevice<T>(srcDevArray, srcOffset, dstDevArray, dstOffset, count);
        }

        public void CopyOnDeviceAsync<T>(T[, ,] srcDevArray, int srcOffset, T[, ,] dstDevArray, int dstOffset, int count, int streamId) where T : struct
        {
            DoCopyOnDeviceAsync<T>(srcDevArray, srcOffset, dstDevArray, dstOffset, count, streamId);
        }

        /// <summary>
        /// Copies between preallocated arrays on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The source device array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstDevArray">The destination device array.</param>
        /// <param name="dstOffet">The destination offet.</param>
        /// <param name="count">The number of element.</param>
        protected abstract void DoCopyOnDevice<T>(Array srcDevArray, int srcOffset, Array dstDevArray, int dstOffet, int count) where T : struct;

        protected abstract void DoCopyOnDevice<T>(DevicePtrEx srcDevArray, int srcOffset, DevicePtrEx dstDevArray, int dstOffet, int count) where T : struct;

        protected abstract void DoCopyOnDeviceAsync<T>(Array srcDevArray, int srcOffset, Array dstDevArray, int dstOffet, int count, int streamId) where T : struct;

        protected abstract void DoCopyOnDeviceAsync<T>(DevicePtrEx srcDevArray, int srcOffset, DevicePtrEx dstDevArray, int dstOffet, int count, int streamId) where T : struct;

        internal SmartStage[] _smartInputStages = new SmartStage[0];

        internal SmartStage[] _smartOutputStages = new SmartStage[0];

        internal enum eSmartStageType { Input, Output };

        internal struct SmartStage
        {
            public IntPtr Pointer { get; set; }
            public int Length { get; set; }
            public eSmartStageType Type { get; set; }
        }




        /// <summary>
        /// Gets a value indicating whether this instance is smart copy enabled.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if this instance is smart copy enabled; otherwise, <c>false</c>.
        /// </value>
        public virtual bool IsSmartCopyEnabled { get; protected set; }

        /// <summary>
        /// Gets or sets a value indicating whether device supports smart copy.
        /// </summary>
        /// <value>
        ///   <c>true</c> if supports smart copy; otherwise, <c>false</c>.
        /// </value>
        public virtual bool SupportsSmartCopy
        {
            get { return true; }
        }

        private bool _wasMultithreadingEnabled = false;

        /// <summary>
        /// Enables smart copy. The overloads of CopyToDeviceAsync and CopyFromDeviceAsync using pinned memory staging posts
        /// is now possible. If multithreading is not enabled this will be done automatically.
        /// </summary>
        public virtual void EnableSmartCopy()
        {
            if (IsSmartCopyEnabled)
                throw new CudafyHostException(CudafyHostException.csSMART_COPY_ALREADY_ENABLED);
            _wasMultithreadingEnabled = IsMultithreadingEnabled;
            if (!IsMultithreadingEnabled)
                EnableMultithreading();
            IsSmartCopyEnabled = true;
        }

        /// <summary>
        /// Disables smart copy and multithreading if this was set automatically during smart copy enable.
        /// </summary>
        public virtual void DisableSmartCopy()
        {
            if (!IsSmartCopyEnabled)
                throw new CudafyHostException(CudafyHostException.csSMART_COPY_IS_NOT_ENABLED);
            if (!_wasMultithreadingEnabled)
                DisableMultithreading();
            IsSmartCopyEnabled = false;
        }

        /// <summary>
        /// Allocates on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <returns>Device array of length 1.</returns>
        public virtual T[] Allocate<T>() where T : struct
        {
            return Allocate<T>(1);
        }

        /// <summary>
        /// Allocates array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">Length of 1D array.</param>
        /// <returns>Device array of length x.</returns>
        public abstract T[] Allocate<T>(int x) where T : struct;

        /// <summary>
        /// Allocates array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">The x dimension.</param>
        /// <param name="y">The y dimension.</param>
        /// <returns>2D device array.</returns>
        public abstract T[,] Allocate<T>(int x, int y) where T : struct;

        /// <summary>
        /// Allocates array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">The x dimension.</param>
        /// <param name="y">The y dimension.</param>
        /// <param name="z">The z dimension.</param>
        /// <returns>3D device array.</returns>
        public abstract T[, ,] Allocate<T>(int x, int y, int z) where T : struct;

        /// <summary>
        /// Allocates array on device of same size as supplied host array.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>1D device array.</returns>
        public abstract T[] Allocate<T>(T[] hostArray) where T : struct;

        /// <summary>
        /// Allocates array on device of same size as supplied host array.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>1D device array.</returns>
        public abstract T[,] Allocate<T>(T[,] hostArray) where T : struct;

        /// <summary>
        /// Allocates array on device of same size as supplied host array.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>1D device array.</returns>
        public abstract T[, ,] Allocate<T>(T[, ,] hostArray) where T : struct;


        /// <summary>
        /// Sets the specified device array to zero.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        public void Set<T>(T[] devArray) where T : struct
        {
            DoSet<T>(devArray);
        }

        /// <summary>
        /// Sets the specified device array to zero.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        public void Set<T>(T[,] devArray) where T : struct
        {
            DoSet<T>(devArray);
        }

        /// <summary>
        /// Sets the specified device array to zero.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        public void Set<T>(T[, ,] devArray) where T : struct
        {
            DoSet<T>(devArray);
        }

        /// <summary>
        /// Sets the specified device array to zero.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="count">The number of elements.</param>
        public void Set<T>(T[] devArray, int offset, int count) where T : struct
        {
            DoSet<T>(devArray, offset, count);
        }

        /// <summary>
        /// Sets the specified device array to zero.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="count">The number of elements.</param>
        public void Set<T>(T[,] devArray, int offset, int count) where T : struct
        {
            DoSet<T>(devArray, offset, count);
        }

        /// <summary>
        /// Sets the specified device array to zero.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="count">The number of elements.</param>
        public void Set<T>(T[, ,] devArray, int offset, int count) where T : struct
        {
            DoSet<T>(devArray, offset, count);
        }


        /// <summary>
        /// Does the set.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="count">The count.</param>
        protected abstract void DoSet<T>(Array devArray, int offset = 0, int count = 0) where T : struct;

        /// <summary>
        /// Frees the specified data array on device.
        /// </summary>
        /// <param name="devArray">The device array to free.</param>
        public abstract void Free(object devArray);

        /// <summary>
        /// Frees all data arrays on device.
        /// </summary>
        public abstract void FreeAll();

        /// <summary>
        /// Verifies launch arguments are on GPU and are supported.
        /// </summary>
        /// <param name="args">The arguments.</param>
        /// <exception cref="ArgumentException">Argument is either not on GPU or not supported.</exception>
        protected void VerifyMembersAreOnGPU(params object[] args)
        {
            int i = 1;
            lock (_lock)
            {
                foreach (object o in args)
                {
                    Type type = o.GetType();
                    //if (type == typeof(uint) || type == typeof(float) || type == typeof(int) || type == typeof(GThread))
                    if(type.IsValueType || type == typeof(GThread))
                        continue;

                    if (!_deviceMemory.ContainsKey(o))
                        throw new ArgumentException(string.Format("Argument {0} of type {1} is not on the GPU or not supported.", i, type));

                    i++;
                }
            }
        }

        /// <summary>
        /// Verifies the specified data is on GPU.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <exception cref="CudafyHostException">Data is not on GPU.</exception>
        public void VerifyOnGPU(object data)
        {
            if (!IsOnGPU(data))
                throw new CudafyHostException(CudafyHostException.csDATA_IS_NOT_ON_GPU);
        }


        /// <summary>
        /// Determines whether the specified data is on GPU.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <returns>
        /// 	<c>true</c> if the specified data is on GPU; otherwise, <c>false</c>.
        /// </returns>
        public bool IsOnGPU(object data)
        {
            lock (_lock)
            {
                return data != null && _deviceMemory.ContainsKey(data);
            }
        }

        /// <summary>
        /// Gets the pointer to the native GPU data.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <returns>Pointer to the actual data.</returns>
        public virtual object GetGPUData(object data)
        {
            VerifyOnGPU(data);
            return _deviceMemory[data];
        }


#warning OpenCL might change this
        /// <summary>
        /// Gets the size of the type specified. Note that this differs from Marshal.SizeOf for System.Char (it returns 2 instead of 1).
        /// </summary>
        /// <param name="t">The type to get the size of.</param>
        /// <returns>Size of type in bytes.</returns>
        public static int MSizeOf(Type t)
        {
            if (t == typeof(char))
                return 2;
            else
                return Marshal.SizeOf(t);
        }

        /// <summary>
        /// Gets the version.
        /// </summary>
        /// <returns></returns>
        public virtual int GetDriverVersion()
        {
            return 1010;
        }

        //public virtual void HostRegister<T>(T[] hostArray)
        //{           
        //}

        //public virtual void Unregister<T>(T[] hostArray)
        //{
        //}

        ///// <summary>
        ///// Convert 2D to 1D array.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="array2d">The 2D array.</param>
        ///// <returns>1D array.</returns>
        //protected T[] Convert2Dto1DArray<T>(T[,] array2d)
        //{
        //    int x = array2d.GetUpperBound(0) - 1;
        //    int y = array2d.GetUpperBound(1) - 1;
        //    T[] array1d = new T[x * y];
        //    int dstIndex = 0;
        //    for (int i = 0; i < x; i++)
        //    {
        //        for (int j = 0; j < y; j++)
        //        {
        //            array1d[dstIndex] = array2d[x, y];
        //            dstIndex++;
        //        }
        //    }
        //    return array1d;
        //}

        ///// <summary>
        ///// Convert 3D to 1D array.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="array3d">The 3D array.</param>
        ///// <returns>1D array.</returns>
        //protected T[] Convert3Dto1DArray<T>(T[,,] array3d)
        //{
        //    int x = array3d.GetUpperBound(0) - 1;
        //    int y = array3d.GetUpperBound(1) - 1;
        //    int z = array3d.GetUpperBound(2) - 1;
        //    T[] array1d = new T[x * y * z];
        //    int dstIndex = 0;
        //    for (int i = 0; i < x; i++)
        //    {
        //        for (int j = 0; j < y; j++)
        //        {
        //            for (int k = 0; k < y; k++)
        //            {
        //                array1d[dstIndex] = array3d[x, y, k];
        //                dstIndex++;
        //            }
        //        }
        //    }
        //    return array1d;
        //}

        ///// <summary>
        /////Convert 2D to 1D array.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="array2d">The 2D array.</param>
        ///// <returns>1D array.</returns>
        //protected T[] Convert2Dto1DArrayPrimitive<T>(T[,] array2d)
        //{
        //    int len = array2d.Length;
        //    T[] array1d = new T[array2d.Length];
        //    System.Buffer.BlockCopy(array2d, 0, array1d, 0, len * Marshal.SizeOf(typeof(T))); 

        //    return array1d;
        //}

        ///// <summary>
        ///// Convert 3D to 1D array.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="array3d">The 3D array.</param>
        ///// <returns>1D array.</returns>
        //protected T[] Convert3Dto1DArrayPrimitive<T>(T[,,] array3d)
        //{
        //    int len = array3d.Length;
        //    T[] array1d = new T[array3d.Length];
        //    System.Buffer.BlockCopy(array3d, 0, array1d, 0, len * Marshal.SizeOf(typeof(T)));

        //    return array1d;
        //}

    }

    /// <summary>
    /// Base class for Device data pointers
    /// </summary>
    public abstract class DevicePtrEx
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DevicePtrEx"/> class.
        /// </summary>
        public DevicePtrEx()
        {
            Disposed = false;
            _children = new List<DevicePtrEx>();
        }
        
        /// <summary>
        /// Gets the size of the X.
        /// </summary>
        /// <value>
        /// The size of the X.
        /// </value>
        public int XSize { get; protected set; }
        /// <summary>
        /// Gets the size of the Y.
        /// </summary>
        /// <value>
        /// The size of the Y.
        /// </value>
        public int YSize { get; protected set; }
        /// <summary>
        /// Gets the size of the Z.
        /// </summary>
        /// <value>
        /// The size of the Z.
        /// </value>
        public int ZSize { get; protected set; }
 
        /// <summary>
        /// Gets the number of dimensions (rank).
        /// </summary>
        public int Dimensions { get; protected set; }

        /// <summary>
        /// Gets the total size.
        /// </summary>
        public int TotalSize
        {
            get { return XSize * YSize * ZSize; }
        }

        /// <summary>
        /// Gets the offset1 D.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <returns></returns>
        public int GetOffset1D(int x)
        {
            return x;
        }

        /// <summary>
        /// Gets the offset1 D.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        public int GetOffset1D(int x, int y)
        {
            int v = (x * YSize) + y;
            return v;
        }

        /// <summary>
        /// Gets the offset1 D.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public int GetOffset1D(int x, int y, int z)
        {
            return (x * YSize * ZSize) + (y * ZSize) + z;//i*length*width + j*width + k
        }

        //public DevicePtrEx Original { get; set; }

        /// <summary>
        /// Gets the pointer when overridden.
        /// </summary>
        public virtual IntPtr Pointer
        {
            get { return IntPtr.Zero; }
        }

        /// <summary>
        /// Gets or sets the offset.
        /// </summary>
        /// <value>
        /// The offset.
        /// </value>
        public virtual int Offset { get; protected set; }

        /// <summary>
        /// Gets or sets a value indicating whether this <see cref="DevicePtrEx"/> is disposed.
        /// </summary>
        /// <value>
        ///   <c>true</c> if disposed; otherwise, <c>false</c>.
        /// </value>
        public bool Disposed { get; set; }

        /// <summary>
        /// Gets the dimensions.
        /// </summary>
        /// <returns></returns>
        public int[] GetDimensions()
        {
            int[] dims = new int[Dimensions];
            if (Dimensions > 0)
                dims[0] = XSize;
            if (Dimensions > 1)
                dims[1] = YSize;
            if (Dimensions > 2)
                dims[2] = ZSize;
            return dims;
        }

        /// <summary>
        /// Adds the child.
        /// </summary>
        /// <param name="ptrEx">The PTR ex.</param>
        public void AddChild(DevicePtrEx ptrEx)
        {
            _children.Add(ptrEx);
        }

        /// <summary>
        /// Removes the children.
        /// </summary>
        public void RemoveChildren()
        {
            foreach (var ptr in _children)
                ptr.RemoveChildren();
            _children.Clear();
        }

        /// <summary>
        /// Gets the level 1 children.
        /// </summary>
        public IEnumerable<DevicePtrEx> Children
        {
            get { return _children; }
        }

        /// <summary>
        /// Gets all children.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<DevicePtrEx> GetAllChildren()
        {
            foreach (var ptr in _children)
            {
                yield return ptr;
                foreach (var child in ptr.GetAllChildren())
                    yield return child;
            }
            
        }

        private List<DevicePtrEx> _children;

        /// <summary>
        /// Gets a value indicating whether created from cast.
        /// </summary>
        /// <value>
        ///   <c>true</c> if created from cast; otherwise, <c>false</c>.
        /// </value>
        public bool CreatedFromCast { get; protected set; }
    }
}
