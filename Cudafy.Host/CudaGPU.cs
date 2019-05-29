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
using System.Reflection;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Threading;
using GASS.CUDA;
using GASS.CUDA.Tools;
using GASS.CUDA.Types;
using GASS.Types;
namespace Cudafy.Host
{
    public enum eMemoryType
    {
        Host = 1,
        Device,
        Array,
        Unified
    }

    public struct Peer2PeerTokens
    {
        public ulong p2pToken;
        public uint vaSpaceToken;
    };
    
    /// <summary>
    /// Represents a Cuda GPU.
    /// </summary>
    public sealed class CudaGPU : GPGPU
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CudaGPU"/> class.
        /// </summary>
        /// <param name="deviceId">The device id.</param>
        public CudaGPU(int deviceId = 0) : base(deviceId)
        {
            try
            {
                _cuda = new CUDA(deviceId, true);
                _ccs = null;
                if (this.GetDriverVersion() < 5000)
                    throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, "CUDA Driver Version 4.2 or earlier");
                if (IntPtr.Size == 8)
                    _runtimeDriver = new CUDARuntime64();
                else
                    _runtimeDriver = new CUDARuntime32();
            }
            catch (IndexOutOfRangeException)
            {
                throw new CudafyHostException(CudafyHostException.csDEVICE_ID_OUT_OF_RANGE);
            }
            _hostHandles = new Dictionary<IntPtr, CUcontext>();
        }

        private ICUDARuntime _runtimeDriver;

        private CUDA _cuda;

        private CUevent _startEvent;

        private CUevent _stopEvent;

        private CUDAContextSynchronizer _ccs;

        /// <summary>
        /// Enables peer access from within a kernel. Only supported on Tesla devices and Linux or Windows TCC.
        /// </summary>
        /// <param name="peer">Peer to access. This is a one-way relationship.</param>
        public override void EnablePeerAccess(GPGPU peer)
        {
            if (!(peer is CudaGPU))
                throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, peer.GetType());
            CudaGPU cudaPeer = peer as CudaGPU;
            _cuda.EnablePeerAccess(cudaPeer.GetDeviceContext(), 0);
        }


        /// <summary>
        /// Disables peer access.
        /// </summary>
        /// <param name="peer">Accessible peer to disable access to.</param>
        public override void DisablePeerAccess(GPGPU peer)
        {
            if (!(peer is CudaGPU))
                throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, peer.GetType());
            CudaGPU cudaPeer = peer as CudaGPU;
            _cuda.DisablePeerAccess(cudaPeer.GetDeviceContext());
        }

        /// <summary>
        /// Use this to check if device supports direct access from kernel to another device.
        /// Only supported on Tesla devices and Linux or Windows TCC.
        /// </summary>
        /// <param name="peer">Peer to access.</param>
        /// <returns>
        /// True if access is possible, else false.
        /// </returns>
        public override bool CanAccessPeer(GPGPU peer)
        {
            if(!(peer is CudaGPU))
                return false;
            CUDA dstCUDA = (CUDA)(peer as CudaGPU).CudaDotNet;
            CUdevice srcDevice = _cuda.GetDevice(this.DeviceId);
            CUdevice peerDevice = dstCUDA.GetDevice(peer.DeviceId);
            bool res = _cuda.DeviceCanAccessPeer(srcDevice, peerDevice);
            return res;
        }

        public eMemoryType GetPointerMemoryType<T>(T[] data)
        {
            return DoGetPointerMemoryType<T>(data);
        }

        public eMemoryType GetPointerMemoryType<T>(T[,] data)
        {
            return DoGetPointerMemoryType<T>(data);
        }

        public eMemoryType GetPointerMemoryType<T>(T[,,] data)
        {
            return DoGetPointerMemoryType<T>(data);
        }

        private eMemoryType DoGetPointerMemoryType<T>(Array data)
        {
            CUDevicePtrEx ptrEx = TryGetDeviceMemory(data) as CUDevicePtrEx;
            eMemoryType rc = eMemoryType.Host;
            if (ptrEx != null)
            {
                CUMemoryType mt = _cuda.GetPointerMemoryType(CUPointerAttribute.MemoryType, ptrEx.DevPtr);
                rc = (eMemoryType)(int)mt;
            }
            return rc;
        }

        public Peer2PeerTokens GetPointerP2PTokens<T>(T[] data)
        {
            return DoGetPointerP2PTokens<T>(data);
        }

        public Peer2PeerTokens GetPointerP2PTokens<T>(T[,] data)
        {
            return DoGetPointerP2PTokens<T>(data);
        }

        public Peer2PeerTokens GetPointerP2PTokens<T>(T[, ,] data)
        {
            return DoGetPointerP2PTokens<T>(data);
        }

        private Peer2PeerTokens DoGetPointerP2PTokens<T>(Array data)
        {
            CUDevicePtrEx ptrEx = (CUDevicePtrEx)GetDeviceMemory(data);
            CUP2PTokens t = _cuda.GetP2PTokens(ptrEx.DevPtr);
            Peer2PeerTokens p2pt = new Peer2PeerTokens()
            {
                p2pToken = t.p2pToken,
                vaSpaceToken = t.vaSpaceToken
            };
            return p2pt;
        }

        //protected override IntPtr DoGetPointerAttribute<T>(ePointerAttribute attr, Array devArray)
        //{
        //    CUPointerAttribute devAttr = (CUPointerAttribute)(int)attr;
        //    CUdeviceptr devPtr = ((CUDevicePtrEx)GetDeviceMemory(devArray)).DevPtr;

        //    CUMemoryType mt = _cuda.GetPointerMemoryType(devAttr, devPtr);
        //    CUcontext ctx = _cuda.GetPointerContext(devPtr);
        //    return _cuda.GetPointerAttribute(devAttr, devPtr);
        //}

        internal CUcontext GetPointerContext<T>(Array data)
        {
            CUDevicePtrEx ptrEx = (CUDevicePtrEx)GetDeviceMemory(data);
            CUcontext ctx = _cuda.GetPointerContext(ptrEx.DevPtr);
            return ctx;
        }

        /// <summary>
        /// Gets the CUstream object identified by streamId.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        /// <returns>CUstream object.</returns>
        public override object GetStream(int streamId)
        {
            lock (_lock)
            {
                CUstream cuStr = new CUstream();
                if (streamId >= 0 && !_streams.ContainsKey(streamId))
                {
                    try
                    {
                        cuStr = _cuda.CreateStream();
                    }
                    catch (CUDAException ex)
                    {
                        HandleCUDAException(ex);
                    }
                    _streams.Add(streamId, cuStr);
                }
                else if (streamId >= 0)
                {
                    cuStr = (CUstream)_streams[streamId];
                }

                return cuStr;
            }
        }

        /// <summary>
        /// Locks this instance.
        /// </summary>
        /// <exception cref="CudafyHostException">Multithreading is not enabled.</exception>
        public override void Lock()
        {
            if (_ccs == null)
                throw new CudafyHostException(CudafyHostException.csMULTITHREADING_IS_NOT_ENABLED);
            try
            {
                //Debug.WriteLine("Locking");
                //var curCtx = _cuda.GetCurrentContext();
                //if (curCtx.Pointer != IntPtr.Zero && _cuda.DeviceContext.Pointer != curCtx.Pointer)
                    _ccs.Lock();
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
        }

        /// <summary>
        /// Unlocks this instance.
        /// </summary>
        /// <exception cref="CudafyHostException">Device is not locked.</exception>
        public override void Unlock()
        {
            if (_ccs == null)
                throw new CudafyHostException(CudafyHostException.csMULTITHREADING_IS_NOT_ENABLED);
            try
            {
                //Debug.WriteLine("Unlocking");
                //if(_ccs.IsLocked)
                    _ccs.Unlock();
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
        }

        /// <summary>
        /// Gets a value indicating whether this instance is locked.
        /// </summary>
        /// <value>
        ///   <c>true</c> if this instance is locked; otherwise, <c>false</c>.
        /// </value>
        public override bool IsLocked
        {
            get
            {
                return _ccs != null && _ccs.IsLocked;
            }
            protected set
            {
            }
        }

        /// <summary>
        /// Allows multiple threads to access this GPU.
        /// </summary>
        public override void EnableMultithreading()
        {
            if (_ccs == null)
            {
                //_ccs = new CUDAContextSynchronizer(_cuda.CurrentContext);
                _ccs = new CUDAContextSynchronizer(_cuda.DeviceContext);
                try
                {
                    _cc = _ccs.MakeFloating();
                }
                catch (CUDAException ex)
                {
                    HandleCUDAException(ex);
                }
            }
        }

        /// <summary>
        /// Called once multiple threads have completed work.
        /// </summary>
        public override void DisableMultithreading()
        {
            if (_ccs != null)
            {
                try
                {
                    _ccs.StopFloating(_cc);
                }
                catch (CUDAException ex)
                {
                    HandleCUDAException(ex);
                }
                _ccs = null;
            }
        }

        /// <summary>
        /// Gets a value indicating whether this instance has multithreading enabled.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if this instance is multithreading enabled; otherwise, <c>false</c>.
        /// </value>
        public override bool IsMultithreadingEnabled
        {
            get { return _ccs != null; }
        }

        private CUcontext _cc;

        /// <summary>
        /// Sets the current context to the context associated with this device when it was created.
        /// Use of this method is vitally important when working with multiple GPUs.
        /// </summary>
        public override void SetCurrentContext()
        {
            _cuda.SetCurrentContext(_cuda.DeviceContext);
        }

        internal CUcontext GetDeviceContext()
        {
            return _cuda.DeviceContext;
        }

        /// <summary>
        /// Gets a value indicating whether this instance is current context. You must ensure this is true before
        /// attempting communication with device.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if this instance is current context; otherwise, <c>false</c>.
        /// </value>
        public override bool IsCurrentContext
        {
            get
            {
                CUcontext curCtx = CUDA.GetCurrentContext();
                CUcontext devCtx = GetDeviceContext();
                bool res = curCtx.Pointer == devCtx.Pointer;
                return res;
            }
        }

        /// <summary>
        /// Gets the CUDA.NET handle. You can cast this to CUDA in the GASS.CUDA namespace.
        /// See http://www.hoopoe-cloud.com/Solutions/CUDA.NET/Default.aspx
        /// </summary>
        public object CudaDotNet
        {
            get { return _cuda; }
        }

        /// <summary>
        /// Gets the device properties.
        /// </summary>
        /// <param name="useAdvanced">If true then also get properties via cudart.dll (e.g. MultiProcessorCount).</param>
        /// <returns>Device properties instance.</returns>
        /// <exception cref="CudafyHostException">Failed to get properties.</exception>
        /// <exception cref="DllNotFoundException">Library named cudart.dll is needed for advanced properties and was not found.</exception>
        public override GPGPUProperties GetDeviceProperties(bool useAdvanced = true)
        {
            Device dev = _cuda.Devices[DeviceId];//CurrentDevice;
            
            GPGPUProperties props = new GPGPUProperties();
            props.UseAdvanced = useAdvanced;
            props.Capability = dev.ComputeCapability;
            props.PlatformName = "CUDA " + GetDriverVersion().ToString();
            props.Name = dev.Name;
            props.DeviceId = dev.Ordinal;
            props.TotalMemory = dev.TotalMemory;
            DeviceProperties dp = dev.Properties;
            props.ClockRate = dp.ClockRate;
            props.MaxGridSize = new dim3(dp.MaxGridSize[0], dp.MaxGridSize[1]);
            props.MaxThreadsSize = new dim3(dp.MaxThreadsDim[0], dp.MaxThreadsDim[1]);
            props.MaxThreadsPerBlock = dp.MaxThreadsPerBlock;
            props.MemoryPitch = dp.MemoryPitch;
            props.RegistersPerBlock = dp.RegistersPerBlock;
            props.SharedMemoryPerBlock = dp.SharedMemoryPerBlock;
            props.WarpSize = dp.SIMDWidth;
            props.TotalConstantMemory = dp.TotalConstantMemory;
            props.TextureAlignment = dp.TextureAlign;
            props.MultiProcessorCount = 0;
            
            if (useAdvanced)
            {
                cudaDeviceProp devProps = new cudaDeviceProp();
                try
                {
                    cudaError rc = _runtimeDriver.GetDeviceProperties(ref devProps, props.DeviceId);
                    if (rc == cudaError.cudaSuccess)
                    {
                        props.MultiProcessorCount = devProps.multiProcessorCount;
                        props.MaxThreadsPerMultiProcessor = devProps.maxThreadsPerMultiProcessor;
                        props.CanMapHostMemory = devProps.canMapHostMemory == 0 ? false : true;
                        props.ConcurrentKernels = devProps.concurrentKernels;
                        props.ComputeMode = devProps.computeMode;
                        props.DeviceOverlap = devProps.deviceOverlap == 0 ? false : true;
                        props.ECCEnabled = devProps.ECCEnabled == 0 ? false : true;
                        props.Integrated = devProps.integrated == 0 ? false : true;
                        props.KernelExecTimeoutEnabled = devProps.kernelExecTimeoutEnabled == 0 ? false : true;
                        props.PciBusID = devProps.pciBusID;
                        props.PciDeviceID = devProps.pciDeviceID;
                        props.TotalGlobalMem = devProps.totalGlobalMem;
                        props.AsynchEngineCount = devProps.asyncEngineCount;
#if LINUX 
                        props.HighPerformanceDriver = true;
#else
                        props.HighPerformanceDriver = devProps.tccDriver == 1;
#endif
                        
                    }
                    else
                        throw new CudafyHostException(CudafyHostException.csFAILED_TO_GET_PROPERIES_X, rc);
                }
                catch (CudafyHostException)
                {
                    throw;
                }
                catch (DllNotFoundException ex)
                {
                    props.Message = string.Format("Dll {0} not found.", ex.Message);
                    props.UseAdvanced = false;
                    Debug.WriteLine(props.Message);
                    throw;
                }
            }

            return props;
        }

        /// <summary>
        /// Synchronizes context.
        /// </summary>
        public override void Synchronize()
        {
            try
            {
                _cuda.SynchronizeContext();
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
        }

        private void HandleCUDAException(CUDAException ex)
        {
            string addInfo = string.Empty;
            switch (ex.CUDAError)
            {
                case CUResult.ECCUncorrectable:
                    break;
                case CUResult.ErrorAlreadyAcquired:
                    break;
                case CUResult.ErrorAlreadyMapped:
                    break;
                case CUResult.ErrorArrayIsMapped:
                    break;
                case CUResult.ErrorContextAlreadyCurrent:
                    break;
                case CUResult.ErrorDeinitialized:
                    break;
                case CUResult.ErrorFileNotFound:
                    break;
                case CUResult.ErrorInvalidContext:
                    break;
                case CUResult.ErrorInvalidDevice:
                    break;
                case CUResult.ErrorInvalidHandle:
                    break;
                case CUResult.ErrorInvalidImage:
                    break;
                case CUResult.ErrorInvalidSource:
                    break;
                case CUResult.ErrorInvalidValue:
                    break;
                case CUResult.ErrorLaunchFailed:
                    break;
                case CUResult.ErrorLaunchIncompatibleTexturing:
                    break;
                case CUResult.ErrorLaunchOutOfResources:
                    break;
                case CUResult.ErrorLaunchTimeout:
                    break;
                case CUResult.ErrorMapFailed:
                    break;
                case CUResult.ErrorNoBinaryForGPU:
                    addInfo = "Ensure that compiled architecture version is suitable for device"; 
                    break;
                case CUResult.ErrorNoDevice:
                    break;
                case CUResult.ErrorNotFound:
                    break;
                case CUResult.ErrorNotInitialized:
                    addInfo = "Ensure that suitable GPU and CUDA is correctly installed. If newly installed then reboot may be necessary.";
                    break;
                case CUResult.ErrorNotMapped:
                    break;
                case CUResult.ErrorNotReady:
                    break;
                case CUResult.ErrorOutOfMemory:
                    addInfo = "Ensure that memory on GPU is being explicitly released with Free()"; 
                    break;
                case CUResult.ErrorUnknown:
                    break;
                case CUResult.ErrorUnmapFailed:
                    break;
                case CUResult.NotMappedAsArray:
                    break;
                case CUResult.NotMappedAsPointer:
                    break;
                case CUResult.PointerIs64Bit:
                    break;
                case CUResult.SizeIs64Bit:
                    break;
                case CUResult.Success:
                    break;
                default:
                    break;
            }
            if(string.IsNullOrEmpty(addInfo))
                throw new CudafyHostException(CudafyHostException.csCUDA_EXCEPTION_X, ex.CUDAError.ToString());
            else
                throw new CudafyHostException(CudafyHostException.csCUDA_EXCEPTION_X_X, ex.CUDAError.ToString(), addInfo);
        }



        /// <summary>
        /// Starts the timer.
        /// </summary>
        public override void StartTimer()
        {
            try
            {
                _startEvent = _cuda.CreateEvent();
                _cuda.RecordEvent(_startEvent);
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
        }

        /// <summary>
        /// Stops the timer.
        /// </summary>
        /// <returns>Elapsed time.</returns>
        public override float StopTimer()
        {
            float time = 0;
            try
            {
                _stopEvent = _cuda.CreateEvent();
                _cuda.RecordEvent(_stopEvent);
                _cuda.SynchronizeEvent(_stopEvent);
                time = _cuda.ElapsedTime(_startEvent, _stopEvent);
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
            return time;
        }

        /// <summary>
        /// Gets the free memory.
        /// </summary>
        /// <value>The free memory.</value>
        public override ulong FreeMemory
        {
            get 
            {
                if (!IsCurrentContext)
                    return 0;
                return _cuda.FreeMemory; 
            }
        }

        /// <summary>
        /// Gets the total memory.
        /// </summary>
        /// <value>
        /// The total memory.
        /// </value>
        public override ulong TotalMemory
        {
            get 
            {
                if (!IsCurrentContext)
                    return 0;
                return _cuda.TotalMemory; 
            }
        }

        /// <summary>
        /// Explicitly creates a stream.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        public override void CreateStream(int streamId)
        {
            if (streamId < 0)
                throw new ArgumentOutOfRangeException("streamId must be greater than or equal to zero");
            if (_streams.ContainsKey(streamId))
                throw new CudafyHostException(CudafyHostException.csSTREAM_X_ALREADY_SET, streamId);

            var cuStr = _cuda.CreateStream();
            _streams.Add(streamId, cuStr);

        }

        #region Launch

        private int AlignUp(int offset, int alignment)
        {
            int newoffset = ((offset) + (alignment) - 1) & ~((alignment) - 1);
            return newoffset;
        }

        /// <summary>
        /// Does the launch.
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id, or -1 for non-async.</param>
        /// <param name="gpuMethodInfo">The gpu method info.</param>
        /// <param name="arguments">The arguments.</param>
        protected override void DoLaunch(dim3 gridSize, dim3 blockSize, int streamId, KernelMethodInfo gpuMethodInfo, params object[] arguments)
        {
#warning TODO- Call cuPrintfInit if necessary (_module.CanPrint;)( http://www.jeremykemp.co.uk/08/02/2010/cuda-cuprintf/ )

            //if (_module.CanPrint)
            //{
            //    CUfunction printInit = _cuda.GetModuleFunction("cudaPrintfInit");
            //    _cuda.Launch(printInit);
            //}
            if (IsSmartCopyEnabled)
            {
                lock (_smartCopyLock)
                {

                    while (_streamsPending.Where(sp => sp.StreamId == streamId).Count() > 0)
                        Monitor.Wait(_smartCopyLock);
                    Monitor.Pulse(_smartCopyLock);
                }
                Lock();
            }

#warning TODO Set Shared memory size
            
            int ptrSize = CUdeviceptr.Size;
                
            CUfunction function = _cuda.GetModuleFunction((CUmodule)_module.Tag, gpuMethodInfo.Method.Name);
            _cuda.SetFunctionBlockShape(function, blockSize.x, blockSize.y, blockSize.z);
            int offset = 0;
            foreach (object o in arguments)
            {
                Type type = o.GetType();                
                if (o is GThread)
                {
                    throw new CudafyFatalException("GThread should not be passed to launch!");
                }
                try
                {
                    //if (type == typeof(int))
                    //{
                    //    offset = AlignUp(offset, 4);
                    //    _cuda.SetParameter(function, offset, (uint)((int)o));// (uint)(long)
                    //    offset += 4; //;
                    //}
                    if (type == typeof(uint))
                    {
                        offset = AlignUp(offset, 4);
                        _cuda.SetParameter(function, offset, (uint)o);// (uint)(long)
                        offset += 4; //;
                    }
                    else if (type == typeof(float))
                    {
                        offset = AlignUp(offset, 4);
                        _cuda.SetParameter(function, offset, (float)o);
                        offset += 4; //4;
                    }
                    else if (type == typeof(long))
                    {
                        offset = AlignUp(offset, 8);
                        _cuda.SetParameter(function, offset, (long)o);//(uint)((int)o));// (uint)(long)
                        offset += 8; //;
                    }
                    else if (type.IsValueType && type != typeof(char))
                    {
                        int size = MSizeOf(o.GetType());
                        offset = AlignUp(offset, size);
                        _cuda.SetParameter(function, offset, o);
                        offset += size;
                    }
                    else if(type == typeof(char))
                    {
                        offset = AlignUp(offset, 2);
                        byte[] ba = Encoding.Unicode.GetBytes(new char[] { (char)o });
                        ushort shrt = BitConverter.ToUInt16(ba, 0);
                        _cuda.SetParameter(function, offset, shrt);
                        offset += 2;
                    }
                    else
                    {
                        CUDevicePtrEx ptrEx = (CUDevicePtrEx)GetDeviceMemory(o);
                        CUdeviceptr ptr = ptrEx.DevPtr;

                        offset = AlignUp(offset, ptrSize);
                        if (ptrSize == 8)
                            _cuda.SetParameter(function, offset, (ulong)ptr.Pointer);
                        else
                            _cuda.SetParameter(function, offset, (uint)ptr.Pointer);
                        offset += ptrSize;
                        // Dummy methods do not support smart arrays
                        if (!gpuMethodInfo.IsDummy)
                        {
                            if (ptrEx.Dimensions > 0)
                            {
                                offset = AlignUp(offset, 4);
                                _cuda.SetParameter(function, offset, (uint)ptrEx.XSize);
                                offset += 4;//4;
                            }
                            if (ptrEx.Dimensions > 1)
                            {
                                offset = AlignUp(offset, 4);
                                _cuda.SetParameter(function, offset, (uint)ptrEx.YSize);
                                offset += 4;//4;
                            }
                            if (ptrEx.Dimensions > 2)
                            {
                                offset = AlignUp(offset, 4);
                                _cuda.SetParameter(function, offset, (uint)ptrEx.ZSize);
                                offset += 4;//4;
                            }
                        }
                    }
                }
                catch (CUDAException ex)
                {
                    HandleCUDAException(ex);
                }

            }
            try
            {
                _cuda.SetParameterSize(function, (uint)(offset));
                if(streamId < 0)
                    _cuda.Launch(function, gridSize.x, gridSize.y);
                else
                {
                    CUstream cuStr = new CUstream();
                    cuStr.Pointer = IntPtr.Zero;
                    if (streamId > 0 && !_streams.ContainsKey(streamId))
                    {
                        cuStr = _cuda.CreateStream();
                        _streams.Add(streamId, cuStr);
                    }
                    else if(streamId > 0)
                        cuStr = (CUstream)_streams[streamId];
                    _cuda.LaunchAsync(function, gridSize.x, gridSize.y, cuStr);
                }
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
            if (IsSmartCopyEnabled)
                Unlock();
        }

        #endregion

        #region Constant Memory
#pragma warning disable 1591

        //protected override void DoCopyToDevice<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count)
        //{
        //    CUdeviceptr devPtr = ((CUDevicePtrEx)GetDeviceMemory(devArray)).DevPtr;
        //    Type type = typeof(T);
        //    unsafe
        //    {
        //        GCHandle handle = GCHandle.Alloc(hostArray, GCHandleType.Pinned);
        //        try
        //        {
        //            int size = MSizeOf(typeof(T));
        //            IntPtr hostPtr = handle.AddrOfPinnedObject();
        //            IntPtr hostOffsetPtr = new IntPtr(hostPtr.ToInt64() + hostOffset * size);
        //            CUdeviceptr devOffsetPtr = devPtr + (long)(devOffset * size);
        //            _cuda.CopyHostToDevice(devOffsetPtr, hostOffsetPtr, (uint)(count * size));
        //        }
        //        catch (CUDAException ex)
        //        {
        //            HandleCUDAException(ex);
        //        }
        //        finally
        //        {
        //            handle.Free();
        //        }
        //    }
        //}

        protected override void DoCopyToConstantMemoryAsync<T>(IntPtr hostArray, int hostOffset, Array devArray, int devOffset, int count, KernelConstantInfo ci, int streamId)
        {
            Type type = typeof(T);
            CUdeviceptr devPtr = (CUdeviceptr)ci.CudaPointer;
            int size = MSizeOf(typeof(T));
            if (count <= 0)
                throw new ArgumentOutOfRangeException("count");
            unsafe
            {
                GCHandle handle = GCHandle.Alloc(hostArray, GCHandleType.Pinned);
                try
                {
                    IntPtr hostPtr = hostArray;
                    IntPtr hostOffsetPtr = new IntPtr(hostPtr.ToInt64() + hostOffset * size);
                    CUdeviceptr devOffsetPtr = devPtr + (long)(devOffset * size);
                    CUstream stream = (CUstream)GetStream(streamId);
                    _cuda.CopyHostToDeviceAsync(devOffsetPtr, hostOffsetPtr, (uint)(count * size), stream);
                }
                catch (CUDAException ex)
                {
                    HandleCUDAException(ex);
                }
                finally
                {
                    handle.Free();
                }
            }
        }

        protected override void DoCopyToConstantMemory<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, KernelConstantInfo ci)
        {
            Type type = typeof(T);
            CUdeviceptr devPtr = (CUdeviceptr)ci.CudaPointer;
            int size = MSizeOf(typeof(T));
            if (count == 0)
                count = hostArray.Length;
            unsafe
            {
                GCHandle handle = GCHandle.Alloc(hostArray, GCHandleType.Pinned);
                try
                {
                    IntPtr hostPtr = handle.AddrOfPinnedObject();
                    IntPtr hostOffsetPtr = new IntPtr(hostPtr.ToInt64() + hostOffset * size);
                    CUdeviceptr devOffsetPtr = devPtr + (long)(devOffset * size);
                    _cuda.CopyHostToDevice(devOffsetPtr, hostOffsetPtr, (uint)(count * size));
                }
                catch (CUDAException ex)
                {
                    HandleCUDAException(ex);
                }
                finally
                {
                    handle.Free();
                }
            }
        }

        #endregion

        protected override Array DoCast<T,U>(int offset, Array devArray, int n)
        {
            U[] devMem = new U[0];
            CUDevicePtrEx devPtrEx = (CUDevicePtrEx)GetDeviceMemory(devArray);
            CUDevicePtrEx newDevPtrEx = devPtrEx.Cast<U>(devPtrEx, offset, n);
            AddToDeviceMemory(devMem, newDevPtrEx);
            return devMem;
        }

        protected override Array DoCast<T,U>(int offset, Array devArray, int x, int y)
        {
            U[,] devMem = new U[0, 0];
            CUDevicePtrEx devPtrEx = (CUDevicePtrEx)GetDeviceMemory(devArray);
            CUDevicePtrEx newDevPtrEx = devPtrEx.Cast<U>(devPtrEx, offset, x, y);
            AddToDeviceMemory(devMem, newDevPtrEx);
            return devMem;
        }

        protected override Array DoCast<T,U>(int offset, Array devArray, int x, int y, int z)
        {
            U[, ,] devMem = new U[0, 0, 0];
            CUDevicePtrEx devPtrEx = (CUDevicePtrEx)GetDeviceMemory(devArray);
            CUDevicePtrEx newDevPtrEx = devPtrEx.Cast<U>(devPtrEx, offset, x, y, z);
            AddToDeviceMemory(devMem, newDevPtrEx);
            return devMem;
        }

        //public override void HostRegister<T>(T[] hostArray)
        //{
        //    GCHandle handle = GCHandle.Alloc(hostArray, GCHandleType.Pinned);
        //    int size = MSizeOf(typeof(T));
        //    _cuda.HostRegister(handle.AddrOfPinnedObject(), size * hostArray.Length, 0);
        //}

        //public override void Unregister<T>(T[] hostArray)
        //{
        //    GCHandle handle = GCHandle.Alloc(hostArray, GCHandleType.Pinned);
        //    _cuda.HostUnregister(handle.AddrOfPinnedObject());
        //}

        protected override void DoCopyToDevice<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count)
        {
            DoCopyToDeviceAsync<T>(hostArray, hostOffset, devArray, devOffset, count, 0);
        }

        protected override void DoCopyToDeviceAsync<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, int streamId)
        {
            CUdeviceptr devPtr = ((CUDevicePtrEx)GetDeviceMemory(devArray)).DevPtr;
            Type type = typeof(T);
            unsafe
            {
                GCHandle handle = GCHandle.Alloc(hostArray, GCHandleType.Pinned);
                try
                {                 
                    int size = MSizeOf(typeof(T));
                    IntPtr hostPtr = handle.AddrOfPinnedObject();
                    IntPtr hostOffsetPtr = new IntPtr(hostPtr.ToInt64() + hostOffset * size);
                    CUdeviceptr devOffsetPtr = devPtr + (long)(devOffset * size);
                    if(streamId > 0)
                    {
                        CUstream stream = (CUstream)GetStream(streamId);
                        _cuda.CopyHostToDeviceAsync(devOffsetPtr, hostOffsetPtr, (uint)(count * size), stream);
                    }
                    else
                        _cuda.CopyHostToDevice(devOffsetPtr, hostOffsetPtr, (uint)(count * size));
                }
                catch (CUDAException ex)
                {
                    HandleCUDAException(ex);
                }
                finally
                {
                    handle.Free();
                }
            }
        }

        protected override void DoCopyFromDevice<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count)
        {
            DoCopyFromDeviceAsync<T>(devArray, devOffset, hostArray, hostOffset, count, -1);
        }

        protected override void DoCopyFromDeviceAsync<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count, int streamId)
        {
            CUdeviceptr devPtr = ((CUDevicePtrEx)GetDeviceMemory(devArray)).DevPtr;
            Type type = typeof(T);
            unsafe
            {
                GCHandle handle = GCHandle.Alloc(hostArray, GCHandleType.Pinned);
                try
                {
                    int size = MSizeOf(typeof(T)); //MSizeOf(typeof(T));
                    IntPtr hostPtr = handle.AddrOfPinnedObject();
                    IntPtr hostOffsetPtr = new IntPtr(hostPtr.ToInt64() + hostOffset * size);
                    CUdeviceptr devOffsetPtr = devPtr + (long)(devOffset * size);
                    if (streamId > 0)
                    {
                        CUstream stream = (CUstream)GetStream(streamId);
                        _cuda.CopyDeviceToHostAsync(devOffsetPtr, hostOffsetPtr, (uint)(count * size), stream);
                    }
                    else
                        _cuda.CopyDeviceToHost(devOffsetPtr, hostOffsetPtr, (uint)(count * size));
                }
                catch (CUDAException ex)
                {
                    HandleCUDAException(ex);
                }
                finally
                {
                    handle.Free();
                }
            }
        }

        protected override void DoCopyFromDevice<T>(Array devArray, Array hostArray)
        {
            CUDevicePtrEx ptrEx = ((CUDevicePtrEx)GetDeviceMemory(devArray));
            DoCopyFromDevice<T>(devArray, 0, hostArray, 0, ptrEx.TotalSize);
        }



        protected override void DoCopyToDeviceAsync<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, int streamId, IntPtr stagingPost, bool isConstantMemory = false)
        {
            Lock();
            CUstream stream = (CUstream)GetStream(streamId);
            Unlock();
            var streamDesc = new StreamDesc() { Stream = stream, StreamId = streamId };

            lock (_smartCopyLock)
            {
                while (_streamsPending.Where(sp => sp.StreamId == streamId).Count() > 0)
                    Monitor.Wait(_smartCopyLock);
                _streamsPending.Add(streamDesc);
                Monitor.Pulse(_smartCopyLock);
            }

            DoCopyOnHostM2NDelegate<T> copyM2N = new DoCopyOnHostM2NDelegate<T>(DoCopyOnHost<T>);
            var cdp = new CopyDeviceParams<T>() { count = count, devArray = devArray, devOffset = devOffset,
                                                  stagingPost = stagingPost,
                                                  streamId = streamDesc,
                                                  copyToDevice = true,
                                                  hostArray = hostArray,
                                                  hostOffset = hostOffset,
                                                  IsConstantMemory = isConstantMemory
            };
            AsyncCallback callback = OnCopyOnHostCompleted<T>;
            IAsyncResult res = copyM2N.BeginInvoke(hostArray, hostOffset, stagingPost, 0, count, callback,
                new DelegateStateM2N<T>() { Params = cdp, Dlgt = copyM2N });
        }

        protected override void DoCopyFromDeviceAsync<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count, int streamId, IntPtr stagingPost)
        {
            Lock();
            CUstream stream = (CUstream)GetStream(streamId);
            Unlock();
            var streamDesc = new StreamDesc() { Stream = stream, StreamId = streamId };

            lock (_smartCopyLock)
            {               
                while (_streamsPending.Where(sp => sp.StreamId == streamId).Count() > 0)
                    Monitor.Wait(_smartCopyLock);
                _streamsPending.Add(streamDesc);
                Monitor.Pulse(_smartCopyLock);
            }
            DoCopyFromDeviceAsyncDelegate<T> copyFrom = new DoCopyFromDeviceAsyncDelegate<T>(DoCopyFromDeviceAsyncEx<T>);
            AsyncCallback callback = OnCopyFromDeviceAsyncCompleted<T>;

            var cdp = new CopyDeviceParams<T>()
            {
                count = count,
                devArray = devArray,
                devOffset = devOffset,
                stagingPost = stagingPost,
                streamId = streamDesc,
                copyToDevice = false,
                hostArray = hostArray,
                hostOffset = hostOffset
            };
            IAsyncResult res = copyFrom.BeginInvoke(devArray, devOffset, stagingPost, 0, count, streamId, callback,
                new DelegateStateCFDA<T>() { Params = cdp, Dlgt = copyFrom });
        }

        /// <summary>
        /// Synchronizes the stream.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        public override void SynchronizeStream(int streamId = 0)
        {
            if (IsSmartCopyEnabled)
            {
                Lock();
                CUstream stream = (CUstream)GetStream(streamId);
                Unlock();
                var streamDesc = new StreamDesc() { Stream = stream, StreamId = streamId };
                lock (_smartCopyLock)
                {

                    while (_streamsPending.Where(sp => sp.StreamId == streamId).Count() > 0)
                        Monitor.Wait(_smartCopyLock);
                    Monitor.Pulse(_smartCopyLock);
                }
            }

            CUstream cuStr = new CUstream();
            cuStr.Pointer = IntPtr.Zero;
            if (streamId > 0)
            {
                if(!_streams.ContainsKey(streamId))
                    throw new CudafyHostException(CudafyHostException.csSTREAM_X_NOT_SET, streamId);
                cuStr = (CUstream)_streams[streamId];
            }
            else
                cuStr.Pointer = IntPtr.Zero;
            try
            {
                if (IsSmartCopyEnabled)
                {
                    Lock();
                    _cuda.SynchronizeStream(cuStr);
                    Unlock();
                }
                else
                    _cuda.SynchronizeStream(cuStr);
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
        }


        private struct StreamDesc
        {
            public int StreamId;
            public CUstream Stream;
        }

        private object _smartCopyLock = new object();

        private List<StreamDesc> _streamsPending = new List<StreamDesc>();

        private void DoCopyFromDeviceAsyncEx<T>(Array devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId) where T : struct
        {
            if(IsSmartCopyEnabled)
                Lock();
            DoCopyFromDeviceAsync<T>(devArray, devOffset, hostArray, hostOffset, count, streamId);
            if (IsSmartCopyEnabled)
                Unlock();
        }

        private void OnCopyOnHostCompleted<T>(IAsyncResult result) where T : struct
        {
            DelegateStateM2N<T> dlgtStaten = (DelegateStateM2N<T>)result.AsyncState;
            CopyDeviceParams<T> cdp = dlgtStaten.Params;
            DoCopyOnHostM2NDelegate<T> dlgt = dlgtStaten.Dlgt;
            dlgt.EndInvoke(result);
            if (IsSmartCopyEnabled)
                Lock();
            if (cdp.IsConstantMemory)
            {
                KernelConstantInfo ci = InitializeCopyToConstantMemory(null, cdp.hostOffset, cdp.devArray, cdp.devOffset, ref cdp.count);
                DoCopyToConstantMemoryAsync<T>(cdp.stagingPost, 0, cdp.devArray, cdp.devOffset, cdp.count, ci, cdp.streamId.StreamId);
            }
            else
                DoCopyToDeviceAsync<T>(cdp.stagingPost, 0, cdp.devArray, cdp.devOffset, cdp.count, cdp.streamId.StreamId);
            if (IsSmartCopyEnabled)
                Unlock();
            lock (_smartCopyLock)
            {
                bool removed = _streamsPending.Remove(cdp.streamId);
                Debug.Assert(removed);
                Monitor.Pulse(_smartCopyLock);
            }
            
        }

        private void OnCopyFromDeviceAsyncCompleted<T>(IAsyncResult result)
        {
            DelegateStateCFDA<T> dlgtState = (DelegateStateCFDA<T>)result.AsyncState;
            CopyDeviceParams<T> cdp = dlgtState.Params;
            DoCopyFromDeviceAsyncDelegate<T> dlgt = dlgtState.Dlgt;
            dlgt.EndInvoke(result);

            if (IsSmartCopyEnabled)
                Lock();
            _cuda.SynchronizeStream(cdp.streamId.Stream);
            if (IsSmartCopyEnabled)
                Unlock();
            lock(_smartCopyLock)
            {
                DoCopyOnHost<T>(cdp.stagingPost, 0, cdp.hostArray, cdp.hostOffset, cdp.count);
                bool removed = _streamsPending.Remove(cdp.streamId);
                Debug.Assert(removed);
                Monitor.Pulse(_smartCopyLock);
            }
        }

        private struct DelegateStateM2N<T>
        {
            public CopyDeviceParams<T> Params;
            public DoCopyOnHostM2NDelegate<T> Dlgt;
        }

        private struct DelegateStateCFDA<T>
        {
            public CopyDeviceParams<T> Params;
            public DoCopyFromDeviceAsyncDelegate<T> Dlgt;
        }

        private struct CopyDeviceParams<T>
        {
            public Array hostArray;
            public int hostOffset;
            public Array devArray;
            public int devOffset;
            public int count;
            public StreamDesc streamId;
            public bool copyToDevice;
            public IntPtr stagingPost;
            public bool IsConstantMemory;
        }

        private delegate void DoCopyFromDeviceAsyncDelegate<T>(Array devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId);

        private delegate void DoCopyOnHostM2NDelegate<T>(Array srcArray, int srcOffset, IntPtr dstArray, int dstOffset, int count);

        private delegate void DoCopyOnHostN2MDelegate<T>(IntPtr srcArray, int srcOffset, Array dstArray, int dstOffset, int count);

        protected override void DoCopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, Array devArray, int devOffset, int count, int streamId)
        {
            DevicePtrEx ptrEx = (CUDevicePtrEx)GetDeviceMemory(devArray);
            DoCopyToDeviceAsync<T>(hostArray, hostOffset, ptrEx, devOffset, count, streamId);
        }

        protected override void DoCopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, DevicePtrEx ptrEx, int devOffset, int count, int streamId)
        {
            if (count <= 0)
                throw new ArgumentOutOfRangeException("count");
            CUdeviceptr ptr = (ptrEx as CUDevicePtrEx).DevPtr;
            Type type = typeof(T);
            CUstream cuStr = GetCUstream(streamId);
            unsafe
            {
                int size = MSizeOf(type);
                IntPtr hostPtr = hostArray;
                IntPtr hostOffsetPtr = new IntPtr(hostPtr.ToInt64() + hostOffset * size);
                CUdeviceptr devOffsetPtr = ptr + (long)(devOffset * size);
                try
                {
                    if (streamId >= 0)
                    {
                        _cuda.CopyHostToDeviceAsync(devOffsetPtr, hostOffsetPtr, (uint)(count * size), cuStr);
                    }
                    else
                    {
                        _cuda.CopyHostToDevice(devOffsetPtr, hostOffsetPtr, (uint)(count * size));
                    }
                }
                catch (CUDAException ex)
                {
                    HandleCUDAException(ex);
                }
            }
        }

        private CUstream GetCUstream(int streamId)
        {
            CUstream cuStr = new CUstream();
            cuStr.Pointer = IntPtr.Zero;
            if (streamId > 0 && !_streams.ContainsKey(streamId))
            {
                try
                {
                    cuStr = _cuda.CreateStream();
                }
                catch (CUDAException ex)
                {
                    HandleCUDAException(ex);
                }
                _streams.Add(streamId, cuStr);
            }
            else if (streamId > 0)
            {
                cuStr = (CUstream)_streams[streamId];
            }
            return cuStr;
        }

        protected override void DoCopyFromDeviceAsync<T>(Array devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId)
        {
            DevicePtrEx ptrEx = (CUDevicePtrEx)GetDeviceMemory(devArray);
            DoCopyFromDeviceAsync<T>(ptrEx, devOffset, hostArray, hostOffset, count, streamId);
        }

        protected override void DoCopyFromDeviceAsync<T>(DevicePtrEx devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId)
        {

            CUdeviceptr ptr = (devArray as CUDevicePtrEx).DevPtr;
            Type type = typeof(T);
            CUstream cuStr = new CUstream();
            cuStr.Pointer = IntPtr.Zero;
            if (streamId > 0 && !_streams.ContainsKey(streamId))
            {
                try
                {
                    cuStr = _cuda.CreateStream();
                }
                catch (CUDAException ex)
                {
                    HandleCUDAException(ex);
                }
                _streams.Add(streamId, cuStr);
            }
            else if(streamId > 0)
            {
                cuStr = (CUstream)_streams[streamId];
            }
            unsafe
            {
                int size = MSizeOf(type);
                IntPtr hostPtr = hostArray;
                IntPtr hostOffsetPtr = new IntPtr(hostPtr.ToInt64() + hostOffset * size);
                CUdeviceptr devOffsetPtr = ptr + (long)(devOffset * size);
                try
                {
                    if (streamId >= 0)
                    {
                        _cuda.CopyDeviceToHostAsync(devOffsetPtr, hostOffsetPtr, (uint)(count * size), cuStr);
                    }
                    else
                    {
                        _cuda.CopyDeviceToHost(devOffsetPtr, hostOffsetPtr, (uint)(count * size));
                    }
                }
                catch (CUDAException ex)
                {
                    HandleCUDAException(ex);
                }
            }
        }
#pragma warning restore 1591


        private new Dictionary<IntPtr, CUcontext> _hostHandles;

        /// <summary>
        /// Destroys the stream.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        public override void DestroyStream(int streamId)
        {
            if (!_streams.ContainsKey(streamId))
            {
                Debug.WriteLine(string.Format("Warning: DestroyStream(int streamId) streamId {0} does not exist"));
                return;
            }
                //throw new CudafyHostException(CudafyHostException.csSTREAM_X_NOT_SET, streamId);
            CUstream cuStr = (CUstream)_streams[streamId];
            try
            {
                _cuda.DestroyStream(cuStr);
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
            _streams.Remove(streamId);
        }

        /// <summary>
        /// Destroys all streams.
        /// </summary>
        public override void DestroyStreams()
        {
            try
            {
                foreach (var cuStr in _streams.Values)
                    _cuda.DestroyStream((CUstream)cuStr);
            }
            catch (CUDAException ex)
            {
                Debug.WriteLine(ex.Message);
            }
            _streams.Clear();
        }

        /// <summary>
        /// Performs a default host memory allocation.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">The x size.</param>
        /// <returns>
        /// Pointer to allocated memory.
        /// </returns>
        public override IntPtr HostAllocate<T>(int x)
        {
            int bytes = MSizeOf(typeof(T)) * x;
            IntPtr ptr = IntPtr.Zero;
            try
            {
                ptr = _cuda.HostAllocate((uint)bytes, CUDADriver.CU_MEMHOSTALLOC_PORTABLE);
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
            _hostHandles.Add(ptr, _cuda.CurrentContext);
            return ptr;
        }

        /// <summary>
        /// Frees memory allocated by HostAllocate.
        /// </summary>
        /// <param name="ptr">The pointer.</param>
        /// <exception cref="CudafyHostException">Pointer not found.</exception>
        public override void HostFree(IntPtr ptr)
        {
            if (!_hostHandles.Keys.Contains(ptr))
                throw new CudafyHostException(CudafyHostException.csPOINTER_NOT_FOUND);
            try
            {
                _cuda.PushCurrentContext(_hostHandles[ptr]);
                _cuda.FreeHost(ptr);
                _cuda.PopCurrentContext();
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
            _hostHandles.Remove(ptr);
        }

        /// <summary>
        /// Frees all memory allocated by HostAllocate. Disables smart copy.
        /// </summary>
        public override void HostFreeAll()
        {
            if (_hostHandles != null)
            {
                lock (_lock)
                {
                    foreach (var v in _hostHandles)
                    {
                        try
                        {
                            _cuda.PushCurrentContext((CUcontext)v.Value);
                            _cuda.FreeHost(v.Key);
                            _cuda.PopCurrentContext();
                        }
                        catch (CUDAException ex)
                        {
                            HandleCUDAException(ex);
                        }
                    }
                    _hostHandles.Clear();
                    if (IsSmartCopyEnabled)
                        DisableSmartCopy();
                }
            }
        }

        /// <summary>
        /// Allocates array on device, copies to device and returns pointer.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>The device array.</returns>
        public override T[] CopyToDevice<T>(T[] hostArray)
        {
            T[] devArray = Allocate<T>(hostArray.Length);
            DoCopyToDevice<T>(hostArray, 0, devArray, 0, hostArray.Length);
            return devArray;
        }

        /// <summary>
        /// Allocates array on device, copies to device and returns pointer.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>The device array.</returns>
        public override T[,] CopyToDevice<T>(T[,] hostArray)
        {
            T[,] devArray = Allocate<T>(hostArray.GetLength(0), hostArray.GetLength(1));
            DoCopyToDevice<T>(hostArray, 0, devArray, 0, hostArray.Length);
            return devArray;
        }

        /// <summary>
        /// Allocates array on device, copies to device and returns pointer.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray"></param>
        /// <returns>The device array.</returns>
        public override T[,,] CopyToDevice<T>(T[,,] hostArray)
        {
            T[,,] devArray = Allocate<T>(hostArray.GetLength(0), hostArray.GetLength(1), hostArray.GetLength(2));
            DoCopyToDevice<T>(hostArray, 0, devArray, 0, hostArray.Length);
            return devArray;
        }

        ///// <summary>
        ///// Copies from device.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="devArray">The dev array.</param>
        ///// <param name="hostData">The host data.</param>
        //public override void CopyFromDevice<T>(T devArray, out T hostData)
        //{
        //    T[] hostArray = new T[1];
        //    CUDevicePtrEx ptrEx = (CUDevicePtrEx)GetDeviceMemory(devArray);
        //    CUdeviceptr ptr = ptrEx.DevPtr;
        //    try
        //    {
        //        _cuda.CopyDeviceToHost(ptr, hostArray);
        //    }
        //    catch (CUDAException ex)
        //    {
        //        HandleCUDAException(ex);
        //    }
        //    hostData = hostArray[0];
        //}

        /// <summary>
        /// Copies between preallocated arrays on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The source device array.</param>
        /// <param name="dstDevArray">The destination device array.</param>
        public override void CopyOnDevice<T>(T[] srcDevArray, T[] dstDevArray)
        {
            CUDevicePtrEx ptrSrcEx = (CUDevicePtrEx)GetDeviceMemory(srcDevArray);
            CUDevicePtrEx ptrDstEx = (CUDevicePtrEx)GetDeviceMemory(dstDevArray);

            uint bytes = (uint)ptrSrcEx.TotalSize * (uint)(MSizeOf(typeof(T)));
            try
            {
                _cuda.CopyDeviceToDevice(ptrSrcEx.DevPtr, ptrDstEx.DevPtr, bytes);
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
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
        //public override void CopyOnDevice<T>(T[] srcDevArray, int srcOffset, T[] dstDevArray, int dstOffet, int count)
        //{
        //    DoCopyOnDevice<T>(
        //}

        protected override void DoCopyOnDevice<T>(DevicePtrEx srcDevArray, int srcOffset, DevicePtrEx dstDevArray, int dstOffet, int count)
        {
            CUDevicePtrEx ptrSrcEx = ((CUDevicePtrEx)srcDevArray);
            CUDevicePtrEx ptrDstEx = ((CUDevicePtrEx)dstDevArray);
            int size = MSizeOf(typeof(T));
            CUdeviceptr ptrSrcOffset = ptrSrcEx.DevPtr + (long)(srcOffset * size);
            CUdeviceptr ptrDstOffset = ptrDstEx.DevPtr + (long)(dstOffet * size);

            uint bytes = (uint)count * (uint)size;
            try
            {
                _cuda.CopyDeviceToDevice(ptrSrcOffset, ptrDstOffset, bytes);
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
        }

        protected override void DoCopyDeviceToDevice<T>(Array srcDevArray, int srcOffset, GPGPU peer, Array dstDevArray, int dstOffet, int count)
        {          
            CUDevicePtrEx ptrSrcEx = (CUDevicePtrEx)GetDeviceMemory(srcDevArray);
            CUDevicePtrEx ptrDstEx = (CUDevicePtrEx)peer.GetDeviceMemory(dstDevArray);
            int size = MSizeOf(typeof(T));
            CUdeviceptr ptrSrcOffset = ptrSrcEx.DevPtr + (long)(srcOffset * size);
            CUdeviceptr ptrDstOffset = ptrDstEx.DevPtr + (long)(dstOffet * size);

            uint bytes = (uint)count * (uint)size;
            try
            {
                _cuda.CopyPeerToPeer(ptrDstOffset, (peer as CudaGPU).GetDeviceContext(), ptrSrcOffset, _cuda.CurrentContext, count * size);
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
        }

        protected override void DoCopyDeviceToDeviceAsync<T>(Array srcDevArray, int srcOffset, GPGPU peer, Array dstDevArray, int dstOffet, int count, int streamId)
        {
            CUDevicePtrEx ptrSrcEx = (CUDevicePtrEx)GetDeviceMemory(srcDevArray);
            CUDevicePtrEx ptrDstEx = (CUDevicePtrEx)peer.GetDeviceMemory(dstDevArray);
            int size = MSizeOf(typeof(T));
            CUdeviceptr ptrSrcOffset = ptrSrcEx.DevPtr + (long)(srcOffset * size);
            CUdeviceptr ptrDstOffset = ptrDstEx.DevPtr + (long)(dstOffet * size);
            CUstream stream = (CUstream)GetStream(streamId);
            uint bytes = (uint)count * (uint)size;
            try
            {
                _cuda.CopyPeerToPeerAsync(ptrDstOffset, GetDeviceContext(), ptrSrcOffset, (peer as CudaGPU).GetDeviceContext(), count, stream);
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
        }


        protected override void DoCopyOnDevice<T>(Array srcDevArray, int srcOffset, Array dstDevArray, int dstOffet, int count)
        {
            DevicePtrEx ptrSrcEx = (CUDevicePtrEx)GetDeviceMemory(srcDevArray);
            DevicePtrEx ptrDstEx = (CUDevicePtrEx)GetDeviceMemory(dstDevArray);
            DoCopyOnDevice<T>(ptrSrcEx, srcOffset, ptrDstEx, dstOffet, count);
        }

        protected override void DoCopyOnDeviceAsync<T>(DevicePtrEx srcDevArray, int srcOffset, DevicePtrEx dstDevArray, int dstOffet, int count, int streamId)
        {
            CUDevicePtrEx ptrSrcEx = ((CUDevicePtrEx)srcDevArray);
            CUDevicePtrEx ptrDstEx = ((CUDevicePtrEx)dstDevArray);
            int size = MSizeOf(typeof(T));
            CUdeviceptr ptrSrcOffset = ptrSrcEx.DevPtr + (long)(srcOffset * size);
            CUdeviceptr ptrDstOffset = ptrDstEx.DevPtr + (long)(dstOffet * size);
            CUstream cuStr = GetCUstream(streamId);
            uint bytes = (uint)count * (uint)size;
            try
            {
                _cuda.CopyDeviceToDeviceAsync(ptrSrcOffset, ptrDstOffset, bytes, cuStr);
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
        }


        protected override void DoCopyOnDeviceAsync<T>(Array srcDevArray, int srcOffset, Array dstDevArray, int dstOffet, int count, int streamId)
        {
            DevicePtrEx ptrSrcEx = (CUDevicePtrEx)GetDeviceMemory(srcDevArray);
            DevicePtrEx ptrDstEx = (CUDevicePtrEx)GetDeviceMemory(dstDevArray);
            DoCopyOnDeviceAsync<T>(ptrSrcEx, srcOffset, ptrDstEx, dstOffet, count, streamId);
        }

        /// <summary>
        /// Allocates on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <returns>Device array of length 1.</returns>
        public override T[] Allocate<T>()
        {
            T[] devMem = new T[0];
            CUdeviceptr ptr = new CUdeviceptr();
            try
            {
                ptr = _cuda.Allocate((uint)MSizeOf(typeof(T)));//  (uint)(MSizeOf(typeof(T)) * 1));
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
            AddToDeviceMemory(devMem, new CUDevicePtrEx(ptr, _cuda.CurrentContext));
            return devMem;
        }

        /// <summary>
        /// Allocates array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">Length of 1D array.</param>
        /// <returns>Device array of length x.</returns>
        public override T[] Allocate<T>(int x)
        {
            T[] devMem = new T[0];
            CUdeviceptr ptr = new CUdeviceptr();
            try
            {
                ptr = _cuda.Allocate((uint)(MSizeOf(typeof(T)) * x));
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
            AddToDeviceMemory(devMem, new CUDevicePtrEx(ptr, x, _cuda.CurrentContext));
            return devMem;
        }

        /// <summary>
        /// Allocates array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">The x dimension.</param>
        /// <param name="y">The y dimension.</param>
        /// <returns>2D device array.</returns>
        public override T[,] Allocate<T>(int x, int y)
        {
            T[,] devMem = new T[0,0];
            CUdeviceptr ptr = new CUdeviceptr();
            try
            {
                ptr = _cuda.Allocate((uint)(MSizeOf(typeof(T)) * x * y));
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
            AddToDeviceMemory(devMem, new CUDevicePtrEx(ptr, x, y, _cuda.CurrentContext));
            return devMem;
        }

        /// <summary>
        /// Allocates array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">The x dimension.</param>
        /// <param name="y">The y dimension.</param>
        /// <param name="z">The z dimension.</param>
        /// <returns>3D device array.</returns>
        public override T[,,] Allocate<T>(int x, int y, int z)
        {
            T[,,] devMem = new T[0, 0, 0];
            CUdeviceptr ptr = new CUdeviceptr();
            try
            {
                ptr = _cuda.Allocate((uint)(MSizeOf(typeof(T)) * x * y * z));
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
            AddToDeviceMemory(devMem, new CUDevicePtrEx(ptr, x, y, z, _cuda.CurrentContext));
            return devMem;
        }

        /// <summary>
        /// Allocates array on device of same size as supplied host array.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>1D device array.</returns>
        public override T[] Allocate<T>(T[] hostArray)
        {
            T[] devMem = new T[0];
            CUdeviceptr ptr = new CUdeviceptr();
            try
            {
                ptr = _cuda.Allocate(hostArray);
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
            AddToDeviceMemory(devMem, new CUDevicePtrEx(ptr, hostArray.Length, _cuda.CurrentContext));
            return devMem;
        }
#pragma warning disable 1591
        public override T[,] Allocate<T>(T[,] hostArray)
        {
            T[,] devMem = new T[0,0];
            CUdeviceptr ptr = new CUdeviceptr();
            try
            {
                ptr = _cuda.Allocate((uint)(hostArray.Length * MSizeOf(typeof(T))));
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
            AddToDeviceMemory(devMem, new CUDevicePtrEx(ptr, hostArray.GetLength(0), hostArray.GetLength(1), _cuda.CurrentContext));
            return devMem;
        }

        public override T[,,] Allocate<T>(T[,,] hostArray)
        {
            T[,,] devMem = new T[0, 0, 0];
            CUdeviceptr ptr = new CUdeviceptr();
            try
            {
                ptr = _cuda.Allocate((uint)(hostArray.Length * MSizeOf(typeof(T))));
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
            AddToDeviceMemory(devMem, new CUDevicePtrEx(ptr, hostArray.GetLength(0), hostArray.GetLength(1), hostArray.GetLength(2), _cuda.CurrentContext));
            return devMem;
        }

        protected override void DoSet<T>(Array devArray, int offset = 0, int count = 0)
        {
            CUDevicePtrEx ptrEx = (CUDevicePtrEx)GetDeviceMemory(devArray); 
            int size = MSizeOf(typeof(T));
            CUdeviceptr ptr = ptrEx.DevPtr + (long)(size * offset);
            if (count <= 0)
                count = ptrEx.TotalSize;
            try
            {
                _cuda.Memset(ptr, (byte)0, (uint)(count * size));
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }
        }

        /// <summary>
        /// Frees the specified data array on device.
        /// </summary>
        /// <param name="devArray">The device array to free.</param>
        public override void Free(object devArray)
        {
            CUDevicePtrEx ptrEx = (CUDevicePtrEx)GetDeviceMemory(devArray);
            try
            {
                if (!ptrEx.CreatedFromCast && DeviceMemoryValueExists(ptrEx))
                {
                    var curCtx = _cuda.GetCurrentContextV1();
                    if (ptrEx.Context.Value.Pointer != curCtx.Pointer)
                    {
                        _cuda.SetCurrentContext(ptrEx.Context.Value);
                        _cuda.Free(ptrEx.DevPtr);
                        _cuda.SetCurrentContext(curCtx);
                    }
                    else
                        _cuda.Free(ptrEx.DevPtr);
                    ptrEx.Disposed = true;
                }
                else
                { 
                    Debug.WriteLine(string.Format("ptrEx.CreatedFromCast={0} && _cuda{1} && ptrEx.Disposed={2}",
                        ptrEx.CreatedFromCast, _cuda == null ? "==null" : "!=null", ptrEx.Disposed));
                }
                RemoveFromDeviceMemoryEx(ptrEx);
                ptrEx.RemoveChildren();
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }            
        }

        /// <summary>
        /// Frees all data arrays on device.
        /// </summary>
        public override void FreeAll()
        {
            lock (_lock)
            {
                foreach (CUDevicePtrEx ptrEx in GetDeviceMemories())
                {
                    try
                    {
                        if (!ptrEx.CreatedFromCast && _cuda != null && !ptrEx.Disposed)
                        {
                            var curCtx = _cuda.DeviceContext;//_cuda.GetCurrentContext();
                            if (ptrEx.Context.Value.Pointer != curCtx.Pointer)
                            {
                                _cuda.SetCurrentContext(ptrEx.Context.Value);
                                _cuda.Free(ptrEx.DevPtr);
                                _cuda.SetCurrentContext(curCtx);
                            }
                            else if (ptrEx.Context.Value.Pointer != _cuda.GetCurrentContextV1().Pointer)
                            {
                                _cuda.SetCurrentContext(ptrEx.Context.Value);
                                _cuda.Free(ptrEx.DevPtr);
                                _cuda.SetCurrentContext(curCtx);
                            }
                            else
                            {
                                _cuda.Free(ptrEx.DevPtr);
                            }
                            ptrEx.Disposed = true;
                        }
                        else
                            Debug.WriteLine(string.Format("ptrEx.CreatedFromCast={0} && _cuda{1} && ptrEx.Disposed={2}",
                                ptrEx.CreatedFromCast, _cuda == null ? "==null" : "!=null", ptrEx.Disposed));
                    }
                    catch (CUDAException ex)
                    {                        
#if DEBUG
                        HandleCUDAException(ex);
#endif
                        Trace.WriteLine(ex.Message);
                    }
                }
                ClearDeviceMemory();
            }
        }

        public override void LoadModule(CudafyModule module, bool unload = true)
        {
            if (!IsCurrentContext)
                throw new CudafyHostException(CudafyHostException.csCONTEXT_IS_NOT_CURRENT);
            if (unload)
                UnloadModules();
            else
                CheckForDuplicateMembers(module);

            if(!module.HasPTX && !module.HasBinary)
                throw new CudafyHostException(CudafyHostException.csNO_X_PRESENT_IN_CUDAFY_MODULE, "PTX or binary");
            PTXModule ptxModule = module.PTX;
            BinaryModule binModule = module.Binary;
            if (ptxModule == null && binModule == null)
                throw new CudafyHostException(CudafyHostException.csNO_SUITABLE_X_PRESENT_IN_CUDAFY_MODULE, "PTX or binary");

            try
            {
                byte[] bytes = binModule == null ? Encoding.ASCII.GetBytes(ptxModule.PTX) : binModule.Binary;
                CUmodule cumod = _cuda.LoadModule(bytes);
                _module = module;
                _module.Tag = cumod;
            }
            catch (CUDAException ex)
            {
                HandleCUDAException(ex);
            }

            _modules.Add(module);

            // Load constants
            foreach (var kvp in module.Constants)
            {
                if (!kvp.Value.IsDummy)
                {
                    CUdeviceptr ptr = new CUdeviceptr();
                    try
                    {
                        ptr = _cuda.GetModuleGlobal(kvp.Key);
                    }
                    catch (CUDAException ex)
                    {
                        HandleCUDAException(ex);
                    }
                    module.Constants[kvp.Key].CudaPointer = ptr;
                }
            }
        }

        public override void UnloadModule(CudafyModule module)
        {
            if (!_modules.Remove(module))
                throw new CudafyHostException(CudafyHostException.csMODULE_NOT_FOUND);
            if (_module == module)
                _module = null;
            if(module.Tag != null)
                _cuda.UnloadModule((CUmodule)module.Tag);
            
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
                CudaGPU gpu = CudafyHost.GetDevice(eGPUType.Cuda) as CudaGPU;
                cnt = (gpu.CudaDotNet as CUDA).GetDeviceCount();              
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




        /// <summary>
        /// Gets the pointer to the native GPU data.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <returns>
        /// Pointer to the actual data. This can be cast to GASS.CUDA.Types.CUdeviceptr.
        /// </returns>
        public override object GetGPUData(object data)
        {
            return ((CUDevicePtrEx)base.GetGPUData(data)).DevPtr;
        }

        
        public override int GetDriverVersion()
        {
            return _cuda.GetDriverVersion();
        }



#pragma warning restore 1591



    }



    /// <summary>
    /// Internal use.
    /// </summary>
    public class CUDevicePtrEx : DevicePtrEx
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
        public CUDevicePtrEx(CUdeviceptr devPtr, CUcontext? context)
            : this(devPtr, 1, 1, 1, context)
        {
            Dimensions = 0;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="CUDevicePtrEx"/> class.
        /// </summary>
        /// <param name="devPtr">The dev PTR.</param>
        /// <param name="xSize">Size of the x.</param>
        /// <param name="context">The context.</param>     
        public CUDevicePtrEx(CUdeviceptr devPtr, int xSize, CUcontext? context)
            : this(devPtr, xSize, 1, 1, context)
        {
            Dimensions = 1;
        }

        /// <summary>
        /// Casts the specified pointer.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ptrEx">The pointer.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="xSize">Size of the x.</param>
        /// <returns></returns>
        public CUDevicePtrEx Cast<T>(CUDevicePtrEx ptrEx, int offset, int xSize)
        {
            int size = GPGPU.MSizeOf(typeof(T));
            CUdeviceptr ptrOffset = ptrEx.DevPtr + (long)(offset * size);
            CUDevicePtrEx newPtrEx = new CUDevicePtrEx(ptrOffset, xSize, ptrEx.Context);
            newPtrEx.CreatedFromCast = true;
            ptrEx.AddChild(newPtrEx);
            return newPtrEx;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="CUDevicePtrEx"/> class.
        /// </summary>
        /// <param name="devPtr">The dev PTR.</param>
        /// <param name="xSize">Size of the x.</param>
        /// <param name="ySize">Size of the y.</param>
        /// <param name="context">The context.</param>       
        public CUDevicePtrEx(CUdeviceptr devPtr, int xSize, int ySize, CUcontext? context)
            : this(devPtr, xSize, ySize, 1, context)
        {
            Dimensions = 2;
        }

        /// <summary>
        /// Casts the specified pointer.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ptrEx">The pointer.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="xSize">Size of the x.</param>
        /// <param name="ySize">Size of the y.</param>
        /// <returns></returns>
        public CUDevicePtrEx Cast<T>(CUDevicePtrEx ptrEx, int offset, int xSize, int ySize)
        {
            int size = GPGPU.MSizeOf(typeof(T));
            CUdeviceptr ptrOffset = ptrEx.DevPtr + (long)(offset * size);
            CUDevicePtrEx newPtrEx = new CUDevicePtrEx(ptrOffset, xSize, ySize, ptrEx.Context);
            newPtrEx.CreatedFromCast = true;
            ptrEx.AddChild(newPtrEx);
            return newPtrEx;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="CUDevicePtrEx"/> class.
        /// </summary>
        /// <param name="devPtr">The dev PTR.</param>
        /// <param name="xSize">Size of the x.</param>
        /// <param name="ySize">Size of the y.</param>
        /// <param name="zSize">Size of the z.</param>
        /// <param name="context">The context.</param>       
        public CUDevicePtrEx(CUdeviceptr devPtr, int xSize, int ySize, int zSize, CUcontext? context)
        {
            CreatedFromCast = false;
            DevPtr = devPtr;
            XSize = xSize;
            YSize = ySize;
            ZSize = zSize;
            Dimensions = 3;
            Context = context;
            //OriginalSize = originalSize < 0 ? TotalSize : originalSize;
        }

        /// <summary>
        /// Casts the specified pointer.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ptrEx">The pointer.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="xSize">Size of the x.</param>
        /// <param name="ySize">Size of the y.</param>
        /// <param name="zSize">Size of the z.</param>
        /// <returns></returns>
        public CUDevicePtrEx Cast<T>(CUDevicePtrEx ptrEx, int offset, int xSize, int ySize, int zSize)
        {
            int size = GPGPU.MSizeOf(typeof(T));
            CUdeviceptr ptrOffset = ptrEx.DevPtr + (long)(offset * size);
            CUDevicePtrEx newPtrEx = new CUDevicePtrEx(ptrOffset, xSize, ySize, zSize, ptrEx.Context);
            newPtrEx.CreatedFromCast = true;
            ptrEx.AddChild(newPtrEx);
            return newPtrEx;
        }


        /// <summary>
        /// Gets the dev PTR.
        /// </summary>
        public CUdeviceptr DevPtr { get; private set; }

        /// <summary>
        /// Gets the IntPtr in DevPtr.
        /// </summary>
        public override IntPtr Pointer
        {
            get { return DevPtr.Pointer; }
        }


        /// <summary>
        /// Gets the context.
        /// </summary>
        public CUcontext? Context { get; private set; }


    }
}
