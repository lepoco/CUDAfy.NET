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
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Diagnostics;


namespace Cudafy.Host
{
    /// <summary>
    /// Represents an emulated Cuda GPU.
    /// </summary>
    public sealed class EmulatedGPU : GPGPU
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EmulatedGPU"/> class.
        /// </summary>
        /// <param name="deviceId">The device id.</param>
        public EmulatedGPU(int deviceId = 0)
            : base(deviceId)
        {            
            _lock = new object();
            try
            {
                _availableBytesPerfctr = new PerformanceCounter("Memory", "Available Bytes");
            }
            catch (Exception ex)
            {
                _availableBytesPerfctr = null;
                Debug.WriteLine(ex.Message);
#if DEBUG
                throw;
#endif 
            }
        }

        ///// <summary>
        ///// Releases unmanaged resources and performs other cleanup operations before the
        ///// <see cref="EmulatedGPU"/> is reclaimed by garbage collection.
        ///// </summary>
        //~EmulatedGPU()
        //{
        //    lock (_lock)
        //    {
        //        FreeAll();
        //        HostFreeAll();
        //    }
        //}

        
#warning Make configurable at EmulatedGPU level
#pragma warning disable 1591
        /// <summary>
        /// Gets the device properties.
        /// </summary>
        /// <returns>Device properties instance.</returns>
        public override GPGPUProperties GetDeviceProperties(bool useAdvanced = true)
        {
            int i1K = 1024;
            int i64K = 64 * i1K;
            int i32K = 32 * i1K;

            //Computer comp = new Computer();
            GPGPUProperties props = new GPGPUProperties();
            props.UseAdvanced = useAdvanced;
            props.Capability = new Version(0, 1, 0, 0);
            props.Name = "Emulated GPGPU Kernel";
            props.DeviceId = DeviceId;

            ulong freeMem = FreeMemory;
            props.TotalMemory = freeMem;
            props.ClockRate = 0;
            props.IsSimulated = true;
            props.TotalConstantMemory = i64K;
            props.MaxGridSize = new dim3(i64K, i64K);
            props.MaxThreadsSize = new dim3(i1K, i1K);
            props.MaxThreadsPerBlock = i1K;
            props.MemoryPitch = 2147483647;
            props.RegistersPerBlock = i32K;
            props.SharedMemoryPerBlock = 49152;
            props.WarpSize = 32;
            props.TextureAlignment = 512;
            if (useAdvanced)
                props.MultiProcessorCount = Environment.ProcessorCount;
            else
                props.MultiProcessorCount = 0;
            return props;
        }

        private PerformanceCounter _availableBytesPerfctr;

        /// <summary>
        /// Gets the device count.
        /// </summary>
        /// <returns>Number of devices of this type.</returns>
        public new static int GetDeviceCount()
        {
            int cnt = 0;
            try
            {
                EmulatedGPU gpu = CudafyHost.GetDevice(eGPUType.Emulator) as EmulatedGPU;
                cnt = CudafyHost.GetDeviceCount(eGPUType.Emulator);
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
            }
            return cnt;
        }

        /// <summary>
        /// Synchronizes context.
        /// </summary>
        public override void Synchronize()
        {
        }



        /// <summary>
        /// Gets the free memory.
        /// </summary>
        /// <value>The free memory.</value>
        public override ulong FreeMemory
        {
            get 
            { 
                if(_availableBytesPerfctr == null)
                    _availableBytesPerfctr = new PerformanceCounter("Memory", "Available Bytes");
                return (ulong)_availableBytesPerfctr.NextValue(); 
            }
        }

        // http://stackoverflow.com/questions/105031/how-do-you-get-total-amount-of-ram-the-computer-has
        static ulong GetTotalMemoryInBytes()
        {
            return new Microsoft.VisualBasic.Devices.ComputerInfo().TotalPhysicalMemory;
        }

        /// <summary>
        /// Gets the total memory.
        /// </summary>
        /// <value>
        /// The total memory.
        /// </value>
        public override ulong TotalMemory
        {
            get { return GetTotalMemoryInBytes(); }
        }

        public override bool CanAccessPeer(GPGPU peer)
        {
            lock (_peerAccessLock)
            {
                return peer != this && peer is EmulatedGPU;
            }
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
            if (streamId > -1 && !_streams.ContainsKey(streamId))
                _streams.Add(streamId, streamId);

            MethodInfo mi = gpuMethodInfo.Method;
            if (mi == null)
                throw new CudafyHostException(CudafyHostException.csX_NOT_SET, gpuMethodInfo.Name);
            object instance = mi.IsStatic ? null : Activator.CreateInstance(mi.DeclaringType);
            if (gpuMethodInfo.IsDummy)
            {
                object[] argsCopy = new object[arguments.Length];
                for (int i = 0; i < arguments.Length; i++)
                {
                    if (arguments[i].GetType().IsArray)
                    {
                        var v = TryGetDeviceMemory(arguments[i]) as EmuDevicePtrEx;
                        if (v != null)
                        {
                            if (v.Offset == 0)
                            {
                                argsCopy[i] = v.DevPtr;
                            }
                            else
                            {
                                throw new CudafyHostException(CudafyHostException.csX_NOT_CURRENTLY_SUPPORTED, "Offsets in arrays passed to dummy functions");
                            }
                        }
                        else
                            argsCopy[i] = arguments[i];
                    }
                    else
                        argsCopy[i] = arguments[i];
                }
                mi.Invoke(instance, argsCopy);
                return;
            }

            GGrid grid = new GGrid(gridSize);
            Dictionary<Array, EmuDevicePtrEx> dic;
            object[] pList = BuildParameterList2(mi, arguments, out dic);
            //object[] pListCopy = new object[0];
            if (gridSize.z > 1)
                throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, "3D grid sizes");
            if (blockSize.z > 1)
                throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, "3D block sizes");
            for (int x = 0; x < gridSize.x; x++)
            {
                for (int y = 0; y < gridSize.y; y++)
                {
                    int totalSize = blockSize.x * blockSize.y * blockSize.z;
                    Thread[] threads = new Thread[totalSize];
                    GBlock blk2lnch = new GBlock(grid, blockSize, x, y);
                    int tCtr = 0;

                    int pListLen = pList.Length;
                    for (int tx = 0; tx < blockSize.x; tx++)
                    {
                        for (int ty = 0; ty < blockSize.y; ty++)
                        {
                            GThread ht = new GThread(tx, ty, blk2lnch);
                            object[] pListCopy = new object[pListLen];
                            for (int pc = 0; pc < pListLen; pc++)
                                if (pList[pc] is GThread)
                                    pListCopy[pc] = ht;
                                else
                                    pListCopy[pc] = pList[pc];

#warning OPTIMIZATION if there is no synchronize then start and join threads in multiple of processor count - check this in disassembly and put flag in gpuMethodInfo
                            threads[tCtr] = new Thread(() =>
                            {
                                mi.Invoke(instance, pListCopy);
                            });

                            threads[tCtr].Name = string.Format("Grid_{0}_{1}_Thread_{2}_{3}", x, y, tx, ty);
                            threads[tCtr].Start();
                            //if (ctr % 16 == 0)
                            //    Console.WriteLine("Ctr=" + ctr.ToString());
                            //ctr++;
                            tCtr++;
                        }
                    }

                    for (int i = 0; i < totalSize; i++)
                    {
                        threads[i].Join();
                        //Console.WriteLine("Thread {0} exited.", threads[i].Name);

                    }
                }
            }


            int iArgs = 0;
            ParameterInfo[] piArray = mi.GetParameters();
            for (int iParams = 0; iParams < piArray.Length; iParams++)
            {
                ParameterInfo pi = piArray[iParams];
                if (pi.ParameterType == typeof(GThread))
                    continue;
                else if (iArgs < pList.Length)
                {
                    object o = pList[iArgs++];
                    if (!(o is GThread))
                    {
                        if (o.GetType().IsArray)
                        {
                            if (dic.ContainsKey(o as Array))
                            {
                                EmuDevicePtrEx ptrEx = dic[o as Array];
                                DoCopy(o as Array, 0, ptrEx.DevPtr, ptrEx.Offset, ptrEx.TotalSize, pi.ParameterType.GetElementType());
                            }
                        }
                    }
                    else
                        iParams--;
                }
            }
        }

        //        protected override void DoLaunch(dim3 gridSize, dim3 blockSize, int streamId, KernelMethodInfo gpuMethodInfo, params object[] arguments)
        //        {
        //            if (streamId > -1 && !_streams.ContainsKey(streamId))
        //                _streams.Add(streamId, streamId);


        //            MethodInfo mi = gpuMethodInfo.Method;
        //            if (mi == null)
        //                throw new CudafyHostException(CudafyHostException.csX_NOT_SET, gpuMethodInfo.Name);
        //            bool isStatic = mi.IsStatic;
        //            object instance = isStatic ? null : Activator.CreateInstance(mi.DeclaringType);
        //            if (gpuMethodInfo.IsDummy)
        //            {
        //                mi.Invoke(instance, arguments);
        //                return;
        //            }
        //            //List<Type> paramTypes = new List<Type>();
        //            //mi.Parameters().ToList().ForEach(p => paramTypes.Add(p.ParameterType));
        //            MethodInvoker imi = null;
        //            //mi.DeclaringType.DelegateForCallMethod(mi.Name,
        //                //typeof(GThread), typeof(byte[]), typeof(long), typeof(uint[]));   //mi.DelegateForCallMethod();
        //            StaticMethodInvoker smi = null;
        //            if(isStatic)
        //                smi = mi.DelegateForCallStaticMethod();
        //            else
        //                imi = mi.DelegateForCallMethod();

        //            GGrid grid = new GGrid(gridSize);
        //            for (int x = 0; x < gridSize.x; x++)
        //            {
        //                for (int y = 0; y < gridSize.y; y++)
        //                {
        //                    int totalSize = blockSize.x * blockSize.y * blockSize.z;
        //                    Thread[] threads = new Thread[totalSize];
        //                    IAsyncResult[] ars = new IAsyncResult[totalSize]; 
        //                    GBlock blk2lnch = new GBlock(grid, blockSize, x, y);
        //                    int tCtr = 0;
        //                    for (int tx = 0; tx < blockSize.x; tx++)
        //                    {
        //                        for (int ty = 0; ty < blockSize.y; ty++)
        //                        {
        //                            GThread ht = new GThread(tx, ty, blk2lnch);
        //                            object[] pList = BuildParameterList(mi, ht, arguments);

        //#warning OPTIMIZATION if there is no synchronize then start and join threads in multiple of processor count - check this in disassembly and put flag in gpuMethodInfo
        //                            //threads[tCtr] = new Thread(() =>
        //                            //{
        //                            IAsyncResult ar = null;    
        //                            if(isStatic)
        //                                ar = smi.BeginInvoke(pList, null, null);
        //                            else
        //                                ar = imi.BeginInvoke(instance, pList, null, null);
        //                                //if (mi.IsStatic)
        //                                //    mi.Call(pList);
        //                                //else
        //                                //    mi.Call(instance, pList);
        //                           // });

        //                            //mi.Call(instance, pList);
        //                            //threads[tCtr].Name = string.Format("Grid_{0}_{1}_Thread_{2}_{3}", x, y, tx, ty);
        //                            //threads[tCtr].Start();
        //                            //if (ctr % 16 == 0)
        //                            //    Console.WriteLine("Ctr=" + ctr.ToString());
        //                            //ctr++;
        //                            ars[tCtr] = ar;
        //                            tCtr++;
        //                        }
        //                    }

        //                    for (int i = 0; i < totalSize; i++)
        //                    {
        //                        //threads[i].Join();
        //                        //Console.WriteLine("Thread {0} exited.", threads[i].Name);
        //                        if (isStatic)
        //                            smi.EndInvoke(ars[i]);
        //                        else
        //                            imi.EndInvoke(ars[i]);
        //                    }
        //                }
        //            }
        //        }

        private object[] BuildParameterList2(MethodInfo mi, object[] userArgs, out Dictionary<Array, EmuDevicePtrEx> dic)
        {
            dic = new Dictionary<Array, EmuDevicePtrEx>();
            List<object> prms = new List<object>();
            int iArgs = 0;
            ParameterInfo[] piArray = mi.GetParameters();
            for (int iParams = 0; iParams < piArray.Length; iParams++)
            {
                ParameterInfo pi = piArray[iParams];
                if (pi.ParameterType == typeof(GThread))
                    prms.Add(new GThread(0, 0, null));
                else if (iArgs < userArgs.Length)
                {
                    object o = userArgs[iArgs++];
                    if (!(o is GThread))
                    {
                        if (!pi.ParameterType.IsArray && o.GetType().IsArray && !pi.IsOut && !pi.ParameterType.IsByRef)
                        {
                            EmuDevicePtrEx ptrEx = (EmuDevicePtrEx)GetDeviceMemory(o);
                            prms.Add(ptrEx.DevPtr.GetValue(0));
                            //prms.Add((o as Array).GetValue(0));
                        }
                        else if (pi.IsOut)
                            throw new CudafyHostException(CudafyHostException.csPARAMETER_PASSED_BY_REFERENCE_X_NOT_CURRENTLY_SUPPORTED, "out");
                        else if (pi.ParameterType.IsByRef)
                            throw new CudafyHostException(CudafyHostException.csPARAMETER_PASSED_BY_REFERENCE_X_NOT_CURRENTLY_SUPPORTED, "ref");
                        else if (o.GetType().IsArray)
                        {
                            EmuDevicePtrEx ptrEx = (EmuDevicePtrEx)GetDeviceMemory(o);
                            if (ptrEx.Offset == 0 && ptrEx.DevPtr.Rank == pi.ParameterType.GetArrayRank())
                            {
                                prms.Add(ptrEx.DevPtr);
                            }
                            else
                            {
                                Array tempArray = Array.CreateInstance(pi.ParameterType.GetElementType(), ptrEx.GetDimensions());
                                DoCopy(ptrEx.DevPtr, ptrEx.Offset, tempArray, 0, ptrEx.TotalSize, pi.ParameterType.GetElementType());
                                prms.Add(tempArray);
                                dic.Add(tempArray, ptrEx);
                            }
                        }
                        else
                            prms.Add(o);
                    }
                    else
                        iParams--;
                }

            }
            return prms.ToArray();
        }

        //private object[] BuildParameterList(MethodInfo mi, GThread ht, object[] userArgs)
        //{
        //    List<object> prms = new List<object>();
        //    int iArgs = 0;
        //    ParameterInfo[] piArray = mi.GetParameters();
        //    for (int iParams = 0; iParams < piArray.Length; iParams++)
        //    {
        //        ParameterInfo pi = piArray[iParams];
        //        if (pi.ParameterType == typeof(GThread))
        //            prms.Add(ht);
        //        else if (iArgs < userArgs.Length)
        //        {
        //            object o = userArgs[iArgs++];
        //            if (!(o is GThread))
        //            {
        //                if (!pi.ParameterType.IsArray && o.GetType().IsArray && !pi.IsOut && !pi.ParameterType.IsByRef)
        //                {
        //                    EmuDevicePtrEx ptrEx = (EmuDevicePtrEx)GetDeviceMemory(o);
        //                    prms.Add(ptrEx.DevPtr.GetValue(0));
        //                    //prms.Add((o as Array).GetValue(0));
        //                }
        //                else if (pi.IsOut)
        //                    throw new CudafyHostException(CudafyHostException.csPARAMETER_PASSED_BY_REFERENCE_X_NOT_CURRENTLY_SUPPORTED, "out");
        //                else if (pi.ParameterType.IsByRef)
        //                    throw new CudafyHostException(CudafyHostException.csPARAMETER_PASSED_BY_REFERENCE_X_NOT_CURRENTLY_SUPPORTED, "ref");
        //                else if(o.GetType().IsArray)
        //                {
        //                    EmuDevicePtrEx ptrEx = (EmuDevicePtrEx)GetDeviceMemory(o);
        //                    if (ptrEx.DevPtr.Rank == pi.ParameterType.GetArrayRank())
        //                    {
        //                        prms.Add(ptrEx.DevPtr);
        //                    }
        //                    else
        //                    {                             
        //                        Array tempArray = Array.CreateInstance(pi.ParameterType.GetElementType(), ptrEx.GetDimensions());
        //                        DoCopy(ptrEx.DevPtr, 0, tempArray, 0, ptrEx.TotalSize, pi.ParameterType.GetElementType());
        //                        prms.Add(tempArray);
        //                    }
        //                }
        //                else
        //                    prms.Add(o);
        //            }
        //            else
        //                iParams--;
        //        }

        //    }
        //    return prms.ToArray();
        //}

        protected override Array DoCast<T, U>(int offset, Array devArray, int n)
        {
            if (typeof(T) != typeof(U))
                throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, "Casting between types on Emulator");
            T[] devMemPtr = new T[0];
            EmuDevicePtrEx ptrEx = (EmuDevicePtrEx)GetDeviceMemory(devArray);
            ptrEx = new EmuDevicePtrEx(offset, ptrEx.DevPtr, n);
            AddToDeviceMemory(devMemPtr, ptrEx);
            return devMemPtr;
        }

        protected override Array DoCast<T, U>(int offset, Array devArray, int x, int y)
        {
            if (typeof(T) != typeof(U))
                throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, "Casting between types on Emulator");
            T[,] devMemPtr = new T[0, 0];
            EmuDevicePtrEx ptrEx = (EmuDevicePtrEx)GetDeviceMemory(devArray);
            ptrEx = new EmuDevicePtrEx(offset, ptrEx.DevPtr, x, y);
            AddToDeviceMemory(devMemPtr, ptrEx);
            return devMemPtr;
        }

        protected override Array DoCast<T, U>(int offset, Array devArray, int x, int y, int z)
        {
            if (typeof(T) != typeof(U))
                throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, "Casting between types on Emulator");
            T[, ,] devMemPtr = new T[0, 0, 0];
            EmuDevicePtrEx ptrEx = (EmuDevicePtrEx)GetDeviceMemory(devArray);
            ptrEx = new EmuDevicePtrEx(offset, ptrEx.DevPtr, x, y, z);
            AddToDeviceMemory(devMemPtr, ptrEx);
            return devMemPtr;
        }


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
        protected override void DoCopyToConstantMemory<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, KernelConstantInfo ci)
        {
            Array.Copy(hostArray, hostOffset, devArray, devOffset, count);
        }

        protected override void DoCopyToConstantMemoryAsync<T>(IntPtr hostArray, int hostOffset, Array devArray, int devOffset, int count, KernelConstantInfo ci, int streamId)
        {
            if (streamId >= 0 && !_streams.ContainsKey(streamId))
                _streams.Add(streamId, streamId);
            int size = MSizeOf(typeof(T));
            GCHandle handle = GCHandle.Alloc(devArray, GCHandleType.Pinned);
            IntPtr devArrayPtr = new IntPtr(handle.AddrOfPinnedObject().ToInt64() + (devOffset * size));
            IntPtr hostArrayPtr = new IntPtr(hostArray.ToInt64() + (hostOffset * size));
            CopyMemory(devArrayPtr, hostArrayPtr, (uint)(count * size));
        }




        /// <summary>
        /// Does the copy to device.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        protected override void DoCopyToDevice<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count)
        {
            EmuDevicePtrEx devPtr = (EmuDevicePtrEx)GetDeviceMemory(devArray);
            DoCopy<T>(hostArray, hostOffset, devPtr.DevPtr, devPtr.Offset + devOffset, count);
        }

        protected override void DoCopyToDeviceAsync<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, int streamId)
        {
            GetStream(streamId);
            DoCopyToDevice<T>(hostArray, hostOffset, devArray, devOffset, count);
        }

        protected override void DoCopyFromDeviceAsync<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count, int streamId)
        {
            GetStream(streamId);
            DoCopyFromDevice<T>(devArray, devOffset, hostArray, hostOffset, count);
        }



        /// <summary>
        /// Does the copy from device.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The count.</param>
        protected override void DoCopyFromDevice<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count)
        {
            EmuDevicePtrEx devPtr = (EmuDevicePtrEx)GetDeviceMemory(devArray);
            DoCopy<T>(devPtr.DevPtr, devPtr.Offset + devOffset, hostArray, hostOffset, count);
            //Array.Copy(devArray, devOffset, hostArray, hostOffset, count);
        }

        protected override void DoCopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, Array devArray, int devOffset, int count, int streamId)
        {
            if (streamId >= 0 && !_streams.ContainsKey(streamId))
                _streams.Add(streamId, streamId);
            var ptrEx = GetDeviceMemory(devArray) as EmuDevicePtrEx;
            DoCopyToDeviceAsync<T>(hostArray, hostOffset, ptrEx, devOffset, count, streamId);
        }

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
        protected override void DoCopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, DevicePtrEx devArray, int devOffset, int count, int streamId)
        {
            var ptrEx = devArray as EmuDevicePtrEx;
            int size = MSizeOf(typeof(T));
            if (!_hostHandles.ContainsKey(hostArray))
                throw new CudafyHostException(CudafyHostException.csDATA_IS_NOT_HOST_ALLOCATED);
            IntPtr hostArrayPtr = _hostHandles[hostArray].AddrOfPinnedObject();
            IntPtr devOffsetPtr = new IntPtr(ptrEx.Pointer.ToInt64() + devOffset * size);
            IntPtr hostOffsetPtr = new IntPtr(hostArrayPtr.ToInt64() + hostOffset * size);
            try
            {
                CopyMemory(devOffsetPtr, hostOffsetPtr, (uint)(count * MSizeOf(typeof(T))));
            }
            finally
            {
                ptrEx.FreeHandle();
            }
        }

        protected override void DoCopyFromDeviceAsync<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count, int streamId, IntPtr stagingPost)
        {
            DoCopyFromDeviceAsync<T>(devArray, devOffset, stagingPost, 0, count, streamId);
            DoCopyOnHost<T>(stagingPost, 0, hostArray, hostOffset, count);
        }

        protected override void DoCopyToDeviceAsync<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, int streamId, IntPtr stagingPost, bool isConstantMemory = false)
        {
            DoCopyOnHost<T>(hostArray, hostOffset, stagingPost, 0, count);
            if(!isConstantMemory)
                DoCopyToDeviceAsync<T>(stagingPost, 0, devArray, devOffset, count, streamId);
            else
                DoCopyToConstantMemoryAsync<T>(stagingPost, 0, devArray, devOffset, count, null, streamId);  
        }

        protected override void DoCopyFromDeviceAsync<T>(Array devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId)
        {
            if (streamId >= 0 && !_streams.ContainsKey(streamId))
                _streams.Add(streamId, streamId);
            VerifyOnGPU(devArray);
            var ptrEx = GetDeviceMemory(devArray) as EmuDevicePtrEx;
            DoCopyFromDeviceAsync<T>(ptrEx, devOffset, hostArray, hostOffset, count, streamId);
        }

        public override void CreateStream(int streamId)
        {
            if (streamId > -1 && !_streams.ContainsKey(streamId))
                _streams.Add(streamId, streamId);
        }

        protected override void DoCopyOnDevice<T>(DevicePtrEx srcDevArray, int srcOffset, DevicePtrEx dstDevArray, int dstOffet, int count)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyOnDeviceAsync<T>(Array srcDevArray, int srcOffset, Array dstDevArray, int dstOffet, int count, int streamId)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyOnDeviceAsync<T>(DevicePtrEx srcDevArray, int srcOffset, DevicePtrEx dstDevArray, int dstOffet, int count, int streamId)
        {
            throw new NotImplementedException();
        }


        /// <summary>
        /// Does the copy from device async.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="streamId">The stream id.</param>
        protected override void DoCopyFromDeviceAsync<T>(DevicePtrEx devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId)
        {
            var ptrEx = devArray as EmuDevicePtrEx;
            int size = MSizeOf(typeof(T));
            //if (!_hostHandles.ContainsKey(hostArray))
            //    throw new CudafyHostException(CudafyHostException.csDATA_IS_NOT_HOST_ALLOCATED);
            GCHandle handle = _hostHandles[hostArray];
            IntPtr hostArrayPtr = handle.AddrOfPinnedObject();
            IntPtr devOffsetPtr = new IntPtr(ptrEx.Pointer.ToInt64() + devOffset * size);
            IntPtr hostOffsetPtr = new IntPtr(hostArrayPtr.ToInt64() + hostOffset * size);
            try
            {
                CopyMemory(hostOffsetPtr, devOffsetPtr, (uint)(count * MSizeOf(typeof(T))));
            }
            finally
            {
                ptrEx.FreeHandle();
            }
        }



        /// <summary>
        /// Does the copy from device.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="hostArray">The host array.</param>
        protected override void DoCopyFromDevice<T>(Array devArray, Array hostArray)
        {
            DoCopyFromDevice<T>(devArray, 0, hostArray, 0, hostArray.Length);
        }

        /// <summary>
        /// Allocates array on device, copies to device and returns pointer.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>The device array.</returns>
        public override T[] CopyToDevice<T>(T[] hostArray)
        {
            T[] devMemPtr = Allocate<T>(hostArray);
            //EmuDevicePtrEx devMem = (EmuDevicePtrEx)GetDeviceMemory(devMemPtr);
            DoCopyToDevice<T>(hostArray, 0, devMemPtr, 0, hostArray.Length);
            return devMemPtr;
        }

        /// <summary>
        /// Allocates array on device, copies to device and returns pointer.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>The device array.</returns>
        public override T[,] CopyToDevice<T>(T[,] hostArray)
        {
            T[,] devMemPtr = Allocate<T>(hostArray);
            //EmuDevicePtrEx devMem = (EmuDevicePtrEx)GetDeviceMemory(devMemPtr);
            DoCopyToDevice<T>(hostArray, 0, devMemPtr, 0, hostArray.Length);
            return devMemPtr;
        }

        /// <summary>
        /// Allocates array on device, copies to device and returns pointer.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray"></param>
        /// <returns>The device array.</returns>
        public override T[, ,] CopyToDevice<T>(T[, ,] hostArray)
        {
            T[, ,] devMemPtr = Allocate<T>(hostArray);
            //EmuDevicePtrEx devMem = (EmuDevicePtrEx)GetDeviceMemory(devMemPtr);
            DoCopyToDevice<T>(hostArray, 0, devMemPtr, 0, hostArray.Length);
            return devMemPtr;
        }

        ///// <summary>
        ///// Copies from device.
        ///// </summary>
        ///// <typeparam name="T"></typeparam>
        ///// <param name="devArray">The dev array.</param>
        ///// <param name="hostData">The host data.</param>
        //public override void CopyFromDevice<T>(T devArray, out T hostData)
        //{
        //    VerifyOnGPU(devArray);
        //    EmuDevicePtrEx ptrEx = (EmuDevicePtrEx)GetDeviceMemory(devArray);
        //    hostData = (T)ptrEx.DevPtr.GetValue(ptrEx.Offset);
        //}


        /// <summary>
        /// Allocates array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">Length of 1D array.</param>
        /// <returns>Device array of length x.</returns>
        public override T[] Allocate<T>(int x)
        {
            T[] devMemPtr = new T[0];
            T[] devMem = new T[x];
            AddToDeviceMemory(devMemPtr, new EmuDevicePtrEx(0, devMem, x));
            return devMemPtr;
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
            T[] devMem = new T[x * y];
            T[,] devMemPtr = new T[0, 0];
            AddToDeviceMemory(devMemPtr, new EmuDevicePtrEx(0, devMem, x, y));
            return devMemPtr;
        }

        /// <summary>
        /// Allocates array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">The x dimension.</param>
        /// <param name="y">The y dimension.</param>
        /// <param name="z">The z dimension.</param>
        /// <returns>3D device array.</returns>
        public override T[, ,] Allocate<T>(int x, int y, int z)
        {
            T[] devMem = new T[x * y * z];
            T[, ,] devMemPtr = new T[0, 0, 0];
            AddToDeviceMemory(devMemPtr, new EmuDevicePtrEx(0, devMem, x, y, z));
            return devMemPtr;
        }

        /// <summary>
        /// Allocates array on device of same size as supplied host array.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>1D device array.</returns>
        public override T[] Allocate<T>(T[] hostArray)
        {
            return Allocate<T>(hostArray.Length);
        }

        /// <summary>
        /// Allocates array on device of same size as supplied host array.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>2D device array.</returns>
        public override T[,] Allocate<T>(T[,] hostArray)
        {
            return Allocate<T>(hostArray.GetLength(0), hostArray.GetLength(1));
        }

        /// <summary>
        /// Allocates array on device of same size as supplied host array.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>3D device array.</returns>
        public override T[, ,] Allocate<T>(T[, ,] hostArray)
        {
            return Allocate<T>(hostArray.GetLength(0), hostArray.GetLength(1), hostArray.GetLength(2));
        }

        /// <summary>
        /// Does the set.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="count">The count.</param>
        protected override void DoSet<T>(Array devArray, int offset = 0, int count = 0)
        {
            VerifyOnGPU(devArray);
            EmuDevicePtrEx ptrEx = (EmuDevicePtrEx)GetDeviceMemory(devArray);
            if (count == 0)
                count = ptrEx.TotalSize;
            Array.Clear(ptrEx.DevPtr, ptrEx.Offset + offset, count);
        }

        /// <summary>
        /// Frees the specified data array on device.
        /// </summary>
        /// <param name="devArray">The device array to free.</param>
        public override void Free(object devArray)
        {
            VerifyOnGPU(devArray);
            RemoveFromDeviceMemory(devArray);
        }

        public override void FreeAll()
        {
            ClearDeviceMemory();
        }

        public override void LoadModule(CudafyModule module, bool unload)
        {
            if (unload)
                UnloadModules();
            else
                CheckForDuplicateMembers(module);
            _module = module;
            _modules.Add(module);
        }

        public override void UnloadModule(CudafyModule module)
        {
            if (!_modules.Remove(module))
                throw new CudafyHostException(CudafyHostException.csMODULE_NOT_FOUND);
            if (_module == module)
                _module = null;
        }

        /// <summary>
        /// Copies between preallocated arrays on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The source device array.</param>
        /// <param name="dstDevArray">The destination device array.</param>
        public override void CopyOnDevice<T>(T[] srcDevArray, T[] dstDevArray)
        {
            EmuDevicePtrEx srcPtrEx = (EmuDevicePtrEx)GetDeviceMemory(srcDevArray);
            EmuDevicePtrEx dstPtrEx = (EmuDevicePtrEx)GetDeviceMemory(dstDevArray);
            Array.Copy(srcPtrEx.DevPtr, srcPtrEx.Offset, dstPtrEx.DevPtr, dstPtrEx.Offset, Math.Min(srcPtrEx.TotalSize, dstPtrEx.TotalSize));
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
        protected override void DoCopyOnDevice<T>(Array srcDevArray, int srcOffset, Array dstDevArray, int dstOffet, int count)
        {
            EmuDevicePtrEx srcPtrEx = (EmuDevicePtrEx)GetDeviceMemory(srcDevArray);
            EmuDevicePtrEx dstPtrEx = (EmuDevicePtrEx)GetDeviceMemory(dstDevArray);
            Array.Copy(srcPtrEx.DevPtr, srcPtrEx.Offset + srcOffset, dstPtrEx.DevPtr, dstPtrEx.Offset + dstOffet, count);
        }

        protected override void DoCopyDeviceToDevice<T>(Array srcDevArray, int srcOffset, GPGPU peer, Array dstDevArray, int dstOffet, int count)
        {
            EmuDevicePtrEx srcPtrEx = (EmuDevicePtrEx)GetDeviceMemory(srcDevArray);
            EmuDevicePtrEx dstPtrEx = (EmuDevicePtrEx)peer.GetDeviceMemory(dstDevArray);
            Array.Copy(srcPtrEx.DevPtr, srcPtrEx.Offset + srcOffset, dstPtrEx.DevPtr, dstPtrEx.Offset + dstOffet, count);
        }

        protected override void DoCopyDeviceToDeviceAsync<T>(Array srcDevArray, int srcOffset, GPGPU peer, Array dstDevArray, int dstOffet, int count, int stream) 
        {
            DoCopyDeviceToDevice<T>(srcDevArray, srcOffset, peer, dstDevArray, dstOffet, count);
        }

        /// <summary>
        /// Synchronizes the stream.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        public override void SynchronizeStream(int streamId = 0)
        {
            if (streamId > 0 && !_streams.ContainsKey(streamId))
                throw new CudafyHostException(CudafyHostException.csSTREAM_X_NOT_SET, streamId);
        }

        /// <summary>
        /// Destroys the stream.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        public override void DestroyStream(int streamId)
        {
            if (!_streams.Remove(streamId))
            {
                Debug.WriteLine(string.Format("Warning: DestroyStream(int streamId) streamId {0} does not exist"));
            }
        }

        /// <summary>
        /// Destroys all streams.
        /// </summary>
        public override void DestroyStreams()
        {
            _streams.Clear();
        }

 
#pragma warning restore 1591







    }

    /// <summary>
    /// Internal use.
    /// </summary>
    public class EmuDevicePtrEx : DevicePtrEx
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EmuDevicePtrEx"/> class.
        /// </summary>
        /// <param name="offset">Offset in samples.</param>
        /// <param name="devPtr">The dev pointer.</param>
        /// <param name="dimensions">The dimensions.</param>
        public EmuDevicePtrEx(int offset, Array devPtr, params int[] dimensions)// EmuDevicePtrEx original)//int originalSize = -1)
        {
            DevPtr = devPtr;
            Dimensions = dimensions.Length;
            Debug.Assert(Dimensions > 0);
            Debug.Assert(Dimensions < 4);
            XSize = dimensions[0];
            YSize = Dimensions > 1 ? dimensions[1] : 1;
            ZSize = Dimensions > 2 ? dimensions[2] : 1;
            Offset = offset;
        }


        /// <summary>
        /// Gets the dev PTR.
        /// </summary>
        public Array DevPtr { get; private set; }


        /// <summary>
        /// Gets the offset in bytes.
        /// </summary>
        public int OffsetBytes
        {
            get
            {
                int size = DevPtr.Length == 0 ? 0 : GPGPU.MSizeOf(DevPtr.GetValue(0).GetType());
                return Offset * size;
            }
        }


        private GCHandle _devPtrHandle;

        /// <summary>
        /// Gets the native pointer to the data. FreeHandle() must be called afterwards.
        /// </summary>
        /// <param name="offset">Offset in bytes.</param>
        /// <returns>Pointer</returns>
        public IntPtr GetDevPtrPtr(long offset)
        {
            _devPtrHandle = GCHandle.Alloc(DevPtr, GCHandleType.Pinned);
            long address = _devPtrHandle.AddrOfPinnedObject().ToInt64();
            IntPtr offsetPtr = new IntPtr(address + offset + OffsetBytes);
            return offsetPtr;
        }

        /// <summary>
        /// Gets the native pointer.
        /// </summary>
        public override IntPtr Pointer
        {
            get { return GetDevPtrPtr(0); }
        }

        /// <summary>
        /// Frees the handle allocated by GetDevPtrPtr
        /// </summary>
        public void FreeHandle()
        {
            if (_devPtrHandle.IsAllocated)
                _devPtrHandle.Free();
        }
    }
}
