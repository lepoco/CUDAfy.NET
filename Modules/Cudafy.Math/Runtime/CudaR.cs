using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using GASS.CUDA;
using GASS.CUDA.Types;

using Cudafy.Host;
namespace Cudafy.Maths.Runtime
{
    /// <summary>
    /// Implements a wrapper for the CUDA Runtime API.
    /// </summary>
    public class CudaR : GPGPUR
    {
        internal CudaR()
        {
            _deviceMemory = new Dictionary<object, object>();
        }

        private void HandleError(cudaError error)
        {
            if (error != cudaError.cudaSuccess)
                throw new CudafyHostException(error.ToString());
        }

        //private int MSizeOf<T>()
        //{
        //    return Marshal.SizeOf(typeof(T));
        //}

        /// <summary>
        /// Frees the specified data array on device.
        /// </summary>
        /// <param name="devArray">The device array to free.</param>
        public override void Free(object devArray)
        {
            VerifyOnGPU(devArray);
            CUDevicePtrEx ptrEx = (CUDevicePtrEx)_deviceMemory[devArray];
            HandleError(CUDARuntime.cudaFree(ptrEx.DevPtr));
            _deviceMemory.Remove(devArray);
        }

        /// <summary>
        /// Allocates array on device of same size as supplied host array.
        /// </summary>
        /// <typeparam name="T">Float, Double ComplexF or ComplexD.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>1D device array.</returns>
        public override T[] Allocate<T>(T[] hostArray)
        {
            return Allocate<T>(hostArray.Length);
        }

        /// <summary>
        /// Allocates vector on device.
        /// </summary>
        /// <typeparam name="T">Float, Double ComplexF or ComplexD.</typeparam>
        /// <param name="x">Length of 1D array.</param>
        /// <returns>1D device array.</returns>
        public override T[] Allocate<T>(int x)
        {
            T[] devMem = new T[0];
            CUdeviceptr ptr = new CUdeviceptr();
            HandleError(CUDARuntime.cudaMalloc(ref ptr, x * CUDA.MSizeOf(typeof(T)))); 
            _deviceMemory.Add(devMem, new CUDevicePtrEx(ptr, x, null));
            return devMem;
        }

        /// <summary>
        /// Allocates array on device.
        /// </summary>
        /// <typeparam name="T">Float, Double ComplexF or ComplexD.</typeparam>
        /// <param name="rows">The x dimension.</param>
        /// <param name="columns">The y dimension.</param>
        /// <returns>2D matrix.</returns>
        public override T[,] Allocate<T>(int rows, int columns)
        {
            T[,] devMem = new T[0, 0];
            CUdeviceptr ptr = new CUdeviceptr();
            HandleError(CUDARuntime.cudaMalloc(ref ptr, rows * columns * CUDA.MSizeOf(typeof(T)))); 
            _deviceMemory.Add(devMem, new CUDevicePtrEx(ptr, rows, columns, null));
            return devMem;
        }

        /// <summary>
        /// Sets the device.
        /// </summary>
        /// <param name="id">The id.</param>
        public override void SetDevice(int id)
        {
             HandleError(CUDARuntime.cudaSetDevice(id));
        }

        /// <summary>
        /// Copies to device.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        public override void CopyToDevice<T>(T[] hostArray, int hostOffset, T[] devArray, int devOffset, int count)
        {
            CopyToDevice<T>(hostArray, hostOffset, devArray, devOffset, count);
        }

        /// <summary>
        /// Copies to device.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        private void CopyToDevice<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count)
        {
            CUDevicePtrEx devPtrEx = GetDeviceMemory(devArray) as CUDevicePtrEx;
            int n = hostArray.Length;
            Type type = typeof(T);
            int elemSize = Marshal.SizeOf(type);
            unsafe
            {
                GCHandle handle = GCHandle.Alloc(hostArray, GCHandleType.Pinned);
                IntPtr hostPtr = new IntPtr(handle.AddrOfPinnedObject().ToInt64() + hostOffset * elemSize);
                CUdeviceptr devPtr = devPtrEx.DevPtr + devOffset * elemSize;
                cudaError rc = CUDARuntime.cudaMemcpy(devPtr, hostPtr, elemSize * n, cudaMemcpyKind.cudaMemcpyHostToDevice);
                handle.Free();
                HandleError(rc);
            }
        }

        /// <summary>
        /// Copies from device.
        /// </summary>
        /// <typeparam name="T">Float, Double ComplexF or ComplexD.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The count.</param>
        public override void CopyFromDevice<T>(T[] devArray, int devOffset, T[] hostArray, int hostOffset, int count)
        {
            CopyFromDevice<T>(devArray, devOffset, hostArray, hostOffset, count);
        }

        private void CopyFromDevice<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count)
        {
            CUDevicePtrEx devPtrEx = GetDeviceMemory(devArray) as CUDevicePtrEx;
            int n = hostArray.Length;
            Type type = typeof(T);
            int elemSize = CUDA.MSizeOf(type);
            unsafe
            {
                GCHandle handle = GCHandle.Alloc(hostArray, GCHandleType.Pinned);
                IntPtr hostPtr = new IntPtr(handle.AddrOfPinnedObject().ToInt64() + hostOffset * elemSize);
                CUdeviceptr devPtr = devPtrEx.DevPtr + devOffset * elemSize;
                cudaError rc = CUDARuntime.cudaMemcpy(hostPtr, devPtr, elemSize * n, cudaMemcpyKind.cudaMemcpyDeviceToHost);
                handle.Free();
                HandleError(rc);
            }
        }


        /// <summary>
        /// Allocates the specified array.
        /// </summary>
        /// <typeparam name="T">Float, Double ComplexF or ComplexD.</typeparam>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public override T[,,] Allocate<T>(int x, int y, int z)
        {
            T[,,] devMem = new T[0, 0, 0];
            CUdeviceptr ptr = new CUdeviceptr();
            HandleError(CUDARuntime.cudaMalloc(ref ptr, x * y * z * CUDA.MSizeOf(typeof(T))));
            _deviceMemory.Add(devMem, new CUDevicePtrEx(ptr, x, y, z, null));
            return devMem;
        }

        /// <summary>
        /// Copies to device.
        /// </summary>
        /// <typeparam name="T">Float, Double ComplexF or ComplexD.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        public override void CopyToDevice<T>(T[,] hostArray, int hostOffset, T[,] devArray, int devOffset, int count)
        {
            CopyToDevice<T>(hostArray, hostOffset, devArray, devOffset, count);
        }

        /// <summary>
        /// Copies from device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The count.</param>
        public override void CopyFromDevice<T>(T[,] devArray, int devOffset, T[,] hostArray, int hostOffset, int count)
        {
            CopyFromDevice(devArray, devOffset, hostArray, hostOffset, count);
        }

        /// <summary>
        /// Copies to device.
        /// </summary>
        /// <typeparam name="T">Float, Double ComplexF or ComplexD.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        public override void CopyToDevice<T>(T[, ,] hostArray, int hostOffset, T[, ,] devArray, int devOffset, int count)
        {
            CopyToDevice<T>(hostArray, hostOffset, devArray, devOffset, count);
        }

        /// <summary>
        /// Copies from device.
        /// </summary>
        /// <typeparam name="T">Float, Double ComplexF or ComplexD.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The count.</param>
        public override void CopyFromDevice<T>(T[, ,] devArray, int devOffset, T[, ,] hostArray, int hostOffset, int count)
        {
            CopyFromDevice(devArray, devOffset, hostArray, hostOffset, count);
        }
    }
}
