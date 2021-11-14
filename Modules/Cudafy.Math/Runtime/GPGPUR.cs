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
    /// Abstract base class for run-time API.
    /// </summary>
    public abstract class GPGPUR
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GPGPUR"/> class.
        /// </summary>
        protected GPGPUR()
        {
            _deviceMemory = new Dictionary<object, object>();
        }

        /// <summary>
        /// Creates the specified gpu type.
        /// </summary>
        /// <param name="gpuType">Type of the gpu.</param>
        /// <returns></returns>
        public static GPGPUR Create(eGPUType gpuType)
        {
            if (gpuType == eGPUType.Cuda)
                return new CudaR();
            else
                throw new NotSupportedException(gpuType.ToString());
        }

        private void HandleError(cudaError error)
        {
            if (error != cudaError.cudaSuccess)
                throw new CudafyHostException(error.ToString());
        }

        private int MSizeOf<T>()
        {
            return Marshal.SizeOf(typeof(T));
        }

        /// <summary>
        /// Stores pointers to data on the device.
        /// </summary>
        protected Dictionary<object, object> _deviceMemory;


        /// <summary>
        /// Gets the device memory for key specified.
        /// </summary>
        /// <param name="devArray">The dev array.</param>
        /// <returns></returns>
        public object GetDeviceMemory(object devArray)
        {
            VerifyOnGPU(devArray);
            object ptr = _deviceMemory[devArray];
            return ptr;
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
            return _deviceMemory.ContainsKey(data);
        }

        /// <summary>
        /// Frees the specified data array on device.
        /// </summary>
        /// <param name="devArray">The device array to free.</param>
        public abstract void Free(object devArray);

        /// <summary>
        /// Allocates array on device of same size as supplied host array.
        /// </summary>
        /// <typeparam name="T">Float, Double ComplexF or ComplexD.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>1D device array.</returns>
        public abstract T[] Allocate<T>(T[] hostArray);

        /// <summary>
        /// Allocates vector on device.
        /// </summary>
        /// <typeparam name="T">Float, Double ComplexF or ComplexD.</typeparam>
        /// <param name="x">Length of 1D array.</param>
        /// <returns>1D device array.</returns>
        public abstract T[] Allocate<T>(int x);

        /// <summary>
        /// Allocates array on device.
        /// </summary>
        /// <typeparam name="T">Float, Double ComplexF or ComplexD.</typeparam>
        /// <param name="rows">The x dimension.</param>
        /// <param name="columns">The y dimension.</param>
        /// <returns>2D matrix.</returns>
        public abstract T[,] Allocate<T>(int rows, int columns);

        /// <summary>
        /// Allocates the specified x.
        /// </summary>
        /// <typeparam name="T">Float, Double ComplexF or ComplexD.</typeparam>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public abstract T[,,] Allocate<T>(int x, int y, int z);

        /// <summary>
        /// Sets the device.
        /// </summary>
        /// <param name="id">The id.</param>
        public abstract void SetDevice(int id);

        /// <summary>
        /// Copies to device.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        public abstract void CopyToDevice<T>(T[] hostArray, int hostOffset, T[] devArray, int devOffset, int count);

        /// <summary>
        /// Copies from device.
        /// </summary>
        /// <typeparam name="T">Float, Double ComplexF or ComplexD.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The count.</param>
        public abstract void CopyFromDevice<T>(T[] devArray, int devOffset, T[] hostArray, int hostOffset, int count);

        /// <summary>
        /// Copies to device.
        /// </summary>
        /// <typeparam name="T">Float, Double ComplexF or ComplexD.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        public abstract void CopyToDevice<T>(T[,] hostArray, int hostOffset, T[,] devArray, int devOffset, int count);

        /// <summary>
        /// Copies from device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The count.</param>
        public abstract void CopyFromDevice<T>(T[,] devArray, int devOffset, T[,] hostArray, int hostOffset, int count);

        /// <summary>
        /// Copies to device.
        /// </summary>
        /// <typeparam name="T">Float, Double ComplexF or ComplexD.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        public abstract void CopyToDevice<T>(T[,,] hostArray, int hostOffset, T[,,] devArray, int devOffset, int count);

        /// <summary>
        /// Copies from device.
        /// </summary>
        /// <typeparam name="T">Float, Double ComplexF or ComplexD.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The count.</param>
        public abstract void CopyFromDevice<T>(T[,,] devArray, int devOffset, T[,,] hostArray, int hostOffset, int count);


    }
}
