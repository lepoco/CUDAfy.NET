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
using Cudafy.Types;
using Cudafy.Host;

using GASS.CUDA;
using GASS.CUDA.Types;
using GASS.CUDA.FFT;
using GASS.CUDA.FFT.Types;
namespace Cudafy.Maths.FFT
{
    /// <summary>
    /// FFT wrapper for Cuda GPUs.
    /// </summary>
    internal class CudaFFT : GPGPUFFT
    {
        internal CudaFFT(CudaGPU gpu)
        {
            _gpu = gpu;
            if (IntPtr.Size == 8)
                _driver = new CUFFTDriver64();
            else
                throw new NotSupportedException();
                //_driver = new CUFFTDriver32();
        }

        private ICUFFTDriver _driver;

        /// <summary>
        /// Sets the stream.
        /// </summary>
        /// <param name="plan">The plan to set the stream for.</param>
        /// <param name="streamId">The stream id.</param>
        public override void SetStream(FFTPlan plan, int streamId)
        {
            if (streamId < 0)
                throw new ArgumentOutOfRangeException("streamId");
            CUstream cus = (CUstream)_gpu.GetStream(streamId);
            FFTPlanEx planEx = Plans[plan];
            cudaStream cs = new cudaStream();
            //cs.Value = cus.Pointer.ToInt32();
            CUFFTResult res = _driver.cufftSetStream(planEx.CudaFFTHandle, cs);
            if (res != CUFFTResult.Success)
                throw new CudafyMathException(CudafyMathException.csCUDA_EXCEPTION_X, res);
        }


        /// <summary>
        /// Creates a 1D plan.
        /// </summary>
        /// <param name="fftType">Type of FFT.</param>
        /// <param name="dataType">Data type.</param>
        /// <param name="nx">The length in samples.</param>
        /// <param name="batchSize">The number of FFTs in batch.</param>
        /// <returns>Plan.</returns>
        public override FFTPlan1D Plan1D(eFFTType fftType, eDataType dataType, int nx, int batchSize)
        {
            int insize, outsize;
            CUFFTType cuFFTType = VerifyTypes(fftType, dataType, out insize, out outsize);
            cufftHandle handle = new cufftHandle();
            CUFFTResult res;
            if (batchSize <= 1)
                res = _driver.cufftPlan1d(ref handle, nx, cuFFTType, batchSize);
            else
                res = _driver.cufftPlanMany(ref handle, 1, new int[] { nx },
                    null,   //inembed
                    1,      //istride 1
                    0,      //idist 0
                    null,   //onembed
                    1,      //ostride 1
                    0,      //odist 0
                    cuFFTType, 
                    batchSize);
            
            if (res != CUFFTResult.Success)
                throw new CudafyHostException(res.ToString());
            FFTPlan1D plan = new FFTPlan1D(nx, batchSize, this);
            FFTPlan1DEx planEx = new FFTPlan1DEx(plan) { CudaFFTHandle = handle, CudaFFTType = cuFFTType, DataType = dataType };
            Plans.Add(plan, planEx);
            return plan;
        }

        public override FFTPlan1D Plan1D(eFFTType fftType, eDataType dataType, int nx, int batchSize, int istride, int idist, int ostride, int odist)
        {
            int insize, outsize;
            CUFFTType cuFFTType = VerifyTypes(fftType, dataType, out insize, out outsize);
            cufftHandle handle = new cufftHandle();
            CUFFTResult res;
            if (batchSize <= 1)
                res = _driver.cufftPlan1d(ref handle, nx, cuFFTType, batchSize);
            else
                res = _driver.cufftPlanMany(ref handle, 1, new int[] { nx },
                    new int[] { idist },   //inembed
                    istride,      //istride
                    idist,      //idist
                    new int[] { odist },   //onembed
                    ostride,      //ostride
                    odist,      //odist
                    cuFFTType,
                    batchSize);

            if (res != CUFFTResult.Success)
                throw new CudafyHostException(res.ToString());
            FFTPlan1D plan = new FFTPlan1D(nx, batchSize, this);
            FFTPlan1DEx planEx = new FFTPlan1DEx(plan) { CudaFFTHandle = handle, CudaFFTType = cuFFTType, DataType = dataType };
            Plans.Add(plan, planEx);
            return plan;
        }

        /// <summary>
        /// Creates a 2D plan.
        /// </summary>
        /// <param name="fftType">Type of FFT.</param>
        /// <param name="dataType">Data type.</param>
        /// <param name="nx">The number of samples in x dimension.</param>
        /// <param name="ny">The number of samples in y dimension.</param>
        /// <param name="batchSize">Size of batch.</param>
        /// <returns>Plan.</returns>
        public override FFTPlan2D Plan2D(eFFTType fftType, eDataType dataType, int nx, int ny, int batchSize)
        {
            int insize, outsize;
            CUFFTType cuFFTType = VerifyTypes(fftType, dataType, out insize, out outsize);
            cufftHandle handle = new cufftHandle();
            CUFFTResult res;
            if(batchSize <= 1)
                res = _driver.cufftPlan2d(ref handle, nx, ny, cuFFTType);
            else
                res = _driver.cufftPlanMany(ref handle, 2, new int[] { nx, ny }, null, 1, 0, null, 1, 0, cuFFTType, batchSize);
            if (res != CUFFTResult.Success)
                throw new CudafyHostException(res.ToString());
            FFTPlan2D plan = new FFTPlan2D(nx, ny, batchSize, this);
            FFTPlan2DEx planEx = new FFTPlan2DEx(plan) { CudaFFTHandle = handle, CudaFFTType = cuFFTType, DataType = dataType };
            Plans.Add(plan, planEx);
            return plan;
        }

        /// <summary>
        /// Creates a 3D plan.
        /// </summary>
        /// <param name="fftType">Type of FFT.</param>
        /// <param name="dataType">Data type.</param>
        /// <param name="nx">The number of samples in x dimension.</param>
        /// <param name="ny">The number of samples in y dimension.</param>
        /// <param name="nz">The number of samples in z dimension.</param>
        /// <param name="batchSize">Size of batch.</param>
        /// <returns>Plan.</returns>
        public override FFTPlan3D Plan3D(eFFTType fftType, eDataType dataType, int nx, int ny, int nz, int batchSize)
        {
            int insize, outsize;
            CUFFTType cuFFTType = VerifyTypes(fftType, dataType, out insize, out outsize);
            cufftHandle handle = new cufftHandle();

            CUFFTResult res;
            if (batchSize <= 1)
                res = _driver.cufftPlan3d(ref handle, nx, ny, nz, cuFFTType);
            else
                res = _driver.cufftPlanMany(ref handle, 3, new int[] { nx, ny, nz }, null, 1, 0, null, 1, 0, cuFFTType, batchSize);

            if (res != CUFFTResult.Success)
                throw new CudafyHostException(res.ToString());
            FFTPlan3D plan = new FFTPlan3D(nx, ny, nz, batchSize, this);
            FFTPlan3DEx planEx = new FFTPlan3DEx(plan) { CudaFFTHandle = handle, CudaFFTType = cuFFTType, DataType = dataType };
            Plans.Add(plan, planEx);
            return plan;
        }

        /// <summary>
        /// Executes the specified plan.
        /// </summary>
        /// <typeparam name="T">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <typeparam name="U">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <param name="plan">The plan.</param>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <param name="inverse">if set to <c>true</c> inverse.</param>
        public override void Execute<T,U>(FFTPlan plan, T[] input, U[] output, bool inverse = false)
        {
            DoExecute(plan, input, output, inverse);           
        }

        private void DoExecute(FFTPlan plan, object input, object output, bool inverse = false)
        {
            FFTPlanEx planEx = Plans[plan];
            CUDevicePtrEx inPtrEx;
            CUDevicePtrEx outPtrEx;

            inPtrEx = _gpu.GetDeviceMemory(input) as CUDevicePtrEx;
            outPtrEx = _gpu.GetDeviceMemory(output) as CUDevicePtrEx;

            CUFFTDirection dir = inverse ? CUFFTDirection.Inverse : CUFFTDirection.Forward;
            CUFFTResult res = CUFFTResult.ExecFailed;
            if (planEx.CudaFFTType == CUFFTType.C2C)
                res = _driver.cufftExecC2C(planEx.CudaFFTHandle, inPtrEx.DevPtr, outPtrEx.DevPtr, dir);
            else if (planEx.CudaFFTType == CUFFTType.C2R)
                res = _driver.cufftExecC2R(planEx.CudaFFTHandle, inPtrEx.DevPtr, outPtrEx.DevPtr);
            else if (planEx.CudaFFTType == CUFFTType.D2Z)
                res = _driver.cufftExecD2Z(planEx.CudaFFTHandle, inPtrEx.DevPtr, outPtrEx.DevPtr);
            else if (planEx.CudaFFTType == CUFFTType.R2C)
                res = _driver.cufftExecR2C(planEx.CudaFFTHandle, inPtrEx.DevPtr, outPtrEx.DevPtr);
            else if (planEx.CudaFFTType == CUFFTType.Z2D)
                res = _driver.cufftExecZ2D(planEx.CudaFFTHandle, inPtrEx.DevPtr, outPtrEx.DevPtr);
            else if (planEx.CudaFFTType == CUFFTType.Z2Z)
                res = _driver.cufftExecZ2Z(planEx.CudaFFTHandle, inPtrEx.DevPtr, outPtrEx.DevPtr, dir);
            if (res != CUFFTResult.Success)
                throw new CudafyMathException(res.ToString());
        }

        /// <summary>
        /// Executes the specified plan.
        /// </summary>
        /// <typeparam name="T">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <typeparam name="U">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <param name="plan">The plan.</param>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <param name="inverse">if set to <c>true</c> inverse.</param>
        public override void Execute<T,U>(FFTPlan plan, T[,] input, U[,] output, bool inverse = false)
        {
            DoExecute(plan, input, output, inverse); 
        }

        /// <summary>
        /// Executes the specified plan.
        /// </summary>
        /// <typeparam name="T">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <typeparam name="U">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <param name="plan">The plan.</param>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <param name="inverse">if set to <c>true</c> inverse.</param>
        public override void Execute<T,U>(FFTPlan plan, T[,,] input, U[,,] output, bool inverse = false)
        {
            DoExecute(plan, input, output, inverse);
        }

        /// <summary>
        /// Frees the specified plan.
        /// </summary>
        /// <param name="plan">The plan.</param>
        public override void Remove(FFTPlan plan)
        {
            
            FFTPlanEx planEx = Plans[plan];

            CUFFTResult res = _driver.cufftDestroy(planEx.CudaFFTHandle);
            if (res != CUFFTResult.Success)
                //throw new CudafyHostException(res.ToString());
                Debug.WriteLine("remove plan failed: "+res.ToString());
            else
                Debug.WriteLine("remove plan succeeded: " + res.ToString());
            Plans.Remove(plan);
        }

        /// <summary>
        /// Gets the version of CUFFT (CUDA 5.0 only)
        /// </summary>
        /// <returns>Version of library or -1 if not supported or available.</returns>
        public override int GetVersion()
        {
            int version = -1;
            CUFFTResult res = _driver.cufftGetVersion(ref version);
            if (res != CUFFTResult.Success)
                version = -1;
            return version;
        }

        /// <summary>
        /// Configures the layout of CUFFT output in FFTW‐compatible modes.
        /// When FFTW compatibility is desired, it can be configured for padding
        /// only, for asymmetric complex inputs only, or to be fully compatible.
        /// </summary>
        /// <param name="plan">The plan.</param>
        /// <param name="mode">The mode.</param>
        public override void SetCompatibilityMode(FFTPlan plan, eCompatibilityMode mode)
        {
            CUFFTCompatibility cumode = (CUFFTCompatibility)mode;
            FFTPlanEx planEx = Plans[plan];
            CUFFTResult res = _driver.cufftSetCompatibilityMode(planEx.CudaFFTHandle, cumode);
            if (res != CUFFTResult.Success)
                throw new CudafyHostException(res.ToString());
        }
    }     
}
