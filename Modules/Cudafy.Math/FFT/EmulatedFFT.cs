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

using Cudafy.Host;
using GASS.CUDA.FFT;
using GASS.CUDA.FFT.Types;
namespace Cudafy.Maths.FFT
{

    /// <summary>
    /// Implements emulation of GPU FFT library.
    /// </summary>
    internal class HostFFT : GPGPUFFT
    {

        internal HostFFT(EmulatedGPU gpu)
        {
            _gpu = gpu;
        }

        /// <summary>
        /// Creates a 1D plan.
        /// </summary>
        /// <param name="fftType">Type of FFT.</param>
        /// <param name="dataType">The data type.</param>
        /// <param name="nx">The length in samples.</param>
        /// <param name="batch">The number of FFTs in batch.</param>
        /// <returns>
        /// Plan.
        /// </returns>
        public override FFTPlan1D Plan1D(eFFTType fftType, eDataType dataType, int nx, int batch)
        {
            int insize, outsize;
            CUFFTType cuFFTType = VerifyTypes(fftType, dataType, out insize, out outsize);
            //Console.WriteLine(size);
            IntPtr pin = fftwf.malloc(nx * insize * batch);
            IntPtr pout = fftwf.malloc(nx * outsize * batch);

            Ifftw_plan fwdPlan;
            Ifftw_plan invPlan;

            if (dataType == eDataType.Single)
            {
                if (batch == 1)
                {
                    fwdPlan = fftwf_plan.dft_1d(fftType, nx, pin, pout, fftw_direction.Forward, fftw_flags.Estimate);
                    invPlan = fftwf_plan.dft_1d(fftType, nx, pin, pout, fftw_direction.Backward, fftw_flags.Estimate);
                }
                else
                {
                    fwdPlan = fftwf_plan.dft_many(fftType, 1, new int[] { nx }, batch,
                        pin, null, 1, nx,
                        pout, null, 1, nx, fftw_direction.Forward, fftw_flags.Estimate);
                    invPlan = fftwf_plan.dft_many(fftType, 1, new int[] { nx }, batch,
                        pin, null, 1, nx,
                        pout, null, 1, nx, fftw_direction.Backward, fftw_flags.Estimate);
                }
            }
            else
            {
                if (batch == 1)
                {
                    fwdPlan = fftw_plan.dft_1d(fftType, nx, pin, pout, fftw_direction.Forward, fftw_flags.Estimate);
                    invPlan = fftw_plan.dft_1d(fftType, nx, pin, pout, fftw_direction.Backward, fftw_flags.Estimate);
                }
                else
                {
                    fwdPlan = fftw_plan.dft_many(fftType, 1, new int[] { nx }, batch,
                        pin, null, 1, nx,
                        pout, null, 1, nx, fftw_direction.Forward, fftw_flags.Estimate);
                    invPlan = fftw_plan.dft_many(fftType, 1, new int[] { nx }, batch,
                        pin, null, 1, nx,
                        pout, null, 1, nx, fftw_direction.Backward, fftw_flags.Estimate);
                }
            }
         
            FFTPlan1D plan = new FFTPlan1D(nx, batch, this);
            FFTPlan1DEx planEx = new FFTPlan1DEx(plan) { FFTWFwdPlan = fwdPlan, FFTWInvPlan = invPlan, N = nx, DataType = dataType }; 
            Plans.Add(plan, planEx);
            return plan;
        }

        /// <summary>
        /// Creates a 2D plan.
        /// </summary>
        /// <param name="fftType">Type of FFT.</param>
        /// <param name="dataType">The data type.</param>
        /// <param name="nx">The x length in samples.</param>
        /// <param name="ny">The y length in samples.</param>
        /// <param name="batch">The number of FFTs in batch.</param>
        /// <returns>
        /// Plan.
        /// </returns>
        public override FFTPlan2D Plan2D(eFFTType fftType, eDataType dataType, int nx, int ny, int batch)
        {
            int insize, outsize;
            int totalSize = nx * ny;
            CUFFTType cuFFTType = VerifyTypes(fftType, dataType, out insize, out outsize);
            IntPtr pin = fftwf.malloc(totalSize * insize * batch);
            IntPtr pout = fftwf.malloc(totalSize * outsize * batch);

            Ifftw_plan fwdPlan;
            Ifftw_plan invPlan;

            if (dataType == eDataType.Single)
            {
                if (batch == 1)
                {
                    fwdPlan = fftwf_plan.dft_2d(fftType, nx, ny, pin, pout, fftw_direction.Forward, fftw_flags.Estimate);
                    invPlan = fftwf_plan.dft_2d(fftType, nx, ny, pin, pout, fftw_direction.Backward, fftw_flags.Estimate);
                }                                
                else
                {
                    fwdPlan = fftwf_plan.dft_many(fftType, 2, new int[] { nx, ny }, batch,
                        pin, null, 1, nx * ny,
                        pout, null, 1, nx * ny, fftw_direction.Forward, fftw_flags.Estimate);
                    invPlan = fftwf_plan.dft_many(fftType, 2, new int[] { nx, ny }, batch,
                        pin, null, 1, nx * ny,
                        pout, null, 1, nx * ny, fftw_direction.Backward, fftw_flags.Estimate);
                }
            }
            else
            {
                if (batch == 1)
                {
                    fwdPlan = fftw_plan.dft_2d(fftType, nx, ny, pin, pout, fftw_direction.Forward, fftw_flags.Estimate);
                    invPlan = fftw_plan.dft_2d(fftType, nx, ny, pin, pout, fftw_direction.Backward, fftw_flags.Estimate);
                }
                else
                {
                    fwdPlan = fftw_plan.dft_many(fftType, 2, new int[] { nx, ny }, batch,
                        pin, null, 1, nx * ny,
                        pout, null, 1, nx * ny, fftw_direction.Forward, fftw_flags.Estimate);
                    invPlan = fftw_plan.dft_many(fftType, 2, new int[] { nx, ny }, batch,
                        pin, null, 1, nx * ny,
                        pout, null, 1, nx * ny, fftw_direction.Backward, fftw_flags.Estimate);
                }
            }

            FFTPlan2D plan = new FFTPlan2D(nx, ny, batch, this);
            FFTPlan2DEx planEx = new FFTPlan2DEx(plan) { FFTWFwdPlan = fwdPlan, FFTWInvPlan = invPlan, N = totalSize, DataType = dataType };
            Plans.Add(plan, planEx);
            return plan;
        }

        /// <summary>
        /// Creates a 3D plan.
        /// </summary>
        /// <param name="fftType">Type of FFT.</param>
        /// <param name="dataType">The data type.</param>
        /// <param name="nx">The x length in samples.</param>
        /// <param name="ny">The y length in samples.</param>
        /// <param name="nz">The z length in samples.</param>
        /// <param name="batch">The number of FFTs in batch.</param>
        /// <returns>
        /// Plan.
        /// </returns>
        public override FFTPlan3D Plan3D(eFFTType fftType, eDataType dataType, int nx, int ny, int nz, int batch = 1)
        {
            int insize, outsize;
            int totalSize = nx * ny * nz;
            CUFFTType cuFFTType = VerifyTypes(fftType, dataType, out insize, out outsize);
            //Console.WriteLine(size);
            IntPtr pin = fftwf.malloc(totalSize * insize * batch);
            IntPtr pout = fftwf.malloc(totalSize * outsize * batch);

            Ifftw_plan fwdPlan;
            Ifftw_plan invPlan;
            if (dataType == eDataType.Single)
            {
                if (batch == 1)
                {
                    fwdPlan = fftwf_plan.dft_3d(fftType, nx, ny, nz, pin, pout, fftw_direction.Forward, fftw_flags.Estimate);
                    invPlan = fftwf_plan.dft_3d(fftType, nx, ny, nz, pin, pout, fftw_direction.Backward, fftw_flags.Estimate);
                }
                else
                {
                    fwdPlan = fftwf_plan.dft_many(fftType, 3, new int[] { nx, ny, nz }, batch,
                        pin, null, 1, nx * ny * nz,
                        pout, null, 1, nx * ny * nz, fftw_direction.Forward, fftw_flags.Estimate);
                    invPlan = fftwf_plan.dft_many(fftType, 3, new int[] { nx, ny, nz }, batch,
                        pin, null, 1, nx * ny * nz,
                        pout, null, 1, nx * ny * nz, fftw_direction.Backward, fftw_flags.Estimate);
                }
            }
            else
            {
                if (batch == 1)
                {
                    fwdPlan = fftw_plan.dft_3d(fftType, nx, ny, nz, pin, pout, fftw_direction.Forward, fftw_flags.Estimate);
                    invPlan = fftw_plan.dft_3d(fftType, nx, ny, nz, pin, pout, fftw_direction.Backward, fftw_flags.Estimate);
                }
                else
                {
                    fwdPlan = fftw_plan.dft_many(fftType, 3, new int[] { nx, ny, nz }, batch,
                        pin, null, 1, nx * ny * nz,
                        pout, null, 1, nx * ny * nz, fftw_direction.Forward, fftw_flags.Estimate);
                    invPlan = fftw_plan.dft_many(fftType, 3, new int[] { nx, ny, nz }, batch,
                        pin, null, 1, nx * ny * nz,
                        pout, null, 1, nx * ny * nz, fftw_direction.Backward, fftw_flags.Estimate);
                }
            }

            FFTPlan3D plan = new FFTPlan3D(nx, ny, nz, batch, this);
            FFTPlan3DEx planEx = new FFTPlan3DEx(plan) { FFTWFwdPlan = fwdPlan, FFTWInvPlan = invPlan, N = totalSize, DataType = dataType };
            Plans.Add(plan, planEx);
            return plan;
        }

        private void DoExecute<T, U>(FFTPlan plan, Array input, Array output, bool inverse = false)
        {
            //_gpu.VerifyOnGPU(input);
            //_gpu.VerifyOnGPU(output);
            EmuDevicePtrEx inputPtr = _gpu.GetDeviceMemory(input) as EmuDevicePtrEx;
            EmuDevicePtrEx outputPtr = _gpu.GetDeviceMemory(output) as EmuDevicePtrEx;
            FFTPlanEx planEx = Plans[plan];
            Ifftw_plan fftwplan = inverse ? planEx.FFTWInvPlan : planEx.FFTWFwdPlan;
            int insize = Marshal.SizeOf(typeof(T));
            int inoffset = inputPtr.OffsetBytes;
            int outoffset = outputPtr.OffsetBytes;
            int outsize = Marshal.SizeOf(typeof(U));
            int batchSize = plan.BatchSize;
            int planLength = plan.Length;
            int N = planEx.N;
            unsafe
            {
                GCHandle inhandle = new GCHandle();
                GCHandle outhandle = new GCHandle();
                try
                {
                    inhandle = GCHandle.Alloc(inputPtr.DevPtr, GCHandleType.Pinned);
                    outhandle = GCHandle.Alloc(outputPtr.DevPtr, GCHandleType.Pinned);

                    long srcAddress = inhandle.AddrOfPinnedObject().ToInt64();
                    long dstAddress = outhandle.AddrOfPinnedObject().ToInt64();

                    int srcOffset = 0;
                    int dstOffset = 0;
                    //for (int b = 0; b < batchSize; b++)
                    {
                        IntPtr srcOffsetPtr = new IntPtr(srcAddress + inoffset + (srcOffset * insize));
                        GPGPU.CopyMemory(fftwplan.Input, srcOffsetPtr, (uint)(N * insize * batchSize));

                        fftwplan.Execute();
                        IntPtr dstIntPtrOffset = new IntPtr(dstAddress + outoffset + (dstOffset * outsize));
                        GPGPU.CopyMemory(dstIntPtrOffset, fftwplan.Output, (uint)(N * outsize * batchSize));

                        //srcOffset += planLength;
                       // dstOffset += planLength;
                    }
                }
                finally
                {
                    inhandle.Free();
                    outhandle.Free();
                }
            }
        }

        /// <summary>
        /// Executes the specified plan.
        /// </summary>
        /// <typeparam name="T">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <typeparam name="U">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <param name="plan">The plan.</param>
        /// <param name="input">The input data.</param>
        /// <param name="output">The output data.</param>
        /// <param name="inverse">if set to <c>true</c> inverse.</param>
        public override void Execute<T,U>(FFTPlan plan, T[] input, U[] output, bool inverse = false)
        {
            DoExecute<T, U>(plan, input, output, inverse);
        }

        /// <summary>
        /// Executes the specified plan.
        /// </summary>
        /// <typeparam name="T">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <typeparam name="U">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <param name="plan">The plan.</param>
        /// <param name="input">The input data.</param>
        /// <param name="output">The output data.</param>
        /// <param name="inverse">if set to <c>true</c> inverse.</param>
        public override void Execute<T,U>(FFTPlan plan, T[,] input, U[,] output, bool inverse = false)
        {
            DoExecute<T, U>(plan, input, output, inverse);
        }

        /// <summary>
        /// Executes the specified plan.
        /// </summary>
        /// <typeparam name="T">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <typeparam name="U">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <param name="plan">The plan.</param>
        /// <param name="input">The input data.</param>
        /// <param name="output">The output data.</param>
        /// <param name="inverse">if set to <c>true</c> inverse.</param>
        public override void Execute<T,U>(FFTPlan plan, T[, ,] input, U[, ,] output, bool inverse = false)
        {
            DoExecute<T, U>(plan, input, output, inverse);
        }

        //[DllImport("kernel32.dll", EntryPoint = "RtlMoveMemory")]
        //private static extern void CopyMemory(IntPtr Destination, IntPtr Source, uint Length);

        /// <summary>
        /// Frees the specified plan.
        /// </summary>
        /// <param name="plan">The plan.</param>
        public override void Remove(FFTPlan plan)
        {
            FFTPlanEx planEx = Plans[plan];
            if (planEx.DataType == eDataType.Single)
            {
                fftwf.destroy_plan(planEx.FFTWFwdPlan.Handle);
                fftwf.destroy_plan(planEx.FFTWInvPlan.Handle);
            }
            else if (planEx.DataType == eDataType.Double)
            {
                fftw.destroy_plan(planEx.FFTWFwdPlan.Handle);
                fftw.destroy_plan(planEx.FFTWInvPlan.Handle);
            }
            Plans.Remove(plan);
        }

        /// <summary>
        /// Creates a 1D plan.
        /// </summary>
        /// <param name="fftType">Type of FFT.</param>
        /// <param name="dataType">The data type.</param>
        /// <param name="nx">The length in samples.</param>
        /// <param name="batch">The number of FFTs in batch.</param>
        /// <param name="istride">The istride.</param>
        /// <param name="idist">The idist.</param>
        /// <param name="ostride">The ostride.</param>
        /// <param name="odist">The odist.</param>
        /// <returns>Plan.</returns>
        public override FFTPlan1D Plan1D(eFFTType fftType, eDataType dataType, int nx, int batch, int istride, int idist, int ostride, int odist)
        {
            int insize, outsize;
            CUFFTType cuFFTType = VerifyTypes(fftType, dataType, out insize, out outsize);
            //Console.WriteLine(size);
            IntPtr pin = fftwf.malloc(nx * insize * batch);
            IntPtr pout = fftwf.malloc(nx * outsize * batch);

            Ifftw_plan fwdPlan;
            Ifftw_plan invPlan;

            if (dataType == eDataType.Single)
            {
                fwdPlan = fftwf_plan.dft_many(fftType, 1, new int[] { nx }, batch,
                    pin, null, istride, idist,
                    pout, null, ostride, odist, fftw_direction.Forward, fftw_flags.Estimate);
                invPlan = fftwf_plan.dft_many(fftType, 1, new int[] { nx }, batch,
                    pin, null, istride, idist,
                    pout, null, ostride, odist, fftw_direction.Backward, fftw_flags.Estimate);
            }
            else
            {
                fwdPlan = fftw_plan.dft_many(fftType, 1, new int[] { nx }, batch,
                    pin, null, istride, idist,
                    pout, null, ostride, odist, fftw_direction.Forward, fftw_flags.Estimate);
                invPlan = fftw_plan.dft_many(fftType, 1, new int[] { nx }, batch,
                    pin, null, istride, idist,
                    pout, null, ostride, odist, fftw_direction.Backward, fftw_flags.Estimate);
            }

            FFTPlan1D plan = new FFTPlan1D(nx, batch, this);
            FFTPlan1DEx planEx = new FFTPlan1DEx(plan) { FFTWFwdPlan = fwdPlan, FFTWInvPlan = invPlan, N = nx, DataType = dataType };
            Plans.Add(plan, planEx);
            return plan;
        }
    }
}
