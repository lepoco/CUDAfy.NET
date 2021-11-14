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
using Cudafy.Host;
using GASS.CUDA.Types;
namespace Cudafy.Maths.RAND
{
    internal abstract class CudaRAND : GPGPURAND
    {
        internal CudaRAND(GPGPU gpu, curandRngType rng_type)
        {
            _gpu = gpu;
            if (IntPtr.Size == 8)
            {
                _driver = new CURANDDriver64();
            }
            else
            {
                throw new NotSupportedException();
                //_driver = new CURANDDriver32();
            }
        }


        protected override void Shutdown()
        {
            try
            {
                SafeCall(_driver.DestroyGenerator(_gen));
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                throw;
            }
        }

        protected RandGenerator _gen;

        protected GPGPU _gpu;

        protected ICURANDDriver _driver;

        protected void SafeCall(curandStatus status, DevicePtrEx ptrEx = null)
        {
            if(ptrEx != null)
                Free(ptrEx);
            if (status != curandStatus.CURAND_STATUS_SUCCESS)
                throw new CudafyMathException(CudafyMathException.csRAND_ERROR_X, status.ToString());
        }

        protected abstract DevicePtrEx GetDevicePtr(Array array, ref int n);

        protected abstract void Free(DevicePtrEx ptrEx);

        public override void SetPseudoRandomGeneratorSeed(ulong seed)
        {
            SafeCall(_driver.SetPseudoRandomGeneratorSeed(_gen, seed));
        }

        public override void GenerateUniform(float[] array, int n = 0)
        {
            DevicePtrEx ptrEx = GetDevicePtr(array, ref n);
            SafeCall(_driver.GenerateUniform(_gen, ptrEx.Pointer, n), ptrEx);
        }

        public override void GenerateUniform(double[] array, int n = 0)
        {
            DevicePtrEx ptrEx = GetDevicePtr(array, ref n);
            SafeCall(_driver.GenerateUniformDouble(_gen, ptrEx.Pointer, n), ptrEx);
        }

        public override void Generate(uint[] array, int n = 0)
        {
            DevicePtrEx ptrEx = GetDevicePtr(array, ref n);
            SafeCall(_driver.Generate(_gen, ptrEx.Pointer, n), ptrEx);
        }

        public override void GenerateLogNormal(float[] array, float mean, float stddev, int n = 0)
        {
            DevicePtrEx ptrEx = GetDevicePtr(array, ref n);
            SafeCall(_driver.GenerateLogNormal(_gen, ptrEx.Pointer, n, mean, stddev), ptrEx);
        }

        public override void GenerateLogNormal(double[] array, double mean, double stddev, int n = 0)
        {
            DevicePtrEx ptrEx = GetDevicePtr(array, ref n);
            SafeCall(_driver.GenerateLogNormalDouble(_gen, ptrEx.Pointer, n, mean, stddev), ptrEx);
        }

        public override void Generate(ulong[] array, int n = 0)
        {
            DevicePtrEx ptrEx = GetDevicePtr(array, ref n);
            SafeCall(_driver.Generate(_gen, ptrEx.Pointer, n), ptrEx);
        }

        public override void GenerateNormal(float[] array, float mean, float stddev, int n = 0)
        {
            DevicePtrEx ptrEx = GetDevicePtr(array, ref n);
            SafeCall(_driver.GenerateNormal(_gen, ptrEx.Pointer, n, mean, stddev), ptrEx);
        }

        public override void GenerateNormal(double[] array, float mean, float stddev, int n = 0)
        {
            DevicePtrEx ptrEx = GetDevicePtr(array, ref n);
            SafeCall(_driver.GenerateNormalDouble(_gen, ptrEx.Pointer, n, mean, stddev), ptrEx);
        }

        public override void GenerateSeeds()
        {
            SafeCall(_driver.GenerateSeeds(_gen));
        }

        public override RandDirectionVectors32 GetDirectionVectors32(curandDirectionVectorSet set)
        {
            RandDirectionVectors32 vectors = new RandDirectionVectors32();
            SafeCall(_driver.GetDirectionVectors32(ref vectors, set));
            return vectors;
        }

        public override RandDirectionVectors64 GetDirectionVectors64(curandDirectionVectorSet set)
        {
            RandDirectionVectors64 vectors = new RandDirectionVectors64();
            SafeCall(_driver.GetDirectionVectors64(ref vectors, set));
            return vectors;
        }

        /// <summary>
        /// Copies memory.
        /// </summary>
        /// <param name="Destination">The destination.</param>
        /// <param name="Source">The source.</param>
        /// <param name="Length">The length.</param>
        [DllImport("kernel32.dll", EntryPoint = "RtlMoveMemory")]
        private static extern void CopyMemory(IntPtr Destination, IntPtr Source, uint Length);

        public override uint[] GetScrambleConstants32(int n)
        {
            IntPtr ptr = IntPtr.Zero;
            SafeCall(_driver.GetScrambleConstants32(ref ptr));
            uint[] constants = new uint[n];
            GCHandle handle = GCHandle.Alloc(constants, GCHandleType.Pinned);
            CopyMemory(handle.AddrOfPinnedObject(), ptr, (uint)n * sizeof(uint));
            handle.Free();
            return constants;
            
        }

        public override ulong[] GetScrambleConstants64(int n)
        {
            IntPtr ptr = IntPtr.Zero;
            SafeCall(_driver.GetScrambleConstants64(ref ptr));
            ulong[] constants = new ulong[n];
            GCHandle handle = GCHandle.Alloc(constants, GCHandleType.Pinned);
            CopyMemory(handle.AddrOfPinnedObject(), ptr, (uint)n * sizeof(ulong));
            handle.Free();
            return constants;
        }

        public override int GetVersion()
        {
            int version = 0;
            SafeCall(_driver.GetVersion(ref version));
            return version;
        }

        public override void SetGeneratorOffset(ulong offset)
        {
            SafeCall(_driver.SetGeneratorOffset(_gen, offset));
        }

        public override void SetGeneratorOrdering(curandOrdering order)
        {
            SafeCall(_driver.SetGeneratorOrdering(_gen, order));
        }

        public override void SetQuasiRandomGeneratorDimensions(uint num_dimensions)
        {
            SafeCall(_driver.SetQuasiRandomGeneratorDimensions(_gen, num_dimensions));
        }

        public override void SetStream(int streamId)
        {
            if (streamId < 0)
                throw new ArgumentOutOfRangeException("streamId");
            CUstream cus = (CUstream)_gpu.GetStream(streamId);
            SafeCall(_driver.SetStream(_gen, cus)); 
        }
    }
}
