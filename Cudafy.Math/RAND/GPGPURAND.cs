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
using Cudafy.Host;

namespace Cudafy.Maths.RAND
{
    /// <summary>
    /// RAND wrapper for Cuda GPUs.
    /// </summary>
    public abstract class GPGPURAND
    {
        internal GPGPURAND()
        {
            _lock = new object();
        }

        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="GPGPURAND"/> is reclaimed by garbage collection.
        /// </summary>
        ~GPGPURAND()
        {
            Dispose(false);
        }

        /// <summary>
        /// Releases unmanaged and - optionally - managed resources
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            // This object will be cleaned up by the Dispose method.
            // Therefore, you should call GC.SupressFinalize to
            // take this object off the finalization queue
            // and prevent finalization code for this object
            // from executing a second time.
            GC.SuppressFinalize(this);
        }

        private object _lock;

        // Track whether Dispose has been called.
        private bool _disposed = false;

        /// <summary>
        /// Releases unmanaged and - optionally - managed resources
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            lock (_lock)
            {
                Debug.WriteLine(string.Format("GPGPURAND::Dispose({0})", disposing));
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
                    Shutdown();

                    // Note disposing has been done.
                    _disposed = true;

                }
                else
                    Debug.WriteLine("Already disposed");
            }
        }

        /// <summary>
        /// Creates an instance based on the specified gpu with pseudo random generator.
        /// </summary>
        /// <param name="gpu">The gpu.</param>
        /// <param name="host">if set to <c>true</c> the uses generator on the host (if applicable).</param>
        /// <returns>New instance.</returns>
        public static GPGPURAND Create(GPGPU gpu, bool host = false)
        {
            return Create(gpu, curandRngType.CURAND_RNG_PSEUDO_DEFAULT, host);
        }

        /// <summary>
        /// Creates an instance based on the specified gpu.
        /// </summary>
        /// <param name="gpu">The gpu.</param>
        /// <param name="rng_type">The type of generator.</param>
        /// <param name="host">if set to <c>true</c> the uses generator on the host (if applicable).</param>
        /// <returns>New instance.</returns>
        public static GPGPURAND Create(GPGPU gpu, curandRngType rng_type, bool host = false)
        {
            GPGPURAND rand;
            if (!host && gpu is CudaGPU)
                rand = new CudaDeviceRAND(gpu, rng_type);
            else if (gpu is CudaGPU)
                rand = new CudaHostRAND(gpu, rng_type);
            else
                rand = new HostRAND(gpu, rng_type);
          
            return rand;
        }

        /// <summary>
        /// Shutdowns this instance.
        /// </summary>
        protected abstract void Shutdown();

        /// <summary>
        /// Sets the pseudo random generator seed.
        /// </summary>
        /// <param name="seed">The seed.</param>
        public abstract void SetPseudoRandomGeneratorSeed(ulong seed);

        /// <summary>
        /// Generates random data.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <param name="n">Count</param>
        public abstract void GenerateUniform(float[] array, int n = 0);

        /// <summary>
        /// Generates random data.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <param name="n">Count</param>
        public abstract void GenerateUniform(double[] array, int n = 0);

        /// <summary>
        /// Generates random data.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <param name="n">Count</param>
        public abstract void Generate(uint[] array, int n = 0);

        /// <summary>
        /// Generates random data.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stddev">The stddev.</param>
        /// <param name="n">Count</param>
        public abstract void GenerateLogNormal(float[] array, float mean, float stddev, int n = 0);

        /// <summary>
        /// Generates random data.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stddev">The stddev.</param>
        /// <param name="n">Count</param>
        public abstract void GenerateLogNormal(double[] array, double mean, double stddev, int n = 0);

        /// <summary>
        /// Generates random data.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <param name="n">Count</param>
        public abstract void Generate(ulong[] array, int n = 0);

        /// <summary>
        /// Generates random data.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stddev">The stddev.</param>
        /// <param name="n">Count</param>
        public abstract void GenerateNormal(float[] array, float mean, float stddev, int n = 0);

        /// <summary>
        /// Generates random data.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stddev">The stddev.</param>
        /// <param name="n">Count</param>
        public abstract void GenerateNormal(double[] array, float mean, float stddev, int n = 0);

        /// <summary>
        /// Generates seeds.
        /// </summary>
        public abstract void GenerateSeeds();

        /// <summary>
        /// Gets the direction vectors for 32-bit.
        /// </summary>
        /// <param name="set">The set.</param>
        /// <returns></returns>
        public abstract RandDirectionVectors32 GetDirectionVectors32(curandDirectionVectorSet set);

        /// <summary>
        /// Gets the direction vectors for 64-bit.
        /// </summary>
        /// <param name="set">The set.</param>
        /// <returns></returns>
        public abstract RandDirectionVectors64 GetDirectionVectors64(curandDirectionVectorSet set);

        /// <summary>
        /// Gets the scramble constants for 32-bit.
        /// </summary>
        /// <param name="n">Count</param>
        /// <returns></returns>
        public abstract uint[] GetScrambleConstants32(int n);

        /// <summary>
        /// Gets the scramble constants for 64-bit.
        /// </summary>
        /// <param name="n">Count</param>
        /// <returns></returns>
        public abstract ulong[] GetScrambleConstants64(int n);

        /// <summary>
        /// Gets the version of library.
        /// </summary>
        /// <returns></returns>
        public abstract int GetVersion();

        /// <summary>
        /// Sets the generator offset.
        /// </summary>
        /// <param name="offset">The offset.</param>
        public abstract void SetGeneratorOffset(ulong offset);

        /// <summary>
        /// Sets the generator ordering.
        /// </summary>
        /// <param name="order">The order.</param>
        public abstract void SetGeneratorOrdering(curandOrdering order);

        /// <summary>
        /// Sets the quasi random generator dimensions.
        /// </summary>
        /// <param name="num_dimensions">The num_dimensions.</param>
        public abstract void SetQuasiRandomGeneratorDimensions(uint num_dimensions);

        /// <summary>
        /// Sets the stream.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        public abstract void SetStream(int streamId);

    }
}
