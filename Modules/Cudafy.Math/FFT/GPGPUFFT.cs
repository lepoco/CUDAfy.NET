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
using Cudafy.Types;
using Cudafy.Host;

using GASS.CUDA.FFT;
using GASS.CUDA.FFT.Types;
namespace Cudafy.Maths.FFT
{
    /// <summary>
    /// FFT Type enumeration.
    /// </summary>
    public enum eFFTType 
    {
        /// <summary>
        /// Real to complex.
        /// </summary>
        Real2Complex,
        /// <summary>
        /// Complex to real.
        /// </summary>
        Complex2Real,
        /// <summary>
        /// Complex to complex.
        /// </summary>
        Complex2Complex 
    };

    /// <summary>
    /// Data type enumeration.
    /// </summary>
    public enum eDataType 
    {
        /// <summary>
        /// Double floating point.
        /// </summary>
        Double,
        /// <summary>
        /// Single floating point.
        /// </summary>
        Single 
    };

    /// <summary>
    /// FFTW compatibility mode.
    /// </summary>
    [Flags]
    public enum eCompatibilityMode
    {
        /// <summary>
        /// </summary>
        Native = 0,
        /// <summary>
        ///      Inserts extra padding between packed in-place transforms for
        ///      batched transforms with power-of-2 size. (default)
        /// </summary>
        FFTW_Padding = 1,
        /// <summary>
        ///      Guarantees FFTW-compatible output for non-symmetric complex inputs
        ///      for transforms with power-of-2 size. This is only useful for
        ///      artificial (i.e. random) datasets as actual data will always be
        ///      symmetric if it has come from the real plane. If you don't
        ///      understand what this means, you probably don't have to use it.
        /// </summary>
        FFTW_Asymmetric = 2,
        /// <summary>
        ///     For convenience, enables all FFTW compatibility modes at once.
        /// </summary>
        FFTW_All = 3
    }

    /// <summary>
    /// FFT wrapper.
    /// </summary>
    public abstract class GPGPUFFT
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GPGPUFFT"/> class.
        /// </summary>
        protected GPGPUFFT()
        {
            Plans = new Dictionary<FFTPlan, FFTPlanEx>();
        }

        /// <summary>
        /// GPU instance on which the FFT instance was made.
        /// </summary>
        protected GPGPU _gpu;

        internal Dictionary<FFTPlan, FFTPlanEx> Plans;


        /// <summary>
        /// Verifies the types.
        /// </summary>
        /// <param name="fftType">Type of the FFT.</param>
        /// <param name="dataType">Type of the data.</param>
        /// <param name="inSize">Size of input elements.</param>
        /// <param name="outSize">Size of output elements.</param>
        /// <returns>The CUFFTType.</returns>
        protected CUFFTType VerifyTypes(eFFTType fftType, eDataType dataType, out int inSize, out int outSize)
        {
            inSize = 8;
            outSize = 8;
            bool isDouble = dataType == eDataType.Double;

            CUFFTType cuFftType;
            if (fftType == eFFTType.Complex2Complex)
                cuFftType = isDouble ? CUFFTType.Z2Z : CUFFTType.C2C;
            else if (fftType == eFFTType.Complex2Real)
                cuFftType = isDouble ? CUFFTType.Z2D : CUFFTType.C2R;
            else //if (fftType == eFFTType.Real2Complex)
                cuFftType = isDouble ? CUFFTType.D2Z : CUFFTType.R2C;

            inSize = ((fftType == eFFTType.Complex2Complex || fftType == eFFTType.Complex2Real) ? 8 : 4) * (isDouble ? 2 : 1);
            outSize = ((fftType == eFFTType.Complex2Complex || fftType == eFFTType.Real2Complex) ? 8 : 4) * (isDouble ? 2 : 1);

            return cuFftType;
        }

        /// <summary>
        /// Sets the stream.
        /// </summary>
        /// <param name="plan">The plan to set the stream for.</param>
        /// <param name="streamId">The stream id.</param>
        public virtual void SetStream(FFTPlan plan, int streamId)
        {
            if (!Plans.ContainsKey(plan))
                throw new CudafyMathException(CudafyMathException.csPLAN_NOT_FOUND);
        }

        /// <summary>
        /// Creates a GPGPUFFT based on the supplied GPGPU instance (e.g. CudaFFT or EmulatedGPU).
        /// </summary>
        /// <param name="gpu">The gpu instance.</param>
        /// <returns></returns>
        public static GPGPUFFT Create(GPGPU gpu)
        {
            if (gpu is CudaGPU)
                return new CudaFFT(gpu as CudaGPU);
            else
                return new HostFFT(gpu as EmulatedGPU);
        }

        /// <summary>
        /// Frees the specified plan.
        /// </summary>
        /// <param name="plan">The plan.</param>
        public abstract void Remove(FFTPlan plan);

        /// <summary>
        /// Destroys all plans.
        /// </summary>
        public virtual void RemoveAll()
        {
            List<FFTPlan> plans = Plans.Keys.ToList();
            foreach (var v in plans)
                v.Dispose();
        }

        /// <summary>
        /// Creates a 1D plan.
        /// </summary>
        /// <param name="fftType">Type of FFT.</param>
        /// <param name="dataType">The data type.</param>
        /// <param name="nx">The length in samples.</param>
        /// <param name="batch">The number of FFTs in batch.</param>
        /// <returns>Plan.</returns>
        public abstract FFTPlan1D Plan1D(eFFTType fftType, eDataType dataType, int nx, int batch = 1);

        /// <summary>
        /// Plan1s the D.
        /// </summary>
        /// <param name="fftType">Type of the FFT.</param>
        /// <param name="dataType">Type of the data.</param>
        /// <param name="nx">The nx.</param>
        /// <param name="batchSize">Size of the batch.</param>
        /// <param name="istride">The istride.</param>
        /// <param name="idist">The idist.</param>
        /// <param name="ostride">The ostride.</param>
        /// <param name="odist">The odist.</param>
        /// <returns></returns>
        public abstract FFTPlan1D Plan1D(eFFTType fftType, eDataType dataType, int nx, int batchSize, int istride, int idist, int ostride, int odist);

        /// <summary>
        /// Creates a 2D plan.
        /// </summary>
        /// <param name="fftType">Type of FFT.</param>
        /// <param name="dataType">The data type.</param>
        /// <param name="nx">The x length in samples.</param>
        /// <param name="ny">The y length in samples.</param>
        /// <param name="batch">The number of FFTs in batch.</param>
        /// <returns>Plan.</returns>
        public abstract FFTPlan2D Plan2D(eFFTType fftType, eDataType dataType, int nx, int ny, int batch = 1);

        /// <summary>
        /// Creates a 3D plan.
        /// </summary>
        /// <param name="fftType">Type of FFT.</param>
        /// <param name="dataType">The data type.</param>
        /// <param name="nx">The x length in samples.</param>
        /// <param name="ny">The y length in samples.</param>
        /// <param name="nz">The z length in samples.</param>
        /// <param name="batch">The number of FFTs in batch.</param>
        /// <returns>Plan.</returns>
        public abstract FFTPlan3D Plan3D(eFFTType fftType, eDataType dataType, int nx, int ny, int nz, int batch = 1);

        /// <summary>
        /// Executes the specified plan.
        /// </summary>
        /// <typeparam name="T">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <typeparam name="U">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <param name="plan">The plan.</param>
        /// <param name="input">The input data.</param>
        /// <param name="output">The output data.</param>
        /// <param name="inverse">if set to <c>true</c> inverse.</param>
        public abstract void Execute<T,U>(FFTPlan plan, T[] input, U[] output, bool inverse = false);

        /// <summary>
        /// Executes the specified plan.
        /// </summary>
        /// <typeparam name="T">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <typeparam name="U">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <param name="plan">The plan.</param>
        /// <param name="input">The input data.</param>
        /// <param name="output">The output data.</param>
        /// <param name="inverse">if set to <c>true</c> inverse.</param>
        public abstract void Execute<T,U>(FFTPlan plan, T[,] input, U[,] output, bool inverse = false);

        /// <summary>
        /// Executes the specified plan.
        /// </summary>
        /// <typeparam name="T">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <typeparam name="U">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <param name="plan">The plan.</param>
        /// <param name="input">The input data.</param>
        /// <param name="output">The output data.</param>
        /// <param name="inverse">if set to <c>true</c> inverse.</param>
        public abstract void Execute<T,U>(FFTPlan plan, T[,,] input, U[,,] output, bool inverse = false);

        /// <summary>
        /// Gets the version of library wrapped by this library.
        /// </summary>
        /// <returns>Version of library or -1 if not supported or available.</returns>
        public virtual int GetVersion()
        {
            return -1;
        }


        /// <summary>
        /// Configures the layout of CUFFT output in FFTW‐compatible modes.
        /// When FFTW compatibility is desired, it can be configured for padding
        /// only, for asymmetric complex inputs only, or to be fully compatible.
        /// </summary>
        /// <param name="plan">The plan.</param>
        /// <param name="mode">The mode.</param>
        public virtual void SetCompatibilityMode(FFTPlan plan, eCompatibilityMode mode)
        {            
        }
    }
}
