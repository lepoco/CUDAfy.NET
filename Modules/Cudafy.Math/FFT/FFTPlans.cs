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
using System.Diagnostics;
using GASS.CUDA.FFT;
using GASS.CUDA.FFT.Types;

namespace Cudafy.Maths.FFT
{
    /// <summary>
    /// Abstract base class for FFT plans.
    /// </summary>
    public abstract class FFTPlan : IDisposable
    {
        internal FFTPlan(int batch, GPGPUFFT gpuFft)
        {
            BatchSize = batch;
            GPUFFT = gpuFft;
            _lock = new object();
        }

        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="FFTPlan"/> is reclaimed by garbage collection.
        /// </summary>
        ~FFTPlan()
        {
            //Dispose(false);
        }

        // Lock
        private object _lock;

        // Track whether Dispose has been called.
        private bool _disposed = false;

        /// <summary>
        /// Gets a value indicating whether this instance is disposed.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if this instance is disposed; otherwise, <c>false</c>.
        /// </value>
        public bool IsDisposed
        {
            get { lock (_lock) { return _disposed; } }
        }

        /// <summary>
        /// Gets or sets the size of the batch.
        /// </summary>
        /// <value>The size of the batch.</value>
        public int BatchSize { get; protected set; }
        internal GPGPUFFT GPUFFT { get; private set; }
        /// <summary>
        /// Gets the length when overridden.
        /// </summary>
        /// <value>The length.</value>
        public abstract int Length { get; }

        ///// <summary>
        ///// Destroys this instance.
        ///// </summary>
        //public abstract void Destroy();

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
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

        /// <summary>
        /// Releases unmanaged and - optionally - managed resources
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            lock (_lock)
            {
                Debug.WriteLine(string.Format("FFTPlan::Dispose({0})", disposing));
                // Check to see if Dispose has already been called.
                if (!this._disposed)
                {
                    Debug.WriteLine("Disposing");
                    // If disposing equals true, dispose all managed
                    // and unmanaged resources.
                    if (disposing)
                    {
                        // Dispose managed resources.
                        GPUFFT.Remove(this);
                    }

                    // Call the appropriate methods to clean up
                    // unmanaged resources here.
                    // If disposing is false,
                    // only the following code is executed.
                    //Destroy();
                    

                    // Note disposing has been done.
                    _disposed = true;

                }
                else
                    Debug.WriteLine("FFTPlan already disposed");
            }
        }

        //void IDisposable.Dispose()
        //{
        //    throw new NotImplementedException();
        //}
    }

    /// <summary>
    /// Represents a 1D FFT plan.
    /// </summary>
    public class FFTPlan1D : FFTPlan
    {
        internal FFTPlan1D(int nx, int batch, GPGPUFFT gpuFft)
            : base(batch, gpuFft)
        {
            XSize = nx;
        }

        /// <summary>
        /// Gets or sets the size of the X dimension.
        /// </summary>
        /// <value>The size of the X dimension.</value>
        public int XSize { get; protected set; }
        /// <summary>
        /// Executes the FFT.
        /// </summary>
        /// <typeparam name="T">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <typeparam name="U">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <param name="inverse">if set to <c>true</c> inverse.</param>
        public virtual void Execute<T,U>(T[] input, U[] output, bool inverse = false)
        {
            GPUFFT.Execute(this, input, output, inverse);
        }

        /// <summary>
        /// Gets the length (XSize).
        /// </summary>
        /// <value>The length.</value>
        public override int Length
        {
            get { return XSize; }
        }

        /// <summary>
        /// Configures the layout of CUFFT output in FFTW‐compatible modes.
        /// When FFTW compatibility is desired, it can be configured for padding
        /// only, for asymmetric complex inputs only, or to be fully compatible.
        /// </summary>
        /// <param name="mode">The mode.</param>
        public void SetCompatibilityMode(eCompatibilityMode mode)
        {
            GPUFFT.SetCompatibilityMode(this, mode);
        }
    }

    /// <summary>
    /// Represents a 2D FFT plan.
    /// </summary>
    public class FFTPlan2D : FFTPlan1D
    {
        internal FFTPlan2D(int nx, int ny, int batch, GPGPUFFT gpuFft)
            : base(nx, batch, gpuFft)
        {
            YSize = ny;
        }

        /// <summary>
        /// Gets or sets the size of the Y dimension.
        /// </summary>
        /// <value>The size of the Y dimension.</value>
        public int YSize { get; protected set; }

        /// <summary>
        /// Executes the FFT.
        /// </summary>
        /// <typeparam name="T">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <typeparam name="U">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <param name="inverse">if set to <c>true</c> inverse.</param>
        public void Execute<T,U>(T[,] input, U[,] output, bool inverse = false)
        {
            GPUFFT.Execute(this, input, output, inverse);
        }

        /// <summary>
        /// Gets the length (XSize * YSize).
        /// </summary>
        /// <value>The length.</value>
        public override int Length
        {
            get { return XSize * YSize; }
        }
    }

    /// <summary>
    /// Represents a 3D FFT plan.
    /// </summary>
    public class FFTPlan3D : FFTPlan2D
    {
        internal FFTPlan3D(int nx, int ny, int nz, int batch, GPGPUFFT gpuFft)
            : base(nx, ny, batch, gpuFft)
        {
            ZSize = nz;
        }

        /// <summary>
        /// Executes the specified input.
        /// </summary>
        /// <typeparam name="T">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <typeparam name="U">Data format: Double, Single, ComplexD or ComplexF.</typeparam>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <param name="inverse">if set to <c>true</c> [inverse].</param>
        public void Execute<T, U>(T[,,] input, U[,,] output, bool inverse = false)
        {
            GPUFFT.Execute(this, input, output, inverse);
        }

        /// <summary>
        /// Gets or sets the size of the Z dimension.
        /// </summary>
        /// <value>The size of the Z dimension.</value>
        public int ZSize { get; internal set; }

        /// <summary>
        /// Gets the length (XSize * YSize * ZSize).
        /// </summary>
        /// <value>The length.</value>
        public override int Length
        {
            get { return XSize * YSize * ZSize; }
        }
    }

    internal abstract class FFTPlanEx
    {
        public cufftHandle CudaFFTHandle { get; set; }
        public CUFFTType CudaFFTType { get; set; }
        public Ifftw_plan FFTWFwdPlan { get; set; }
        public Ifftw_plan FFTWInvPlan { get; set; }
        public int N { get; set; }
        public eDataType DataType { get; set; }
    }

    internal class FFTPlan1DEx : FFTPlanEx
    {
        public FFTPlan1DEx(FFTPlan1D plan)
        {
            Plan = plan;
        }

        public FFTPlan1D Plan { get; set; }

    }

    internal class FFTPlan2DEx : FFTPlanEx
    {
        public FFTPlan2DEx(FFTPlan2D plan)
        {
            Plan = plan;
        }

        public FFTPlan1D Plan { get; set; }
    }

    internal class FFTPlan3DEx : FFTPlanEx
    {
        public FFTPlan3DEx(FFTPlan3D plan)
        {
            Plan = plan;
        }

        public FFTPlan3D Plan { get; set; }
    }
}
