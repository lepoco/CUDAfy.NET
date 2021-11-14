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
using Cudafy.Types;
using Cudafy.Maths.BLAS.Types;

using GASS.CUDA.BLAS;


namespace Cudafy.Maths.BLAS
{
    internal enum eDataType { S, C, D, Z };

    /// <summary>
    /// Abstract base class for devices supporting BLAS.
    /// Warning: This code has received limited testing.
    /// </summary>
    public abstract class GPGPUBLAS : IDisposable
    {
        /// <summary>
        /// Creates a BLAS wrapper based on the specified gpu.
        /// </summary>
        /// <param name="gpu">The gpu.</param>
        /// <returns></returns>
        public static GPGPUBLAS Create(GPGPU gpu)
        {
            if (gpu is CudaGPU)
                return new CudaBLAS(gpu);
            else
                return new HostBLAS(gpu);
                //throw new NotImplementedException(gpu.ToString());
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="GPGPUBLAS"/> class.
        /// </summary>
        protected GPGPUBLAS()
        {
            _lock = new object();
        }

        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="GPGPUBLAS"/> is reclaimed by garbage collection.
        /// </summary>
        ~GPGPUBLAS()
        {
            Dispose(false);
        }

        public abstract int GetVersion();

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
        /// Shutdowns this instance.
        /// </summary>
        protected abstract void Shutdown();

        /// <summary>
        /// Releases unmanaged and - optionally - managed resources
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            lock (_lock)
            {
                Debug.WriteLine(string.Format("GPGPUBLAS::Dispose({0})", disposing));
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

        #region Matrix Helper Functions
        public int GetIndexColumnMajor(int i, int j, int m)
        {
            return i + j * m;
        }

        public int GetIndexPackedSymmetric(int i, int j, int n, cublasFillMode fillMode)
        {
            if (fillMode == cublasFillMode.Lower)
            {
                if (i < j)
                {
                    throw new ArgumentOutOfRangeException("Please set i >= j in Lower fill mode.");
                }

                return i + ((2 * n - j - 1) * j) / 2;
            }
            else
            {
                if (i > j)
                {
                    throw new ArgumentOutOfRangeException("Please set i <= j in Upper fill mode.");
                }
                return i + (j * (j + 1)) / 2;
            }
        }
        #endregion

        #region BLAS Level 1

        #region Max

        /// <summary>
        /// IAMAXs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract int IAMAX(float[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// IAMAXs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract int IAMAX(double[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// IAMAXs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract int IAMAX(ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// IAMAXs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract int IAMAX(ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        #endregion

        #region Min

        /// <summary>
        /// IAMINs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract int IAMIN(float[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// IAMINs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract int IAMIN(double[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// IAMINs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract int IAMIN(ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// IAMINs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract int IAMIN(ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        #endregion

        #region Sum

        /// <summary>
        /// ASUMs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract float ASUM(float[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// ASUMs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract double ASUM(double[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// ASUMs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract float ASUM(ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// ASUMs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract double ASUM(ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1);
                   
        #endregion

        #region AXPY

        /// <summary>
        /// AXPYs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void AXPY(float alpha, float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            AXPY(new [] { alpha }, vectorx, vectory, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// AXPYs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void AXPY(double alpha, double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            AXPY(new [] { alpha }, vectorx, vectory, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// AXPYs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void AXPY(ComplexF alpha, ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            AXPY(new [] { alpha }, vectorx, vectory, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// AXPYs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void AXPY(ComplexD alpha, ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            AXPY(new[] { alpha }, vectorx, vectory, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// AXPYs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        protected abstract void AXPY(float[] alpha, float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// AXPYs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        protected abstract void AXPY(double[] alpha, double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// AXPYs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        protected abstract void AXPY(ComplexF[] alpha, ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// AXPYs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        protected abstract void AXPY(ComplexD[] alpha, ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        #endregion

        #region Copy

        /// <summary>
        /// COPYs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void COPY(float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// COPYs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void COPY(double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// COPYs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void COPY(ComplexF[] vectorx, ComplexF[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// COPYs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void COPY(ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        #endregion

        #region Dot

        /// <summary>
        /// DOTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        /// <returns></returns>
        public abstract float DOT(float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// DOTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        /// <returns></returns>
        public abstract double DOT(double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// DOTUs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        /// <returns></returns>
        public abstract ComplexF DOTU(ComplexF[] vectorx, ComplexF[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// DOTCs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        /// <returns></returns>
        public abstract ComplexF DOTC(ComplexF[] vectorx, ComplexF[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// DOTUs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        /// <returns></returns>
        public abstract ComplexD DOTU(ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// DOTCs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        /// <returns></returns>
        public abstract ComplexD DOTC(ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        #endregion

        #region NRM2

        /// <summary>
        /// NRs the m2.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract float NRM2(float[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// NRs the m2.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract double NRM2(double[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// NRs the m2.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract float NRM2(ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// NRs the m2.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract double NRM2(ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        #endregion

        #region ROT

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void ROT(float[] vectorx, float[] vectory, float c, float s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            ROT(vectorx, vectory, new[] { c }, new[] { s }, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void ROT(double[] vectorx, double[] vectory, double c, double s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            ROT(vectorx, vectory, new[] { c }, new[] { s }, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void ROT(ComplexF[] vectorx, ComplexF[] vectory, float c, ComplexF s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            ROT(vectorx, vectory, new[] { c }, new[] { s }, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void ROT(ComplexF[] vectorx, ComplexF[] vectory, float c, float s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            ROT(vectorx, vectory, new[] { c }, new[] { s }, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void ROT(ComplexD[] vectorx, ComplexD[] vectory, double c, ComplexD s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            ROT(vectorx, vectory, new[] { c }, new[] { s }, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void ROT(ComplexD[] vectorx, ComplexD[] vectory, double c, double s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            ROT(vectorx, vectory, new[] { c }, new[] { s }, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void ROT(float[] vectorx, float[] vectory, float[] c, float[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void ROT(double[] vectorx, double[] vectory, double[] c, double[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void ROT(ComplexF[] vectorx, ComplexF[] vectory, float[] c, ComplexF[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void ROT(ComplexF[] vectorx, ComplexF[] vectory, float[] c, float[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void ROT(ComplexD[] vectorx, ComplexD[] vectory, double[] c, ComplexD[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void ROT(ComplexD[] vectorx, ComplexD[] vectory, double[] c, double[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        #endregion

        #region ROTG

        //public void ROTG(ref float a, ref float b, ref float c, ref float s)
        //{
        //    ROTG(new[] { a }, new[] { b }, new[] { c }, new[] { s });
        //}

        //public void ROTG(ref double a, ref double b, ref double c, ref double s)
        //{
        //    ROTG(new[] { a }, new[] { b }, new[] { c }, new[] { s });
        //}

        //public void ROTG(ref ComplexF a, ref ComplexF b, ref float c, ref ComplexF s)
        //{
        //    ROTG(new[] { a }, new[] { b }, new[] { c }, new[] { s });
        //}

        //public void ROTG(ref ComplexD a, ref ComplexD b, ref double c, ref ComplexD s)
        //{
        //    ROTG(new[] { a }, new[] { b }, new[] { c }, new[] { s });
        //}

        /// <summary>
        /// ROTGs the specified a.
        /// </summary>
        /// <param name="a">A.</param>
        /// <param name="b">The b.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        public abstract void ROTG(float[] a, float[] b, float[] c, float[] s);

        /// <summary>
        /// ROTGs the specified a.
        /// </summary>
        /// <param name="a">A.</param>
        /// <param name="b">The b.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        public abstract void ROTG(double[] a, double[] b,double[] c, double[] s);

        /// <summary>
        /// ROTGs the specified a.
        /// </summary>
        /// <param name="a">A.</param>
        /// <param name="b">The b.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        public abstract void ROTG(ComplexF[] a, ComplexF[] b, float[] c, ComplexF[] s);

        /// <summary>
        /// ROTGs the specified a.
        /// </summary>
        /// <param name="a">A.</param>
        /// <param name="b">The b.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        public abstract void ROTG(ComplexD[] a, ComplexD[] b, double[] c, ComplexD[] s);

        #endregion

        #region ROTM

        /// <summary>
        /// ROTMs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="param">The param.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void ROTM(float[] vectorx, float[] vectory, float[] param, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// ROTMs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="param">The param.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void ROTM(double[] vectorx, double[] vectory, double[] param, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        #endregion

        #region ROTMG

        /// <summary>
        /// ROTMGs the specified d1.
        /// </summary>
        /// <param name="d1">The d1.</param>
        /// <param name="d2">The d2.</param>
        /// <param name="x1">The x1.</param>
        /// <param name="y1">The y1.</param>
        /// <param name="param">The param.</param>
        public abstract void ROTMG(ref float d1, ref float d2, ref float x1, ref float y1, float[] param);

        /// <summary>
        /// ROTMGs the specified d1.
        /// </summary>
        /// <param name="d1">The d1.</param>
        /// <param name="d2">The d2.</param>
        /// <param name="x1">The x1.</param>
        /// <param name="y1">The y1.</param>
        /// <param name="param">The param.</param>
        public abstract void ROTMG(ref double d1, ref double d2, ref double x1, ref double y1, double[] param);

        /// <summary>
        /// ROTMGs the specified d1.
        /// </summary>
        /// <param name="d1">The d1.</param>
        /// <param name="d2">The d2.</param>
        /// <param name="x1">The x1.</param>
        /// <param name="y1">The y1.</param>
        /// <param name="param">The param.</param>
        public abstract void ROTMG(float[] d1, float[] d2, float[] x1, float[] y1, float[] param);

        /// <summary>
        /// ROTMGs the specified d1.
        /// </summary>
        /// <param name="d1">The d1.</param>
        /// <param name="d2">The d2.</param>
        /// <param name="x1">The x1.</param>
        /// <param name="y1">The y1.</param>
        /// <param name="param">The param.</param>
        public abstract void ROTMG(double[] d1, double[] d2, double[] x1, double[] y1, double[] param);

        #endregion

        #region SCAL

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public void SCAL(float alpha, float[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            SCAL(new []{ alpha }, vectorx, n, rowx, incx);
        }

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public void SCAL(double alpha, double[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            SCAL(new []{ alpha }, vectorx, n, rowx, incx);
        }

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public void SCAL(ComplexF alpha, ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            SCAL(new []{ alpha }, vectorx, n, rowx, incx);
        }

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public void SCAL(float alpha, ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            SCAL(new []{ alpha }, vectorx, n, rowx, incx);
        }

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public void SCAL(ComplexD alpha, ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            SCAL(new []{ alpha }, vectorx, n, rowx, incx);
        }

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public void SCAL(double alpha, ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            SCAL(new[] { alpha }, vectorx, n, rowx, incx);
        }

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public abstract void SCAL(float[] alpha, float[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public abstract void SCAL(double[] alpha, double[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public abstract void SCAL(ComplexF[] alpha, ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public abstract void SCAL(float[] alpha, ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public abstract void SCAL(ComplexD[] alpha, ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public abstract void SCAL(double[] alpha, ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        #endregion

        #region SWAP

        /// <summary>
        /// SWAPs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void SWAP(float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// SWAPs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void SWAP(double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// SWAPs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void SWAP(ComplexF[] vectorx, ComplexF[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// SWAPs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void SWAP(ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        #endregion

        #endregion 

        #region BLAS Level 2
        #region GBMV
        /// <summary>
        /// Performs the banded matrix-vector multiplication.
        /// y = alpha * op(A) * x + beta * y
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="kl">number of subdiagonals of matrix A.</param>
        /// <param name="ku">number of superdiagonals of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">array of dimentions (kl + ku + 1) * n. This must be packed by column by column method.</param>
        /// <param name="x">vector with n elements if trans = N, m elements otherwise.</param>
        /// <param name="beta">scalar used for multiplication, if beta = 0 then y does not have to be a valid input.</param>
        /// <param name="y">vector with n elements if trans = N, m elements otherwise.</param>
        /// <param name="op">operation op(A) that is non- or(conj.) transpose.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. This typically be kl + ku + 1.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        /// <param name="incy">stride between consecutive elements of y.</param>
        public abstract void GBMV(int m, int n, int kl, int ku, float alpha, float[] A, float[] x, float beta, float[] y, cublasOperation op = cublasOperation.N, int lda = 0, int incx = 1, int incy = 1);

        /// <summary>
        /// Performs the banded matrix-vector multiplication.
        /// y = alpha * op(A) * x + beta * y
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="kl">number of subdiagonals of matrix A.</param>
        /// <param name="ku">number of superdiagonals of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">array of dimentions (kl + ku + 1) * n. This must be packed by column by column method.</param>
        /// <param name="x">vector with n elements if trans = N, m elements otherwise.</param>
        /// <param name="beta">scalar used for multiplication, if beta = 0 then y does not have to be a valid input.</param>
        /// <param name="y">vector with n elements if trans = N, m elements otherwise.</param>
        /// <param name="op">operation op(A) that is non- or(conj.) transpose.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if lda = 0, lda is automatically be kl + ku + 1.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        /// <param name="incy">stride between consecutive elements of y.</param>
        public abstract void GBMV(int m, int n, int kl, int ku, double alpha, double[] A, double[] x, double beta, double[] y, cublasOperation op = cublasOperation.N, int lda = 0, int incx = 1, int incy = 1);
        #endregion

        #region GEMV
        /// <summary>
        /// Performs the matrix-vector multiplication.
        /// y = alpha * op(A) * x + beta * y
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">Array of dimension m * n.</param>
        /// <param name="x">vector with n elements if trans = N, m elements otherwise.</param>
        /// <param name="beta">scalar used for multiplication, if beta = 0 then y does not have to be a valid input.</param>
        /// <param name="y">vector with n elements if trans = N, m elements otherwise.</param>
        /// <param name="op">operation op(A) that is non- or(conj.) transpose.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if lda = 0, lda is automatically tuned.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        /// <param name="incy">stride between consecutive elements of y.</param>
        public abstract void GEMV(int m, int n, float alpha, float[] A, float[] x, float beta, float[] y, cublasOperation op = cublasOperation.N, int lda = 0, int incx = 1, int incy = 1);

        /// <summary>
        /// Performs the matrix-vector multiplication.
        /// y = alpha * op(A) * x + beta * y
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">Array of dimension m * n.</param>
        /// <param name="x">vector with n elements if trans = N, m elements otherwise.</param>
        /// <param name="beta">scalar used for multiplication, if beta = 0 then y does not have to be a valid input.</param>
        /// <param name="y">vector with n elements if trans = N, m elements otherwise.</param>
        /// <param name="op">operation op(A) that is non- or(conj.) transpose.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if lda = 0, lda is automatically tuned.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        /// <param name="incy">stride between consecutive elements of y.</param>
        public abstract void GEMV(int m, int n, double alpha, double[] A, double[] x, double beta, double[] y, cublasOperation op = cublasOperation.N, int lda = 0, int incx = 1, int incy = 1);
        #endregion

        #region GER
        /// <summary>
        /// Performs the rank-1 update.
        /// A = alpha * x * transpose(y) + A
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">vector with m elements.</param>
        /// <param name="y">vector with n elements.</param>
        /// <param name="A">array of dimension m * n.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if lda = 0, lda is automatically tuned.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        /// <param name="incy">stride between consecutive elements of y.</param>
        public abstract void GER(int m, int n, float alpha, float[] x, float[] y, float[] A, int lda = 0, int incx = 1, int incy = 1);

        /// <summary>
        /// Performs the rank-1 update.
        /// A = alpha * x * transpose(y) + A
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">vector with m elements.</param>
        /// <param name="y">vector with n elements.</param>
        /// <param name="A">array of dimension m * n.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if lda = 0, lda is automatically tuned.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        /// <param name="incy">stride between consecutive elements of y.</param>
        public abstract void GER(int m, int n, double alpha, double[] x, double[] y, double[] A, int lda = 0, int incx = 1, int incy = 1);
        #endregion

        #region SBMV
        /// <summary>
        /// Performs the symmetric banded matrix-vector multiplication.
        /// y = alpha * A * x + beta * y
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="k">number of subdiagonals and superdiagonals of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">array of dimentions (k + 1) * n. This must be packed by column by column method.</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="beta">scalar used for multiplication, if beta = 0 then y does not have to be a valid input.</param>
        /// <param name="y">vector with n elements.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other symmetric part is not referenced and is inferred frorm the stored elements.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if lda = 0, lda is automatically be k + 1.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        /// <param name="incy">stride between consecutive elements of y.</param>
        public abstract void SBMV(int n, int k, float alpha, float[] A, float[] x, float beta, float[] y, cublasFillMode uplo = cublasFillMode.Lower, int lda = 0, int incx = 1, int incy = 1);
        /// <summary>
        /// Performs the symmetric banded matrix-vector multiplication.
        /// y = alpha * A * x + beta * y
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="k">number of subdiagonals and superdiagonals of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">array of dimentions (k + 1) * n. This must be packed by column by column method.</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="beta">scalar used for multiplication, if beta = 0 then y does not have to be a valid input.</param>
        /// <param name="y">vector with n elements.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other symmetric part is not referenced and is inferred frorm the stored elements.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if lda = 0, lda is automatically be k + 1.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        /// <param name="incy">stride between consecutive elements of y.</param>
        public abstract void SBMV(int n, int k, double alpha, double[] A, double[] x, double beta, double[] y, cublasFillMode uplo = cublasFillMode.Lower, int lda = 0, int incx = 1, int incy = 1);
        #endregion

        #region SPMV
        /// <summary>
        /// Performs the symmetric packed matrix-vector multiplication.
        /// y = alpha * A * x + beta + y
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="Ap">array with A stored in packed format.</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="y">vector with n elements.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other symmetric part is not referenced and is inferred frorm the stored elements.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        /// <param name="incy">stride between consecutive elements of y.</param>
        public abstract void SPMV(int n, float alpha, float[] Ap, float[] x, float beta, float[] y, cublasFillMode uplo = cublasFillMode.Lower, int incx = 1, int incy = 1);
        
        /// <summary>
        /// Performs the symmetric packed matrix-vector multiplication.
        /// y = alpha * A * x + beta + y
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="Ap">array with A stored in packed format.</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="y">vector with n elements.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other symmetric part is not referenced and is inferred frorm the stored elements.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        /// <param name="incy">stride between consecutive elements of y.</param>
        public abstract void SPMV(int n, double alpha, double[] Ap, double[] x, double beta, double[] y, cublasFillMode uplo = cublasFillMode.Lower, int incx = 1, int incy = 1);
        #endregion

        #region SPR
        /// <summary>
        /// Performs the packed symmetric rank-1 update.
        /// A = alpha * x * transpose(x) + A
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="ap">array with A stored in packed format.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other symmetric part is not referenced and is inferred frorm the stored elements.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        public abstract void SPR(int n, float alpha, float[] x, float[] ap, cublasFillMode uplo = cublasFillMode.Lower, int incx = 1);
        
        /// <summary>
        /// Performs the packed symmetric rank-1 update.
        /// A = alpha * x * transpose(x) + A
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="ap">array with A stored in packed format.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other symmetric part is not referenced and is inferred frorm the stored elements.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        public abstract void SPR(int n, double alpha, double[] x, double[] ap, cublasFillMode uplo = cublasFillMode.Lower, int incx = 1);
        #endregion

        #region SPR2
        /// <summary>
        /// Performs the packed symmetric rank-2 update.
        /// A = alpha * (x * transpose(y) + y * transpose(x)) + A
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="y">vector with n elements.</param>
        /// <param name="ap">array with A stored in packed format.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other symmetric part is not referenced and is inferred frorm the stored elements.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        /// <param name="incy">stride between consecutive elements of y.</param>
        public abstract void SPR2(int n, float alpha, float[] x, float[] y, float[] ap, cublasFillMode uplo = cublasFillMode.Lower, int incx = 1, int incy = 1);

        /// <summary>
        /// Performs the packed symmetric rank-2 update.
        /// A = alpha * (x * transpose(y) + y * transpose(x)) + A
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="y">vector with n elements.</param>
        /// <param name="ap">array with A stored in packed format.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other symmetric part is not referenced and is inferred frorm the stored elements.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        /// <param name="incy">stride between consecutive elements of y.</param>
        public abstract void SPR2(int n, double alpha, double[] x, double[] y, double[] ap, cublasFillMode uplo = cublasFillMode.Lower, int incx = 1, int incy = 1);
        #endregion

        #region SYMV
        /// <summary>
        /// Performs the symmetric matrix-vector multiplication.
        /// y = alpha * A * x + beta * y
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">array of dimension lda + n with lda >= max(1, n).</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="y">vector with n elements.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other symmetric part is not referenced and is inferred frorm the stored elements.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        /// <param name="incy">stride between consecutive elements of y.</param>
        public abstract void SYMV(int n, float alpha, float[] A, float[] x, float beta, float[] y, cublasFillMode uplo = cublasFillMode.Lower, int lda = 0, int incx = 1, int incy = 1);
        
        /// <summary>
        /// Performs the symmetric matrix-vector multiplication.
        /// y = alpha * A * x + beta * y
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">array of dimension lda + n with lda >= max(1, n).</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="y">vector with n elements.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other symmetric part is not referenced and is inferred frorm the stored elements.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        /// <param name="incy">stride between consecutive elements of y.</param>
        public abstract void SYMV(int n, double alpha, double[] A, double[] x, double beta, double[] y, cublasFillMode uplo = cublasFillMode.Lower, int lda = 0, int incx = 1, int incy = 1);
        #endregion

        #region SYR
        /// <summary>
        /// Performs the symmetric rank-1 update.
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="A">array of dimension lda + n with lda >= max(1, n).</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other symmetric part is not referenced and is inferred frorm the stored elements.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        public abstract void SYR(int n, float alpha, float[] x, float[] A, cublasFillMode uplo = cublasFillMode.Lower, int lda = 0, int incx = 1);
        
        /// <summary>
        /// Performs the symmetric rank-1 update.
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="A">array of dimension lda + n with lda >= max(1, n).</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other symmetric part is not referenced and is inferred frorm the stored elements.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        public abstract void SYR(int n, double alpha, double[] x, double[] A, cublasFillMode uplo = cublasFillMode.Lower, int lda = 0, int incx = 1);
        #endregion

        #region SYR2
        /// <summary>
        /// Performs the symmetric rank-2 update.
        /// A = alpha * (x * transpose(y) + y * transpose(x)) + A
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="y">vector with n elements.</param>
        /// <param name="A">array of dimension lda + n with lda >= max(1, n).</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other symmetric part is not referenced and is inferred frorm the stored elements.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        /// <param name="incy">stride between consecutive elements of y.</param>
        public abstract void SYR2(int n, float alpha, float[] x, float[] y, float[] A, cublasFillMode uplo = cublasFillMode.Lower, int lda = 0, int incx = 1, int incy = 1);

        /// <summary>
        /// Performs the symmetric rank-2 update.
        /// A = alpha * (x * transpose(y) + y * transpose(x)) + A
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="y">vector with n elements.</param>
        /// <param name="A">array of dimension lda + n with lda >= max(1, n).</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other symmetric part is not referenced and is inferred frorm the stored elements.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        /// <param name="incy">stride between consecutive elements of y.</param>
        public abstract void SYR2(int n, double alpha, double[] x, double[] y, double[] A, cublasFillMode uplo = cublasFillMode.Lower, int lda = 0, int incx = 1, int incy = 1);
        #endregion

        #region TBMV
        /// <summary>
        /// Performs the triangular banded matrix-vector multiplication.
        /// x = op(A) * x
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="k">number of subdiagonals and superdiagonals of matrix A.</param>
        /// <param name="A">array of dimension lda * n with lda >= k+1</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
        /// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if lda = 0, lda is automatically be k + 1.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        public abstract void TBMV(int n, int k, float[] A, float[] x, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, cublasDiagType diag = cublasDiagType.NonUnit, int lda = 0, int incx = 1);
        
        /// <summary>
        /// Performs the triangular banded matrix-vector multiplication.
        /// x = op(A) * x
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="k">number of subdiagonals and superdiagonals of matrix A.</param>
        /// <param name="A">array of dimension lda * n with lda >= k+1</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
        /// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if lda = 0, lda is automatically be k + 1.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        public abstract void TBMV(int n, int k, double[] A, double[] x, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, cublasDiagType diag = cublasDiagType.NonUnit, int lda = 0, int incx = 1);
        #endregion

        #region TBSV
        /// <summary>
        /// Solves the triangular banded linear system with a single right-hand-side.
        /// x = op(A)^(-1) * x
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="k">number of subdiagonals and superdiagonals of matrix A.</param>
        /// <param name="A">array of dimension lda * n with lda >= k+1</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
        /// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if lda = 0, lda is automatically be k + 1.</param>
        /// <param name="icx">stride between consecutive elements of x.</param>
        public abstract void TBSV(int n, int k, float[] A, float[] x, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, cublasDiagType diag = cublasDiagType.NonUnit, int lda = 0, int incx = 1);
        
        /// <summary>
        /// Solves the triangular banded linear system with a single right-hand-side.
        /// x = op(A)^(-1) * x
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="k">number of subdiagonals and superdiagonals of matrix A.</param>
        /// <param name="A">array of dimension lda * n with lda >= k+1</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
        /// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if lda = 0, lda is automatically be k + 1.</param>
        /// <param name="icx">stride between consecutive elements of x.</param>
        public abstract void TBSV(int n, int k, double[] A, double[] x, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, cublasDiagType diag = cublasDiagType.NonUnit, int lda = 0, int incx = 1);
        #endregion

        #region TPMV
        /// <summary>
        /// Performs the triangular packed matrix-vector multiplication.
        /// x = op(A) * x
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="AP">array with A stored in packed format.</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
        /// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        public abstract void TPMV(int n, float[] AP, float[] x, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, cublasDiagType diag = cublasDiagType.NonUnit, int incx = 1);
        
        /// <summary>
        /// Performs the triangular packed matrix-vector multiplication.
        /// x = op(A) * x
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="AP">array with A stored in packed format.</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
        /// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        public abstract void TPMV(int n, double[] AP, double[] x, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, cublasDiagType diag = cublasDiagType.NonUnit, int incx = 1);
        #endregion

        #region TPSV
        /// <summary>
        /// Solves the packed triangular linear system with a single right-hand-side.
        /// x = op(A)^-1 * x 
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="AP">array with A stored in packed format.</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
        /// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        public abstract void TPSV(int n, float[] AP, float[] x, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, cublasDiagType diag = cublasDiagType.NonUnit, int incx = 1);

        /// <summary>
        /// Solves the packed triangular linear system with a single right-hand-side.
        /// x = op(A)^-1 * x 
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="AP">array with A stored in packed format.</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
        /// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        public abstract void TPSV(int n, double[] AP, double[] x, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, cublasDiagType diag = cublasDiagType.NonUnit, int incx = 1);
        #endregion

        #region TRMV
        /// <summary>
        /// Performs the triangular matrix-vector multiplication.
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="a">array of dimensions lda * n with lda >= max(1, n).</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
        /// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if lda = 0, lda is automatically be n.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        public abstract void TRMV(int n, float[] a, float[] x, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, cublasDiagType diag = cublasDiagType.NonUnit, int lda = 0, int incx = 1);
        
        /// <summary>
        /// Performs the triangular matrix-vector multiplication.
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="a">array of dimensions lda * n with lda >= max(1, n).</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
        /// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if lda = 0, lda is automatically be n.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        public abstract void TRMV(int n, double[] a, double[] x, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, cublasDiagType diag = cublasDiagType.NonUnit, int lda = 0, int incx = 1);
        #endregion

        #region TRSV
        /// <summary>
        /// Solves the triangular linear system with a single right-hand-side.
        /// x = op(A)^-1 * x
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="A">array of dimensions lda * n with lda >= max(1, n).</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
        /// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if lda = 0, lda is automatically be n.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        public abstract void TRSV(int n, float[] A, float[] x, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, cublasDiagType diag = cublasDiagType.NonUnit, int lda = 0, int incx = 1);

        /// <summary>
        /// Solves the triangular linear system with a single right-hand-side.
        /// x = op(A)^-1 * x
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="A">array of dimensions lda * n with lda >= max(1, n).</param>
        /// <param name="x">vector with n elements.</param>
        /// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
        /// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if lda = 0, lda is automatically be n.</param>
        /// <param name="incx">stride between consecutive elements of x.</param>
        public abstract void TRSV(int n, double[] A, double[] x, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, cublasDiagType diag = cublasDiagType.NonUnit, int lda = 0, int incx = 1);
        #endregion
        #endregion

        #region BLAS Level 3
        #region GEMM
        /// <summary>
        /// Performs the matrix-matrix multiplication.
        /// C = alpha * op(A) * op(B) + beta * C
        /// </summary>
        /// <param name="m">number of rows of matrix op(A) and C.</param>
        /// <param name="k">number of columns of matix op(A) and rows of op(B).</param>
        /// <param name="n">number of columns of matix op(B) and C.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">arrasy of dimensions m * k.</param>
        /// <param name="B">array of dimension k * n.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="C">array of dimension m * n.</param>
        /// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store the matrix A.</param>
        /// <param name="ldb">leading dimension of two-dimensional array used to store the matrix B.</param>
        /// <param name="ldc">leading dimension of two-dimensional array used to store the matrix C.</param>
        public abstract void GEMM(int m, int k, int n, float alpha, float[] A, float[] B, float beta, float[] C, cublasOperation transa = cublasOperation.N, cublasOperation transb = cublasOperation.N, int lda = 0, int ldb = 0, int ldc = 0);

        /// <summary>
        /// Performs the matrix-matrix multiplication.
        /// C = alpha * op(A) * op(B) + beta * C
        /// </summary>
        /// <param name="m">number of rows of matrix op(A) and C.</param>
        /// <param name="k">number of columns of matix op(A) and rows of op(B).</param>
        /// <param name="n">number of columns of matix op(B) and C.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">arrasy of dimensions m * k.</param>
        /// <param name="B">array of dimension k * n.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="C">array of dimension m * n.</param>
        /// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store the matrix A.</param>
        /// <param name="ldb">leading dimension of two-dimensional array used to store the matrix B.</param>
        /// <param name="ldc">leading dimension of two-dimensional array used to store the matrix C.</param>
        public abstract void GEMM(int m, int k, int n, double alpha, double[] A, double[] B, double beta, double[] C, cublasOperation transa = cublasOperation.N, cublasOperation transb = cublasOperation.N, int lda = 0, int ldb = 0, int ldc = 0);
        #endregion

        #region SYMM
        /// <summary>
        /// Performs symmetric matrix-matrix multiplication.
        /// C = alpha * A * B + beta * C (side left),
        /// C = alpha * B * A + beta * C (side right)
        /// </summary>
        /// <param name="m">number of rows of matrix C and B, with matrix A sized accordingly.</param>
        /// <param name="n">number of columns of matrix C and B, with matrix A sized accordingly.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">array of dimension m * m with side left, and n * n otherwise.</param>
        /// <param name="B">array of dimension m * n.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="C">array of dimension m * n.</param>
        /// <param name="side">indicates if matrix A is on the left or right of B.</param>
        /// <param name="uplo">indicates if matrix A lower of upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
        /// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
        /// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
        public abstract void SYMM(int m, int n, float alpha, float[] A, float[] B, float beta, float[] C, cublasSideMode side = cublasSideMode.Left, cublasFillMode uplo = cublasFillMode.Lower, int lda = 0, int ldb = 0, int ldc = 0);
        
        /// <summary>
        /// Performs symmetric matrix-matrix multiplication.
        /// C = alpha * A * B + beta * C (side left),
        /// C = alpha * B * A + beta * C (side right)
        /// </summary>
        /// <param name="m">number of rows of matrix C and B, with matrix A sized accordingly.</param>
        /// <param name="n">number of columns of matrix C and B, with matrix A sized accordingly.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">array of dimension m * m with side left, and n * n otherwise.</param>
        /// <param name="B">array of dimension m * n.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="C">array of dimension m * n.</param>
        /// <param name="side">indicates if matrix A is on the left or right of B.</param>
        /// <param name="uplo">indicates if matrix A lower of upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
        /// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
        /// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
        public abstract void SYMM(int m, int n, double alpha, double[] A, double[] B, double beta, double[] C, cublasSideMode side = cublasSideMode.Left, cublasFillMode uplo = cublasFillMode.Lower, int lda = 0, int ldb = 0, int ldc = 0);
        #endregion

        #region SYRK
        /// <summary>
        /// Performs the symmetric rank-k update.
        /// C = alpha * op(A) * transpose(op(A)) + beta * C
        /// </summary>
        /// <param name="n">number of rows of matrix op(A) and C.</param>
        /// <param name="k">number of columns of matrix op(A).</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">array of dimension n * k.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="C">array of dimension n * n.</param>
        /// <param name="trans">operation op(A) that is non- or transpose.</param>
        /// <param name="uplo">indicates if matrix A lower of upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
        /// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
        public abstract void SYRK(int n, int k, float alpha, float[] A, float beta, float[] C, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, int lda = 0, int ldc = 0);

        /// <summary>
        /// Performs the symmetric rank-k update.
        /// C = alpha * op(A) * transpose(op(A)) + beta * C
        /// </summary>
        /// <param name="n">number of rows of matrix op(A) and C.</param>
        /// <param name="k">number of columns of matrix op(A).</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">array of dimension n * k.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="C">array of dimension n * n.</param>
        /// <param name="trans">operation op(A) that is non- or transpose.</param>
        /// <param name="uplo">indicates if matrix C lower of upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
        /// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
        public abstract void SYRK(int n, int k, double alpha, double[] A, double beta, double[] C, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, int lda = 0, int ldc = 0);
        #endregion

        #region SYR2K
        /// <summary>
        /// Performs the symmetric rank-2k update.
        /// C = alpha * (op(A) * transpose(op(B)) + op(B) * transpose(op(A))) + beta * C
        /// </summary>
        /// <param name="n">number of rows of matrix op(A), op(B) and C.</param>
        /// <param name="k">number of columns of matrix op(A) and op(B).</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">array of dimension n * k.</param>
        /// <param name="B">array of dimension n * k.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="C">array of dimension n * n.</param>
        /// <param name="trans">operation op(A), op(B) that is non- or transpose.</param>
        /// <param name="uplo">indicates if matrix C lower of upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
        /// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
        /// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
        public abstract void SYR2K(int n, int k, float alpha, float[] A, float[] B, float beta, float[] C, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, int lda = 0, int ldb = 0, int ldc = 0);

        /// <summary>
        /// Performs the symmetric rank-2k update.
        /// C = alpha * (op(A) * transpose(op(B)) + op(B) * transpose(op(A))) + beta * C
        /// </summary>
        /// <param name="n">number of rows of matrix op(A), op(B) and C.</param>
        /// <param name="k">number of columns of matrix op(A) and op(B).</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">array of dimension n * k.</param>
        /// <param name="B">array of dimension n * k.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="C">array of dimension n * n.</param>
        /// <param name="trans">operation op(A), op(B) that is non- or transpose.</param>
        /// <param name="uplo">indicates if matrix C lower of upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
        /// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
        /// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
        public abstract void SYR2K(int n, int k, double alpha, double[] A, double[] B, double beta, double[] C, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, int lda = 0, int ldb = 0, int ldc = 0);
        #endregion

        #region TRMM
        /// <summary>
        /// Performs the triangular matrix-matrix multiplication.
        /// C = alpha * op(A) * B (side left),
        /// C = alpha * B * op(A) (side right)
        /// </summary>
        /// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
        /// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">array of dimension m * m.</param>
        /// <param name="B">array of dimension m * n.</param>
        /// <param name="C">array of dimension m * n.</param>
        /// <param name="side">indicates if matrix A is on the left or right of B.</param>
        /// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not refernced and is inferred from the stored elements.</param>
        /// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
        /// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
        /// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
        public abstract void TRMM(int m, int n, float alpha, float[] A, float[] B, float[] C, cublasSideMode side = cublasSideMode.Left, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, cublasDiagType diag = cublasDiagType.NonUnit, int lda = 0, int ldb = 0, int ldc = 0);

        /// <summary>
        /// Performs the triangular matrix-matrix multiplication.
        /// C = alpha * op(A) * B (side left),
        /// C = alpha * B * op(A) (side right)
        /// </summary>
        /// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
        /// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">array of dimension m * m.</param>
        /// <param name="B">array of dimension m * n.</param>
        /// <param name="C">array of dimension m * n.</param>
        /// <param name="side">indicates if matrix A is on the left or right of B.</param>
        /// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not refernced and is inferred from the stored elements.</param>
        /// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
        /// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
        /// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
        public abstract void TRMM(int m, int n, double alpha, double[] A, double[] B, double[] C, cublasSideMode side = cublasSideMode.Left, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, cublasDiagType diag = cublasDiagType.NonUnit, int lda = 0, int ldb = 0, int ldc = 0);
        #endregion

        #region TRSM
        /// <summary>
        /// Solves the triangular linear system with multiple right-hand-sides.
        /// B = alpha * (op(A))^-1 * B (left side),
        /// B = alpha * B * (op(A))^-1 (right side)
        /// </summary>
        /// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
        /// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">array of dimension m * m (n * n right side).</param>
        /// <param name="B">array of dimension m * n.</param>
        /// <param name="side">indicates if matrix A is on the left or right of B.</param>
        /// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not refernced and is inferred from the stored elements.</param>
        /// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
        /// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
        public abstract void TRSM(int m, int n, float alpha, float[] A, float[] B, cublasSideMode side = cublasSideMode.Left, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, cublasDiagType diag = cublasDiagType.NonUnit, int lda = 0, int ldb = 0);

        /// <summary>
        /// Solves the triangular linear system with multiple right-hand-sides.
        /// B = alpha * (op(A))^-1 * B (left side),
        /// B = alpha * B * (op(A))^-1 (right side)
        /// </summary>
        /// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
        /// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">array of dimension m * m (n * n right side).</param>
        /// <param name="B">array of dimension m * n.</param>
        /// <param name="side">indicates if matrix A is on the left or right of B.</param>
        /// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
        /// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not refernced and is inferred from the stored elements.</param>
        /// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
        /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
        /// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
        public abstract void TRSM(int m, int n, double alpha, double[] A, double[] B, cublasSideMode side = cublasSideMode.Left, cublasOperation trans = cublasOperation.N, cublasFillMode uplo = cublasFillMode.Lower, cublasDiagType diag = cublasDiagType.NonUnit, int lda = 0, int ldb = 0);
        #endregion
        #endregion

        //public void Copy<T>(T[,] src, T[,] dst, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        //{
        //    Copy<T>(src, dst, n, rowx, incx, rowy, incy);
        //}

        //protected abstract void COPY<T>(object src, object dst, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        //public abstract T DOT<T>(T[] vectorx, T[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        //public abstract T NRM2<T>(T[] vectorx, int n = 0, int rowx = 0, int incx = 1);


        //#region Max
        ///// <summary>
        ///// Gets the index of the maximum value in the specified array.
        ///// </summary>
        ///// <typeparam name="T">One of the supported types.</typeparam>
        ///// <param name="devArray"></param>
        ///// <param name="n"></param>
        ///// <param name="row"></param>
        ///// <param name="incx"></param>
        ///// <returns></returns>
        //public int IAMAX<T>(T[] devArray, int n = 0, int row = 0, int incx = 1)
        //{
        //    return IAMAXEx<T>(devArray, n, row, incx);
        //}

        ////public abstract Tuple<int, int> IAMAX<T>(T[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = false, int incx = 1);

        //protected abstract int IAMAXEx<T>(object devArray, int n = 0, int row = 0, int incx = 1);

        //#endregion

        //#region Min

        //public int IAMIN<T>(T[] devArray, int n = 0, int row = 0, int incx = 1)
        //{
        //    return IAMINEx<T>(devArray, n, row, incx);
        //}

        //public int Min<T>(T[] devArray)
        //{
        //    return IAMINEx<T>(devArray) - 1;
        //}

        //protected abstract int IAMINEx<T>(object devArray, int n = 0, int row = 0, int incx = 1);

        ////public abstract int IAMIN<T>(T[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1);

        //#endregion

        //public void SCAL<T>(T alpha, T[] vector, int n = 0, int row = 0, int incx = 1)
        //{
        //    SCALEx<T>(alpha, vector, n, row, incx);
        //}

        //public abstract void SCALEx<T>(T alpha, object vector, int n = 0, int row = 0, int incx = 1);

        ////public abstract void SCAL<T>(T alpha, T[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1);

        //public abstract T DOTC<T>(T[] vectorx, T[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);


        //public abstract void ROT(float[] vectorx, float[] vectory, float sc, float ss, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        //public abstract void ROT(double[] vectorx, double[] vectory, double sc, double ss, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        //public abstract void ROT(ComplexF[] vectorx, ComplexF[] vectory, float sc, ComplexF ss, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        //public abstract void ROT(ComplexD[] vectorx, ComplexD[] vectory, float sc, ComplexD cs, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);


        //public abstract void ROTM(float[] vectorx, float[] vectory, float[] sparam, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        //public abstract void ROTM(double[] vectorx, double[] vectory, double[] sparam, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);


        //public abstract void SWAP<T>(T[] vectorx, T[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1); 

        //public abstract void ROTG(float[] host_sa, float[] host_sb, float[] host_sc, float[] host_ss)
        //{
        //    throw new NotImplementedException();
        //}

        //public abstract void ROTG(double[] host_da, double[] host_db, double[] host_dc, double[] host_ds)
        //{
        //    throw new NotImplementedException();
        //}

        //public abstract void ROTG(ComplexF[] host_ca, ComplexF[] host_cb, float[] host_sc, float[] host_ss)
        //{
        //    throw new NotImplementedException();
        //}

        //public abstract void ROTG(ComplexD[] host_ca, ComplexD[] host_cb, double[] host_dc, double[] host_ds)
        //{
        //    throw new NotImplementedException();
        //}

        //void IDisposable.Dispose()
        //{
        //    throw new NotImplementedException();
        //}
    }
}
