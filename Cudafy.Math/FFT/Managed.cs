// Copyright 2006 - Tamas Szalay (http://www.sdss.jhu.edu/~tamas/bytes/fftwcsharp.html)
// Modified by Hybrid DSP Systems 2011 http://www.hybriddsp.com
using System;
using System.Collections.Generic;
using System.Text;
namespace Cudafy.Maths.FFT
{
    #region Single Precision
    /// <summary>
    /// Creates, stores, and destroys fftw plans
    /// </summary>
    public class fftwf_plan : Ifftw_plan
    {
        /// <summary>
        /// Native handle.
        /// </summary>
        protected IntPtr _handle;
        /// <summary>
        /// Gets the handle.
        /// </summary>
        /// <value>The handle.</value>
        public IntPtr Handle
        { 
            get 
            { 
                return _handle; 
            } 
        }

        /// <summary>
        /// Gets or sets the input.
        /// </summary>
        /// <value>The input.</value>
        public IntPtr Input { get; internal set; }

        /// <summary>
        /// Gets or sets the output.
        /// </summary>
        /// <value>The output.</value>
        public IntPtr Output { get; internal set; }

        /// <summary>
        /// Executes this instance.
        /// </summary>
        public void Execute()
        {
            fftwf.execute(_handle);
        }

        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="fftwf_plan"/> is reclaimed by garbage collection.
        /// </summary>
        ~fftwf_plan()
        {
            //fftwf.destroy_plan(_handle);
        }

        /// <summary>
        /// Creates plan..
        /// </summary>
        /// <param name="fftType">Type of fft.</param>
        /// <param name="n">The n.</param>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <param name="direction">The direction.</param>
        /// <param name="flags">The flags.</param>
        /// <returns></returns>
        public static fftwf_plan dft_1d(eFFTType fftType, int n, IntPtr input, IntPtr output, fftw_direction direction, fftw_flags flags)
        {
            fftwf_plan p = new fftwf_plan();
            if (fftType == eFFTType.Complex2Complex)
                p._handle = fftwf.dft_1d(n, input, output, direction, flags);
            else if (fftType == eFFTType.Real2Complex)
                p._handle = fftwf.dft_r2c_1d(n, input, output, flags);
            else if (fftType == eFFTType.Complex2Real)
                p._handle = fftwf.dft_c2r_1d(n, input, output, flags);
            p.Input = input;
            p.Output = output;
            return p;
        }

        public static fftwf_plan dft_many(eFFTType fftType, int rank, int[] n, int batch,
                                  IntPtr input, int[] inembed,
                                  int istride, int idist,
                                  IntPtr output, int[] onembed,
                                  int ostride, int odist,
                                  fftw_direction sign, fftw_flags flags)
        {
            fftwf_plan p = new fftwf_plan();
            if (fftType == eFFTType.Complex2Complex)
                p._handle = fftwf.many_dft(rank, n, batch, input, inembed, istride, idist, output, onembed, ostride, odist, sign, flags);
            else if (fftType == eFFTType.Real2Complex)
                p._handle = fftwf.many_dft_r2c(rank, n, batch, input, inembed, istride, idist, output, onembed, ostride, odist, flags);
            else if (fftType == eFFTType.Complex2Real)
                p._handle = fftwf.many_dft_c2r(rank, n, batch, input, inembed, istride, idist, output, onembed, ostride, odist, flags);
            p.Input = input;
            p.Output = output;
            return p;
        }

        /// <summary>
        /// DFT_2Ds the specified FFT type.
        /// </summary>
        /// <param name="fftType">Type of the FFT.</param>
        /// <param name="nx">The nx.</param>
        /// <param name="ny">The ny.</param>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <param name="direction">The direction.</param>
        /// <param name="flags">The flags.</param>
        /// <returns></returns>
        public static fftwf_plan dft_2d(eFFTType fftType, int nx, int ny, IntPtr input, IntPtr output, fftw_direction direction, fftw_flags flags)
        {
            fftwf_plan p = new fftwf_plan();
            //p._handle = fftwf.dft_2d(nx, ny, input, output, direction, flags);
            if (fftType == eFFTType.Complex2Complex)
                p._handle = fftwf.dft_2d(nx, ny, input, output, direction, flags);
            else if (fftType == eFFTType.Real2Complex)
                p._handle = fftwf.dft_r2c_2d(nx, ny, input, output, flags);
            else if (fftType == eFFTType.Complex2Real)
                p._handle = fftwf.dft_c2r_2d(nx, ny, input, output, flags);
            p.Input = input;
            p.Output = output;
            return p;
        }

        /// <summary>
        /// DFT_3Ds the specified FFT type.
        /// </summary>
        /// <param name="fftType">Type of the FFT.</param>
        /// <param name="nx">The nx.</param>
        /// <param name="ny">The ny.</param>
        /// <param name="nz">The nz.</param>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <param name="direction">The direction.</param>
        /// <param name="flags">The flags.</param>
        /// <returns></returns>
        public static fftwf_plan dft_3d(eFFTType fftType, int nx, int ny, int nz, IntPtr input, IntPtr output, fftw_direction direction, fftw_flags flags)
        {
            fftwf_plan p = new fftwf_plan();
           // p._handle = fftwf.dft_3d(nx, ny, nz, input, output, direction, flags);
            if (fftType == eFFTType.Complex2Complex)
                p._handle = fftwf.dft_3d(nx, ny, nz, input, output, direction, flags);
            else if (fftType == eFFTType.Real2Complex)
                p._handle = fftwf.dft_r2c_3d(nx, ny, nz, input, output, flags);
            else if (fftType == eFFTType.Complex2Real)
                p._handle = fftwf.dft_c2r_3d(nx, ny, nz, input, output, flags);
            p.Input = input;
            p.Output = output;
            return p;
        }
    }
    #endregion

    #region Double Precision
    /// <summary>
    /// Creates, stores, and destroys fftw plans
    /// </summary>
    public class fftw_plan : Ifftw_plan
    {
        /// <summary>
        /// Native handle.
        /// </summary>
        protected IntPtr _handle;
        /// <summary>
        /// Gets the handle.
        /// </summary>
        /// <value>The handle.</value>
        public IntPtr Handle
        {
            get
            {
                return _handle;
            }
        }

        /// <summary>
        /// Gets or sets the input.
        /// </summary>
        /// <value>The input.</value>
        public IntPtr Input { get; internal set; }

        /// <summary>
        /// Gets or sets the output.
        /// </summary>
        /// <value>The output.</value>
        public IntPtr Output { get; internal set; }

        /// <summary>
        /// Executes this instance.
        /// </summary>
        public void Execute()
        {
            fftw.execute(_handle);
        }

        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="fftwf_plan"/> is reclaimed by garbage collection.
        /// </summary>
        ~fftw_plan()
        {
            //fftw.destroy_plan(_handle);
        }

        public static fftw_plan dft_many(eFFTType fftType, int rank, int[] n, int batch,
                          IntPtr input, int[] inembed,
                          int istride, int idist,
                          IntPtr output, int[] onembed,
                          int ostride, int odist,
                          fftw_direction sign, fftw_flags flags)
        {
            fftw_plan p = new fftw_plan();
            if (fftType == eFFTType.Complex2Complex)
                p._handle = fftw.many_dft(rank, n, batch, input, inembed, istride, idist, output, onembed, ostride, odist, sign, flags);
            else if (fftType == eFFTType.Real2Complex)
                p._handle = fftw.many_dft_r2c(rank, n, batch, input, inembed, istride, idist, output, onembed, ostride, odist, flags);
            else if (fftType == eFFTType.Complex2Real)
                p._handle = fftw.many_dft_c2r(rank, n, batch, input, inembed, istride, idist, output, onembed, ostride, odist, flags);
            p.Input = input;
            p.Output = output;
            return p;
        }

        /// <summary>
        /// Creates plan..
        /// </summary>
        /// <param name="fftType">FFT type.</param>
        /// <param name="n">The n.</param>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <param name="direction">The direction.</param>
        /// <param name="flags">The flags.</param>
        /// <returns></returns>
        public static fftw_plan dft_1d(eFFTType fftType, int n, IntPtr input, IntPtr output, fftw_direction direction, fftw_flags flags)
        {
            fftw_plan p = new fftw_plan();
            if (fftType == eFFTType.Complex2Complex)
                p._handle = fftw.dft_1d(n, input, output, direction, flags);
            else if (fftType == eFFTType.Real2Complex)
                p._handle = fftw.dft_r2c_1d(n, input, output, flags);
            else if (fftType == eFFTType.Complex2Real)
                p._handle = fftw.dft_c2r_1d(n, input, output, flags);
            p.Input = input;
            p.Output = output;
            return p;
        }

        /// <summary>
        /// DFT_2Ds the specified FFT type.
        /// </summary>
        /// <param name="fftType">Type of the FFT.</param>
        /// <param name="nx">The nx.</param>
        /// <param name="ny">The ny.</param>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <param name="direction">The direction.</param>
        /// <param name="flags">The flags.</param>
        /// <returns></returns>
        public static fftw_plan dft_2d(eFFTType fftType, int nx, int ny, IntPtr input, IntPtr output, fftw_direction direction, fftw_flags flags)
        {
            fftw_plan p = new fftw_plan();
            //p._handle = fftwf.dft_2d(nx, ny, input, output, direction, flags);
            if (fftType == eFFTType.Complex2Complex)
                p._handle = fftw.dft_2d(nx, ny, input, output, direction, flags);
            else if (fftType == eFFTType.Real2Complex)
                p._handle = fftw.dft_r2c_2d(nx, ny, input, output, flags);
            else if (fftType == eFFTType.Complex2Real)
                p._handle = fftw.dft_c2r_2d(nx, ny, input, output, flags);
            p.Input = input;
            p.Output = output;
            return p;
        }

        /// <summary>
        /// DFT_3Ds the specified FFT type.
        /// </summary>
        /// <param name="fftType">Type of the FFT.</param>
        /// <param name="nx">The nx.</param>
        /// <param name="ny">The ny.</param>
        /// <param name="nz">The nz.</param>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <param name="direction">The direction.</param>
        /// <param name="flags">The flags.</param>
        /// <returns></returns>
        public static fftw_plan dft_3d(eFFTType fftType, int nx, int ny, int nz, IntPtr input, IntPtr output, fftw_direction direction, fftw_flags flags)
        {
            fftw_plan p = new fftw_plan();
            // p._handle = fftwf.dft_3d(nx, ny, nz, input, output, direction, flags);
            if (fftType == eFFTType.Complex2Complex)
                p._handle = fftw.dft_3d(nx, ny, nz, input, output, direction, flags);
            else if (fftType == eFFTType.Real2Complex)
                p._handle = fftw.dft_r2c_3d(nx, ny, nz, input, output, flags);
            else if (fftType == eFFTType.Complex2Real)
                p._handle = fftw.dft_c2r_3d(nx, ny, nz, input, output, flags);
            p.Input = input;
            p.Output = output;
            return p;
        }
    }
    #endregion
}
