// Copyright 2006 - Tamas Szalay (http://www.sdss.jhu.edu/~tamas/bytes/fftwcsharp.html)
using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;

namespace FFTW.NET
{
    /// <summary>
    /// To simplify FFTW memory management
    /// </summary>
    public abstract class fftwf_complexarray
    {
        private IntPtr handle;
        /// <summary>
        /// Gets the handle.
        /// </summary>
        /// <value>The handle.</value>
        public IntPtr Handle
        { 
            get 
            { 
                return handle; 
            } 
        }

        private int length;
        /// <summary>
        /// Gets the length.
        /// </summary>
        /// <value>The length.</value>
        public int Length
        { 
            get 
            { 
                return length; 
            } 
        }

        /// <summary>
        /// Creates a new array of complex numbers
        /// </summary>
        /// <param name="length">Logical length of the array</param>
        public fftwf_complexarray(int length)
        {
            this.length = length;
            this.handle = fftwf.malloc(this.length * 8);
        }

        /// <summary>
        /// Creates an FFTW-compatible array from array of floats, initializes to single precision only
        /// </summary>
        /// <param name="data">Array of floats, alternating real and imaginary</param>
        public fftwf_complexarray(float[] data)
        {
            this.length = data.Length / 2;
            this.handle = fftwf.malloc(this.length * 8);
            Marshal.Copy(data, 0, handle, this.length * 2);
        }

        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="fftwf_complexarray"/> is reclaimed by garbage collection.
        /// </summary>
        ~fftwf_complexarray()
        {
            fftwf.free(handle);
        }
    }
}
