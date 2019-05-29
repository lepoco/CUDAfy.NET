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
using Cudafy.Host;
namespace Cudafy.Maths
{
    /// <summary>
    /// Exceptions for host.
    /// </summary>
    [global::System.Serializable]
    public class CudafyMathException : CudafyHostException
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CudafyMathException"/> class.
        /// </summary>
        /// <param name="message">The message.</param>
        public CudafyMathException(string message) : base(message) { }
        /// <summary>
        /// Initializes a new instance of the <see cref="CudafyMathException"/> class.
        /// </summary>
        /// <param name="inner">The inner.</param>
        /// <param name="message">The message.</param>
        public CudafyMathException(Exception inner, string message) : base(message, inner) { }
        /// <summary>
        /// Initializes a new instance of the <see cref="CudafyMathException"/> class.
        /// </summary>
        /// <param name="errMsg">The err MSG.</param>
        /// <param name="args">The args.</param>
        public CudafyMathException(string errMsg, params object[] args) : base(string.Format(errMsg, args)) { CheckParamsAreNoExceptions(args); }
        /// <summary>
        /// Initializes a new instance of the <see cref="CudafyMathException"/> class.
        /// </summary>
        /// <param name="inner">The inner exception.</param>
        /// <param name="errMsg">The err message.</param>
        /// <param name="args">The parameters.</param>
        public CudafyMathException(Exception inner, string errMsg, params object[] args) : base(string.Format(errMsg, args)) { CheckParamsAreNoExceptions(args); }

#pragma warning disable 1591

        public const string csPLAN_NOT_FOUND = "Plan not found.";

        public const string csBLAS_ERROR_X = "BLAS error: {0}";

        public const string csFFT_ERROR_X = "FFT error: {0}";

        public const string csRAND_ERROR_X = "RAND error: {0}";

        protected CudafyMathException(
          System.Runtime.Serialization.SerializationInfo info,
          System.Runtime.Serialization.StreamingContext context)
            : base(info, context) { }
#pragma warning restore 1591
    }
}
