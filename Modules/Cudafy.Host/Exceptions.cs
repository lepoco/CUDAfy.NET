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

namespace Cudafy.Host
{
    /// <summary>
    /// Exceptions for host.
    /// </summary>
    [global::System.Serializable]
    public class CudafyHostException : CudafyException
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CudafyHostException"/> class.
        /// </summary>
        /// <param name="message">The message.</param>
        public CudafyHostException(string message) : base(message) { }
        /// <summary>
        /// Initializes a new instance of the <see cref="CudafyHostException"/> class.
        /// </summary>
        /// <param name="inner">The inner.</param>
        /// <param name="message">The message.</param>
        public CudafyHostException(Exception inner, string message) : base(message, inner) { }
        /// <summary>
        /// Initializes a new instance of the <see cref="CudafyHostException"/> class.
        /// </summary>
        /// <param name="errMsg">The err MSG.</param>
        /// <param name="args">The args.</param>
        public CudafyHostException(string errMsg, params object[] args) : base(string.Format(errMsg, args)) { CheckParamsAreNoExceptions(args); }
        /// <summary>
        /// Initializes a new instance of the <see cref="CudafyHostException"/> class.
        /// </summary>
        /// <param name="inner">The inner exception.</param>
        /// <param name="errMsg">The err message.</param>
        /// <param name="args">The parameters.</param>
        public CudafyHostException(Exception inner, string errMsg, params object[] args) : base(string.Format(errMsg, args)) { CheckParamsAreNoExceptions(args); }

#pragma warning disable 1591
        public const string csCONSTANT_MEMORY_NOT_FOUND = "Constant memory not found.";
        public const string csINDEX_OUT_OF_RANGE = "Index out of range.";
        public const string csHOST_AND_DEVICE_ARRAYS_ARE_OF_DIFFERENT_SIZES = "Host and device arrays are of different sizes.";
        public const string csNO_MODULE_LOADED = "No module loaded.";
        public const string csMODULE_ALREADY_LOADED = "Module already loaded.";
        public const string csMODULE_NOT_FOUND = "Module not found.";
        public const string csPARAMETER_PASSED_BY_REFERENCE_X_NOT_CURRENTLY_SUPPORTED = "Parameter passed by reference ({0}) is currently not supported.";
        public const string csRATIO_OF_INPUT_AND_OUTPUT_ARRAY_05_10_20 = "Ratio of input array size to output array size must be 0.5, 1.0 or 2.0.";
        public const string csCAN_ONLY_LAUNCH_GLOBAL_METHODS = "Can only launch global methods.";
        public const string csCANNOT_EMULATE_DUMMY_FUNCTION_X = "Cannot launch dummy function '{0}'.";
        public const string csNO_X_PRESENT_IN_CUDAFY_MODULE = "No {0} present in Cudafy module.";
        //public const string csNO_SUITABLE_X_PRESENT_IN_CUDAFY_MODULE = "No suitable {0} present in Cudafy module.";
        public const string csFAILED_TO_GET_PROPERIES_X = "Failed to get properties: {0}";
        public const string csDATA_IS_NOT_ON_GPU = "Data is not on the device.";
        public const string csDATA_IS_NOT_HOST_ALLOCATED = "Data is not host allocated.";
        public const string csSTREAM_X_ALREADY_SET = "Stream {0} is already set.";
        public const string csSTREAM_X_NOT_SET = "Stream {0} is not set.";
        public const string csPOINTER_NOT_FOUND = "Pointer not found.";
        public const string csDEVICE_ID_OUT_OF_RANGE = "Device ID out of range.";
        public const string csDEVICE_IS_NOT_LOCKED = "Device is not locked.";
        public const string csMULTITHREADING_IS_NOT_ENABLED = "Multithreading is not enabled.";
        public const string csSMART_COPY_ALREADY_ENABLED = "Smart copy is already enabled.";
        public const string csSMART_COPY_IS_NOT_ENABLED = "Smart copy is not enabled.";
        public const string csCUDA_EXCEPTION_X = "CUDA.NET exception: {0}.";
        public const string csCUDA_EXCEPTION_X_X = "CUDA.NET exception: {0} ({1}).";
        public const string csOPENCL_EXCEPTION_X = "OpenCL exception: {0}.";
        public const string csOPENCL_EXCEPTION_X_X = "OpenCL exception: {0} ({1}).";
        public const string csDUPLICATE_X_NAME = "Module already loaded containing a duplicate {0} name.";
        public const string csPEER_ACCESS_ALREADY_ENABLED = "Peer access already enabled";
        public const string csPEER_ACCESS_WAS_NOT_ENABLED = "Peer access was not enabled";
        public const string csPEER_ACCESS_TO_SELF_NOT_ALLOWED = "Peer access cannot be granted to self.";
        protected CudafyHostException(
          System.Runtime.Serialization.SerializationInfo info,
          System.Runtime.Serialization.StreamingContext context)
            : base(info, context) { }
#pragma warning restore 1591
    }
}
