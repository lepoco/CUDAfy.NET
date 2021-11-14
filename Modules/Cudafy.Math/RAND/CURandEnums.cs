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

namespace Cudafy.Maths.RAND
{
    /// <summary>
    /// Rand Direction Vector Set
    /// </summary>
    public enum curandDirectionVectorSet
    {
        /// <summary>
        /// Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
        /// </summary>
        CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101,
        /// <summary>
        /// Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled
        /// </summary>
        CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102,
        /// <summary>
        /// Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
        /// </summary>
        CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103,
        /// <summary>
        /// Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled
        /// </summary>
        CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104
    };

    /// <summary>
    /// Status
    /// </summary>
    internal enum curandStatus
    {
        CURAND_STATUS_SUCCESS = 0, //< No errors
        CURAND_STATUS_VERSION_MISMATCH = 100, //< Header file and linked library version do not match
        CURAND_STATUS_NOT_INITIALIZED = 101, //< Generator not initialized
        CURAND_STATUS_ALLOCATION_FAILED = 102, //< Memory allocation failed
        CURAND_STATUS_TYPE_ERROR = 103, //< Generator is wrong type
        CURAND_STATUS_OUT_OF_RANGE = 104, //< Argument out of range
        CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105, //< Length requested is not a multple of dimension
        CURAND_STATUS_LAUNCH_FAILURE = 201, //< Kernel launch failure
        CURAND_STATUS_PREEXISTING_FAILURE = 202, //< Preexisting failure on library entry
        CURAND_STATUS_INITIALIZATION_FAILED = 203, //< Initialization of CUDA failed
        CURAND_STATUS_ARCH_MISMATCH = 204, //< Architecture mismatch, GPU does not support requested feature
        CURAND_STATUS_INTERNAL_ERROR = 999 //< Internal library error
    };

    /// <summary>
    /// 
    /// </summary>
    public enum curandRngType
    {
        /// <summary>
        /// Internal use.
        /// </summary>
        CURAND_RNG_TEST = 0,
        /// <summary>
        /// Default pseudorandom generator
        /// </summary>
        CURAND_RNG_PSEUDO_DEFAULT = 100,
        /// <summary>
        /// XORWOW pseudorandom generator
        /// </summary>
        CURAND_RNG_PSEUDO_XORWOW = 101, 
        /// <summary>
        /// Default quasirandom generator
        /// </summary>
        CURAND_RNG_QUASI_DEFAULT = 200, 
        /// <summary>
        /// Sobol32 quasirandom generator
        /// </summary>
        CURAND_RNG_QUASI_SOBOL32 = 201, 
        /// <summary>
        /// Scrambled Sobol32 quasirandom generator
        /// </summary>
        CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,  
        /// <summary>
        /// Sobol64 quasirandom generator
        /// </summary>
        CURAND_RNG_QUASI_SOBOL64 = 203, 
        /// <summary>
        /// Scrambled Sobol64 quasirandom generator
        /// </summary>
        CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204 
    };

    /// <summary>
    /// 
    /// </summary>
    public enum curandOrdering
    {
        /// <summary>
        /// Best ordering for pseudorandom results
        /// </summary>
        CURAND_ORDERING_PSEUDO_BEST = 100,
        /// <summary>
        /// Specific default 4096 thread sequence for pseudorandom results
        /// </summary>
        CURAND_ORDERING_PSEUDO_DEFAULT = 101,
        /// <summary>
        /// Specific seeding pattern for fast lower quality pseudorandom results
        /// </summary>
        CURAND_ORDERING_PSEUDO_SEEDED = 102,
        /// <summary>
        /// Specific n-dimensional ordering for quasirandom results
        /// </summary>
        CURAND_ORDERING_QUASI_DEFAULT = 201 
    };
}
