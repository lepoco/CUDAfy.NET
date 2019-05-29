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
#if !NET35
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


using System.Dynamic;
namespace Cudafy.Host
{
    /// <summary>
    /// Allows GPU functions to be called using dynamic language run-time. For example:
    /// gpgpu.Launch(16, 16).myGPUFunction(x, y, res)
    /// </summary>
    public class DynamicLauncher : DynamicObject
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DynamicLauncher"/> class.
        /// </summary>
        /// <param name="gpu">The gpu.</param>
        public DynamicLauncher(GPGPU gpu)
        {
            GPU = gpu;
        }

        /// <summary>
        /// Gets the GPU.
        /// </summary>
        public GPGPU GPU { get; private set; }

        /// <summary>
        /// Gets or sets the size of the grid.
        /// </summary>
        /// <value>
        /// The size of the grid.
        /// </value>
        public dim3 GridSize { get; set; }

        /// <summary>
        /// Gets or sets the size of the block.
        /// </summary>
        /// <value>
        /// The size of the block.
        /// </value>
        public dim3 BlockSize { get; set; }

        /// <summary>
        /// Gets or sets the stream id.
        /// </summary>
        /// <value>
        /// The stream id.
        /// </value>
        public int StreamId { get; set; }

        /// <summary>
        /// Provides the implementation for operations that invoke a member. Classes derived from the <see cref="T:System.Dynamic.DynamicObject"/> class can override this method to specify dynamic behavior for operations such as calling a method.
        /// </summary>
        /// <param name="binder">Provides information about the dynamic operation. The binder.Name property provides the name of the member on which the dynamic operation is performed. For example, for the statement sampleObject.SampleMethod(100), where sampleObject is an instance of the class derived from the <see cref="T:System.Dynamic.DynamicObject"/> class, binder.Name returns "SampleMethod". The binder.IgnoreCase property specifies whether the member name is case-sensitive.</param>
        /// <param name="args">The arguments that are passed to the object member during the invoke operation. For example, for the statement sampleObject.SampleMethod(100), where sampleObject is derived from the <see cref="T:System.Dynamic.DynamicObject"/> class, <paramref name="args"/> is equal to 100.</param>
        /// <param name="result">The result of the member invocation.</param>
        /// <returns>
        /// true if the operation is successful; otherwise, false. If this method returns false, the run-time binder of the language determines the behavior. (In most cases, a language-specific run-time exception is thrown.)
        /// </returns>
        public override bool TryInvokeMember(InvokeMemberBinder binder, object[] args, out object result)
        {
            result = null;
            GPU.LaunchAsync(GridSize, BlockSize, StreamId, binder.Name, args);
            return true;
        }

        /// <summary>
        /// Returns the enumeration of all global functions.
        /// </summary>
        /// <returns>
        /// A sequence that contains global function names.
        /// </returns>
        public override IEnumerable<string> GetDynamicMemberNames()
        {
            return GPU.GetFunctionNames();
        }
    }

}
#endif