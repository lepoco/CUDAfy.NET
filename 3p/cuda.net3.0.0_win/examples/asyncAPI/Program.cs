/*
 * Copyright 2008 Company for Advanced Supercomputing Solutions Ltd (GASS).
 * All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to GASS ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * GASS MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  GASS DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL GASS BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 *
 *
 * This sample illustrates the usage of CUDA events for both GPU timing and
 * overlapping CPU and GPU execution.  Events are insterted into a stream
 * of CUDA calls.  Since CUDA stream calls are asynchronous, the CPU can
 * perform computations while GPU is executing (including DMA memcopies
 * between the host and device).  CPU can query CUDA events to determine
 * whether GPU has completed tasks.
 *
 */

using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

using GASS.CUDA;
using GASS.CUDA.Types;

namespace asyncAPI
{
    class Program
    {
        static bool CorrectOutput(int[] data, int x)
        {
            foreach (int e in data)
            {
                if (e != x)
                    return false;
            }

            return true;
        }

        static void Main(string[] args)
        {
            // Create a new instance of CUDA class, select 1st device.
            CUDA cuda = new CUDA(0, true);

            // Prepare parameters.
            int n = 16 * 1024 * 1024;
            uint nbytes = (uint)(n * sizeof(int));
            int value = 26;

            // allocate host memory
            int[] a = new int[n];

            // allocate device memory
            CUdeviceptr d_a = cuda.Allocate<int>(a);
            CUDADriver.cuMemsetD8(d_a, 0xff, nbytes);

            // load module
            cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "asyncAPI.ptx"));
            CUfunction func = cuda.GetModuleFunction("increment_kernel");

            // set kernel launch configuration
            cuda.SetFunctionBlockShape(func, 512, 1, 1);

            // create cuda event handles
            CUevent start = cuda.CreateEvent();
            CUevent stop = cuda.CreateEvent();

            // asynchronously issue work to the GPU (all to stream 0)
            CUstream stream = new CUstream();
            cuda.RecordEvent(start);
            cuda.CopyHostToDeviceAsync<int>(d_a, a, stream);

            // set parameters for kernel function
            cuda.SetParameter(func, 0, (uint)d_a.Pointer);
            cuda.SetParameter(func, IntPtr.Size, (uint)value);

            cuda.SetParameterSize(func, (uint)(IntPtr.Size + 4));

            // actually launch kernel
            cuda.LaunchAsync(func, n / 512, 1, stream);

            // wait for every thing to finish, then start copy back data
            cuda.CopyDeviceToHostAsync<int>(d_a, a, stream);

            cuda.RecordEvent(stop);

            // print the cpu and gpu times
            Console.WriteLine("time spent executing by the GPU: {0} ms", cuda.ElapsedTime(start, stop));

            // check the output for correctness
            if (CorrectOutput(a, value))
                Console.WriteLine("Test PASSED");
            else
                Console.WriteLine("Test FAILED");

            // release resources
            cuda.DestroyEvent(start);
            cuda.DestroyEvent(stop);
            cuda.Free(d_a);
        }
    }
}
