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
 */

using System;
using System.Collections.Generic;
using System.Text;

using GASS.CUDA;
using GASS.CUDA.Types;
using System.IO;

namespace clock
{
    class Program
    {
        // The following descriptions were taken from NVIDIA example "clock".
        //
        // This example shows how to use the clock function to measure the performance of 
        // a kernel accurately.
        // 
        // Blocks are executed in parallel and out of order. Since there's no synchronization
        // mechanism between blocks, we measure the clock once for each block. The clock 
        // samples are written to device memory.
        private const int NUM_BLOCKS = 64;
        private const int NUM_THREADS = 256;

        // It's interesting to change the number of blocks and the number of threads to 
        // understand how to keep the hardware busy.
        //
        // Here are some numbers I get on my G80:
        //    blocks - clocks
        //    1 - 3096
        //    8 - 3232
        //    16 - 3364
        //    32 - 4615
        //    64 - 9981
        //
        // With less than 16 blocks some of the multiprocessors of the device are idle. With
        // more than 16 you are using all the multiprocessors, but there's only one block per
        // multiprocessor and that doesn't allow you to hide the latency of the memory. With
        // more than 32 the speed scales linearly.
        static void Main(string[] args)
        {
            // Init CUDA, select 1st device.
            CUDA cuda = new CUDA(0, true);

            // load module
            cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "clock_kernel.cubin"));
            CUfunction func = cuda.GetModuleFunction("timedReduction");

            int[] timer = new int[NUM_BLOCKS * 2];
            float[] input = new float[NUM_THREADS * 2];

            for (int i = 0; i < NUM_THREADS * 2; i++)
            {
                input[i] = (float)i;
            }

            CUdeviceptr dinput = cuda.CopyHostToDevice<float>(input);
            CUdeviceptr doutput = cuda.Allocate((uint)(sizeof(float) * NUM_BLOCKS));
            CUdeviceptr dtimer = cuda.Allocate<int>(timer);

            cuda.SetParameter(func, 0, (uint)dinput.Pointer);
            cuda.SetParameter(func, IntPtr.Size, (uint)doutput.Pointer);
            cuda.SetParameter(func, IntPtr.Size*2, (uint)dtimer.Pointer);
            cuda.SetParameterSize(func, (uint)(IntPtr.Size*3));

            //timedReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS>>>(dinput, doutput, dtimer);
            cuda.SetFunctionBlockShape(func, NUM_THREADS, 1, 1);
            cuda.SetFunctionSharedSize(func, (uint)(sizeof(float) * 2 * NUM_THREADS));
            cuda.Launch(func, NUM_BLOCKS, 1);

            cuda.CopyDeviceToHost<int>(dtimer, timer);

            cuda.Free(dinput);
            cuda.Free(doutput);
            cuda.Free(dtimer);

            foreach (int i in timer)
            {
                Console.WriteLine(i);
            }

            Console.WriteLine("Test PASSED");

            int minStart = timer[0];
            int maxEnd = timer[NUM_BLOCKS];
            for (int i = 1; i < NUM_BLOCKS; i++)
            {
                minStart = timer[i] < minStart ? timer[i] : minStart;
                maxEnd = timer[NUM_BLOCKS + i] > maxEnd ? timer[NUM_BLOCKS + i] : maxEnd;
            }

            Console.WriteLine("time = {0}", maxEnd - minStart);
        }
    }
}
