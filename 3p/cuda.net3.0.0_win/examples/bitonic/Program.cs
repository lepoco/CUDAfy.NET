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

namespace bitonic
{
    class Program
    {
        //
        // A sorting network is a sorting algorith, where the sequence of comparisons
        // is not data-dependent. That makes them suitable for parallel implementations.
        //
        // Bitonic sort is one of the fastest sorting networks, consisting of o(n log^2 n)
        // comparators. It has a simple implemention and it's very efficient when sorting 
        // a small number of elements:
        //
        // http://citeseer.ist.psu.edu/blelloch98experimental.html
        //
        // This implementation is based on:
        //
        // http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
        //
        static void Main(string[] args)
        {
            const int NUM = 256;

            // Init CUDA, select 1st device.
            CUDA cuda = new CUDA(0, true);

            // create values
            int[] values = new int[NUM];
            Random rand = new Random();
            for (int i = 0; i < NUM; i++)
            {
                values[i] = rand.Next();
            }

            // allocate memory and copy to device
            CUdeviceptr dvalues = cuda.CopyHostToDevice<int>(values);

            // load module
            cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "bitonic.cubin"));
            CUfunction func = cuda.GetModuleFunction("bitonicSort");

            cuda.SetParameter(func, 0, (uint)dvalues.Pointer);
            cuda.SetParameterSize(func, (uint)IntPtr.Size);

            //bitonicSort<<<1, NUM, sizeof(int) * NUM>>>(dvalues);
            cuda.SetFunctionBlockShape(func, NUM, 1, 1);
            cuda.SetFunctionSharedSize(func, sizeof(int) * NUM);
            cuda.Launch(func, 1, 1);

            cuda.CopyDeviceToHost<int>(dvalues, values);
            cuda.Free(dvalues);

            bool passed = true;
            for (int i = 1; i < NUM; i++)
            {
                if (values[i - 1] > values[i])
                {
                    passed = false;
                    break;
                }
            }

            Console.WriteLine("Test {0}", passed ? "PASSED" : "FAILED");
        }
    }
}
