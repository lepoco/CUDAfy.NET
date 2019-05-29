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

namespace transpose
{
    class Program
    {
        private const int BLOCK_DIM = 16;
        static void Main(string[] args)
        {
            // Init and select 1st device.
            CUDA cuda = new CUDA(0, true);

            // load module
            //cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "transpose_kernel.cubin"));
            cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "transpose_kernel.ptx"));
            CUfunction transpose = cuda.GetModuleFunction("transpose");
            CUfunction transpose_naive = cuda.GetModuleFunction("transpose_naive");

            const int size_x = 4096;
            const int size_y = 4096;
            const int mem_size = sizeof(float) * size_x * size_y;

            float[] h_idata = new float[size_x * size_y];
            for (int i = 0; i < h_idata.Length; i++)
            {
                h_idata[i] = (float)i;
            }

            // allocate device memory
            // copy host memory to device
            CUdeviceptr d_idata = cuda.CopyHostToDevice<float>(h_idata);
            CUdeviceptr d_odata = cuda.Allocate<float>(h_idata);

            // setup execution parameters
            cuda.SetFunctionBlockShape(transpose_naive, BLOCK_DIM, BLOCK_DIM, 1);
            cuda.SetParameter(transpose_naive, 0, (uint)d_odata.Pointer);
            cuda.SetParameter(transpose_naive, IntPtr.Size, (uint)d_idata.Pointer);
            cuda.SetParameter(transpose_naive, IntPtr.Size * 2, (uint)size_x);
            cuda.SetParameter(transpose_naive, IntPtr.Size * 2 + 4, (uint)size_y);
            cuda.SetParameterSize(transpose_naive, (uint)(IntPtr.Size * 2 + 8));

            cuda.SetFunctionBlockShape(transpose, BLOCK_DIM, BLOCK_DIM, 1);
            cuda.SetParameter(transpose, 0, (uint)d_odata.Pointer);
            cuda.SetParameter(transpose, IntPtr.Size, (uint)d_idata.Pointer);
            cuda.SetParameter(transpose, IntPtr.Size * 2, (uint)size_x);
            cuda.SetParameter(transpose, IntPtr.Size * 2 + 4, (uint)size_y);
            cuda.SetParameterSize(transpose, (uint)(IntPtr.Size * 2 + 8));

            // warmup so we don't time CUDA startup
            cuda.Launch(transpose_naive, size_x / BLOCK_DIM, size_y / BLOCK_DIM);
            cuda.Launch(transpose, size_x / BLOCK_DIM, size_y / BLOCK_DIM);
            //System.Threading.Thread.Sleep(10);
            int numIterations = 100;

            Console.WriteLine("Transposing a {0} by {1} matrix of floats...", size_x, size_y);
            CUevent start = cuda.CreateEvent();
            CUevent end = cuda.CreateEvent();
            cuda.RecordEvent(start);
            for (int i = 0; i < numIterations; i++)
            {
                cuda.Launch(transpose_naive, size_x / BLOCK_DIM, size_y / BLOCK_DIM);
            }
            cuda.SynchronizeContext();
            cuda.RecordEvent(end);
            cuda.SynchronizeContext();
            float naiveTime = cuda.ElapsedTime(start, end);
            Console.WriteLine("Naive transpose average time:     {0} ms\n", naiveTime / numIterations);

            cuda.RecordEvent(start);
            for (int i = 0; i < numIterations; i++)
            {
                cuda.Launch(transpose, size_x / BLOCK_DIM, size_y / BLOCK_DIM);
            }
            cuda.SynchronizeContext();
            cuda.RecordEvent(end);
            cuda.SynchronizeContext();
            float optimizedTime = cuda.ElapsedTime(start, end);

            
            Console.WriteLine("Optimized transpose average time:     {0} ms\n", optimizedTime / numIterations);

            float[] h_odata = new float[size_x * size_y];
            cuda.CopyDeviceToHost<float>(d_odata, h_odata);

            float[] reference = new float[size_x * size_y];
            computeGold(reference, h_idata, size_x, size_y);

            bool res = CompareF(reference, h_odata, size_x * size_y);
            Console.WriteLine("Test {0}", res == true? "PASSED":"FAILED");

            cuda.Free(d_idata);
            cuda.Free(d_odata);

            Console.ReadKey();
        }

        private static bool CompareF(float[] reference, float[] h_odata, int p)
        {
            for (int i = 0; i < p; i++)
            {
                if (reference[i] - h_odata[i] > 0.00001f || reference[i] - h_odata[i] < -0.00001f)
                {
                    return false;
                }
            }

            return true;
        }

        private static void computeGold(float[] reference, float[] idata, int size_x, int size_y)
        {
            // transpose matrix
            for (int y = 0; y < size_y; ++y)
            {
                for (int x = 0; x < size_x; ++x)
                {
                    reference[(x * size_y) + y] = idata[(y * size_x) + x];
                }
            }  
        }
    }
}
