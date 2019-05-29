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
using Cudafy.Rand;
using Cudafy.Translator;
namespace Cudafy.Host.UnitTests
{
    public class CURANDTests
    {
        public static void Basics()
        {
            CudafyModule cm = CudafyTranslator.Cudafy(CudafyModes.Architecture);
            Console.WriteLine(cm.CompilerOutput);
            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(cm);

            int i, total;
            RandStateXORWOW[] devStates = gpu.Allocate<RandStateXORWOW>(64 * 64);
            int[] devResults = gpu.Allocate<int>(64 * 64);
            int[] hostResults = new int[64 * 64];

            gpu.Set(devResults);
#if !NET35
            gpu.Launch(64, 64).setup_kernel(devStates);
            for (i = 0; i < 10; i++)
                gpu.Launch(64, 64).generate_kernel(devStates, devResults);
#else
            gpu.Launch(64, 64, "setup_kernel", devStates);
            for (i = 0; i < 10; i++)
                gpu.Launch(64, 64, "generate_kernel", devStates, devResults);
#endif


            gpu.CopyFromDevice(devResults, hostResults);

            total = 0;
            for (i = 0; i < 64 * 64; i++)
                total += hostResults[i];
            Console.WriteLine("Fraction with low bit set was {0}", (float) total / (64.0f * 64.0f * 100000.0f * 10.0f));

            gpu.FreeAll();
        }


        [Cudafy]
        public static void setup_kernel(GThread thread, RandStateXORWOW[] state)
        {
            int id = thread.threadIdx.x + thread.blockIdx.x * 64;
            thread.curand_init(1234, (ulong)id, 0, ref state[id]);
        }

        [Cudafy]
        public static void generate_kernel(GThread thread, RandStateXORWOW[] state, int[] result)
        {
            int id = thread.threadIdx.x + thread.blockIdx .x * 64;
            int count = 0;
            uint x = 0;

            /* Copy state to local memory for efficiency */
            RandStateXORWOW localState = state[id];
            /* Generate pseudo - random unsigned ints */
            for (int n = 0; n < 100000; n++)
            {
                x = thread.curand(ref localState);
                /* Check if low bit set */
                if ((x & 1) == 1)
                {
                    count++;
                }
            }
            /* Copy state back to global memory */
            state[id] = localState;
            /* Store results */
            result[id] += count;
        }
    }
}
