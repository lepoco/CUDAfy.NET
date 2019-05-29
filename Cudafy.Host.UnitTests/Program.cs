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
using System.Reflection;
using System.IO;
using Cudafy.UnitTests;
using GASS.CUDA.FFT;
using NUnit.Framework;
using Cudafy.Host;
using Cudafy.Translator;
using Cudafy.Compilers;
namespace Cudafy.Host.UnitTests
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {             
                CudafyModes.DeviceId = 0;
                GPGPU gpu = CudafyHost.GetDevice(eGPUType.Cuda, CudafyModes.DeviceId);
                CudafyModes.Architecture = gpu.GetArchitecture(); //eArchitecture.sm_35; // *** Change this to the architecture of your target board ***
                CudafyModes.Target = CompilerHelper.GetGPUType(CudafyModes.Architecture);
                Console.WriteLine("{0}: Arch: {1}, Type: {2}, ID: {3}", gpu.GetDeviceProperties(false).Name, CudafyModes.Architecture, CudafyModes.Target, CudafyModes.DeviceId);

                if (CudafyModes.Target == eGPUType.Cuda)
                {
                    CURANDTests.Basics();
                }

                SIMDFunctionTests sft = new SIMDFunctionTests();
                CudafyUnitTest.PerformAllTests(sft);

                StringTests st = new StringTests();
                CudafyUnitTest.PerformAllTests(st);

                BasicFunctionTests bft = new BasicFunctionTests();
                CudafyUnitTest.PerformAllTests(bft);

                GMathUnitTests gmu = new GMathUnitTests();
                CudafyUnitTest.PerformAllTests(gmu);

                MultithreadedTests mtt = new MultithreadedTests();
                CudafyUnitTest.PerformAllTests(mtt);

                CopyTests1D ct1d = new CopyTests1D();
                CudafyUnitTest.PerformAllTests(ct1d);

                GPGPUTests gput = new GPGPUTests();
                CudafyUnitTest.PerformAllTests(gput);

                if (CudafyHost.GetDeviceCount(CudafyModes.Target) > 1)
                {
                    MultiGPUTests mgt = new MultiGPUTests();
                    CudafyUnitTest.PerformAllTests(mgt);
                }

                if (CudafyModes.Architecture >= eArchitecture.sm_30 && CudafyModes.Target == eGPUType.Cuda)
                {
                    WarpShuffleTests wst = new WarpShuffleTests();
                    CudafyUnitTest.PerformAllTests(wst);
                }

                if (CudafyModes.Architecture >= eArchitecture.sm_35 && CudafyModes.Target == eGPUType.Cuda)
                {
                    Compute35Features c35f = new Compute35Features();
                    CudafyUnitTest.PerformAllTests(c35f);
                }

                Console.WriteLine("Done");
                Console.ReadLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                Console.ReadLine();
            }                      
        }
    }
}
