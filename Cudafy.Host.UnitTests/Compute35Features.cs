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
using System.Runtime.InteropServices;
using System.Diagnostics;
using Cudafy.Host;
using Cudafy.UnitTests;
using NUnit.Framework;
using Cudafy.Translator;
using Cudafy.Compilers;
using Cudafy.DynamicParallelism;
namespace Cudafy.Host.UnitTests
{
    [TestFixture]
    public class Compute35Features : CudafyUnitTest, ICudafyUnitTest
    {
        private CudafyModule _cm;

        private GPGPU _gpu;

        private const int N = 1024;

        [TestFixtureSetUp]        
        public void SetUp()
        {
            //var x = CompilerHelper.Create(ePlatform.x64, eArchitecture.OpenCL, eCudafyCompileMode.Default);
            var y = CompilerHelper.Create(ePlatform.x64, CudafyModes.Architecture, eCudafyCompileMode.DynamicParallelism); 
            _cm = CudafyTranslator.Cudafy(new CompileProperties[] {y}, this.GetType());
            Console.WriteLine(_cm.CompilerOutput);
            _cm.Serialize();
            _gpu = CudafyHost.GetDevice(y.Architecture, CudafyModes.DeviceId);
            _gpu.LoadModule(_cm);
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _gpu.FreeAll();
        }

        //[Cudafy]
        //public static void devKernel(GThread thread, int[] a, int[] c, short coeff)
        //{
        //    int tid = thread.blockIdx.x + N - N;
        //    if (tid < a.Length)
        //        c[tid] = a[tid] * coeff;
        //}

        [Cudafy]//(eCudafyType.Device)]
        public static void childKernel(GThread thread, int[] a, int[] c, short coeff)
        {
            int tid = thread.blockIdx.x;// +N - N;
            //if (tid < a.Length)
                c[tid] = a[tid] * coeff;
        }

        [Cudafy]
        public static int numberYouFirstThoughtOf()
        {
            return 42;
        }

        [Cudafy]
        public static void parentKernel(GThread thread, int[] a, int[] c, short coeff)
        {
            //childKernel(thread, a, c, coeff);
            int rc;
            //BROKEN thread.Launch(N / 2, numberYouFirstThoughtOf() * coeff, "childKernel", a, c, numberYouFirstThoughtOf() * coeff + 23 * a[0]);
            thread.Launch(N, 1, "childKernel", a, c, coeff * numberYouFirstThoughtOf());//a[0]);//numberYouFirstThoughtOf() * coeff + 23 * 
            rc = thread.SynchronizeDevice();
            int count = 0;
            rc = thread.GetDeviceCount(ref count);
        }

        [SetUp]
        public void TestSetUp()
        {

        }

        [TearDown]
        public void TestTearDown()
        {

        }

        [Test]
        public void TestDynamicParallelism()
        {
            int[] a = new int[N];
            int[] c = new int[N];
            short coeff = 8;
            for (int i = 0; i < N; i++)
                a[i] = i;
            int[] dev_a = _gpu.CopyToDevice(a);
            int[] dev_c = _gpu.Allocate<int>(c);
            _gpu.Launch(N, 1, "parentKernel", dev_a, dev_c, coeff);
            _gpu.CopyFromDevice(dev_c, c);
            for (int i = 0; i < N; i++)
                Assert.AreEqual(coeff * numberYouFirstThoughtOf() * a[i], c[i]);
            _gpu.Free(dev_a);      
        }
    }
}
