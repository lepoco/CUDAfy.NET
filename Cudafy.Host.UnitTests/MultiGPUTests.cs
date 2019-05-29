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
using System.IO;
using System.Threading;
using System.Diagnostics;
using Cudafy.Types;
using Cudafy.Translator;
using Cudafy.Host;
using Cudafy.UnitTests;
using NUnit.Framework;

using GASS.CUDA;
using GASS.CUDA.Tools;
namespace Cudafy.Host.UnitTests
{
    [TestFixture]
    public class MultiGPUTests : CudafyUnitTest, ICudafyUnitTest
    {
        private const int N = 1024 * 1024;

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu0 = CudafyHost.CreateDevice(CudafyModes.Target, 0);
            _gpu1 = CudafyHost.CreateDevice(CudafyModes.Target, 1);
            _uintBufferIn0 = new uint[N];
            _uintBufferOut0 = new uint[N];
            _uintBufferIn1 = new uint[N];
            _uintBufferOut1 = new uint[N];
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _gpu0.FreeAll();
        }

        private GPGPU _gpu0;

        private GPGPU _gpu1;

        private uint[] _gpuuintBufferIn0;

        private uint[] _uintBufferIn0;

        private uint[] _uintBufferOut0;

        private uint[] _gpuuintBufferIn1;

        private uint[] _uintBufferIn1;

        private uint[] _uintBufferOut1;

        private void SetInputs()
        {
            Random rand = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < N; i++)
            {
                double r = rand.NextDouble();
                _uintBufferIn0[i] = (uint)(r * uint.MaxValue);
            }
        }

        private void ClearOutputsAndGPU(int deviceId)
        {
            if (deviceId == 0)
            {
                Array.Clear(_uintBufferOut0, 0, _uintBufferOut0.Length);
                _gpu0.FreeAll();
            }
            else
            {
                Array.Clear(_uintBufferOut1, 0, _uintBufferOut1.Length);
                _gpu1.FreeAll();
            }
        }

        [Test]
        public void Test_SingleThreadOneGPU_0()
        {
            _gpu0.SetCurrentContext();
            _gpuuintBufferIn0 = _gpu0.CopyToDevice(_uintBufferIn0);
            _gpu0.CopyFromDevice(_gpuuintBufferIn0, _uintBufferOut0);
            Assert.IsTrue(Compare(_uintBufferIn0, _uintBufferOut0));
            ClearOutputsAndGPU(0);
        }

        [Test]
        public void Test_SingleThreadOneGPU_1()
        {
            _gpu1.SetCurrentContext();
            _gpuuintBufferIn0 = _gpu1.CopyToDevice(_uintBufferIn0);
            _gpu1.CopyFromDevice(_gpuuintBufferIn0, _uintBufferOut0);
            Assert.IsTrue(Compare(_uintBufferIn0, _uintBufferOut0));
            ClearOutputsAndGPU(1);
        }

        [Test]
        public void Test_SingleThreadTwoGPU()
        {
            _gpu0.SetCurrentContext();
            _gpuuintBufferIn0 = _gpu0.CopyToDevice(_uintBufferIn0);
            _gpu1.SetCurrentContext();
            _gpuuintBufferIn1 = _gpu1.CopyToDevice(_uintBufferIn1);
            _gpu0.SetCurrentContext();
            _gpu0.CopyFromDevice(_gpuuintBufferIn0, _uintBufferOut0);
            _gpu1.SetCurrentContext();
            _gpu1.CopyFromDevice(_gpuuintBufferIn1, _uintBufferOut1);
            Assert.IsTrue(Compare(_uintBufferIn0, _uintBufferOut0));
            Assert.IsTrue(Compare(_uintBufferIn1, _uintBufferOut1));
            ClearOutputsAndGPU(0);
            ClearOutputsAndGPU(1);
        }

        [Test]
        public void Test_TwoThreadTwoGPU()
        {
            _gpu0 = CudafyHost.CreateDevice(CudafyModes.Target, 0);
            _gpu1 = CudafyHost.CreateDevice(CudafyModes.Target, 1);
            _gpu0.EnableMultithreading();
            _gpu1.EnableMultithreading();
            bool j1 = false;
            bool j2 = false;
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine(i);
                Thread t1 = new Thread(Test_TwoThreadTwoGPU_Thread0);
                Thread t2 = new Thread(Test_TwoThreadTwoGPU_Thread1);
                t1.Start();
                t2.Start();
                j1 = t1.Join(10000);
                j2 = t2.Join(10000);
                if (!j1 || !j2)
                    break;
            }
            _gpu0.DisableMultithreading();
            _gpu0.FreeAll();
            _gpu1.DisableMultithreading();
            _gpu1.FreeAll();
            Assert.IsTrue(j1);
            Assert.IsTrue(j2);
        }

        private void Test_TwoThreadTwoGPU_Thread0()
        {
            _gpu0.Lock();
            _gpuuintBufferIn0 = _gpu0.CopyToDevice(_uintBufferIn0);
            _gpu0.CopyFromDevice(_gpuuintBufferIn0, _uintBufferOut0);
            Assert.IsTrue(Compare(_uintBufferIn0, _uintBufferOut0));
            _gpu0.Free(_gpuuintBufferIn0);
            _gpu0.Unlock();
        }

        private void Test_TwoThreadTwoGPU_Thread1()
        {
            _gpu1.Lock();
            _gpuuintBufferIn1 = _gpu1.CopyToDevice(_uintBufferIn1);
            _gpu1.CopyFromDevice(_gpuuintBufferIn1, _uintBufferOut1);
            Assert.IsTrue(Compare(_uintBufferIn1, _uintBufferOut1));
            _gpu1.Free(_gpuuintBufferIn1);
            _gpu1.Unlock();
        }


        [Test]
        public void Test_TwoThreadTwoGPUVer2()
        {
            eArchitecture arch = CudafyModes.Target == eGPUType.OpenCL ? eArchitecture.OpenCL : eArchitecture.sm_11;
                      
            _gpu0 = CudafyHost.GetDevice(CudafyModes.Target, 0);
            var cm = CudafyTranslator.Cudafy(arch, typeof(MultiGPUTests));
            _gpu0.SetCurrentContext();
            _gpu0.LoadModule(cm);
            _gpuuintBufferIn0 = _gpu0.Allocate(_uintBufferIn0);
            
            _gpu1 = CudafyHost.GetDevice(CudafyModes.Target, 1);
            // Cannot load same module to two devices, therefore need to clone.
            var cm1 = cm.Clone();      
            _gpu1.SetCurrentContext();
            _gpu1.LoadModule(cm1);
            _gpuuintBufferIn1 = _gpu1.Allocate(_uintBufferIn1);

            _gpu0.EnableMultithreading();
            _gpu1.EnableMultithreading();
            bool j1 = false;
            bool j2 = false;
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine(i);
                Thread t1 = new Thread(Test_TwoThreadTwoGPU_Thread0V2);
                Thread t2 = new Thread(Test_TwoThreadTwoGPU_Thread1V2);
                t1.Start();
                t2.Start();
                j1 = t1.Join(10000);
                j2 = t2.Join(10000);
                if (!j1 || !j2)
                    break;
            }
            _gpu0.DisableMultithreading();
            _gpu0.FreeAll();
            _gpu1.DisableMultithreading();
            _gpu1.FreeAll();
            Assert.IsTrue(j1);
            Assert.IsTrue(j2);
        }

        [Cudafy]
        public static void SomeKernel(GThread thread, uint[] a)
        {
            int id = thread.get_global_id(0);
            a[id] = a[id] * 42;
        }

        private void Test_TwoThreadTwoGPU_Thread0V2()
        {
            _gpu0.Lock();
            _gpu0.CopyToDevice(_uintBufferIn0, _gpuuintBufferIn0);
            _gpu0.Launch(1, 1024, "SomeKernel", _gpuuintBufferIn0);
            _gpu0.CopyFromDevice(_gpuuintBufferIn0, _uintBufferOut0);
            Assert.IsTrue(Compare(_uintBufferIn0, _uintBufferOut0));
            _gpu0.Unlock();
        }

        private void Test_TwoThreadTwoGPU_Thread1V2()
        {
            _gpu1.Lock();
            _gpu1.CopyToDevice(_uintBufferIn1, _gpuuintBufferIn1);
            _gpu1.Launch(1, 1024, "SomeKernel", _gpuuintBufferIn1);
            _gpu1.CopyFromDevice(_gpuuintBufferIn1, _uintBufferOut1);
            Assert.IsTrue(Compare(_uintBufferIn1, _uintBufferOut1));
            _gpu1.Unlock();
        }

        [Test]
        public void Test_SingleThreadGPUtoGPU()
        {
            if (!_gpu0.CanAccessPeer(_gpu1))
            {
                Console.WriteLine("Device not supporting this, so skip.");
                return;
            }
            Random r = new Random();
            for (int i = 0; i < _uintBufferIn0.Length; i++)
                _uintBufferIn0[i] = (uint)r.Next(Int32.MaxValue);

            _gpu0.SetCurrentContext();
            _gpuuintBufferIn0 = _gpu0.CopyToDevice(_uintBufferIn0);
            _gpu1.SetCurrentContext();
            _gpuuintBufferIn1 = _gpu1.CopyToDevice(_uintBufferIn1);
            
            _gpu0.SetCurrentContext();
            long loops = 500;
            Stopwatch sw = Stopwatch.StartNew();

            for (int i = 0; i < loops; i++)
                _gpu0.CopyDeviceToDevice(_gpuuintBufferIn0, 0, _gpu1, _gpuuintBufferIn1, 0, _uintBufferIn0.Length);
            sw.Stop();

            float mbps = (float)((long)_uintBufferIn0.Length * sizeof(int) * loops) / (float)(sw.ElapsedMilliseconds * 1000);
            Console.WriteLine(mbps);
            _gpu1.SetCurrentContext();
            _gpu1.CopyFromDevice(_gpuuintBufferIn1, _uintBufferOut1);

            Assert.IsTrue(Compare(_uintBufferIn0, _uintBufferOut1));
            ClearOutputsAndGPU(0);
            ClearOutputsAndGPU(1);
        }

        [Test]
        public void Test_CanAccessPeer()
        {
            _gpu0.SetCurrentContext();
                       
            bool rc = _gpu0.CanAccessPeer(_gpu1);
            if (rc)
            {
                _gpu0.EnablePeerAccess(_gpu1);
                _gpu0.DisablePeerAccess(_gpu1);
            }
        }

        //[Test]
        //public void Test_FourThreadTwoGPU()
        //{
        //    _gpu0 = CudafyHost.CreateDevice(eGPUType.Cuda, 0);
        //    _gpu0.EnableMultithreading();
        //    _gpu1 = CudafyHost.CreateDevice(eGPUType.Cuda, 1);
        //    _gpu1.EnableMultithreading();
        //    bool j1 = false;
        //    bool j2 = false;
        //    for (int i = 0; i < 10; i++)
        //    {
        //        Console.WriteLine(i);
        //        Thread t1 = new Thread(Test_TwoThreadCopy_Thread1);
        //        Thread t2 = new Thread(Test_TwoThreadCopy_Thread2);
        //        t1.Start();
        //        t2.Start();
        //        j1 = t1.Join(10000);
        //        j2 = t2.Join(10000);
        //        if (!j1 || !j2)
        //            break;
        //    }

        //    _gpu0.DisableMultithreading();           
        //    _gpu0.FreeAll();
        //    Assert.IsTrue(j1);
        //    Assert.IsTrue(j2);
        //}

        //private void Test_TwoThreadCopy_Thread1()
        //{
        //    _gpu0.Lock();
        //    _gpuuintBufferIn0 = _gpu0.CopyToDevice(_uintBufferIn0);
        //    _gpu0.CopyFromDevice(_gpuuintBufferIn0, _uintBufferOut0);
        //    Assert.IsTrue(Compare(_uintBufferIn0, _uintBufferOut0));
        //    _gpu0.Free(_gpuuintBufferIn0);
        //    _gpu0.Unlock();
        //}
        
        //private void Test_TwoThreadCopy_Thread2()
        //{
        //    _gpu0.Lock();
        //    _gpuuintBufferIn1 = _gpu0.CopyToDevice(_uintBufferIn1);
        //    _gpu0.CopyFromDevice(_gpuuintBufferIn1, _uintBufferOut1);
        //    Assert.IsTrue(Compare(_uintBufferIn1, _uintBufferOut1));
        //    _gpu0.Free(_gpuuintBufferIn1);
        //    _gpu0.Unlock();
        //}


        public void TestSetUp()
        {
    
        }

        public void TestTearDown()
        {
       
        }
    }
}
