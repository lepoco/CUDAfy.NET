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
using System.Threading;
using System.Diagnostics;
using Cudafy.Types;
using Cudafy.Host;
using Cudafy.UnitTests;
using NUnit.Framework;

using GASS.CUDA;
using GASS.CUDA.Tools;
namespace Cudafy.Host.UnitTests
{
    [TestFixture]
    public class MultithreadedTests : CudafyUnitTest, ICudafyUnitTest
    {
        private const int N = 1024 * 1024;

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.CreateDevice(CudafyModes.Target, CudafyModes.DeviceId);
            _uintBufferIn1 = new uint[N];
            _uintBufferOut1 = new uint[N];
            _uintBufferIn2 = new uint[N];
            _uintBufferOut2 = new uint[N];
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _gpu.FreeAll();
        }

        private GPGPU _gpu;

        private uint[] _gpuuintBufferIn1;

        private uint[] _uintBufferIn1;

        private uint[] _uintBufferOut1;

        private uint[] _gpuuintBufferIn2;

        private uint[] _uintBufferIn2;

        private uint[] _uintBufferOut2;

        private uint[] _gpuuintBufferIn3;

        private uint[] _gpuuintBufferIn4;

        private void SetInputs()
        {
            Random rand = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < N; i++)
            {
                double r = rand.NextDouble();
                _uintBufferIn1[i] = (uint)(r * uint.MaxValue);
                _uintBufferIn2[i] = (uint)(r * uint.MaxValue / 2);
            }
        }

        private void ClearOutputs()
        {
            for (int i = 0; i < N; i++)
            {
                _uintBufferOut1[i] = 0;
                _uintBufferOut2[i] = 0;
            }
        }

        [Test]
        public void Test_SingleThreadCopy()
        {
            _gpuuintBufferIn1 = _gpu.CopyToDevice(_uintBufferIn1);
            _gpu.CopyFromDevice(_gpuuintBufferIn1, _uintBufferOut1);
            Assert.IsTrue(Compare(_uintBufferIn1, _uintBufferOut1));
            ClearOutputs();
            _gpu.FreeAll();
        }

        [Test]
        public void Test_TwoThreadCopy()
        {
            _gpu = CudafyHost.GetDevice(eGPUType.Cuda);
            _gpuuintBufferIn3 = _gpu.Allocate(_uintBufferIn1);
            _gpuuintBufferIn4 = _gpu.Allocate(_uintBufferIn1);
            _gpu.EnableMultithreading();
            bool j1 = false;
            bool j2 = false;
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine(i);
                SetInputs();
                ClearOutputs();
                Thread t1 = new Thread(Test_TwoThreadCopy_Thread1);
                Thread t2 = new Thread(Test_TwoThreadCopy_Thread2);
                t1.Start();
                t2.Start();
                j1 = t1.Join(10000);
                j2 = t2.Join(10000);
                if (!j1 || !j2)
                    break;
            }

            _gpu.DisableMultithreading();           
            _gpu.FreeAll();
            Assert.IsTrue(j1);
            Assert.IsTrue(j2);
        }

        //private CUDAContextSynchronizer _ccs;

        private void Test_TwoThreadCopy_Thread1()
        {
            try
            {
                //Debug.WriteLine("thread 1, A");
                _gpu.Lock();
                //Debug.WriteLine("thread 1, B");
                _gpuuintBufferIn1 = _gpu.CopyToDevice(_uintBufferIn1);
                _gpu.CopyOnDevice(_gpuuintBufferIn1, _gpuuintBufferIn3);
                //Debug.WriteLine("thread 1, C");
                //Debug.WriteLine(string.Format("Thread {0}: {1} ticks", Thread.CurrentThread.ManagedThreadId, Environment.TickCount));
                _gpu.CopyFromDevice(_gpuuintBufferIn3, _uintBufferOut1);
                //Debug.WriteLine("thread 1, D");
                Assert.IsTrue(Compare(_uintBufferIn1, _uintBufferOut1));
                _gpu.Free(_gpuuintBufferIn1);
                //Debug.WriteLine("thread 1, E");
                _gpu.Unlock();
                //Debug.WriteLine("thread 1, F");
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.ToString());
            }
        }
        
        private void Test_TwoThreadCopy_Thread2()
        {
            try
            {
                //Debug.WriteLine("thread 2, A");
                _gpu.Lock();
                //Debug.WriteLine("thread 2, B");
                _gpuuintBufferIn2 = _gpu.CopyToDevice(_uintBufferIn2);
                _gpu.CopyOnDevice(_gpuuintBufferIn2, _gpuuintBufferIn4);
                //Debug.WriteLine("thread 2, C");
                //Debug.WriteLine(string.Format("Thread {0}: {1} ticks", Thread.CurrentThread.ManagedThreadId, Environment.TickCount));
                _gpu.CopyFromDevice(_gpuuintBufferIn4, _uintBufferOut2);
                //Debug.WriteLine("thread 2, D");
                Assert.IsTrue(Compare(_uintBufferIn2, _uintBufferOut2));
                _gpu.Free(_gpuuintBufferIn2);
                //Debug.WriteLine("thread 2, E");
                _gpu.Unlock();
                //Debug.WriteLine("thread 2, F");
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.ToString());
            }
        }


        public void TestSetUp()
        {
         
        }

        public void TestTearDown()
        {
            
        }
    }
}
