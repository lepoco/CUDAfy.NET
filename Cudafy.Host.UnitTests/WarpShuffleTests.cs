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
using Cudafy.Translator;
using Cudafy.UnitTests;
using Cudafy.WarpShuffleFunctions;
using NUnit.Framework;
using System;
using System.Linq;

namespace Cudafy.Host.UnitTests
{
    public class WarpShuffleTests : CudafyUnitTest, ICudafyUnitTest
    {
        const int WARP_SIZE = 32;
        int[] inputIntArray, d_inputIntArray, d_outputIntArray, gpuIntResult, cpuIntResult;
        float[] inputFloatArray, d_inputFloatArray, d_outputFloatArray, gpuFloatResult, cpuFloatResult;

        private CudafyModule _cm;

        private GPGPU _gpu;

        [TestFixtureSetUp]
        public void SetUp()
        {
            //CudafyModes.Architecture = eArchitecture.sm_30;
            _gpu = CudafyHost.GetDevice(eArchitecture.sm_30, CudafyModes.DeviceId);
            Assert.IsFalse(_gpu is OpenCLDevice, "OpenCL devices are not supported.");

            _cm = CudafyModule.TryDeserialize();
            if (_cm == null || !_cm.TryVerifyChecksums())
            {
                _cm = CudafyTranslator.Cudafy(eArchitecture.sm_30);
                Console.WriteLine(_cm.CompilerOutput);
                _cm.TrySerialize();
            }

            _gpu.LoadModule(_cm);

            inputIntArray = new int[] { 0x17, 0x01, 0x7f, 0xd1, 0xfe, 0x23, 0x2c, 0xa0, 0x00, 0xcf, 0xaa, 0x7a, 0x35, 0xf4, 0x04, 0xbc,
                                        0xe9, 0x6d, 0xb2, 0x55, 0xb0, 0xc8, 0x10, 0x49, 0x76, 0x17, 0x92, 0xab, 0xf3, 0xf2, 0xab, 0xcb}; // arbitrary values
            d_inputIntArray = _gpu.CopyToDevice(inputIntArray);
            d_outputIntArray = _gpu.Allocate<int>(WARP_SIZE);
            gpuIntResult = new int[WARP_SIZE];
            cpuIntResult = new int[WARP_SIZE];

            inputFloatArray = new float[] { 1.7f, -37.03f, 2147.6436f, -0.1f, 7.7f, 99.99f, -809.142f, -0.1115f,
                                            1.0f, 2.0f, 3.0f, 5.0f, 7.5f, 0.1001f, 11.119f, -9.0f,
                                            7749.9847f, -860249.118843f, 0.0f, -2727745.586215f, 12.0f, -11.0f, 77.77f, 22.0f,
                                            377.1112f, -377.1112f, 0.12345f, -0.12345f, 0.11111f, -0.11111f, 700000f, -14f}; // arbitrary values
            d_inputFloatArray = _gpu.CopyToDevice(inputFloatArray);
            d_outputFloatArray = _gpu.Allocate<float>(WARP_SIZE);
            gpuFloatResult = new float[WARP_SIZE];
            cpuFloatResult = new float[WARP_SIZE];
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _gpu.FreeAll();
        }

        [SetUp]
        public void TestSetUp() { }

        [TearDown]
        public void TestTearDown() { }

        [Test]
        public void IntShuffle()
        {
            for (int shuffleLane = 0; shuffleLane < WARP_SIZE; shuffleLane++)
            {
                for (int baseLane = 0; baseLane < WARP_SIZE; baseLane++) // cpu
                {
                    cpuIntResult[baseLane] = inputIntArray[shuffleLane];
                }
                _gpu.Launch(1, WARP_SIZE).unitTest_IntShuffle(d_inputIntArray, d_outputIntArray, shuffleLane); // gpu
                _gpu.CopyFromDevice<int>(d_outputIntArray, gpuIntResult);
                Assert.True(gpuIntResult.SequenceEqual(cpuIntResult));
            }
        }
        [Cudafy]
        public static void unitTest_IntShuffle(GThread thread, int[] d_inputArray, int[] d_outputArray, int shuffleLane)
        {
            int input = d_inputArray[thread.threadIdx.x];
            d_outputArray[thread.threadIdx.x] = thread.Shuffle(input, shuffleLane);
        }

        [Test]
        public void IntShuffleUp()
        {
            for (int delta = 0; delta < WARP_SIZE; delta++)
            {
                for (int baseLane = 0; baseLane < WARP_SIZE; baseLane++) // cpu
                {
                    cpuIntResult[baseLane] = inputIntArray[baseLane - delta < 0 ? baseLane : baseLane - delta];
                }
                _gpu.Launch(1, WARP_SIZE).unitTest_IntShuffleUp(d_inputIntArray, d_outputIntArray, delta); // gpu
                _gpu.CopyFromDevice<int>(d_outputIntArray, gpuIntResult);
                Assert.True(gpuIntResult.SequenceEqual(cpuIntResult));
            }
        }
        [Cudafy]
        public static void unitTest_IntShuffleUp(GThread thread, int[] d_inputArray, int[] d_outputArray, uint delta)
        {
            int input = d_inputArray[thread.threadIdx.x];
            d_outputArray[thread.threadIdx.x] = thread.ShuffleUp(input, delta);
        }

        [Test]
        public void IntShuffleDown()
        {
            for (int delta = 0; delta < WARP_SIZE; delta++)
            {
                for (int baseLane = 0; baseLane < WARP_SIZE; baseLane++) // cpu
                {
                    cpuIntResult[baseLane] = inputIntArray[baseLane + delta > WARP_SIZE - 1 ? baseLane : baseLane + delta];
                }
                _gpu.Launch(1, WARP_SIZE).unitTest_IntShuffleDown(d_inputIntArray, d_outputIntArray, delta); // gpu
                _gpu.CopyFromDevice<int>(d_outputIntArray, gpuIntResult);
                Assert.True(gpuIntResult.SequenceEqual(cpuIntResult));
            }
        }
        [Cudafy]
        public static void unitTest_IntShuffleDown(GThread thread, int[] d_inputArray, int[] d_outputArray, uint delta)
        {
            int input = d_inputArray[thread.threadIdx.x];
            d_outputArray[thread.threadIdx.x] = thread.ShuffleDown(input, delta);
        }

        [Test]
        public void IntShuffleXor()
        {
            for (int laneMask = 0; laneMask < WARP_SIZE; laneMask++)
            {
                for (int baseLane = 0; baseLane < WARP_SIZE; baseLane++) // cpu
                {
                    cpuIntResult[baseLane] = inputIntArray[baseLane ^ laneMask];
                }
                _gpu.Launch(1, WARP_SIZE).unitTest_IntShuffleXor(d_inputIntArray, d_outputIntArray, laneMask); // gpu
                _gpu.CopyFromDevice<int>(d_outputIntArray, gpuIntResult);
                Assert.True(gpuIntResult.SequenceEqual(cpuIntResult));
            }
        }
        [Cudafy]
        public static void unitTest_IntShuffleXor(GThread thread, int[] d_inputArray, int[] d_outputArray, int laneMask)
        {
            int input = d_inputArray[thread.threadIdx.x];
            d_outputArray[thread.threadIdx.x] = thread.ShuffleXor(input, laneMask);
        }

        [Test]
        public void FloatShuffle()
        {
            for (int shuffleLane = 0; shuffleLane < WARP_SIZE; shuffleLane++)
            {
                for (int baseLane = 0; baseLane < WARP_SIZE; baseLane++) // cpu
                {
                    cpuFloatResult[baseLane] = inputFloatArray[shuffleLane];
                }
                _gpu.Launch(1, WARP_SIZE).unitTest_FloatShuffle(d_inputFloatArray, d_outputFloatArray, shuffleLane); // gpu
                _gpu.CopyFromDevice<float>(d_outputFloatArray, gpuFloatResult);
                Assert.True(gpuFloatResult.SequenceEqual(cpuFloatResult));
            }
        }
        [Cudafy]
        public static void unitTest_FloatShuffle(GThread thread, float[] d_inputArray, float[] d_outputArray, int shuffleLane)
        {
            float input = d_inputArray[thread.threadIdx.x];
            d_outputArray[thread.threadIdx.x] = thread.Shuffle(input, shuffleLane);
        }

        [Test]
        public void FloatShuffleUp()
        {
            for (int delta = 0; delta < WARP_SIZE; delta++)
            {
                for (int baseLane = 0; baseLane < WARP_SIZE; baseLane++) // cpu
                {
                    cpuFloatResult[baseLane] = inputFloatArray[baseLane - delta < 0 ? baseLane : baseLane - delta];
                }
                _gpu.Launch(1, WARP_SIZE).unitTest_FloatShuffleUp(d_inputFloatArray, d_outputFloatArray, delta); // gpu
                _gpu.CopyFromDevice<float>(d_outputFloatArray, gpuFloatResult);
                Assert.True(gpuFloatResult.SequenceEqual(cpuFloatResult));
            }
        }
        [Cudafy]
        public static void unitTest_FloatShuffleUp(GThread thread, float[] d_inputArray, float[] d_outputArray, uint delta)
        {
            float input = d_inputArray[thread.threadIdx.x];
            d_outputArray[thread.threadIdx.x] = thread.ShuffleUp(input, delta);
        }

        [Test]
        public void FloatShuffleDown()
        {
            for (int delta = 0; delta < WARP_SIZE; delta++)
            {
                for (int baseLane = 0; baseLane < WARP_SIZE; baseLane++) // cpu
                {
                    cpuFloatResult[baseLane] = inputFloatArray[baseLane + delta > WARP_SIZE - 1 ? baseLane : baseLane + delta];
                }
                _gpu.Launch(1, WARP_SIZE).unitTest_FloatShuffleDown(d_inputFloatArray, d_outputFloatArray, delta); // gpu
                _gpu.CopyFromDevice<float>(d_outputFloatArray, gpuFloatResult);
                Assert.True(gpuFloatResult.SequenceEqual(cpuFloatResult));
            }
        }
        [Cudafy]
        public static void unitTest_FloatShuffleDown(GThread thread, float[] d_inputArray, float[] d_outputArray, uint delta)
        {
            float input = d_inputArray[thread.threadIdx.x];
            d_outputArray[thread.threadIdx.x] = thread.ShuffleDown(input, delta);
        }

        [Test]
        public void FloatShuffleXor()
        {
            for (int laneMask = 0; laneMask < WARP_SIZE; laneMask++)
            {
                for (int baseLane = 0; baseLane < WARP_SIZE; baseLane++) // cpu
                {
                    cpuFloatResult[baseLane] = inputFloatArray[baseLane ^ laneMask];
                }
                _gpu.Launch(1, WARP_SIZE).unitTest_FloatShuffleXor(d_inputFloatArray, d_outputFloatArray, laneMask); // gpu
                _gpu.CopyFromDevice<float>(d_outputFloatArray, gpuFloatResult);
                Assert.True(gpuFloatResult.SequenceEqual(cpuFloatResult));
            }
        }
        [Cudafy]
        public static void unitTest_FloatShuffleXor(GThread thread, float[] d_inputArray, float[] d_outputArray, int laneMask)
        {
            float input = d_inputArray[thread.threadIdx.x];
            d_outputArray[thread.threadIdx.x] = thread.ShuffleXor(input, laneMask);
        }

    }
}
