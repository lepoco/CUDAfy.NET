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
using System.Diagnostics;
using System.Runtime.InteropServices;
using Cudafy.Types;
using Cudafy.Host;
using Cudafy.UnitTests;
using Cudafy.Translator;
using NUnit.Framework;

namespace Cudafy.Host.UnitTests
{

    //[Cudafy]
    //public struct StructWithBool
    //{
    //    public bool B;
    //}
    
    [TestFixture]
    public class CopyTests1D : CudafyUnitTest, ICudafyUnitTest
    {
        private CudafyModule _cm;

        private GPGPU _gpu;

        private const int N = 1024 * 1024 * 4;

        private byte[] _byteBufferIn;

        private byte[] _byteBufferOut;

        private byte[] _gpubyteBufferIn;

        private byte[] _gpubyteBufferOut;

        private sbyte[] _sbyteBufferIn;

        private sbyte[] _sbyteBufferOut;

        private sbyte[] _gpusbyteBufferIn;

        private sbyte[] _gpusbyteBufferOut;

        private ushort[] _ushortBufferIn;

        private ushort[] _ushortBufferOut;

        private ushort[] _gpuushortBufferIn;

        private ushort[] _gpuushortBufferOut;

        private uint[] _uintBufferIn;

        private uint[] _uintBufferOut;

        private uint[] _gpuuintBufferIn;

        private uint[] _gpuuintBufferOut;

        private ulong[] _ulongBufferIn;

        private ulong[] _ulongBufferOut;

        private ulong[] _gpuulongBufferIn;

        private ulong[] _gpuulongBufferOut;

        private ComplexD[] _cplxDBufferIn;

        private ComplexD[] _cplxDBufferOut;

        private ComplexD[] _gpucplxDBufferIn;

        private ComplexD[] _gpucplxDBufferOut;

        private ComplexF[] _cplxFBufferIn;

        private ComplexF[] _cplxFBufferOut;

        private ComplexF[] _gpucplxFBufferIn;

        private ComplexF[] _gpucplxFBufferOut;

        private StructWithBool[] _gpuStructWithBool;

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.GetDevice(CudafyModes.Architecture, CudafyModes.DeviceId);

            _byteBufferIn = new byte[N];
            _byteBufferOut = new byte[N];            

            _sbyteBufferIn = new sbyte[N];
            _sbyteBufferOut = new sbyte[N];

            _ushortBufferIn = new ushort[N];
            _ushortBufferOut = new ushort[N];

            _uintBufferIn = new uint[N];
            _uintBufferOut = new uint[N];

            _ulongBufferIn = new ulong[N];
            _ulongBufferOut = new ulong[N];

            _cplxDBufferIn = new ComplexD[N];
            _cplxDBufferOut = new ComplexD[N];

            _cplxFBufferIn = new ComplexF[N];
            _cplxFBufferOut = new ComplexF[N];

            SetInputs();
            ClearOutputsAndGPU();
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _gpu.FreeAll();
        }

        private void SetInputs()
        {
            Random rand = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < N; i++)
            {
                double r = rand.NextDouble() / 4;
                double j = rand.NextDouble() / 4;
                _byteBufferIn[i] = (byte)(r * Byte.MaxValue);
                _sbyteBufferIn[i] = (sbyte)((r * Byte.MaxValue) - SByte.MaxValue);
                _ushortBufferIn[i] = (ushort)(r * ushort.MaxValue);
                _uintBufferIn[i] = (uint)(r * uint.MaxValue);
                _ulongBufferIn[i] = (ulong)(r * ulong.MaxValue);
                _cplxDBufferIn[i].x = r * short.MaxValue;
                _cplxDBufferIn[i].y = j * short.MaxValue - 1.0;
                _cplxFBufferIn[i].x = (float)r * short.MaxValue;
                _cplxFBufferIn[i].y = (float)j * short.MaxValue - 1.0F;
            }  
        }

        private void ClearOutputsAndGPU()
        {
            for (int i = 0; i < N; i++)
            {
                _byteBufferOut[i] = 0;
                _sbyteBufferOut[i] = 0;
                _ushortBufferOut[i] = 0;
                _uintBufferOut[i] = 0;
                _ulongBufferOut[i] = 0;
                _cplxDBufferOut[i].x = 0;
                _cplxDBufferOut[i].y = 0;
                _cplxFBufferOut[i].x = 0;
                _cplxFBufferOut[i].y = 0;
            }
            _gpu.FreeAll();
            _gpu.HostFreeAll();
            GC.Collect();
        }

        [Cudafy]
        public static void DoubleAllValues(GThread t, uint[] input, uint[] output)
        {
            int tid = t.threadIdx.x + t.blockDim.x * t.blockIdx.x;
            if(tid < input.Length)
                output[tid] = input[tid] * 2;
        }

        [Test]
        public void Test_smartCopyToDevice()
        {
            if (_gpu is OpenCLDevice)
            {
                Console.WriteLine("Device not supporting smart copy, so skip.");
                return;
            }
            bool emulated = _gpu is EmulatedGPU;
            var mod = CudafyModule.TryDeserialize();
            if (mod == null || !mod.TryVerifyChecksums())
            {
                mod = CudafyTranslator.Cudafy(CudafyModes.Architecture);
                mod.Serialize();
            }
            _gpu.LoadModule(mod);
            int batchSize = emulated ? 2 : 8;
            int loops = emulated ? 1 : 6;
            int n = emulated ? 2048 : N;
            _gpuuintBufferIn = _gpu.Allocate<uint>(n);
            _gpuuintBufferOut = _gpu.Allocate<uint>(n);

            Stopwatch sw = Stopwatch.StartNew();
            for (int x = 0; x < loops; x++)
            {
                for (int i = 0; i < batchSize; i++)
                {
                    _gpu.CopyToDevice(_uintBufferIn, 0, _gpuuintBufferIn, 0, n);
                    _gpu.Launch(n / 256, 256, "DoubleAllValues", _gpuuintBufferIn, _gpuuintBufferOut);
                    _gpu.CopyFromDevice(_gpuuintBufferOut, 0, _uintBufferOut, 0, n);
                }
            }
            long time = sw.ElapsedMilliseconds;
            Console.WriteLine(time);
            IntPtr[] stagingPostIn = new IntPtr[batchSize];
            IntPtr[] stagingPostOut = new IntPtr[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                stagingPostIn[i] = _gpu.HostAllocate<uint>(n);
                stagingPostOut[i] = _gpu.HostAllocate<uint>(n);
            }
            _gpu.EnableSmartCopy();
            sw.Restart();
            for (int x = 0; x < loops; x++)
            {
                for (int i = 0; i < batchSize; i++)
                    _gpu.CopyToDeviceAsync(_uintBufferIn, 0, _gpuuintBufferIn, 0, n, i + 1, stagingPostIn[i]);
                for (int i = 0; i < batchSize; i++)
                    _gpu.LaunchAsync(n / 256, 256, i + 1, "DoubleAllValues", _gpuuintBufferIn, _gpuuintBufferOut);
                for (int i = 0; i < batchSize; i++)
                    _gpu.CopyFromDeviceAsync(_gpuuintBufferOut, 0, _uintBufferOut, 0, n, i + 1, stagingPostOut[i]);
                for (int i = 0; i < batchSize; i++)
                    _gpu.SynchronizeStream(i + 1);
                //for (int i = 0; i < batchSize; i++)
                //{
                //    _gpu.CopyToDeviceAsync(stagingPostIn[i], 0, _gpuuintBufferIn, 0, N, i+1);
                //    _gpu.LaunchAsync(N / 512, 512, i + 1, "DoubleAllValues", _gpuuintBufferIn, _gpuuintBufferOut);
                //    _gpu.CopyFromDeviceAsync(_gpuuintBufferOut, 0, stagingPostOut[i], 0, N, i + 1);
                //}
                //for (int i = 0; i < batchSize; i++)
                //    _gpu.SynchronizeStream(i + 1);
            }

            time = sw.ElapsedMilliseconds;
            Console.WriteLine(time);
            _gpu.DisableSmartCopy();
            for (int i = 0; i < n; i++)
                _uintBufferIn[i] *= 2;
            Assert.IsTrue(Compare(_uintBufferIn, _uintBufferOut, 0, 0, n));
                       
            ClearOutputsAndGPU();
        }

        public enum MyEnum : int
        {
            mon = 0, tue = 1, wed = 2, thu = 3, fri = 4, sat = 5
        }

        public struct MyEnumStruct
        {
            public MyEnum ME;
        }



        [Test]
        public void TestEnum()
        {
            MyEnumStruct[] enum1 = new MyEnumStruct[10];
            for (int i = 0; i < 10; i++)
            {
                enum1[i].ME = MyEnum.mon;
            }
            MyEnumStruct[] d_enum1 = _gpu.CopyToDevice(enum1);
        }

        [Test]
        public void Test_getValue_int2D()
        {
            int[,] data2D = new int[16,12];
            for (int i = 0, ctr = 0; i < 16; i++)
                for (int j = 0; j < 12; j++)
                    data2D[i, j] = ctr++;
            int[,] dev2D = _gpu.CopyToDevice(data2D);
            int v = _gpu.GetValue(dev2D, 14, 9);
            Assert.AreEqual(data2D[14,9], v);
            ClearOutputsAndGPU();
        }

        //[Test]
        //public void Test_asyncNonpinnedTransfers()
        //{
        //    int[] data = new int[N];
        //    int[] data_d = _gpu.Allocate(data);
        //    int[] res = new int[N];
        //    int[] res_d = _gpu.Allocate(res);
        //    IntPtr resPtr = _gpu.HostAllocate<int>(N);
        //    IntPtr dataPtr = _gpu.HostAllocate<int>(N);
        //    Random r = new Random(543);
        //    for (int i = 0; i < N; i++)
        //        data[i] = r.Next();
        //    _gpu.CreateStream(1);
        //    Stopwatch sw = Stopwatch.StartNew();
        //    //_gpu.CopyToDeviceAsync(dataPtr, 0, data_d, 0, N, 1);
        //    _gpu.CopyToDeviceAsync(data, 0, data_d, 0, N, 1);
        //    _gpu.CopyOnDeviceAsync(data_d, 0, res_d, 0, N, 1);
        //    _gpu.CopyOnDeviceAsync(data_d, 0, res_d, 0, N, 1);
        //    _gpu.CopyOnDeviceAsync(data_d, 0, res_d, 0, N, 1);
        //    _gpu.CopyOnDeviceAsync(data_d, 0, res_d, 0, N, 1);
        //    _gpu.CopyOnDeviceAsync(data_d, 0, res_d, 0, N, 1);
        //    //_gpu.CopyFromDeviceAsync(res_d, 0, res, 0, N, 1);
        //    //_gpu.CopyFromDeviceAsync(res_d, 0, resPtr, 0, N, 1);
        //    long t1 = sw.ElapsedMilliseconds;
        //    //for (int i = N - 1024; i < N; i++)
        //    //    Assert.AreEqual(0, res[i], string.Format("Sample i={0}", i));
        //    _gpu.SynchronizeStream(1);
        //    long t2 = sw.ElapsedMilliseconds;
        //    //for (int i = 0; i < N; i++)
        //    //    Assert.AreEqual(data[i], res[i]);       
        //}

        [Test]
        public void Test_getValue_int3D()
        {
            int[,,] data3D = new int[16, 12, 8];
            for (int i = 0, ctr = 0; i < 16; i++)
                for (int j = 0; j < 12; j++)
                    for (int k = 0; k < 8; k++)
                    data3D[i, j, k] = ctr++;
            int[,,] dev3D = _gpu.CopyToDevice(data3D);
            int v = _gpu.GetValue(dev3D, 14, 9, 6);
            Assert.AreEqual(data3D[14, 9, 6], v);
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_getValue_complexD()
        {
            _gpucplxDBufferIn = _gpu.Allocate(_cplxDBufferIn);
            _gpu.CopyToDevice(_cplxDBufferIn, _gpucplxDBufferIn);
            ComplexD cd = _gpu.GetValue(_gpucplxDBufferIn, N/32);
            Assert.AreEqual(_cplxDBufferIn[N / 32], cd);
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_autoCopyToFromGPU_byte()
        {
            _gpubyteBufferIn = _gpu.CopyToDevice(_byteBufferIn);
            _gpu.CopyFromDevice(_gpubyteBufferIn, _byteBufferOut);
            Assert.IsTrue(Compare(_byteBufferIn, _byteBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_autoCopyToFromGPU_sbyte()
        {
            _gpusbyteBufferIn = _gpu.CopyToDevice(_sbyteBufferIn);
            _gpu.CopyFromDevice(_gpusbyteBufferIn, _sbyteBufferOut);
            Assert.IsTrue(Compare(_sbyteBufferIn, _sbyteBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_autoCopyToFromGPU_ushort()
        {
            _gpuushortBufferIn = _gpu.CopyToDevice(_ushortBufferIn);
            _gpu.CopyFromDevice(_gpuushortBufferIn, _ushortBufferOut);
            Assert.IsTrue(Compare(_ushortBufferIn, _ushortBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_autoCopyToFromGPU_uint()
        {
            _gpuuintBufferIn = _gpu.CopyToDevice(_uintBufferIn);
            _gpu.CopyFromDevice(_gpuuintBufferIn, _uintBufferOut);
            Assert.IsTrue(Compare(_uintBufferIn, _uintBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_autoCopyToFromGPU_ulong()
        {
            _gpuulongBufferIn = _gpu.CopyToDevice(_ulongBufferIn);
            _gpu.CopyFromDevice(_gpuulongBufferIn, _ulongBufferOut);
            Assert.IsTrue(Compare(_ulongBufferIn, _ulongBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_autoCopyToFromGPU_cplxF()
        {
            _gpucplxFBufferIn = _gpu.CopyToDevice(_cplxFBufferIn);
            _gpu.CopyFromDevice(_gpucplxFBufferIn, _cplxFBufferOut);
            Assert.IsTrue(Compare(_cplxFBufferIn, _cplxFBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_autoCopyToFromGPU_cplxD()
        {
            _gpucplxDBufferIn = _gpu.CopyToDevice(_cplxDBufferIn);
            _gpu.CopyFromDevice(_gpucplxDBufferIn, _cplxDBufferOut);
            Assert.IsTrue(Compare(_cplxDBufferIn, _cplxDBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_copyToFromOffsetGPU_byte()
        {
            _gpubyteBufferIn = _gpu.Allocate(_byteBufferIn);
            _gpu.CopyToDevice(_byteBufferIn, _gpubyteBufferIn);
            _gpu.CopyFromDevice(_gpubyteBufferIn, N / 16, _byteBufferOut, N / 8, N / 2);
            Assert.IsTrue(Compare(_byteBufferIn, _byteBufferOut, N / 16, N / 8, N / 2));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_cast_byte()
        {
            if (_gpu is EmulatedGPU || _gpu is OpenCLDevice)
            {
                Console.WriteLine("Emulated not supporting cast with offset, so skip.");
                return;
            }
            _gpubyteBufferIn = _gpu.Allocate(_byteBufferIn);
            _gpu.CopyToDevice(_byteBufferIn, _gpubyteBufferIn);
            byte[] offsetArray = _gpu.Cast(N / 16, _gpubyteBufferIn, N / 2);
            _gpu.CopyFromDevice(offsetArray, 0, _byteBufferOut, 0, N / 2);
            Assert.IsTrue(Compare(_byteBufferIn, _byteBufferOut, N / 16, 0, N / 2));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_cast_cast_byte()
        {
            if (_gpu is EmulatedGPU || _gpu is OpenCLDevice)
            {
                Console.WriteLine("Device not supporting cast with offset, so skip.");
                return;
            }
            _gpubyteBufferIn = _gpu.Allocate(_byteBufferIn);
            _gpu.CopyToDevice(_byteBufferIn, _gpubyteBufferIn);
            byte[] offsetArray = _gpu.Cast(N / 16, _gpubyteBufferIn, N / 2);
            byte[] array2 = _gpu.Cast(0, offsetArray, N / 2);
            _gpu.CopyFromDevice(array2, 0, _byteBufferOut, 0, N / 2);
            Assert.IsTrue(Compare(_byteBufferIn, _byteBufferOut, N / 16, 0, N / 2));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_cast_byte_to_sbyte()
        {
            if (!(_gpu is CudaGPU))
            {
                Console.WriteLine("Device not supporting cast, so skip.");
                return;
            }
            _gpubyteBufferIn = _gpu.Allocate(_byteBufferIn);
            _gpu.CopyToDevice(_byteBufferIn, _gpubyteBufferIn);
            sbyte[] sbyteBufferOut = new sbyte[N];
            sbyte[] offsetArray = _gpu.Cast<byte,sbyte>(_gpubyteBufferIn, N);
            _gpu.CopyFromDevice(offsetArray, sbyteBufferOut);
            for (int i = 0; i < N; i++)
                Assert.AreEqual((sbyte)_byteBufferIn[i], sbyteBufferOut[i]);
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_cast_cplxD_to_double()
        {
            if (!(_gpu is CudaGPU))
            {
                Console.WriteLine("Device not supporting cast, so skip.");
                return;
            }
            _gpucplxDBufferIn = _gpu.Allocate(_cplxDBufferIn);
            _gpu.CopyToDevice(_cplxDBufferIn, _gpucplxDBufferIn);
            double[] doubleBufferOut = new double[N*2];
            double[] offsetArray = _gpu.Cast<ComplexD, double>(_gpucplxDBufferIn, N*2);
            _gpu.CopyFromDevice(offsetArray, doubleBufferOut);
            for (int i = 0; i < N; i++)
            {
                Assert.AreEqual(_cplxDBufferIn[i].x, doubleBufferOut[i * 2]);
                Assert.AreEqual(_cplxDBufferIn[i].y, doubleBufferOut[i * 2 + 1]);
            }
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_copyToFromOffsetGPU_cplxD()
        {
            _gpucplxDBufferIn = _gpu.Allocate(_cplxDBufferIn);
            _gpu.CopyToDevice(_cplxDBufferIn, _gpucplxDBufferIn);
            _gpu.CopyFromDevice(_gpucplxDBufferIn, N / 16, _cplxDBufferOut, N / 8, N / 2);
            Assert.IsTrue(Compare(_cplxDBufferIn, _cplxDBufferOut, N / 16, N / 8, N / 2));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_cast_uint_to_2d()
        {
            if (_gpu is EmulatedGPU || _gpu is OpenCLDevice)
            {
                Console.WriteLine("Device not supporting cast with offset, so skip.");
                return;
            }
            if (N > 32768)
            {
                Debug.WriteLine("Skipping Test_cast_uint_to_2d due to N being too large");
                return;
            }
            _gpuuintBufferIn = _gpu.CopyToDevice(_uintBufferIn);
            uint[,] devarray2D = _gpu.Cast(N / 2, _gpuuintBufferIn, N / 32, N / 64);
            uint[,] hostArray2D = new uint[N / 32, N / 64];
            _gpu.CopyFromDevice(devarray2D, hostArray2D);
            Assert.IsTrue(CompareEx<uint>(_uintBufferIn, hostArray2D, N / 2, 0, N / 2));
            ClearOutputsAndGPU();

        }


        [Test]
        public void Test_set_uint()
        {
            _gpuuintBufferIn = _gpu.CopyToDevice(_uintBufferIn);
            _gpu.Set(_gpuuintBufferIn);
            _gpu.CopyFromDevice(_gpuuintBufferIn, _uintBufferOut);
            Assert.IsTrue(Compare((uint)0, _uintBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_set_cplxF()
        {
            _gpucplxFBufferIn = _gpu.CopyToDevice(_cplxFBufferIn);
            _gpu.Set(_gpucplxFBufferIn);
            _gpu.CopyFromDevice(_gpucplxFBufferIn, _cplxFBufferOut);
            Assert.IsTrue(Compare(new ComplexF(), _cplxFBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_set_selection_uint()
        {
            _gpuuintBufferIn = _gpu.CopyToDevice(_uintBufferIn);
            _gpu.Set(_gpuuintBufferIn, N / 4, N / 2);
            _gpu.CopyFromDevice(_gpuuintBufferIn, _uintBufferOut);
            Assert.IsTrue(Compare(_uintBufferIn, _uintBufferOut, 0, 0, N / 4));
            Assert.IsTrue(Compare((uint)0, _uintBufferOut, N / 4, N / 2));
            Assert.IsTrue(Compare(_uintBufferIn, _uintBufferOut, N - N / 4, N - N / 4, N / 4));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_set_selection_cplxF()
        {
            _gpucplxFBufferIn = _gpu.CopyToDevice(_cplxFBufferIn);
            _gpu.Set(_gpucplxFBufferIn, N / 4, N / 2);
            _gpu.CopyFromDevice(_gpucplxFBufferIn, _cplxFBufferOut);
            Assert.IsTrue(Compare(_cplxFBufferIn, _cplxFBufferOut, 0, 0, N / 4));
            Assert.IsTrue(Compare(new ComplexF(), _cplxFBufferOut, N / 4, N / 2));
            Assert.IsTrue(Compare(_cplxFBufferIn, _cplxFBufferOut, N - N / 4, N - N / 4, N / 4));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_copyToOffsetFromGPU_ulong()
        {
            _gpuulongBufferIn = _gpu.Allocate(_ulongBufferIn);
            _gpu.Set(_gpuulongBufferIn);
            _gpu.CopyToDevice(_ulongBufferIn, N / 4, _gpuulongBufferIn, N / 16, N / 2);
            _gpu.CopyFromDevice(_gpuulongBufferIn, _ulongBufferOut);
            Assert.IsTrue(Compare(_ulongBufferIn, _ulongBufferOut, N / 4, N / 16, N / 2));
        }

        [Test]
        public void Test_copyToOffsetFromGPU_byte()
        {
            _gpubyteBufferIn = _gpu.Allocate(_byteBufferIn);
            _gpu.Set(_gpubyteBufferIn);
            _gpu.CopyToDevice(_byteBufferIn, N / 4, _gpubyteBufferIn, N / 16, N / 2);
            _gpu.CopyFromDevice(_gpubyteBufferIn, _byteBufferOut);
            Assert.IsTrue(Compare(_byteBufferIn, _byteBufferOut, N / 4, N / 16, N / 2));
        }

        [Test]
        public void Test_copyOnHost()
        {
            int len = 35687;
            int[] bufa = new int[len];
            int[] bufb = new int[len];
            Random r = new Random();
            for (int i = 0; i < len; i++)
                bufa[i] = r.Next() + 1;
            IntPtr ha = _gpu.HostAllocate<int>(len);

            ha.Write(bufa, 0, 0, len);
            IntPtr hb = _gpu.HostAllocate<int>(len);
            GPGPU.CopyMemory(hb, ha, (uint)len * sizeof(int));

            hb.Read(bufb, 0, 0, len);
            for (int i = 0; i < len; i++)
            {
                Assert.True(bufa[i] == bufb[i]);
                Assert.False(bufa[i] == 0);
            }
            _gpu.HostFreeAll();
        }

        [Test]
        public void Test_copyOnGPU_cplxD()
        {
            _gpucplxDBufferIn = _gpu.CopyToDevice(_cplxDBufferIn);
            _gpucplxDBufferOut = _gpu.Allocate<ComplexD>(N);
            _gpu.CopyOnDevice(_gpucplxDBufferIn, _gpucplxDBufferOut);
            _gpu.Set(_gpucplxDBufferIn);
            _gpu.CopyFromDevice(_gpucplxDBufferOut, _cplxDBufferOut);
            Assert.IsTrue(Compare(_cplxDBufferIn, _cplxDBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_copyOnOffsetGPU_ushort()
        {
            _gpuushortBufferIn = _gpu.CopyToDevice(_ushortBufferIn);
            _gpuushortBufferOut = _gpu.Allocate<ushort>(N / 2);
            _gpu.CopyOnDevice(_gpuushortBufferIn, N / 4, _gpuushortBufferOut, N / 8, N / 3);
            _gpu.Set(_gpuushortBufferIn);
            _gpu.CopyFromDevice(_gpuushortBufferOut, 0, _ushortBufferOut, 0, N / 2);
            Assert.IsTrue(Compare(_ushortBufferIn, _ushortBufferOut, N / 4, N / 8, N / 3));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_copyFromPinned()
        {
            IntPtr srcPtr = _gpu.HostAllocate<ushort>(N);
            IntPtr dstPtr = _gpu.HostAllocate<ushort>(N);
            srcPtr.Write(_ushortBufferIn);
            _gpuushortBufferIn = _gpu.Allocate<ushort>(N);
            _gpuushortBufferOut = _gpu.Allocate<ushort>(N);
            _gpu.CopyToDeviceAsync(srcPtr, 0, _gpuushortBufferIn, 0, N, 0);
            _gpu.CopyOnDevice(_gpuushortBufferIn, 0, _gpuushortBufferOut, 0, N);
            _gpu.CopyFromDeviceAsync(_gpuushortBufferOut, 0, dstPtr, 0, N, 0);
            _gpu.SynchronizeStream(0);
            dstPtr.Read(_ushortBufferOut);
            _gpu.HostFreeAll();
            Assert.IsTrue(Compare(_ushortBufferIn, _ushortBufferOut, 0, 0, N));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_LongArray()
        {
            int n = 1024 * 1024 * 32;
            int[] longArray = new int[n];
            Random rand = new Random();
            for (int i = 0; i < n; i++)
                longArray[i] = rand.Next();
            int[] output = new int[n];
            int[] longArray_dev = _gpu.CopyToDevice(longArray);
            _gpu.CopyFromDevice(longArray_dev, output);
            _gpu.Free(longArray_dev);
            Assert.IsTrue(Compare(longArray, output, 0, 0, n));
            ClearOutputsAndGPU();
        }


        [StructLayout(LayoutKind.Sequential, Size = 50, Pack = 8, CharSet = CharSet.Unicode)]
        [Cudafy]
        public struct MyStruct
        {
            public int X;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
            public byte[] Data;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 32)]
            public byte[] _Message;
            [CudafyIgnore]
            public string Message
            {
                get { return Encoding.Unicode.GetString(_Message); }
                set
                {
                    if (_Message == null)
                        _Message = new byte[32];
                    Encoding.Unicode.GetBytes(value, 0, 16, _Message, 0);
                }
            }
        }

        //[Test]
        //[ExpectedException(typeof(ArgumentException))]
        //public void Test_structWithBoolean()
        //{
        //    var swbArray = new StructWithBool[1024];

        //    StructWithBool[] swbArray_dev = _gpu.CopyToDevice(swbArray);

        //    ClearOutputsAndGPU();
        //}

        unsafe private IntPtr MarshalArray(ref MyStruct[] items)
        {
            int iSizeOfOneItemPos = Marshal.SizeOf(typeof(MyStruct));
            int iTotalSize = iSizeOfOneItemPos * items.Length;
            IntPtr pUnmanagedItems = Marshal.AllocHGlobal(iTotalSize);
            byte* pbyUnmanagedItems = (byte*)(pUnmanagedItems.ToPointer());

            for (int i = 0; i < items.Length; i++, pbyUnmanagedItems += (iSizeOfOneItemPos))
            {
                IntPtr pOneItemPos = new IntPtr(pbyUnmanagedItems);
                Marshal.StructureToPtr(items[i], pOneItemPos, false);
            }

            return pUnmanagedItems;
        }

        unsafe private IntPtr MarshalArray(ref MyStruct[] items, IntPtr dstPtr)
        {
            int iSizeOfOneItemPos = Marshal.SizeOf(typeof(MyStruct));
            int iTotalSize = iSizeOfOneItemPos * items.Length;
            byte* pbyUnmanagedItems = (byte*)(dstPtr.ToPointer());

            for (int i = 0; i < items.Length; i++, pbyUnmanagedItems += (iSizeOfOneItemPos))
            {
                IntPtr pOneItemPos = new IntPtr(pbyUnmanagedItems);
                Marshal.StructureToPtr(items[i], pOneItemPos, false);
            }

            return dstPtr;
        }

        unsafe private void UnMarshalArray(IntPtr pUnmanagedItems, ref MyStruct[] items)
        {
            int iSizeOfOneItemPos = Marshal.SizeOf(typeof(MyStruct));
            byte* pbyUnmanagedItem = (byte*)(pUnmanagedItems.ToPointer());

            for (int i = 0; i < items.Length; i++, pbyUnmanagedItem += (iSizeOfOneItemPos))
            {
                IntPtr pOneItemPos = new IntPtr(pbyUnmanagedItem);
                items[i] = (MyStruct)(Marshal.PtrToStructure(pOneItemPos, typeof(MyStruct)));
            }
        }

        //[Test]
        //public void Test_myStruct()
        //{
        //    IntPtr host_ptr = _gpu.HostAllocate<MyStruct>(N);
        //    IntPtr host_c_ptr = _gpu.HostAllocate<MyStruct>(N);
        //    MyStruct[] a = new MyStruct[N];

        //    for (int i = 0; i < N; i++)
        //    {
        //        a[i].X = i;
        //        a[i].Message = "0123456789ABCDEF" + i.ToString();
        //    }

        //    PrimitiveStruct[] x = new PrimitiveStruct[N];
        //    PrimitiveStruct[] y = new PrimitiveStruct[N];
        //    for (int i = 0; i < N; i++)
        //    {
        //        //x[i].Value = i;
        //        x[i].Message = "hello " + i.ToString();
        //    }
        //    IntPtr x_ptr = _gpu.HostAllocate<PrimitiveStruct>(N);
        //    Stopwatch sw = Stopwatch.StartNew();
        //    x_ptr.Write(x);
        //    x_ptr.Read(y);
        //    Console.WriteLine(sw.ElapsedMilliseconds);
        //    for (int i = 0; i < N; i++)
        //    {
        //        Assert.AreEqual(x[i], y[i]);
        //        if (i < 8)
        //            Console.WriteLine("\t" + y[i].Message);
        //    }

        //    //host_ptr.Write(a);
        //    //MyStruct[] c = new MyStruct[N];
        //    //MyStruct[] dev_c = _gpu.Allocate<MyStruct>(N);
        //    //MyStruct[] dev_a = _gpu.Allocate<MyStruct>(N);
        //    //_gpu.CopyToDeviceAsync(host_ptr, 0, dev_a, 0, N);
        //    //_gpu.CopyOnDevice(dev_a, dev_c);
        //    //_gpu.CopyFromDeviceAsync(dev_c, 0, host_c_ptr, 0, N);
        //    //_gpu.SynchronizeStream();
        //    //host_c_ptr.Read(c);
        //    _gpu.HostFreeAll();
        //    //_gpu.FreeAll();
        //    //for (int i = 0; i < N; i++)
        //    //{
        //    //    Assert.AreEqual(a[i].X, c[i].X);
        //    //    Assert.AreEqual(a[i].Message, c[i].Message);
        //    //}
        //}


        public void TestSetUp()
        {

        }

        public void TestTearDown()
        {

        }
    }
}
