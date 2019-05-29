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
using System.Reflection;
using System.IO;
using Cudafy.Host;
using Cudafy.UnitTests;
using NUnit.Framework;
using Cudafy.Translator;

namespace Cudafy.cudafycl.UnitTests
{
    [Cudafy]
    [StructLayout(LayoutKind.Sequential, Size = 80, CharSet = CharSet.Unicode)]
    public unsafe struct PrimitiveStruct
    {
        public int Value1;
        public int Value2;
        public int Value3;
        public int Value4;
        public fixed sbyte _message[32];
        public fixed char _messageChars[16];
        [CudafyIgnore]
        public string Message
        {
            get
            {
                fixed (char* ptr = _messageChars)
                {
                    string ts = new string(ptr);
                    return ts;
                }
            }
            set
            {
                fixed (char* srcptr = value)
                {
                    fixed (char* dstptr = _messageChars)
                    {

                        IntPtr src = new IntPtr(srcptr);
                        IntPtr dst = new IntPtr(dstptr);
                        GPGPU.CopyMemory(dst, src, (uint)Math.Min(32, value.Length * 2));
                    }
                }
            }
        }

        [CudafyIgnore]
        public void SetMessage(string value, int length)
        {
            fixed (char* srcptr = value)
            {
                fixed (char* dstptr = _messageChars)
                {
                    IntPtr src = new IntPtr(srcptr);
                    IntPtr dst = new IntPtr(dstptr);
                    GPGPU.CopyMemory(dst, src, (uint)Math.Min(32, length * 2));
                }
            }
        }
    }
    
    public unsafe class EmbeddedCudafyModuleTests :  CudafyUnitTest, ICudafyUnitTest
    {
        private CudafyModule _cm;

        private GPGPU _gpu;

        private const int N = 1024;

        [TestFixtureSetUp]
        public void SetUp()
        {
            //var cm = CudafyTranslator.Cudafy(typeof(PrimitiveStruct), GetType());
            //
            if (CudafyModule.HasCudafyModuleInAssembly())   // Post-build event command line was: cudafycl.exe $(TargetPath)
            {                                               // Do this for Release
                _cm = GetType().Assembly.GetCudafyModule();
            }
            else // Post-build event command line was: cudafycl.exe $(TargetPath) -cdfy
            {    // Do this for Debug
                string name = this.GetType().Assembly.GetName().Name;
                _cm = CudafyModule.TryDeserialize(name);
            }
            Assert.IsFalse(_cm == null);
            Assert.IsTrue(_cm.TryVerifyChecksums());

            _gpu = CudafyHost.GetDevice(CudafyModes.Target);
            if(_cm != null)
                _gpu.LoadModule(_cm);
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _gpu.FreeAll();
        }

        [Test]
        public void Test_processStructure()
        {

            PrimitiveStruct[] x = new PrimitiveStruct[N];
            PrimitiveStruct[] y = new PrimitiveStruct[N];
            int size = Marshal.SizeOf(x[0]);
            for (int i = 0; i < N; i++)
            {
                x[i].Value1 = i;
                x[i].Message = "hello " + i.ToString();
            }
            IntPtr x_ptr = _gpu.HostAllocate<PrimitiveStruct>(N);
            IntPtr y_ptr = _gpu.HostAllocate<PrimitiveStruct>(N);
            PrimitiveStruct[] dev_x = _gpu.Allocate<PrimitiveStruct>(N);
            PrimitiveStruct[] dev_y = _gpu.Allocate<PrimitiveStruct>(N);
            //x_ptr.Write(x);

            //_gpu.CopyToDeviceAsync(x_ptr, 0, dev_x, 0, N);
            _gpu.CopyToDevice(x, 0, dev_x, 0, N);
            _gpu.Launch(1, N, "ProcessStructure", dev_x, dev_y);
            //_gpu.CopyFromDeviceAsync(dev_y, 0, y_ptr, 0, N);
            _gpu.CopyFromDevice(dev_y, 0, y, 0, N);

            _gpu.SynchronizeStream();

            //y_ptr.Read(y);
            _gpu.HostFreeAll();
            _gpu.FreeAll();
            for (int i = 0; i < N; i++)
            {
                Assert.AreEqual(x[i].Value1, y[i].Value1);
                //Assert.AreEqual(x[i].myvalues, y[i].myvalues);
                //Console.WriteLine(y[i].Message);
                Assert.AreEqual(x[i].Message, y[i].Message);
            }
        }

        //       struct __align__(8) float6 {
        //float2 u, v, w;
        //};


        [Cudafy]
        public static void ProcessStructure(GThread thread, PrimitiveStruct[] x, PrimitiveStruct[] y)
        {
            int idx = thread.threadIdx.x;
            y[idx].Value1 = x[idx].Value1;
            y[idx].Value2 = x[idx].Value2;
            y[idx].Value3 = x[idx].Value3;
            y[idx].Value4 = x[idx].Value4;
            fixed (char* xptr = x[idx]._messageChars)
            {
                fixed (char* yptr = y[idx]._messageChars)
                {
                    char* px = xptr;
                    char* py = yptr;
                    for (int i = 0; i < 16; i++)
                    {
                        *py = *px;
                        px++;
                        py++;
                    }
                }
            }
        }


        [Test]
        public void Test_mpyVectorByCoeffShort()
        {
            int[] a = new int[N];
            int[] c = new int[N];
            short coeff = 8;
            for (int i = 0; i < N; i++)
                a[i] = i;
            int[] dev_a = _gpu.CopyToDevice(a);
            int[] dev_c = _gpu.Allocate<int>(c);
            _gpu.Launch(N, 1, "mpyVectorByCoeffShort", dev_a, dev_c, coeff);
            _gpu.CopyFromDevice(dev_c, c);
            for (int i = 0; i < N; i++)
                Assert.AreEqual(i * coeff, c[i]);
            _gpu.Free(dev_a);
        }

        [Test]
        public void Test_mpyVectorByCoeffInt32()
        {
            int[] a = new int[N];
            int[] c = new int[N];
            int coeff = 8;
            for (int i = 0; i < N; i++)
                a[i] = i;
            int[] dev_a = _gpu.CopyToDevice(a);
            int[] dev_c = _gpu.Allocate<int>(c);
            _gpu.Launch(N, 1, "mpyVectorByCoeffInt32", dev_a, dev_c, coeff);
            _gpu.CopyFromDevice(dev_c, c);
            for (int i = 0; i < N; i++)
                Assert.AreEqual(i * coeff, c[i]);
            _gpu.Free(dev_a);
        }

        [Test]
        public void Test_mpyVectorByCoeffSByte()
        {
            int[] a = new int[N];
            int[] c = new int[N];
            sbyte coeff = 8;
            for (int i = 0; i < N; i++)
                a[i] = i;
            int[] dev_a = _gpu.CopyToDevice(a);
            int[] dev_c = _gpu.Allocate<int>(c);
            _gpu.Launch(N, 1, "mpyVectorByCoeffSByte", dev_a, dev_c, coeff);
            _gpu.CopyFromDevice(dev_c, c);
            for (int i = 0; i < N; i++)
                Assert.AreEqual(i * coeff, c[i]);
            _gpu.Free(dev_a);
        }

        [Test]
        public void Test_doubleVectorOffset()
        {
            int[] a = new int[N];
            int[] c = new int[N];
            for (int i = 0; i < N; i++)
                a[i] = i;
            int[] dev_a = _gpu.CopyToDevice(a);
            int[] dev_a_offset = _gpu.Cast(N / 2, dev_a, N / 2);
            _gpu.Launch(N / 2, 1, "doubleVectorOffset", dev_a_offset);
            _gpu.CopyFromDevice(dev_a_offset, c);
            for (int i = N / 2; i < N; i++)
                Assert.AreEqual(i * 2, c[i - N / 2]);
            _gpu.Free(dev_a);
        }


        [Test]
        public void Test_dynamic()
        {
            int a = 1;
            int b = 2;
            int c;
            int[] dev_c = _gpu.Allocate<int>();
#if !NET35
            _gpu.Launch().add(a, b, dev_c);
#else
            _gpu.Launch(1,1,"add", a, b, dev_c);
#endif
            _gpu.CopyFromDevice(dev_c, out c);
            Assert.AreEqual(a + b, c);
            _gpu.Free(dev_c);
        }

        [Test]
        public void Test_add()
        {
            int a = 1;
            int b = 2;
            int c;
            int[] dev_c = _gpu.Allocate<int>();
            _gpu.Launch(1, 1, "add", a, b, dev_c);
            _gpu.CopyFromDevice(dev_c, out c);
            Assert.AreEqual(a + b, c);
            _gpu.Free(dev_c);
        }

        [Test]
        public void Test_sub()
        {
            int a = 1;
            int b = 2;
            int c;
            int[] dev_c = _gpu.Allocate<int>();

            _gpu.Launch(1, 1, "sub", a, b, dev_c);
            _gpu.CopyFromDevice(dev_c, out c);
            Assert.AreEqual(a - b, c);
            _gpu.Free(dev_c);
        }

        [Test]
        public void Test_mpy()
        {
            int a = 3;
            int b = 4;
            int c;
            int[] dev_c = _gpu.Allocate<int>();

            _gpu.Launch(1, 1, "mpy", a, b, dev_c);
            _gpu.CopyFromDevice(dev_c, out c);
            Assert.AreEqual(a * b, c);
            _gpu.Free(dev_c);
        }

        [Test]
        public void Test_doubleVector()
        {
            int[] a = new int[N];
            int[] c = new int[N];
            for (int i = 0; i < N; i++)
                a[i] = i;
            int[] dev_a = _gpu.CopyToDevice(a);
            _gpu.Launch(N, 1, "doubleVector", dev_a);
            _gpu.CopyFromDevice(dev_a, c);
            for (int i = 0; i < N; i++)
                Assert.AreEqual(i * 2, c[i]);
            _gpu.Free(dev_a);
        }



        [Test]
        public void Test_mpyVector()
        {
            int[] a = new int[N];
            int[] c = new int[N];
            for (int i = 0; i < N; i++)
                a[i] = i;
            int[] dev_a = _gpu.CopyToDevice(a);
            _gpu.Launch(N, 1, "mpyVector", 42, dev_a);
            _gpu.CopyFromDevice(dev_a, c);
            for (int i = 0; i < N; i++)
                Assert.AreEqual(i * 42, c[i]);
            _gpu.Free(dev_a);
        }

        [Test]
        public void Test_mpyVectorEx()
        {
            int[] a = new int[N];
            int[] c = new int[N];
            for (int i = 0; i < N; i++)
                a[i] = i;
            int[] dev_a = _gpu.CopyToDevice(a);
            _gpu.Launch(N, 1, "mpyVectorEx", 42, 17, dev_a);
            _gpu.CopyFromDevice(dev_a, c);
            for (int i = 0; i < N; i++)
                Assert.AreEqual(i * 42, c[i]);
            _gpu.Free(dev_a);
        }

        [Test]
        public void Test_doubleVector2()
        {
            int[] a = new int[N];
            int[] c = new int[N];
            for (int i = 0; i < N; i++)
                a[i] = i;
            int[] dev_a = _gpu.CopyToDevice(a);
            int[] dev_c = _gpu.Allocate(a);
            _gpu.Launch(N, 1, "doubleVector2", dev_a, dev_c);
            _gpu.CopyFromDevice(dev_c, c);
            for (int i = 0; i < N; i++)
                Assert.AreEqual(i * 2, c[i]);
            _gpu.FreeAll();
        }

        [Test]
        public void Test_useForeach()
        {
            int[] a = new int[N];
            int[] c = new int[1];
            for (int i = 0; i < N; i++)
                a[i] = i;
            int[] dev_a = _gpu.CopyToDevice(a);
            int[] dev_c = _gpu.Allocate(c);
            _gpu.Launch(1, 1, "useForeach", dev_a, dev_c);
            _gpu.CopyFromDevice(dev_c, c);
            int sum = a.Sum();
            _gpu.FreeAll();
            Assert.AreEqual(sum, c[0]);
        }

        [Test]
        public void Test_useForeachSByte()
        {
            sbyte[] a = new sbyte[N];
            int[] c = new int[1];
            for (int i = 0; i < N; i++)
                a[i] = (sbyte)i;
            sbyte[] dev_a = _gpu.CopyToDevice(a);
            int[] dev_c = _gpu.Allocate(c);
            _gpu.Launch(1, 1, "useForeachSByte", dev_a, dev_c, (sbyte)42);
            _gpu.CopyFromDevice(dev_c, c);

            int[] int_a = new int[N];
            for (int i = 0; i < N; i++)
                int_a[i] = (int)a[i];
            int sum = int_a.Sum();
            _gpu.FreeAll();
            Assert.AreEqual(sum, c[0]);
        }

        [Test]
        public void Test_addVectoradd()
        {
            Test_addVector(eAddVectorMode.GlobalVar);
        }

        [Test]
        public void Test_addVectoraddSmart()
        {
            Test_addVector(eAddVectorMode.Smart);
        }

        [Test]
        public void Test_addVectoraddDevice()
        {
            Test_addVector(eAddVectorMode.GlobalVarDevice);
        }

        [Test]
        public void Test_2DAddressing()
        {
            int w = 256;
            int h = 128;
            int[,] input = new int[w, h];
            int[] output = new int[w * h];

            int i = 0;
            for (int x = 0; x < w; x++)
                for (int y = 0; y < h; y++)
                    input[x, y] = i++;

            int[,] input_dev = _gpu.CopyToDevice(input);
            int[] output_dev = _gpu.Allocate<int>(w * h);
            int coeff = 42;
            _gpu.Launch(1, 1, "twoDAddressingWrong", input_dev, coeff, output_dev);
            _gpu.CopyFromDevice(output_dev, output);

            i = 0;
            for (int x = 0; x < w; x++)
                for (int y = 0; y < h; y++)
                    Assert.AreEqual(input[x, y] * 42, output[i++]);
        }

        private enum eAddVectorMode { GlobalVar, Smart, GlobalVarDevice };

        private void Test_addVector(eAddVectorMode addMode)
        {
            int[] a = new int[N];
            int[] b = new int[N];
            int[] c = new int[N];

            // allocate the memory on the GPU
            int[] dev_c = _gpu.Allocate<int>(c);

            // fill the arrays 'a' and 'b' on the CPU
            for (int i = 0; i < N; i++)
            {
                a[i] = -i;
                b[i] = i * i;
            }

            // copy the arrays 'a' and 'b' to the GPU
            int[] dev_a = _gpu.CopyToDevice(a);
            int[] dev_b = _gpu.CopyToDevice(b);

            if (addMode == eAddVectorMode.Smart)
                _gpu.Launch(N, 1, "addVectorSmart", dev_a, dev_b, dev_c);
            else if (addMode == eAddVectorMode.GlobalVar)
                _gpu.Launch(N, 1, "addVector", dev_a, dev_b, dev_c);
            else if (addMode == eAddVectorMode.GlobalVarDevice)
                _gpu.Launch(N, 1, "addVectorDevice", dev_a, dev_b, dev_c);
            else
                throw new NotSupportedException(addMode.ToString());

            // copy the array 'c' back from the GPU to the CPU
            _gpu.CopyFromDevice(dev_c, c);

            // test the results
            for (int i = 0; i < N; i++)
            {
                Assert.AreEqual(a[i] + b[i], c[i]);
            }

            // free the memory allocated on the GPU
            _gpu.FreeAll();
        }

        private const int threadsPerBlock = 256;

        public const int blocksPerGrid = 4;

        [Test]
        public void TestSharedMemory()
        {
            float c;

            // allocate memory on the cpu side
            float[] a = new float[N];
            float[] b = new float[N];
            float[] partial_c = new float[blocksPerGrid];

            // allocate the memory on the GPU
            float[] dev_a = _gpu.Allocate<float>(N);
            float[] dev_b = _gpu.Allocate<float>(N);
            float[] dev_partial_c = _gpu.Allocate<float>(blocksPerGrid);

            float[] dev_test = _gpu.Allocate<float>(blocksPerGrid * blocksPerGrid);

            // fill in the host memory with data
            for (int i = 0; i < N; i++)
            {
                a[i] = i;
                b[i] = i * 2;
            }

            // copy the arrays 'a' and 'b' to the GPU
            _gpu.CopyToDevice(a, dev_a);
            _gpu.CopyToDevice(b, dev_b);

            _gpu.Launch(blocksPerGrid, threadsPerBlock, "dot", dev_a, dev_b, dev_partial_c);

            // copy the array 'c' back from the GPU to the CPU
            _gpu.CopyFromDevice(dev_partial_c, partial_c);

            // finish up on the CPU side
            c = 0;
            for (int i = 0; i < blocksPerGrid; i++)
            {
                c += partial_c[i];
            }

            // free memory on the gpu side
            _gpu.FreeAll();

            Assert.AreEqual(714779648.0, c, 0.00001);
        }

        private static float sum_squares(float x)
        {
            return (x * (x + 1) * (2 * x + 1) / 6);
        }


        [Cudafy]
        public static void add(int a, int b, int[] c)
        {
            c[0] = a + b;
        }

        [Cudafy]
        public static void sub(int a, int b, int[] c)
        {
            c[0] = a - b;
        }

        [Cudafy]
        public static void mpy(int a, int b, int[] c)
        {
            c[0] = a * b;
        }

        [Cudafy]
        public static void doubleVector(GThread thread, int[] a)
        {
            int tid = thread.blockIdx.x;
            if (tid < N)
                a[tid] *= 2;
        }

        [Cudafy]
        public static void doubleVectorOffset(GThread thread, int[] a)
        {
            int tid = thread.blockIdx.x;
            if (tid < a.Length)
                a[tid] *= 2;
        }

        [Cudafy]
        public static void mpyVector(GThread thread, int coeff, int[] a)
        {
            int tid = thread.blockIdx.x;
            if (tid < N)
                a[tid] *= coeff;
        }

        [Cudafy]
        public static void mpyVectorEx(GThread thread, int coeff, int notused, int[] a)
        {
            int tid = thread.blockIdx.x;
            if (tid < N)
                a[tid] *= coeff;
        }

        [Cudafy]
        public static void doubleVector2(GThread thread, int[] a, int[] c)
        {
            int tid = thread.blockIdx.x;
            if (tid < N)
                c[tid] = a[tid] * 2;
        }

        [Cudafy]
        public static void addVector(GThread thread, int[] a, int[] b, int[] c)
        {
            int tid = thread.blockIdx.x;
            if (tid < N)
                c[tid] = a[tid] + b[tid];
        }

        [Cudafy]
        public static void addVectorSmart(GThread thread, int[] a, int[] b, int[] c)
        {
            int tid = thread.blockIdx.x;
            if (tid < a.Length)
                c[tid] = a[tid] + b[tid];
        }

        [Cudafy]
        public static void addVectorDevice(GThread thread, int[] a, int[] b, int[] c)
        {
            int tid = thread.blockIdx.x;
            if (tid < N)
                c[tid] = addDevice(thread, a[tid], b[tid]);
        }

        [Cudafy]
        public static void mpyVectorByCoeffShort(GThread thread, int[] a, int[] c, short coeff)
        {
            int tid = thread.blockIdx.x;
            if (tid < a.Length)
                c[tid] = a[tid] * coeff;
        }

        [Cudafy]
        public static void mpyVectorByCoeffInt32(GThread thread, int[] a, int[] c, int coeff)
        {
            int tid = thread.blockIdx.x;
            if (tid < a.Length)
                c[tid] = a[tid] * coeff;
        }

        [Cudafy]
        public static void mpyVectorByCoeffSByte(GThread thread, int[] a, int[] c, sbyte coeff)
        {
            int tid = thread.blockIdx.x;
            if (tid < a.Length)
                c[tid] = a[tid] * coeff;
        }

        [Cudafy]
        public static int addDevice(GThread t, int a, int b)
        {
            //int coef = t.threadIdx.x / t.threadIdx.x;
            return a + b;// *coef;
        }

        [Cudafy]
        public static void useForeach(int[] a, int[] c)
        {
            int total = 0;
            foreach (int i in a)
                total += i;
            c[0] = total;
        }



        [Cudafy]
        public static void dot(GThread thread, float[] a, float[] b, float[] c)
        {
            float[] cache = thread.AllocateShared<float>("cache", threadsPerBlock);

            int tid = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int cacheIndex = thread.threadIdx.x;

            float temp = 0;
            while (tid < N)
            {
                temp += a[tid] * b[tid];
                tid += thread.blockDim.x * thread.gridDim.x;
            }

            // set the cache values
            cache[cacheIndex] = temp;

            // synchronize threads in this block
            thread.SyncThreads();

            // for reductions, threadsPerBlock must be a power of 2
            // because of the following code
            int i = thread.blockDim.x / 2;
            while (i != 0)
            {
                if (cacheIndex < i)
                    cache[cacheIndex] += cache[cacheIndex + i];
                thread.SyncThreads();
                i /= 2;
            }

            if (cacheIndex == 0)
                c[thread.blockIdx.x] = cache[0];
        }

        [Cudafy]
        public static void twoDAddressing(GThread thread, int[,] input, int coeff, int[] output)
        {
            int x = 0;
            for (int dx = 0; dx < input.GetLength(0); dx++)
            {
                for (int dy = 0; dy < input.GetLength(1); dy++)
                {
                    output[x++] = input[dx, dy] * coeff;
                }
            }
        }

        [Cudafy]
        public static void twoDAddressingWrong(GThread thread, int[,] input, int coeff, int[] output)
        {
            int x = 0;
            for (int dx = -1; dx < input.GetLength(0) - 1; dx++)
            {
                for (int dy = -1; dy < input.GetLength(1) - 1; dy++)
                {
                    output[x++] = input[dx + 1, dy + 1] * coeff;
                }
            }
        }


        [Cudafy]
        public static void useForeachSByte(sbyte[] a, int[] c, sbyte s)
        {
            int total = 0;
            foreach (sbyte i in a)
                total += i;
            c[0] = total;
        }

        [SetUp]
        public void TestSetUp()
        {

        }

        [TearDown]
        public void TestTearDown()
        {

        }
    }
}
