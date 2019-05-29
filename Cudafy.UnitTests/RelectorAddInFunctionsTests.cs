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
using NUnit.Framework;
using Cudafy.Translator;
namespace Cudafy.UnitTests
{

    [Cudafy]
    public struct TestStruct
    {
        public int value;

        public float doit(float ox1, float oy1, ref float n1)
        {
            n1 = ox1 + oy1;
            return 1.0F;
        }
    }
    
    [TestFixture]
    public class RelectorAddInFunctionsTests : CudafyUnitTest, ICudafyUnitTest
    {
        private CudafyModule _cm;

        private const int N = 1024;

        [SetUp]
        public void SetUp()
        {
            _cm = CudafyTranslator.Cudafy(eArchitecture.sm_20);//typeof(RelectorAddInFunctionsTests));
        }

        [TearDown]
        public void TearDown()
        {
        }

        [Test]
        public void TestHas_getLength()
        {
            Assert.Contains("getLength", _cm.Functions.Keys);
        }

        [Test]
        public void TestHas_getTotalLength()
        {
            Assert.Contains("getTotalLength", _cm.Functions.Keys);
        }

        [Test]
        public void TestHas_getRank()
        {
            Assert.Contains("getRank", _cm.Functions.Keys);
        }

        [Test]
        public void TestHas_add()
        {
            Assert.Contains("add", _cm.Functions.Keys); 
        }

        [Test]
        public void TestHas_sub()
        {
            Assert.Contains("sub", _cm.Functions.Keys);
        }

        [Test]
        public void TestHas_mpy()
        {
            Assert.Contains("mpy", _cm.Functions.Keys);
        }

        [Test]
        public void TestHas_addVector()
        {
            Assert.Contains("addVector", _cm.Functions.Keys);
        }

        [Test]
        public void TestHas_addVectorSmart()
        {
            Assert.Contains("addVectorSmart", _cm.Functions.Keys);
        }

        [Test]
        public void TestIsGlobal()
        {
            Assert.Contains("add", _cm.Functions.Keys);
            Assert.AreEqual(eKernelMethodType.Global, _cm.Functions["add"].MethodType);
        }

        [Test]
        public void TestIsDevice()
        {
            Assert.Contains("addDevice", _cm.Functions.Keys);
            Assert.AreEqual(eKernelMethodType.Device, _cm.Functions["addDevice"].MethodType);
        }

        [Test]
        public void TestSharedMemory()
        {
            Assert.Contains("dot", _cm.Functions.Keys);
        }

        [Test]
        public void TestFixed()
        {
            Assert.Contains("fixedTest", _cm.Functions.Keys);
        }

        [Test]
        public void TestEnum()
        {
            Assert.Contains("enumTest", _cm.Functions.Keys);
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
        public static int addDevice(int a, int b)
        {
            return a + b;
        }

        [Cudafy]
        public static int getRank(int[, ,] a)
        {
            return a.Rank;
        }

        [Cudafy]
        public static int getTotalLength(int[, ,] a)
        {
            return a.Length;
        }

        [Cudafy]
        public static int getLength(int[, ,] a)
        {
            return a.GetLength(2);
        }


        [Cudafy]
        public static double[] myField1D = new double[16];

        [Cudafy]
        public static double[,] myField2D = new double[16, 32];

        [Cudafy]
        public static double[, ,] myField3D = new double[4, 8, 12];

        private const int threadsPerBlock = 256;

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

            //callWithShared(cache);
        }

        [Cudafy]
        public static unsafe void fixedTest(byte[] buffer)
        {
            fixed (byte* p = buffer)
            {
                byte* offset = p + 1;
                int* ip = (int*)(void*)(offset);
                *ip = 42;
            }
        }

        [Cudafy]
        public static unsafe void fixedTest2(byte[] buffer)
        {
            fixed (byte* p = buffer)
            {
                int* ip = (int*)(void*)(p + 1);
                *ip = 42;
            }
        }

        [Cudafy]
        public static void enumTest(GThread thread, MyEnum test, int[] data)
        {
            if (test == MyEnum.A)
                data[thread.threadIdx.x] *= (int)test;            
        }

        [Cudafy]
        public enum MyEnum { A = 1, B = 2, C = 4, D = 8 };

        [Cudafy]
        public static void AllAnyBallotTests(GThread thread, float[] data)
        {
            thread.SyncThreadsCount(true);
            thread.All(true);
            thread.Any(true);
            thread.Ballot(true);

        }

        //[Cudafy]
        //public static void TestTryCatchFinally()
        //{
        //    int x = 42;
        //    try
        //    {
        //        for (int i = 0; i < 100; i++)
        //            i = i + 1;
        //    }
        //    catch (Exception ex)
        //    {

        //    }
        //    finally
        //    {
        //        if (x == 42)
        //            x++;
        //    }
        //}

        //[Cudafy]
        //public static void TestThrow()
        //{
        //    throw new NotImplementedException();
        //}

        //[Cudafy]
        //public static void TestContinue()
        //{
        //    int ctr = 0;
        //    for (int i = 0; i < 100; i++)
        //    {
        //        if (i < 50)
        //            continue;
        //        if( i > 60)
        //            ctr+=i;
        //        if (i > 70)
        //            ctr++;
        //    }
        //}

        //[Cudafy]
        //public void TestSphereNull(Sphere[] spheres)
        //{
        //    for (int i = 0; i < spheres.Length; i++)
        //        if (this == null)
        //            i++;
        //}


        //[Cudafy]
        //public static double doSquareRoot(float f)
        //{
        //    //f *= (float)Math.PI;
        //    double d = Math.E + Math.PI;
        //    f += (float)(Math.E + d);
        //    return Math.Sqrt(f);
        //}

        //[CudafyDummy]
        //public static int iamadummy(int a, int b)
        //{
        //    return a + b;
        //}

        //[Cudafy]
        //public static TestStruct callTestStruct(TestStruct ts, ref TestStruct refTs, ref float v)
        //{
        //    refTs.value = (ts.value)++;
        //    //ts.value++;
        //    callWithStruct(ref ts);
        //    ts.doit(1.0F, 2.0F, ref v);
        //    return ts;
        //}

        //[Cudafy(eCudafyType.Device)]
        //public static void callWithShared(float[] cacheRef)
        //{
        //    cacheRef[5] = 42.0F;
        //}

        //[Cudafy(eCudafyType.Device)]
        //public static void callWithStruct(ref TestStruct refTs)
        //{
        //    refTs.value = 42;
        //}


        public void TestSetUp()
        {
         
        }

        public void TestTearDown()
        {
           
        }
    }
}
