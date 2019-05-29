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
using Cudafy.Host;
using Cudafy.Atomics;
using Cudafy.IntegerIntrinsics;
using Cudafy.UnitTests;
using NUnit.Framework;
using Cudafy.Translator;

namespace Cudafy.Host.UnitTests
{
    [TestFixture]
    public abstract class CudafiedUnitTests : CudafyUnitTest, ICudafyUnitTest
    {
        private CudafyModule _cm;

        protected GPGPU _gpu;

        public bool SupportsDouble { get; private set; }

        [TestFixtureSetUp]
        public virtual void SetUp()
        {
            _gpu = CudafyHost.GetDevice(CudafyModes.Architecture, CudafyModes.DeviceId);
            var types = new List<Type>();
            types.Add(this.GetType());
            types.Add(typeof(MathSingleTest));
            SupportsDouble = _gpu.GetDeviceProperties().SupportsDoublePrecision;
            if (SupportsDouble)
                types.Add(typeof(MathDoubleTest));

            _cm = CudafyTranslator.Cudafy(CudafyModes.Architecture, types.ToArray());
            Debug.WriteLine(_cm.SourceCode);
            _gpu.LoadModule(_cm);
        }

        [TestFixtureTearDown]
        public virtual void TearDown()
        {
            _gpu.FreeAll();
        }

        [SetUp]
        public virtual void TestSetUp()
        {

        }

        [TearDown]
        public virtual void TestTearDown()
        {

        }
    }  
    
    [TestFixture]
    public class GMathUnitTests : CudafiedUnitTests, ICudafyUnitTest
    {

        private const int N = 64;

        [Test]
        public void Test_Math()
        {
            if (!SupportsDouble)
            {
                Console.WriteLine("Device does not support double precision, skipping test...");
                return;
            }
            double[] data = new double[N]; 
            double[] dev_data = _gpu.CopyToDevice(data);
#if !NET35
            _gpu.Launch().mathtest(dev_data);
#else
            _gpu.Launch(1, 1, "mathtest", dev_data);
#endif
            _gpu.CopyFromDevice(dev_data, data);
            double[] control = new double[N];
            MathDoubleTest.mathtest(control);
            for (int i = 0; i < N-4; i++)
                Assert.AreEqual(control[i], data[i], 0.00005, "Index={0}", i);
            if (_gpu is CudaGPU)
            {
                Assert.AreEqual(9.2188684372274053E18, data[60], 0.0000000000000001E18);
                Assert.AreEqual(1.8442240474082181E19, data[61], 0.0000000000000001E19);
            }
            else
            {
                Assert.IsTrue(Double.IsInfinity(data[60]));
                Assert.IsTrue(Double.IsInfinity(data[61]));
            }
            Assert.IsTrue(Double.IsNaN(data[62]));
        }

        [Test]
        public void Test_GMath()
        {
            float[] data = new float[N];
            float[] dev_data = _gpu.CopyToDevice(data);
#if !NET35
            _gpu.Launch().gmathtest(dev_data);
#else
            _gpu.Launch(1, 1, "gmathtest", dev_data);
#endif
            _gpu.CopyFromDevice(dev_data, data);
            float[] control = new float[N];
            MathSingleTest.gmathtest(control);
            for (int i = 0; i < N-4; i++)
                Assert.AreEqual(control[i], data[i], 0.00005, "Index={0}", i);
            if (_gpu is CudaGPU)
            {
                Assert.AreEqual(2.14643507E9, data[60], 0.00000001E9);
                Assert.AreEqual(4.29391872E9, data[61], 0.00000001E9);
            }
            else
            {
                Assert.IsTrue(Single.IsInfinity(data[60]));
                Assert.IsTrue(Single.IsInfinity(data[61]));
            }
            Assert.IsTrue(Single.IsNaN(data[62]));
        }

        [Test]
        public void Test_atomicsUInt32()
        {
            uint[] input = new uint[N];
            uint[] output = new uint[N];
            uint[] dev_input = _gpu.CopyToDevice(input);
            uint[] dev_output = _gpu.CopyToDevice(output);
            _gpu.Launch(1, 1, "atomicsTestUInt32", dev_input, dev_output);
            _gpu.CopyFromDevice(dev_input, input);
            _gpu.CopyFromDevice(dev_output, output);

            uint[] inputControl = new uint[N];
            uint[] outputControl = new uint[N];
            atomicsTestUInt32(new GThread(0, 0, new GBlock(new GGrid(1), 1, 0, 0)), inputControl, outputControl);
            for (int i = 0; i < N; i++)
                Assert.AreEqual(inputControl[i], input[i], "Input Index={0}", i);
            for (int i = 0; i < N; i++)
                Assert.AreEqual(outputControl[i], output[i], "Output Index={0}", i);
            _gpu.FreeAll();
        }

        [Test]
        public void Test_integerIntrinsicsInt32()
        {
            int[] input = new int[N];
            int[] output = new int[N];
            int[] dev_input = _gpu.CopyToDevice(input);
            int[] dev_output = _gpu.CopyToDevice(output);
            _gpu.Launch(1, 1, "integerIntrinsicsInt32", dev_input, dev_output);
            _gpu.CopyFromDevice(dev_input, input);
            _gpu.CopyFromDevice(dev_output, output);

            int[] inputControl = new int[N];
            int[] outputControl = new int[N];
            integerIntrinsicsInt32(new GThread(0, 0, new GBlock(new GGrid(1), 1, 0, 0)), inputControl, outputControl);
            for (int i = 0; i < N; i++)
                Assert.AreEqual(outputControl[i], output[i], "Output Index={0}", i);
            _gpu.FreeAll();
        }

        [Cudafy]
        public static void integerIntrinsicsInt32(GThread thread, int[] input, int[] output)
        {
            int i = 0;
            int x = 0;
            output[i++] = thread.popcount((uint)0x55555555);// 16
            output[i++] = thread.clz(0x1FFFE00);            // 7
            output[i++] = (int)thread.mul24((int)0x00000042, (int)0x00000042);
            output[i++] = (int)thread.umul24((uint)0x00000042, (uint)0x00000042);
            output[i++] = (int)thread.mulhi(0x0AFFEEDD, 0x0DEEFFAA);
            output[i++] = (int)thread.umulhi((uint)0x0AFFEEDD, (uint)0x0DEEFFAA);
        }

        [Test]
        public void Test_integerIntrinsicsInt64()
        {
            long[] input = new long[N];
            long[] output = new long[N];
            long[] dev_input = _gpu.CopyToDevice(input);
            long[] dev_output = _gpu.CopyToDevice(output);
            _gpu.Launch(1, 1, "integerIntrinsicsInt64", dev_input, dev_output);
            _gpu.CopyFromDevice(dev_input, input);
            _gpu.CopyFromDevice(dev_output, output);

            long[] inputControl = new long[N];
            long[] outputControl = new long[N];
            integerIntrinsicsInt64(new GThread(0, 0, new GBlock(new GGrid(1), 1, 0, 0)), inputControl, outputControl);
            for (int i = 0; i < N; i++)
                Assert.AreEqual(outputControl[i], output[i], "Output Index={0}", i);
            _gpu.FreeAll();
        }

        [Cudafy]
        public static void integerIntrinsicsInt64(GThread thread, long[] input, long[] output)
        {
            int i = 0;
            int x = 0;
            output[i++] = thread.popcountll(0x5555555555555555);  // 32
            output[i++] = thread.clzll(0x1FFFFFFFFF000);          // 15
            output[i++] = (long)thread.umul64hi(0x0FFFFFFFFF000, 0x0555555555555555);
            output[i++] = (long)thread.mul64hi(0x0FFFFFFFFF000, 0x0555555555555555);
        }


        [Cudafy]
        public static void atomicsTestUInt32(GThread thread, uint[] input, uint[] output)
        {
            int i = 0;
            int x = 0;
            output[i++] = thread.atomicAdd(ref input[x], 42); // 42
            output[i++] = thread.atomicSub(ref input[x], 21); // 21
            output[i++] = thread.atomicIncEx(ref input[x]);   // 22
            output[i++] = thread.atomicIncEx(ref input[x]);   // 23
            output[i++] = thread.atomicMax(ref input[x], 50); // 50
            output[i++] = thread.atomicMin(ref input[x], 40); // 40
            output[i++] = thread.atomicOr(ref input[x], 16);  // 56
            output[i++] = thread.atomicAnd(ref input[x], 15); // 8
            output[i++] = thread.atomicXor(ref input[x], 15); // 7
            output[i++] = thread.atomicExch(ref input[x], 88);// 88
            output[i++] = thread.atomicCAS(ref input[x], 88, 123);// 123
            output[i++] = thread.atomicCAS(ref input[x], 321, 222);// 123
            output[i++] = thread.atomicDecEx(ref input[x]);   // 122
        }
    }

    public class MathDoubleTest
    {
        [Cudafy]
        public static void mathtest(double[] c)
        {
            int i = 0;
            c[i++] = Math.Abs((int)-42.3);
            c[i++] = Math.Acos(42.3);
            c[i++] = Math.Asin(42.3);
            c[i++] = Math.Atan(42.3);
            c[i++] = Math.Atan2(42.3, 3.8);
            c[i++] = Math.Ceiling(42.3);
            c[i++] = Math.Cos(42.3);
            c[i++] = Math.Cosh(2.3);
            c[i++] = Math.E;
            c[i++] = Math.Exp(1.3);
            c[i++] = Math.Floor(3.9);
            c[i++] = Math.Log(5.8);
            c[i++] = Math.Log10(3.5);
            c[i++] = Math.Max(4.8, 4.9);
            c[i++] = Math.Min(4.8, 4.9);
            c[i++] = Math.PI;
            c[i++] = Math.Pow(4.4, 2.3);
            c[i++] = Math.Round(5.5);
            c[i++] = Math.Sin(4.2);
            c[i++] = Math.Sinh(3.1);
            c[i++] = Math.Sqrt(8.1);
            c[i++] = Math.Tan(4.3);
            c[i++] = Math.Tanh(8.1);
            c[i++] = Math.Truncate(10.14334325);
            c[i++] = (double)(Double.IsNaN(Math.Sqrt(-1.0)) ? 1.0 : 0.0);
            double zero = 0.0;
            c[i++] = Double.IsInfinity(1/zero) ? 1.0 : 0.0;
            c[60] = Double.PositiveInfinity;
            c[61] = Double.NegativeInfinity;
            c[62] = Double.NaN;
        }
    }

    public class MathSingleTest
    {

        [Cudafy]
        public static void gmathtest(float[] c)
        {
            int i = 0;
            c[i++] = GMath.Abs((long)-42.3F);
            c[i++] = GMath.Acos(42.3F);
            c[i++] = GMath.Asin(42.3F);
            c[i++] = GMath.Atan(42.3F);
            c[i++] = GMath.Atan2(42.3F, 3.8F);
            c[i++] = GMath.Ceiling(42.3F);
            c[i++] = GMath.Cos(42.3F);
            c[i++] = GMath.Cosh(2.3F);
            c[i++] = GMath.E;
            c[i++] = GMath.Exp(1.3F);
            c[i++] = GMath.Floor(3.9F);
            c[i++] = GMath.Log(5.8F);
            c[i++] = GMath.Log10(3.5F);
            c[i++] = GMath.Max(4.8F, 4.9F);
            c[i++] = GMath.Min(4.8F, 4.9F);
            c[i++] = GMath.PI;
            c[i++] = GMath.Pow(4.4F, 2.3F);
            c[i++] = GMath.Round(5.5F);
            c[i++] = GMath.Sin(4.2F);
            c[i++] = GMath.Sinh(3.1F);
            c[i++] = GMath.Sqrt(8.1F);
            c[i++] = GMath.Sqrt(-1.0F);
            c[i++] = GMath.Tan(4.3F);
            c[i++] = GMath.Tanh(8.1F);
            //c[i++] = (float)(Single.IsNaN(GMath.Sqrt(-1.0F)) ? 1.0F : 0.0F);//GMath.Sqrt(-1.0F)
            c[i++] = GMath.Truncate(10.14334325F);
            float zero = 0.0F;
            c[i++] = Single.IsInfinity(1 / zero) ? 1.0F : 0.0F;
            c[60] = Single.PositiveInfinity;
            c[61] = Single.NegativeInfinity;
            c[62] = Single.NaN; 
        }
    }
}
//0x7f800000 = infinity

//0xff800000 = -infinity



//These conform to the ieee floating point specification. You can use the values:



//0x7ff0000000000000 = infinity

//0xfff0000000000000 = -infinity