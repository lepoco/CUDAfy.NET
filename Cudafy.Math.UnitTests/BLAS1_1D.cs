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
using System.Reflection;
using Cudafy.Types;
using Cudafy.Host;
using Cudafy.Maths.BLAS;
using Cudafy.UnitTests;
using NUnit.Framework;
namespace Cudafy.Maths.UnitTests
{    
    [TestFixture]
    public class BLAS1_1D : ICudafyUnitTest
    {

        private float[] _hostInput1;

        private float[] _hostInput2;

        private float[] _hostOutput1;

        private float[] _hostOutput2;

        private float[] _devPtr1;

        private float[] _devPtr2;

        private const int ciN = 1024 * 4;

        private GPGPU _gpu;

        private GPGPUBLAS _blas;

        
        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.CreateDevice(CudafyModes.Target);
            _blas = GPGPUBLAS.Create(_gpu);
            Console.Write("BLAS Version={0}", _blas.GetVersion());
            _hostInput1 = new float[ciN];
            _hostInput2 = new float[ciN];
            _hostOutput1 = new float[ciN];
            _hostOutput2 = new float[ciN];
            _devPtr1 = _gpu.Allocate<float>(_hostInput1);
            _devPtr2 = _gpu.Allocate<float>(_hostOutput1);
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _blas.Dispose();

            _gpu.Free(_devPtr1);
            _gpu.Free(_devPtr2);
        }

        #region MAX

        [Test]
        public void TestMAXInVectorWhole()
        {
            CreateRandomData(_hostInput1);
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            int index = _blas.IAMAX(_devPtr1);
            float max = _hostInput1.Max();
            Assert.AreEqual(max, _hostInput1[index <= 0 ? 0 : index - 1]); // 1-indexed
        }

        [Test]
        public void TestMAXInVectorFirstHalf()
        {
            CreateRamp(_hostInput1);
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            int index = _blas.IAMAX(_devPtr1, ciN / 2);
            float max = _hostInput1.Take(ciN / 2).Max();
            Assert.AreEqual(max, _hostInput1[index - 1]); // 1-indexed
        }

        [Test]
        public void TestMAXInVectorSecondHalf()
        {
            CreateRamp(_hostInput1);
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            int index = _blas.IAMAX(_devPtr1, ciN / 2, ciN / 2);
            _gpu.CopyFromDevice(_devPtr1, _hostOutput1);
            Assert.AreEqual(ciN / 2, index); // 1-indexed
        }

        [Test]
        public void TestMAXInVectorStep()
        {
            CreateRamp(_hostInput1);
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            int index = _blas.IAMAX(_devPtr1, ciN / 2, 0, 2);
            _gpu.CopyFromDevice(_devPtr1, _hostOutput1);
            Assert.AreEqual(ciN / 2, index); // 1-indexed
        }

        #endregion

        #region MIN

        [Test]
        public void TestMINInVectorWhole()
        {
            CreateRandomData(_hostInput1);
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            int index = _blas.IAMIN(_devPtr1);
            float max = _hostInput1.Min();
            Assert.AreEqual(max, _hostInput1[index - 1]); // 1-indexed
        }

        [Test]
        public void TestMINInVectorFirstHalf()
        {
            CreateRamp(_hostInput1);
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            int index = _blas.IAMIN(_devPtr1, ciN / 2);
            float max = _hostInput1.Take(ciN / 2).Min();
            Assert.AreEqual(max, _hostInput1[index - 1]); // 1-indexed
        }

        [Test]
        public void TestMINInVectorSecondHalf()
        {
            CreateRamp(_hostInput1);
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            int index = _blas.IAMIN(_devPtr1, ciN / 2, ciN / 2);
            _gpu.CopyFromDevice(_devPtr1, _hostOutput1);
            Assert.AreEqual(0, index - 1); // 1-indexed
        }

        [Test]
        public void TestMINInVectorStep()
        {
            CreateRamp(_hostInput1);
            _hostInput1 = _hostInput1.Reverse().ToArray();
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            int index = _blas.IAMIN(_devPtr1, ciN / 2, 0, 2);
            _gpu.CopyFromDevice(_devPtr1, _hostOutput1);
            Assert.AreEqual(ciN / 2, index); // 1-indexed
        }

        #endregion

        #region ASUM

        [Test]
        public void TestASUMInVectorWhole()
        {
            CreateRandomData(_hostInput1);
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            float sum = _blas.ASUM(_devPtr1);
            _gpu.CopyFromDevice(_devPtr1, _hostOutput1);
            Assert.AreEqual(_hostInput1.Sum(), sum);
        }

        [Test]
        public void TestASUMInVectorFirstHalf()
        {
            CreateRandomData(_hostInput1);
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            float sum = _blas.ASUM(_devPtr1, ciN / 2, 0);
            _gpu.CopyFromDevice(_devPtr1, _hostOutput1);
            Assert.AreEqual(_hostInput1.Take(ciN / 2).Sum(), sum);
        }

        [Test]
        public void TestASUMInVectorSecondHalf()
        {
            CreateRandomData(_hostInput1);
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            float sum = _blas.ASUM(_devPtr1, ciN / 2, ciN / 2);
            _gpu.CopyFromDevice(_devPtr1, _hostOutput1);
            var scndHalf = _hostInput1.ToList();
            scndHalf.RemoveRange(0, ciN / 2);
            Assert.AreEqual(scndHalf.Sum(), sum);
        }

        [Test]
        public void TestASUMInVectorStep()
        {
            CreateRandomData(_hostInput1);
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            float sum = _blas.ASUM(_devPtr1, ciN/2, 0, 2);
            _gpu.CopyFromDevice(_devPtr1, _hostOutput1);
            float hostsum = 0;
            bool b = true;
            _hostInput1.ToList().ForEach(f => { hostsum += b ? f : 0; b = !b; });
            Assert.AreEqual(hostsum, sum);
        }

        #endregion

        #region AXPY

        [Test]
        public void TestAXPYInVectorWhole()
        {
            CreateRandomData(_hostInput1);
            CreateRandomData(_hostInput2);
            _gpu.CopyToDevice(_hostInput2, _devPtr2);
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            _blas.AXPY(10.0f, _devPtr1, _devPtr2);
            _gpu.CopyFromDevice(_devPtr2, _hostOutput1);
            for (int i = 0; i < ciN; i++)
                Assert.AreEqual(10.0f * _hostInput1[i] + _hostInput2[i], _hostOutput1[i]);
        }

        #endregion       

        #region COPY

        [Test]
        public void TestCOPYInVectorWhole()
        {
            CreateRandomData(_hostInput1);
            CreateRamp(_hostInput2);
            _gpu.CopyToDevice(_hostInput2, _devPtr2);
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            _blas.COPY(_devPtr1, _devPtr2);
            _gpu.CopyFromDevice(_devPtr2, _hostOutput1);
            for (int i = 0; i < ciN; i++)
                Assert.AreEqual(_hostInput1[i], _hostOutput1[i]);
        }

        #endregion

        #region DOT

        [Test]
        public void TestDOTInVectorWhole()
        {
            CreateRandomData(_hostInput1);
            CreateRandomData(_hostInput2);
            _gpu.CopyToDevice(_hostInput2, _devPtr2);
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            float gpuRes = _blas.DOT(_devPtr2, _devPtr1);//, ciN, 0, 1, 0, 1);
            float hostRes = _hostInput1.Zip(_hostInput2, (d1, d2) => d1 * d2).Sum();
            Assert.AreEqual(hostRes, gpuRes, 16);
        }

        #endregion

        #region NRM2

        [Test]
        public void TestNRM2InVectorWhole()
        {
            CreateRandomData(_hostInput1);
             _gpu.CopyToDevice(_hostInput1, _devPtr1);
            float gpuRes = _blas.NRM2(_devPtr1);
            float hostRes = (float)Math.Sqrt(_hostInput1.Sum(f => f * f));
            Assert.AreEqual(hostRes, gpuRes, 0.1);
        }

        #endregion

        #region SCAL

        [Test]
        public void TestSCALInVectorWhole()
        {
            int index = 0;
            CreateRandomData(_hostInput1);
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            _blas.SCAL(10.0f, _devPtr1);
            _gpu.CopyFromDevice(_devPtr1, _hostOutput1);
            foreach (float f in _hostInput1)
                Assert.AreEqual(f * 10.0f, _hostOutput1[index++]);
        }

        #endregion

        #region SWAP

        [Test]
        public void TestSWAPInVectorWhole()
        {
            CreateRandomData(_hostInput1);
            CreateRamp(_hostInput2);
            _gpu.CopyToDevice(_hostInput2, _devPtr2);
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            _blas.SWAP(_devPtr1, _devPtr2);
            _gpu.CopyFromDevice(_devPtr2, _hostOutput2);
            _gpu.CopyFromDevice(_devPtr1, _hostOutput1);
            for (int i = 0; i < ciN; i++)
                Assert.AreEqual(_hostInput1[i], _hostOutput2[i]);
            for (int i = 0; i < ciN; i++)
                Assert.AreEqual(_hostInput2[i], _hostOutput1[i]);
        }

        #endregion 

        //[Test]
        //public void TestAXPYAlphaDeviceInVectorWhole()
        //{
        //    CreateRandomData(_hostInput);
        //    CreateRandomData(_hostInput2);
        //    float[] alpha = new float[1]; 
        //    CreateRandomData(alpha);
        //    _gpu.CopyToDevice(_hostInput2, _devPtr2);
        //    _gpu.CopyToDevice(_hostInput, _devPtr);
        //    float[] devAlpha = _gpu.CopyToDevice(alpha);
        //    _blas.AXPY(devAlpha, _devPtr, _devPtr2);
        //    _gpu.CopyFromDevice(_devPtr2, _hostOutput);
        //    float[] test = new float[1];
        //    _gpu.CopyFromDevice(devAlpha, test);
        //    //for (int i = 0; i < ciN; i++)
        //    //    Assert.AreEqual(alpha[0] * _hostInput[i] + _hostInput2[i], _hostOutput[i]);
        //}

        private void CreateRamp(float[] buffer)
        {
            for (int i = 0; i < buffer.Length; i++)
                buffer[i] = (float)i;
        }

        private void CreateRandomData(float[] buffer)
        {
            Random rand = new Random(Environment.TickCount);
            for (int i = 0; i < buffer.Length; i++)
                buffer[i] = (float)rand.Next(512);
            System.Threading.Thread.Sleep(rand.Next(100));
        }


        public void TestSetUp()
        {
         
        }

        public void TestTearDown()
        {
           
        }
    }
}
