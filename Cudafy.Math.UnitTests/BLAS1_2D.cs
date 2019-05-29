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
    public class BLAS1_2D : ICudafyUnitTest
    {

        private float[,] _hostInput;

        private float[,] _hostInput2;

        private float[,] _hostOutput;

        private float[,] _devPtr;

        private float[,] _devPtr2;

        private const int ciCOLS = 4;

        private const int ciROWS = 6;

        private const int ciTOTAL = ciROWS * ciCOLS;

        private GPGPU _gpu;

        private GPGPUBLAS _blas;


        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.CreateDevice(CudafyModes.Target);
            _blas = GPGPUBLAS.Create(_gpu);
            _hostInput =  new float[ciROWS, ciCOLS];
            _hostInput2 = new float[ciROWS, ciCOLS];
            _hostOutput = new float[ciROWS, ciCOLS];
            _devPtr = _gpu.Allocate<float>(_hostInput);
            _devPtr2 = _gpu.Allocate<float>(_hostOutput);
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _blas.Dispose();

            _gpu.Free(_devPtr);
            _gpu.Free(_devPtr2);
        }

        [Test]
        public void TestMAXInVectorWhole()
        {
            CreateRandomData(_hostInput);
            DebugBuffer(_hostInput);
            _gpu.CopyToDevice(_hostInput, _devPtr);
            float[] castDevPtr = _gpu.Cast(_devPtr, ciTOTAL);
            int index = _blas.IAMAX(castDevPtr);
            var list = _hostInput.Cast<float>().ToList();
            float max = list.Max();
            
            Debug.WriteLine(index);
            Debug.WriteLine(max);
            Assert.AreEqual(max, list[index - 1]); // 1-indexed
        }

        [Test]
        public void TestMAXInFirstRows()
        {
            CreateRandomData(_hostInput);
            _hostInput[4, 2] = 999;
            DebugBuffer(_hostInput);
            _gpu.CopyToDevice(_hostInput, _devPtr);
            float[] castDevPtr = _gpu.Cast(_devPtr, ciTOTAL);
            int index = _blas.IAMAX(castDevPtr, ciTOTAL / 2);
            var list = _hostInput.Cast<float>().ToList();
            float max = list.Take(ciTOTAL / 2).Max();

            Debug.WriteLine(index);
            Debug.WriteLine(max);
            Assert.AreEqual(max, list[index - 1]); // 1-indexed
        }

        [Test]
        public void TestMAXInLastRows()
        {
            CreateRandomData(_hostInput);
            _hostInput[4, 1] = 999;
            DebugBuffer(_hostInput);
            _gpu.CopyToDevice(_hostInput, _devPtr);
            float[] castDevPtr = _gpu.Cast(_devPtr, ciTOTAL);
            int index = _blas.IAMAX(castDevPtr, ciTOTAL / 2, ciTOTAL / 2);
            var list1 = _hostInput.Cast<float>().ToList();
            var list2 = _hostInput.Cast<float>().ToList();
            list2.RemoveRange(0, ciTOTAL / 2); 
            float max = list2.Max();

            Debug.WriteLine(index);
            Debug.WriteLine(max);
            Assert.AreEqual(max, list2[index - 1]); // 1-indexed
        }

        private Tuple<int, int> Get2DLocation(int pos, int width, int height)
        {
            int x = pos / height;
            int y = (pos % height) - 1;
            return new Tuple<int, int>(x, y);
        }

        //private void CreateRamp(float[] buffer)
        //{
        //    for (int i = 0; i < buffer.Length; i++)
        //        buffer[i] = (float)i;
        //}

        private void CreateRandomData(float[,] buffer, int max = 899)
        {
            Random rand = new Random(DateTime.Now.Millisecond);
            int width = buffer.GetLength(1);
            int height = buffer.GetLength(0);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    buffer[y, x] = (float)rand.Next(max);
                    //Debug.Write(string.Format("{0}\t\t", buffer[x,y]));
                }
                //Debug.WriteLine("");
            }
        }

        private void DebugBuffer(float[,] buffer)
        {
            int cols = buffer.GetLength(1);
            int rows = buffer.GetLength(0);
            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < cols; x++)
                {
                    Debug.Write(string.Format("{0}\t\t", buffer[y, x]));
                }
                Debug.WriteLine("");
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
