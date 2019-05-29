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
    public struct UncudafiedStruct
    {
        public int x;
        public int y;
    }

    [Cudafy]
    public struct StructB
    {
        public int x;
        public int y;
    }

    [Cudafy]
    public struct StructA
    {
        public int x;
        public StructB SB;
    }
    

    [Cudafy]
    public struct StructWithBool
    {
        public bool B;
    }
    
    [TestFixture]
    public class RelectorAddInTypeTests : CudafyUnitTest, ICudafyUnitTest
    {
        [Cudafy]
        public struct Sphere
        {
            public float r;
            public float b;
            public float g;
            public float radius;
            public float x;
            public float y;
            public float z;
           // [CudafyIgnore]
            public Sphere(float unwanted)
            {
                r = unwanted;
                b = unwanted;
                g = unwanted;                
                x = unwanted;
                y = unwanted;
                z = unwanted;
                radius = unwanted;
            }

            public float hit(float ox1, float oy1, ref float n1)
            {
                float dx = ox1 - x;
                float dy = oy1 - y;
                if (dx * dx + dy * dy < radius * radius)
                {
                    float dz = GMath.Sqrt(radius * radius - dx * dx - dy * dy);
                    n1 = dz / GMath.Sqrt(radius * radius);
                    return dz + z;
                }
                return -2e10f;
            }

            public float dostuff()
            {
                Sphere s = new Sphere(42);
                return s.r;
            }
        }
        
        private CudafyModule _cm;

        private const int N = 1024;

        [SetUp]
        public void SetUp()
        {
            _cm = CudafyTranslator.Cudafy(typeof(Sphere), typeof(RelectorAddInTypeTests));
        }

        [TearDown]
        public void TearDown()
        {
        }

        [Test]
        public void TestHas_kernel()
        {
            Assert.Contains("kernel", _cm.Functions.Keys);
        }

        [Test]
        public void TestHasSphere()
        {
            Assert.Contains("Cudafy.UnitTests.RelectorAddInTypeTestsSphere", _cm.Types.Keys);
        }

        [Test]
        public void TestHasConstantSphereArray()
        {
            Assert.Contains("constantSphereArray", _cm.Constants.Keys);
        }

        //[Test]
        //[ExpectedException(typeof(CudafyLanguageException))]
        //public void TestCudafyStructWithoutAttribute()
        //{
        //    var mod = CudafyTranslator.Cudafy(typeof(UncudafiedStruct));
        //}

        [Test]
        public void TestStructDependencies()
        {
            var mod = CudafyTranslator.Cudafy(typeof(StructB), typeof(StructA));
            mod.Serialize("TestStructDependencies");
        }

        [Test]
        public void TestStructWithBoolean()
        {
            var mod = CudafyTranslator.Cudafy(typeof(StructWithBool));
            mod.Serialize("TestStructWithBoolean");
        }



        public const int DIM = 1024;
        public const int RAND_MAX = Int32.MaxValue;
        public const float INF = 2e10f;

        public static float rnd(float x)
        {
            float f = x * (float)rand.NextDouble();
            return f;
        }

        public static Random rand = new Random((int)DateTime.Now.Ticks);

        public const int SPHERES = 20;

        [Cudafy]
        public static Sphere[] constantSphereArray = new Sphere[SPHERES];


        [Cudafy]
        public static void kernel(GThread thread, byte[] ptr)
        {
            // map from threadIdx/BlockIdx to pixel position
            int x = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int y = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
            int offset = x + y * thread.blockDim.x * thread.gridDim.x;
            float ox = (x - DIM / 2);
            float oy = (y - DIM / 2);

            float r = 0, g = 0, b = 0;
            float maxz = -INF;

            for (int i = 0; i < SPHERES; i++)
            {
                float n = 0;

                float t = constantSphereArray[i].hit(ox, oy, ref n);
                if (t > maxz)
                {
                    float fscale = n;
                    r = constantSphereArray[i].r * fscale;
                    g = constantSphereArray[i].g * fscale;
                    b = constantSphereArray[i].b * fscale;
                    maxz = t;
                }
            }

            ptr[offset * 4 + 0] = (byte)(r * 255);
            ptr[offset * 4 + 1] = (byte)(g * 255);
            ptr[offset * 4 + 2] = (byte)(b * 255);
            ptr[offset * 4 + 3] = 255;
        }


        public void TestSetUp()
        {
        
        }

        public void TestTearDown()
        {
            
        }
    }

}

