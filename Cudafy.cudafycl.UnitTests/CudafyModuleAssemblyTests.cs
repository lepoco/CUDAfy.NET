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
using System.Diagnostics;
using System.Text;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.IO;
using Cudafy.Host;
using Cudafy.UnitTests;
using NUnit.Framework;
using Cudafy.Translator;
using Cudafy.Compilers;
namespace Cudafy.cudafycl.UnitTests
{
   
    [TestFixture]
    public class CudafyModuleAssemblyTests : CudafyUnitTest, ICudafyUnitTest
    {
        private CudafyModule _cm;

        private GPGPU _gpu;

        private const int N = 1024;

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.GetDevice(CudafyModes.Target, 0);
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _gpu.FreeAll();
        }

        [Test]
        public void GenerateCudafyModuleFile()
        {
            // Ensure that *.cdfy no longer exists
            string fileName = GetType().Assembly.Location;
            string cdfyFileName = Path.ChangeExtension(fileName, "cdfy");
            if (File.Exists(cdfyFileName))
                File.Delete(cdfyFileName);

            string messages = GetType().Assembly.Cudafy();

            Assert.IsTrue(File.Exists(cdfyFileName));
        }
        [Test]
        public void GenerateCudafyModuleFileAndLoadAndTest()
        {
            GenerateCudafyModuleFile();
            string fileName = GetType().Assembly.Location;
            string cdfyFileName = Path.ChangeExtension(fileName, "cdfy");
            var cm = CudafyModule.Deserialize(cdfyFileName);
            _gpu.LoadModule(cm);

            string a = "I believe it costs €155,95 in Düsseldorf";
            char[] dev_a = _gpu.CopyToDevice(a);
            char[] dev_c = _gpu.Allocate(a.ToCharArray());
            char[] host_c = new char[a.Length];
            _gpu.Launch(1, 1, "TransferUnicodeCharArray", dev_a, dev_c);
            _gpu.CopyFromDevice(dev_c, host_c);
            string c = new string(host_c);
            _gpu.FreeAll();
            Assert.AreEqual(a, c);
            Debug.WriteLine(c);

        }

        [Cudafy]
        public static void TransferUnicodeCharArray(char[] a, char[] c)
        {
            for(int i = 0; i < a.Length; i++)
                c[i] = a[i];
        }



        public void TestSetUp()
        {
        
        }

        public void TestTearDown()
        {
          
        }
    }
}

