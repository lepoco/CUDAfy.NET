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
using System.IO;
using Cudafy.Host;
using Cudafy.UnitTests;
using NUnit.Framework;
using Cudafy.Translator;
using Cudafy.Compilers;
using Cudafy.SIMDFunctions; 

namespace Cudafy.Host.UnitTests
{
    [TestFixture]
    public unsafe class SIMDFunctionTests : CudafyUnitTest, ICudafyUnitTest
    {
        private CudafyModule _cm;

        private GPGPU _gpu;

        private GThread _et = new GThread(1, 1, null); // for testing emulator

        uint a, b, expectedResult, gpuResult, cpuResult; // easier to declare these here than as local variables in every [Test]



        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.GetDevice(CudafyModes.Architecture, CudafyModes.DeviceId);

            _cm = CudafyModule.TryDeserialize();
            if (_cm == null || !_cm.TryVerifyChecksums())
            {
                _cm = CudafyTranslator.Cudafy(CudafyModes.Architecture);//typeof(PrimitiveStruct), typeof(BasicFunctionTests));
                Console.WriteLine(_cm.CompilerOutput);
                _cm.TrySerialize();
            }

            _gpu.LoadModule(_cm);

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


        // The tests -- see simd_functions.h for definitions of the gpu functions

        [Test]
        public void vabs2()
        {
            a = 0x7fff8000;
            expectedResult = 0x7fff8000;
            gpuResult = Test("unitTest_vabs2", a);
            cpuResult = _et.vabs2(a);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x8001ffff;
            expectedResult = 0x7fff0001;
            gpuResult = Test("unitTest_vabs2", a);
            cpuResult = _et.vabs2(a);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vabs2(GThread thread, uint[] a, uint[] c)
        {
            c[0] = thread.vabs2(a[0]);
        }


        [Test]
        public void vabsdiffs2()
        {
            a = 0xffff1111;
            b = 0x11112345;
            expectedResult = 0x11121234;
            gpuResult = Test("unitTest_vabsdiffs2", a, b);
            cpuResult = _et.vabsdiffs2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vabsdiffs2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vabsdiffs2(a[0], b[0]);
        }


        [Test]
        public void vabsdiffu2()
        {
            a = 0xffff1111;
            b = 0x11112345;
            expectedResult = 0xeeee1234;
            gpuResult = Test("unitTest_vabsdiffu2", a, b);
            cpuResult = _et.vabsdiffu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vabsdiffu2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vabsdiffu2(a[0], b[0]);
        }


        [Test]
        public void vabsss2()
        {
            a = 0x7fff8000;
            expectedResult = 0x7fff7fff;
            gpuResult = Test("unitTest_vabsss2", a);
            cpuResult = _et.vabsss2(a);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x8001ffff;
            expectedResult = 0x7fff0001;
            gpuResult = Test("unitTest_vabsss2", a);
            cpuResult = _et.vabsss2(a);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vabsss2(GThread thread, uint[] a, uint[] c)
        {
            c[0] = thread.vabsss2(a[0]);
        }


        [Test]
        public void vadd2()
        {
            a = 0xffff1111;
            b = 0x11112222;
            expectedResult = 0x11103333;
            gpuResult = Test("unitTest_vadd2", a, b);
            cpuResult = _et.vadd2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vadd2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vadd2(a[0], b[0]);
        }


        [Test]
        public void vaddss2()
        {
            a = 0x55559999;
            b = 0x4444dddd;
            expectedResult = 0x7fff8000;
            gpuResult = Test("unitTest_vaddss2", a, b);
            cpuResult = _et.vaddss2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x11113333;
            b = 0xeeee2222;
            expectedResult = 0xffff5555;
            gpuResult = Test("unitTest_vaddss2", a, b);
            cpuResult = _et.vaddss2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vaddss2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vaddss2(a[0], b[0]);
        }


        [Test]
        public void vaddus2()
        {
            a = 0x55559999;
            b = 0x4444dddd;
            expectedResult = 0x9999ffff;
            gpuResult = Test("unitTest_vaddus2", a, b);
            cpuResult = _et.vaddus2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vaddus2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vaddus2(a[0], b[0]);
        }


        [Test]
        public void vavgs2()
        {
            a = 0x55559999;
            b = 0x44443333;
            expectedResult = 0x4ccde666;
            gpuResult = Test("unitTest_vavgs2", a, b);
            cpuResult = _et.vavgs2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vavgs2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vavgs2(a[0], b[0]);
        }


        [Test]
        public void vavgu2()
        {
            a = 0x55559999;
            b = 0x4444eeee;
            expectedResult = 0x4ccdc444;
            gpuResult = Test("unitTest_vavgu2", a, b);
            cpuResult = _et.vavgu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x0000ffff;
            b = 0xffffeeee;
            expectedResult = 0x8000f777;
            gpuResult = Test("unitTest_vavgu2", a, b);
            cpuResult = _et.vavgu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vavgu2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vavgu2(a[0], b[0]);
        }


        [Test]
        public void vcmpeq2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0xffff0000;
            gpuResult = Test("unitTest_vcmpeq2", a, b);
            cpuResult = _et.vcmpeq2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmpeq2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmpeq2(a[0], b[0]);
        }


        [Test]
        public void vcmpges2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0xffff0000;
            gpuResult = Test("unitTest_vcmpges2", a, b);
            cpuResult = _et.vcmpges2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0xffffffff;
            gpuResult = Test("unitTest_vcmpges2", a, b);
            cpuResult = _et.vcmpges2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmpges2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmpges2(a[0], b[0]);
        }


        [Test]
        public void vcmpgeu2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0xffffffff;
            gpuResult = Test("unitTest_vcmpgeu2", a, b);
            cpuResult = _et.vcmpgeu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0xffff0000;
            gpuResult = Test("unitTest_vcmpgeu2", a, b);
            cpuResult = _et.vcmpgeu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmpgeu2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmpgeu2(a[0], b[0]);
        }


        [Test]
        public void vcmpgts2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0x00000000;
            gpuResult = Test("unitTest_vcmpgts2", a, b);
            cpuResult = _et.vcmpgts2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0x0000ffff;
            gpuResult = Test("unitTest_vcmpgts2", a, b);
            cpuResult = _et.vcmpgts2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmpgts2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmpgts2(a[0], b[0]);
        }


        [Test]
        public void vcmpgtu2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0x0000ffff;
            gpuResult = Test("unitTest_vcmpgtu2", a, b);
            cpuResult = _et.vcmpgtu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0x00000000;
            gpuResult = Test("unitTest_vcmpgtu2", a, b);
            cpuResult = _et.vcmpgtu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

        }
        [Cudafy]
        public static void unitTest_vcmpgtu2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmpgtu2(a[0], b[0]);
        }


        [Test]
        public void vcmples2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0xffffffff;
            gpuResult = Test("unitTest_vcmples2", a, b);
            cpuResult = _et.vcmples2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0xffff0000;
            gpuResult = Test("unitTest_vcmples2", a, b);
            cpuResult = _et.vcmples2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmples2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmples2(a[0], b[0]);
        }


        [Test]
        public void vcmpleu2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0xffff0000;
            gpuResult = Test("unitTest_vcmpleu2", a, b);
            cpuResult = _et.vcmpleu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0xffffffff;
            gpuResult = Test("unitTest_vcmpleu2", a, b);
            cpuResult = _et.vcmpleu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmpleu2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmpleu2(a[0], b[0]);
        }


        [Test]
        public void vcmplts2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0x0000ffff;
            gpuResult = Test("unitTest_vcmplts2", a, b);
            cpuResult = _et.vcmplts2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0x00000000;
            gpuResult = Test("unitTest_vcmplts2", a, b);
            cpuResult = _et.vcmplts2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmplts2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmplts2(a[0], b[0]);
        }


        [Test]
        public void vcmpltu2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0x00000000;
            gpuResult = Test("unitTest_vcmpltu2", a, b);
            cpuResult = _et.vcmpltu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0x0000ffff;
            gpuResult = Test("unitTest_vcmpltu2", a, b);
            cpuResult = _et.vcmpltu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmpltu2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmpltu2(a[0], b[0]);
        }


        [Test]
        public void vcmpne2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0x0000ffff;
            gpuResult = Test("unitTest_vcmpne2", a, b);
            cpuResult = _et.vcmpne2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0x0000ffff;
            gpuResult = Test("unitTest_vcmpne2", a, b);
            cpuResult = _et.vcmpne2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmpne2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmpne2(a[0], b[0]);
        }


        [Test]
        public void vhaddu2()
        {
            a = 0x55559999;
            b = 0x4444eeee;
            expectedResult = 0x4cccc443;
            gpuResult = Test("unitTest_vhaddu2", a, b);
            cpuResult = _et.vhaddu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x0000ffff;
            b = 0xffffeeee;
            expectedResult = 0x7ffff776;
            gpuResult = Test("unitTest_vhaddu2", a, b);
            cpuResult = _et.vhaddu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vhaddu2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vhaddu2(a[0], b[0]);
        }


        [Test]
        public void vmaxs2()
        {
            a = 0xffff1111;
            b = 0x00002222;
            expectedResult = 0x00002222;
            gpuResult = Test("unitTest_vmaxs2", a, b);
            cpuResult = _et.vmaxs2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vmaxs2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vmaxs2(a[0], b[0]);
        }


        [Test]
        public void vmaxu2()
        {
            a = 0xffff1111;
            b = 0x00002222;
            expectedResult = 0xffff2222;
            gpuResult = Test("unitTest_vmaxu2", a, b);
            cpuResult = _et.vmaxu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vmaxu2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vmaxu2(a[0], b[0]);
        }


        [Test]
        public void vmins2()
        {
            a = 0xffff1111;
            b = 0x00002222;
            expectedResult = 0xffff1111;
            gpuResult = Test("unitTest_vmins2", a, b);
            cpuResult = _et.vmins2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vmins2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vmins2(a[0], b[0]);
        }


        [Test]
        public void vminu2()
        {
            a = 0xffff1111;
            b = 0x00002222;
            expectedResult = 0x00001111;
            gpuResult = Test("unitTest_vminu2", a, b);
            cpuResult = _et.vminu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vminu2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vminu2(a[0], b[0]);
        }


        [Test]
        public void vneg2()
        {
            a = 0x7fff8000;
            expectedResult = 0x80018000;
            gpuResult = Test("unitTest_vneg2", a);
            cpuResult = _et.vneg2(a);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0xffff0001;
            expectedResult = 0x0001ffff;
            gpuResult = Test("unitTest_vneg2", a);
            cpuResult = _et.vneg2(a);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vneg2(GThread thread, uint[] a, uint[] c)
        {
            c[0] = thread.vneg2(a[0]);
        }


        [Test]
        public void vnegss2()
        {
            a = 0x7fff8000;
            expectedResult = 0x80017fff;
            gpuResult = Test("unitTest_vnegss2", a);
            cpuResult = _et.vnegss2(a);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0xffff0001;
            expectedResult = 0x0001ffff;
            gpuResult = Test("unitTest_vnegss2", a);
            cpuResult = _et.vnegss2(a);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vnegss2(GThread thread, uint[] a, uint[] c)
        {
            c[0] = thread.vnegss2(a[0]);
        }


        [Test]
        public void vsads2()
        {
            a = 0xffff1111;
            b = 0x00002222;
            expectedResult = 0x00001112;
            gpuResult = Test("unitTest_vsads2", a, b);
            cpuResult = _et.vsads2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsads2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsads2(a[0], b[0]);
        }


        [Test]
        public void vsadu2()
        {
            a = 0xffff1111;
            b = 0x00002222;
            expectedResult = 0x00011110;
            gpuResult = Test("unitTest_vsadu2", a, b);
            cpuResult = _et.vsadu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsadu2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsadu2(a[0], b[0]);
        }


        [Test]
        public void vseteq2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0x00010000;
            gpuResult = Test("unitTest_vseteq2", a, b);
            cpuResult = _et.vseteq2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vseteq2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vseteq2(a[0], b[0]);
        }


        [Test]
        public void vsetges2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0x00010000;
            gpuResult = Test("unitTest_vsetges2", a, b);
            cpuResult = _et.vsetges2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0x00010001;
            gpuResult = Test("unitTest_vsetges2", a, b);
            cpuResult = _et.vsetges2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetges2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetges2(a[0], b[0]);
        }


        [Test]
        public void vsetgeu2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0x00010001;
            gpuResult = Test("unitTest_vsetgeu2", a, b);
            cpuResult = _et.vsetgeu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0x00010000;
            gpuResult = Test("unitTest_vsetgeu2", a, b);
            cpuResult = _et.vsetgeu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetgeu2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetgeu2(a[0], b[0]);
        }


        [Test]
        public void vsetgts2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0x00000000;
            gpuResult = Test("unitTest_vsetgts2", a, b);
            cpuResult = _et.vsetgts2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0x00000001;
            gpuResult = Test("unitTest_vsetgts2", a, b);
            cpuResult = _et.vsetgts2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetgts2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetgts2(a[0], b[0]);
        }


        [Test]
        public void vsetgtu2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0x00000001;
            gpuResult = Test("unitTest_vsetgtu2", a, b);
            cpuResult = _et.vsetgtu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0x00000000;
            gpuResult = Test("unitTest_vsetgtu2", a, b);
            cpuResult = _et.vsetgtu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetgtu2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetgtu2(a[0], b[0]);
        }


        [Test]
        public void vsetles2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0x00010001;
            gpuResult = Test("unitTest_vsetles2", a, b);
            cpuResult = _et.vsetles2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0x00010000;
            gpuResult = Test("unitTest_vsetles2", a, b);
            cpuResult = _et.vsetles2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetles2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetles2(a[0], b[0]);
        }


        [Test]
        public void vsetleu2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0x00010000;
            gpuResult = Test("unitTest_vsetleu2", a, b);
            cpuResult = _et.vsetleu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0x00010001;
            gpuResult = Test("unitTest_vsetleu2", a, b);
            cpuResult = _et.vsetleu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetleu2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetleu2(a[0], b[0]);
        }


        [Test]
        public void vsetlts2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0x00000001;
            gpuResult = Test("unitTest_vsetlts2", a, b);
            cpuResult = _et.vsetlts2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0x00000000;
            gpuResult = Test("unitTest_vsetlts2", a, b);
            cpuResult = _et.vsetlts2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetlts2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetlts2(a[0], b[0]);
        }


        [Test]
        public void vsetltu2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0x00000000;
            gpuResult = Test("unitTest_vsetltu2", a, b);
            cpuResult = _et.vsetltu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0x00000001;
            gpuResult = Test("unitTest_vsetltu2", a, b);
            cpuResult = _et.vsetltu2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetltu2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetltu2(a[0], b[0]);
        }


        [Test]
        public void vsetne2()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0x00000001;
            gpuResult = Test("unitTest_vsetne2", a, b);
            cpuResult = _et.vsetne2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0x23456789;
            b = 0x2345abcd;
            expectedResult = 0x00000001;
            gpuResult = Test("unitTest_vsetne2", a, b);
            cpuResult = _et.vsetne2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetne2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetne2(a[0], b[0]);
        }


        [Test]
        public void vsub2()
        {
            a = 0x00000000;
            b = 0xffff8000;
            expectedResult = 0x00018000;
            gpuResult = Test("unitTest_vsub2", a, b);
            cpuResult = _et.vsub2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0xffff8000;
            b = 0x00000000;
            expectedResult = 0xffff8000;
            gpuResult = Test("unitTest_vsub2", a, b);
            cpuResult = _et.vsub2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsub2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsub2(a[0], b[0]);
        }


        [Test]
        public void vsubss2()
        {
            a = 0x00000000;
            b = 0xffff8000;
            expectedResult = 0x00017fff;
            gpuResult = Test("unitTest_vsubss2", a, b);
            cpuResult = _et.vsubss2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0xfffe8000;
            b = 0x7fff7fff;
            expectedResult = 0x80008000;
            gpuResult = Test("unitTest_vsubss2", a, b);
            cpuResult = _et.vsubss2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsubss2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsubss2(a[0], b[0]);
        }


        [Test]
        public void vsubus2()
        {
            a = 0x00000000;
            b = 0xffff8000;
            expectedResult = 0x00000000;
            gpuResult = Test("unitTest_vsubus2", a, b);
            cpuResult = _et.vsubus2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");

            a = 0xffff8000;
            b = 0x00000000;
            expectedResult = 0xffff8000;
            gpuResult = Test("unitTest_vsubus2", a, b);
            cpuResult = _et.vsubus2(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsubus2(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsubus2(a[0], b[0]);
        }


        [Test]
        public void vabs4()
        {
            a = 0x7f8081ff;
            expectedResult = 0x7f807f01;
            gpuResult = Test("unitTest_vabs4", a);
            cpuResult = _et.vabs4(a);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vabs4(GThread thread, uint[] a, uint[] c)
        {
            c[0] = thread.vabs4(a[0]);
        }


        [Test]
        public void vabsdiffs4()
        {
            a = 0xffff1111;
            b = 0x11112345;
            expectedResult = 0x12121234;
            gpuResult = Test("unitTest_vabsdiffs4", a, b);
            cpuResult = _et.vabsdiffs4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vabsdiffs4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vabsdiffs4(a[0], b[0]);
        }


        [Test]
        public void vabsdiffu4()
        {
            a = 0xffff1111;
            b = 0x11112345;
            expectedResult = 0xeeee1234;
            gpuResult = Test("unitTest_vabsdiffu4", a, b);
            cpuResult = _et.vabsdiffu4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vabsdiffu4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vabsdiffu4(a[0], b[0]);
        }


        [Test]
        public void vabsss4()
        {
            a = 0x7f8081ff;
            expectedResult = 0x7f7f7f01;
            gpuResult = Test("unitTest_vabsss4", a);
            cpuResult = _et.vabsss4(a);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vabsss4(GThread thread, uint[] a, uint[] c)
        {
            c[0] = thread.vabsss4(a[0]);
        }


        [Test]
        public void vadd4()
        {
            a = 0xffff1111;
            b = 0x11112222;
            expectedResult = 0x10103333;
            gpuResult = Test("unitTest_vadd4", a, b);
            cpuResult = _et.vadd4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vadd4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vadd4(a[0], b[0]);
        }


        [Test]
        public void vaddss4()
        {
            a = 0x55991133;
            b = 0x44ddee22;
            expectedResult = 0x7f80ff55;
            gpuResult = Test("unitTest_vaddss4", a, b);
            cpuResult = _et.vaddss4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vaddss4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vaddss4(a[0], b[0]);
        }


        [Test]
        public void vaddus4()
        {
            a = 0x55559999;
            b = 0x4444dddd;
            expectedResult = 0x9999ffff;
            gpuResult = Test("unitTest_vaddus4", a, b);
            cpuResult = _et.vaddus4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vaddus4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vaddus4(a[0], b[0]);
        }


        [Test]
        public void vavgs4()
        {
            a = 0x55559999;
            b = 0x44443333;
            expectedResult = 0x4d4de6e6;
            gpuResult = Test("unitTest_vavgs4", a, b);
            cpuResult = _et.vavgs4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vavgs4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vavgs4(a[0], b[0]);
        }


        [Test]
        public void vavgu4()
        {
            a = 0x559900ff;
            b = 0x44eeffee;
            expectedResult = 0x4dc480f7;
            gpuResult = Test("unitTest_vavgu4", a, b);
            cpuResult = _et.vavgu4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vavgu4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vavgu4(a[0], b[0]);
        }


        [Test]
        public void vcmpeq4()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0xffff0000;
            gpuResult = Test("unitTest_vcmpeq4", a, b);
            cpuResult = _et.vcmpeq4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmpeq4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmpeq4(a[0], b[0]);
        }


        [Test]
        public void vcmpges4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0xff00ffff;
            gpuResult = Test("unitTest_vcmpges4", a, b);
            cpuResult = _et.vcmpges4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmpges4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmpges4(a[0], b[0]);
        }


        [Test]
        public void vcmpgeu4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0xffffff00;
            gpuResult = Test("unitTest_vcmpgeu4", a, b);
            cpuResult = _et.vcmpgeu4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmpgeu4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmpgeu4(a[0], b[0]);
        }


        [Test]
        public void vcmpgts4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0x000000ff;
            gpuResult = Test("unitTest_vcmpgts4", a, b);
            cpuResult = _et.vcmpgts4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmpgts4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmpgts4(a[0], b[0]);
        }


        [Test]
        public void vcmpgtu4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0x00ff0000;
            gpuResult = Test("unitTest_vcmpgtu4", a, b);
            cpuResult = _et.vcmpgtu4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmpgtu4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmpgtu4(a[0], b[0]);
        }


        [Test]
        public void vcmples4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0xffffff00;
            gpuResult = Test("unitTest_vcmples4", a, b);
            cpuResult = _et.vcmples4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmples4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmples4(a[0], b[0]);
        }


        [Test]
        public void vcmpleu4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0xff00ffff;
            gpuResult = Test("unitTest_vcmpleu4", a, b);
            cpuResult = _et.vcmpleu4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmpleu4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmpleu4(a[0], b[0]);
        }


        [Test]
        public void vcmplts4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0x00ff0000;
            gpuResult = Test("unitTest_vcmplts4", a, b);
            cpuResult = _et.vcmplts4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmplts4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmplts4(a[0], b[0]);
        }


        [Test]
        public void vcmpltu4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0x000000ff;
            gpuResult = Test("unitTest_vcmpltu4", a, b);
            cpuResult = _et.vcmpltu4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmpltu4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmpltu4(a[0], b[0]);
        }


        [Test]
        public void vcmpne4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0x00ff00ff;
            gpuResult = Test("unitTest_vcmpne4", a, b);
            cpuResult = _et.vcmpne4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vcmpne4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vcmpne4(a[0], b[0]);
        }


        [Test]
        public void vhaddu4()
        {
            a = 0x559900ff;
            b = 0x44eeffee;
            expectedResult = 0x4cc37ff6;
            gpuResult = Test("unitTest_vhaddu4", a, b);
            cpuResult = _et.vhaddu4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vhaddu4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vhaddu4(a[0], b[0]);
        }


        [Test]
        public void vmaxs4()
        {
            a = 0xffff1111;
            b = 0x00002222;
            expectedResult = 0x00002222;
            gpuResult = Test("unitTest_vmaxs4", a, b);
            cpuResult = _et.vmaxs4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vmaxs4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vmaxs4(a[0], b[0]);
        }


        [Test]
        public void vmaxu4()
        {
            a = 0xffff1111;
            b = 0x00002222;
            expectedResult = 0xffff2222;
            gpuResult = Test("unitTest_vmaxu4", a, b);
            cpuResult = _et.vmaxu4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vmaxu4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vmaxu4(a[0], b[0]);
        }


        [Test]
        public void vmins4()
        {
            a = 0xffff1111;
            b = 0x00002222;
            expectedResult = 0xffff1111;
            gpuResult = Test("unitTest_vmins4", a, b);
            cpuResult = _et.vmins4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vmins4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vmins4(a[0], b[0]);
        }


        [Test]
        public void vminu4()
        {
            a = 0xffff1111;
            b = 0x00002222;
            expectedResult = 0x00001111;
            gpuResult = Test("unitTest_vminu4", a, b);
            cpuResult = _et.vminu4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vminu4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vminu4(a[0], b[0]);
        }


        [Test]
        public void vneg4()
        {
            a = 0x7f80ff01;
            expectedResult = 0x818001ff;
            gpuResult = Test("unitTest_vneg4", a);
            cpuResult = _et.vneg4(a);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vneg4(GThread thread, uint[] a, uint[] c)
        {
            c[0] = thread.vneg4(a[0]);
        }


        [Test]
        public void vnegss4()
        {
            a = 0x7f80ff01;
            expectedResult = 0x817f01ff;
            gpuResult = Test("unitTest_vnegss4", a);
            cpuResult = _et.vnegss4(a);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vnegss4(GThread thread, uint[] a, uint[] c)
        {
            c[0] = thread.vnegss4(a[0]);
        }


        [Test]
        public void vsads4()
        {
            a = 0xffff1111;
            b = 0x00002222;
            expectedResult = 0x00000024;
            gpuResult = Test("unitTest_vsads4", a, b);
            cpuResult = _et.vsads4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsads4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsads4(a[0], b[0]);
        }


        [Test]
        public void vsadu4()
        {
            a = 0xffff1111;
            b = 0x00002222;
            expectedResult = 0x00000220;
            gpuResult = Test("unitTest_vsadu4", a, b);
            cpuResult = _et.vsadu4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsadu4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsadu4(a[0], b[0]);
        }


        [Test]
        public void vseteq4()
        {
            a = 0x1234abcd;
            b = 0x12345678;
            expectedResult = 0x01010000;
            gpuResult = Test("unitTest_vseteq4", a, b);
            cpuResult = _et.vseteq4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vseteq4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vseteq4(a[0], b[0]);
        }


        [Test]
        public void vsetges4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0x01000101;
            gpuResult = Test("unitTest_vsetges4", a, b);
            cpuResult = _et.vsetges4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetges4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetges4(a[0], b[0]);
        }


        [Test]
        public void vsetgeu4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0x01010100;
            gpuResult = Test("unitTest_vsetgeu4", a, b);
            cpuResult = _et.vsetgeu4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetgeu4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetgeu4(a[0], b[0]);
        }


        [Test]
        public void vsetgts4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0x00000001;
            gpuResult = Test("unitTest_vsetgts4", a, b);
            cpuResult = _et.vsetgts4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetgts4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetgts4(a[0], b[0]);
        }


        [Test]
        public void vsetgtu4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0x00010000;
            gpuResult = Test("unitTest_vsetgtu4", a, b);
            cpuResult = _et.vsetgtu4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetgtu4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetgtu4(a[0], b[0]);
        }


        [Test]
        public void vsetles4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0x01010100;
            gpuResult = Test("unitTest_vsetles4", a, b);
            cpuResult = _et.vsetles4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetles4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetles4(a[0], b[0]);
        }


        [Test]
        public void vsetleu4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0x01000101;
            gpuResult = Test("unitTest_vsetleu4", a, b);
            cpuResult = _et.vsetleu4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetleu4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetleu4(a[0], b[0]);
        }


        [Test]
        public void vsetlts4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0x00010000;
            gpuResult = Test("unitTest_vsetlts4", a, b);
            cpuResult = _et.vsetlts4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetlts4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetlts4(a[0], b[0]);
        }


        [Test]
        public void vsetltu4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0x00000001;
            gpuResult = Test("unitTest_vsetltu4", a, b);
            cpuResult = _et.vsetltu4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetltu4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetltu4(a[0], b[0]);
        }


        [Test]
        public void vsetne4()
        {
            a = 0x12ab1256;
            b = 0x125612ab;
            expectedResult = 0x00010001;
            gpuResult = Test("unitTest_vsetne4", a, b);
            cpuResult = _et.vsetne4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsetne4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsetne4(a[0], b[0]);
        }


        [Test]
        public void vsub4()
        {
            a = 0x0000ff80;
            b = 0xff800000;
            expectedResult = 0x0180ff80;
            gpuResult = Test("unitTest_vsub4", a, b);
            cpuResult = _et.vsub4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsub4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsub4(a[0], b[0]);
        }


        [Test]
        public void vsubss4()
        {
            a = 0x0000fe80;
            b = 0xff807f7f;
            expectedResult = 0x017f8080;
            gpuResult = Test("unitTest_vsubss4", a, b);
            cpuResult = _et.vsubss4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsubss4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsubss4(a[0], b[0]);
        }


        [Test]
        public void vsubus4()
        {
            a = 0x0000ff80;
            b = 0xff800000;
            expectedResult = 0x0000ff80;
            gpuResult = Test("unitTest_vsubus4", a, b);
            cpuResult = _et.vsubus4(a, b);
            Assert.AreEqual(expectedResult, gpuResult, "GPU result:");
            Assert.AreEqual(expectedResult, cpuResult, "CPU result:");
        }
        [Cudafy]
        public static void unitTest_vsubus4(GThread thread, uint[] a, uint[] b, uint[] c)
        {
            c[0] = thread.vsubus4(a[0], b[0]);
        }


        // generic test functions for setting up the gpu

        public uint Test(string kernelName, uint inputValue)
        {
            uint[] a = new uint[] { inputValue };
            uint[] c = new uint[1];
            uint[] da = _gpu.CopyToDevice(a);
            uint[] dc = _gpu.Allocate(c);
            _gpu.Launch(1, 1, kernelName, da, dc);
            _gpu.CopyFromDevice(dc, c);
            return c[0];
        }

        public uint Test(string kernelName, uint inputValue1, uint inputValue2)
        {
            uint[] a = new uint[] { inputValue1 };
            uint[] b = new uint[] { inputValue2 };
            uint[] c = new uint[1];
            uint[] da = _gpu.CopyToDevice(a);
            uint[] db = _gpu.CopyToDevice(b);
            uint[] dc = _gpu.Allocate(c);
            _gpu.Launch(1, 1, kernelName, da, db, dc);
            _gpu.CopyFromDevice(dc, c);
            return c[0];
        }

    }
}
