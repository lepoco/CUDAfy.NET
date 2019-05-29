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
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using Cudafy.Host;
using Cudafy.UnitTests;
using NUnit.Framework;
using Cudafy.Translator;
using Cudafy.Compilers;
namespace Cudafy.Host.UnitTests
{
   
    [TestFixture]
    public class StringTests : CudafyUnitTest, ICudafyUnitTest
    {
        private CudafyModule _cm;

        private GPGPU _gpu;

        private const int N = 1024;

        [TestFixtureSetUp]
        public void SetUp()
        {
            CudafyTranslator.GenerateDebug = true;
            _cm = CudafyModule.TryDeserialize();
            _gpu = CudafyHost.GetDevice(CudafyModes.Architecture, CudafyModes.DeviceId);
            if (_cm == null || !_cm.TryVerifyChecksums())
            {
                _cm = CudafyTranslator.Cudafy(_gpu.GetArchitecture(), this.GetType(), (_gpu is OpenCLDevice) ? null : typeof(StringConstClass));
                _cm.TrySerialize();
            }

            _gpu.LoadModule(_cm);
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _gpu.FreeAll();
        }

        [Test]
        public void TestTransferUnicodeChar()
        {
            char a = '€';
            char c;
            char[] dev_c = _gpu.Allocate<char>();

            _gpu.Launch(1, 1, "TransferUnicodeChar", a, dev_c);
            _gpu.CopyFromDevice(dev_c, out c);
            _gpu.FreeAll();
            Assert.AreEqual(a, c);
            Debug.WriteLine(c);

        }

        [Cudafy]
        public static void TransferUnicodeChar(char a, char[] c)
        {
            c[0] = a;
            //Console.WriteLine("hello from your gpu!");
            //Debug.Assert(c[0] == a);
            //Debug.Assert(c[0] == a, null, "%d == %d is not true!", c[0], a);
        }

        [Test]
        public void TestTransferUnicodeCharArray()
        {
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
            for (int i = 0; i < a.Length; i++)
                c[i] = a[i];
        }

        [Test]
        public void TestTransferASCIIArray()
        {
            string a = "I believe it costs 155,95 in Duesseldorf";
            byte[] bytes = Encoding.ASCII.GetBytes(a);
            byte[] dev_a = _gpu.CopyToDevice(bytes);
            byte[] dev_c = _gpu.Allocate(bytes);
            byte[] host_c = new byte[a.Length];
            _gpu.Launch(1, 1, "TransferASCIIArray", dev_a, dev_c);
            _gpu.CopyFromDevice(dev_c, host_c);
            string c = Encoding.ASCII.GetString(host_c);
            _gpu.FreeAll();
            Assert.AreEqual(a, c);
            Debug.WriteLine(c);
        }

        [Cudafy]
        public static void TransferASCIIArray(byte[] a, byte[] c)
        {
            for (int i = 0; i < a.Length; i++)
                c[i] = a[i];
        }


        [Test]
        public void TestWriteHelloOnGPU()
        {
            string a = "€ello\r\nyou";
            char[] dev_c = _gpu.Allocate<char>(a.Length);
            char[] host_c = new char[a.Length];

            _gpu.Launch(1, 1, "WriteHelloOnGPU", dev_c);
            _gpu.CopyFromDevice(dev_c, host_c);
            string c = new string(host_c);
            _gpu.FreeAll();
            Assert.AreEqual(a, c);
            Debug.WriteLine(c);
        }

        [Cudafy]
        public static void WriteHelloOnGPU(char[] c)
        {
            c[0] = '€';
            c[1] = 'e';
            c[2] = 'l';
            c[3] = 'l';
            c[4] = 'o';
            c[5] = '\r';
            c[6] = '\n';
            c[7] = 'y';
            c[8] = 'o';
            c[9] = 'u';
        }

        [Test]
        public void TestStringSearchv1()
        {
            TestStringSearch(1);
        }

        [Test]
        public void TestStringSearchv2()
        {
            TestStringSearch(2);
        }

        public void TestStringSearch(int version)
        {
            string string2Search = "I believe it costs €155,95 in Düsseldorf";
            char[] string2Search_dev = _gpu.CopyToDevice(string2Search);

            char char2Find = '€';

            int pos = -1;
            int[] pos_dev = _gpu.Allocate<int>();

            _gpu.Launch(1, 1, "StringSearchv" + version.ToString(), string2Search_dev, char2Find, pos_dev);
            _gpu.CopyFromDevice(pos_dev, out pos);
            _gpu.FreeAll();
            Assert.Greater(pos, 0);
            Assert.AreEqual(string2Search.IndexOf(char2Find), pos);
            Debug.WriteLine(pos);
        }

        [Cudafy]
        public static void StringSearchv1(char[] text, char match, int[] pos)
        {
            pos[0] = -1;
            for (int i = 0; i < text.Length; i++)
            {
                if (text[i] == match)
                {
                    pos[0] = i;
                    break;
                }
            }
        }

        [Cudafy]
        public static void StringSearchv2(char[] text, char match, int[] pos)
        {
            pos[0] = GetPos(text, match);
        }

        [Cudafy]
        public static int GetPos(char[] text, char match)
        {
            for (int i = 0; i < text.Length; i++)
            {
                if (text[i] == match)
                {
                    return i;
                }
            }
            return -1;
        }

        [Test]
        public void TestStaticString()
        {
            if (_gpu is OpenCLDevice)
            {
                Console.WriteLine("Device not supporting const string, so skip.");
                return;
            }
            char[] dev_ca = _gpu.Allocate<char>(StringConstClass.constString.Length);
            char[] host_ca = new char[StringConstClass.constString.Length];
            char[] dev_cb = _gpu.Allocate<char>(StringConstClass.constString.Length);
            char[] host_cb = new char[StringConstClass.constString.Length];
            char[] dev_cc = _gpu.Allocate<char>(StringConstClass.constString.Length);
            char[] host_cc = new char[StringConstClass.constString.Length];
            _gpu.Launch(1, 1, "StringConst", dev_ca, dev_cb, dev_cc);
            _gpu.CopyFromDevice(dev_ca, host_ca);
            _gpu.CopyFromDevice(dev_cb, host_cb);
            _gpu.CopyFromDevice(dev_cc, host_cc);
            string ca = new string(host_ca);
            _gpu.FreeAll();
            Assert.AreEqual(StringConstClass.constString, ca, "ca");
            Debug.WriteLine(ca);
            string cb = new string(host_cb);
            _gpu.FreeAll();
            Assert.AreEqual(StringConstClass.constString, cb, "cb");
            Debug.WriteLine(cb);
            string cc = new string(host_cc);
            _gpu.FreeAll();
            Assert.AreEqual(StringConstClass.constString, cc, "cc");
            Debug.WriteLine(cc);
        }





        public void TestSetUp()
        {
        
        }

        public void TestTearDown()
        {
          
        }
    }

    public class StringConstClass
    {
        public const string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        public const string constString = "悪霊退悪霊退散怨霊abcdefghijklmn";
        [Cudafy]
        public static void StringConst(char[] ca, char[] cb, char[] cc)
        {
            int x = 0;
            foreach (char c in constString)
                ca[x++] = c;

            string myString = constString;
            for (int i = 0; i < myString.Length; i++)
                cb[i] = myString[i];

            int len = PassString(constString);
            for (int i = 0; i < len; i++)
                cc[i] = myString[i];
        }

        [Cudafy]
        public static int PassString(string text)
        {
            return text.Length;
        }
    }
}
