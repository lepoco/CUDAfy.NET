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

using Cudafy.Maths.FFT;
using Cudafy.UnitTests;
using NUnit.Framework;
namespace Cudafy.Maths.UnitTests
{
    [TestFixture]
    public class FFTDoubleTests : ICudafyUnitTest
    {

        private double[] _hostInput;

        private ComplexD[] _hostInputCplx;

        private double[] _hostOutput;

        private ComplexD[] _hostOutputCplx;

        private double[] _devInput;

        private ComplexD[] _devInputCplx;

        private ComplexD[] _devInterCplx;

        private double[] _devInter;

        private double[] _devOutput;

        private ComplexD[] _devOutputCplx;

        private const int N = 64 * 64 * 64;

        private const int W = 512;

        private const int H = 512;

        private const int NX = 64;
        private const int NY = 64;
        private const int NZ = 64;

        private const int BATCH = 16;

        private GPGPU _gpu;

        private GPGPUFFT _fft;

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.CreateDevice(CudafyModes.Target);
            
            _hostInput = new double[N * BATCH];
            _hostInputCplx = new ComplexD[N * BATCH];
            _hostOutput = new double[N * BATCH];
            _hostOutputCplx = new ComplexD[N * BATCH];
            _devInput = _gpu.Allocate(_hostInput);
            _devInputCplx = _gpu.Allocate(_hostInputCplx);
            _devInter = _gpu.Allocate<double>(N * 2 * BATCH);
            _devInterCplx = _gpu.Allocate<ComplexD>(N * BATCH);
            _devOutput = _gpu.Allocate(_hostOutput);
            _devOutputCplx = _gpu.Allocate(_hostOutputCplx);
            _fft = GPGPUFFT.Create(_gpu);
            for (int b = 0; b < BATCH; b++)
            {
                for (int i = 0; i < N; i++)
                {
                    ComplexD cf = new ComplexD();
                    cf.x = (double)((10.0F * Math.Sin(100 * 2 * Math.PI * i / N * Math.PI / 180)));
                    cf.y = (double)((10.0F * Math.Sin(200 * 2 * Math.PI * i / N * Math.PI / 180)));
                    _hostInput[i + b * N] = cf.x;
                    _hostInputCplx[i + b * N] = cf;
                }
            }
        }

        
        private void ClearBuffers()
        {
            Array.Clear(_hostOutput, 0, _hostOutput.Length);
            Array.Clear(_hostOutputCplx, 0, _hostOutputCplx.Length);
            _gpu.Set(_devInput);
            _gpu.Set(_devInputCplx);
            _gpu.Set(_devInter);
            _gpu.Set(_devInterCplx);
            _gpu.Set(_devOutput);
            _gpu.Set(_devOutputCplx);
        }

        [Test]
        public void TestReal2Complex2Real1D()
        {
            _gpu.CopyToDevice(_hostInput, _devInput);
            FFTPlan1D planFwd = _fft.Plan1D(eFFTType.Real2Complex, eDataType.Double, N, BATCH);
            planFwd.Execute(_devInput, _devInterCplx);
            FFTPlan1D planRev = _fft.Plan1D(eFFTType.Complex2Real, eDataType.Double, N, BATCH);
            planRev.Execute(_devInterCplx, _devOutput);
            _gpu.CopyFromDevice(_devOutput, _hostOutput);
            for (int i = 0; i < N * BATCH; i++)
                Assert.AreEqual(_hostInput[i], _hostOutput[i] / (double)N, 0.0001, "Index {0}", i);
        }

        [Test]
        public void TestComplex2Complex2Complex1D()
        {
            _gpu.CopyToDevice(_hostInputCplx, _devInputCplx);
            FFTPlan1D planFwd = _fft.Plan1D(eFFTType.Complex2Complex, eDataType.Double, N, BATCH);
            planFwd.Execute(_devInputCplx, _devInterCplx);
            FFTPlan1D planRev = _fft.Plan1D(eFFTType.Complex2Complex, eDataType.Double, N, BATCH);
            planRev.Execute(_devInterCplx, _devOutputCplx, true);
            _gpu.CopyFromDevice(_devOutputCplx, _hostOutputCplx);
            for (int i = 0; i < N * BATCH; i++)
            {
                Assert.AreEqual(_hostInputCplx[i].x, _hostOutputCplx[i].x / (double)N, 0.0001, "Index {0} (x)", i);
                Assert.AreEqual(_hostInputCplx[i].y, _hostOutputCplx[i].y / (double)N, 0.0001, "Index {0} (y)", i);
            }
        }

        [Test]
        public void TestReal2Complex2Real2D()
        {
            int nx = (int)System.Math.Sqrt(N);
            int ny = nx;
            _gpu.CopyToDevice(_hostInput, _devInput);
            FFTPlan2D planFwd = _fft.Plan2D(eFFTType.Real2Complex, eDataType.Double, nx, ny, BATCH);
            planFwd.Execute(_devInput, _devInterCplx);
            FFTPlan2D planRev = _fft.Plan2D(eFFTType.Complex2Real, eDataType.Double, nx, ny, BATCH);
            planRev.Execute(_devInterCplx, _devOutput);
            _gpu.CopyFromDevice(_devOutput, _hostOutput);
            for (int i = 0; i < N * BATCH; i++)
                Assert.AreEqual(_hostInput[i], _hostOutput[i] / (double)N, 0.0001, "Index {0}", i);
        }

        [Test]
        public void TestComplex2Complex2Complex2D()
        {
            int nx = (int)System.Math.Sqrt(N);
            int ny = nx;
            _gpu.CopyToDevice(_hostInputCplx, _devInputCplx);
            FFTPlan2D planFwd = _fft.Plan2D(eFFTType.Complex2Complex, eDataType.Double, nx, ny, BATCH);
            planFwd.Execute(_devInputCplx, _devInterCplx);
            FFTPlan2D planRev = _fft.Plan2D(eFFTType.Complex2Complex, eDataType.Double, nx, ny, BATCH);
            planRev.Execute(_devInterCplx, _devOutputCplx, true);
            _gpu.CopyFromDevice(_devOutputCplx, _hostOutputCplx);
            for (int i = 0; i < N * BATCH; i++)
            {
                Assert.AreEqual(_hostInputCplx[i].x, _hostOutputCplx[i].x / (double)N, 0.0001, "Index {0} (x)", i);
                Assert.AreEqual(_hostInputCplx[i].y, _hostOutputCplx[i].y / (double)N, 0.0001, "Index {0} (y)", i);
            }
        }

        [Test]
        public void TestReal2Complex2Real3D()
        {
            _gpu.CopyToDevice(_hostInput, _devInput);
            FFTPlan3D planFwd = _fft.Plan3D(eFFTType.Real2Complex, eDataType.Double, NX, NY, NZ, BATCH);
            planFwd.Execute(_devInput, _devInterCplx);
            FFTPlan3D planRev = _fft.Plan3D(eFFTType.Complex2Real, eDataType.Double, NX, NY, NZ, BATCH);
            planRev.Execute(_devInterCplx, _devOutput);
            _gpu.CopyFromDevice(_devOutput, _hostOutput);
            for (int i = 0; i < N * BATCH; i++)
                Assert.AreEqual(_hostInput[i], _hostOutput[i] / (double)N, 1, "Index {0}", i);
        }

        [Test]
        public void TestComplex2Complex2Complex3D()
        {
            _gpu.CopyToDevice(_hostInputCplx, _devInputCplx);

            FFTPlan3D planFwd = _fft.Plan3D(eFFTType.Complex2Complex, eDataType.Double, NX, NY, NZ, BATCH);
            planFwd.Execute(_devInputCplx, _devInterCplx);
            FFTPlan3D planRev = _fft.Plan3D(eFFTType.Complex2Complex, eDataType.Double, NX, NY, NZ, BATCH);
            planRev.Execute(_devInterCplx, _devOutputCplx, true);
            _gpu.CopyFromDevice(_devOutputCplx, _hostOutputCplx);
            for (int i = 0; i < N * BATCH; i++)
            {
                Assert.AreEqual(_hostInputCplx[i].x, _hostOutputCplx[i].x / (float)N, 1, "Index {0} (x)", i);
                Assert.AreEqual(_hostInputCplx[i].y, _hostOutputCplx[i].y / (float)N, 1, "Index {0} (y)", i);
            }
        }

        [Test]
        public void TestComplex2Complex2Complex3D_2()
        {
            _gpu.CopyToDevice(_hostInputCplx, _devInputCplx);

            ComplexD[, ,] devInputCplx3D = _gpu.Cast(_devInputCplx, NX, NY, NZ);
            ComplexD[, ,] devInterCplx3D = _gpu.Cast(_devInterCplx, NX, NY, NZ);
            ComplexD[, ,] devOutputCplx3D = _gpu.Cast(_devOutputCplx, NX, NY, NZ);

            FFTPlan3D planFwd = _fft.Plan3D(eFFTType.Complex2Complex, eDataType.Double, NX, NY, NZ, BATCH);
            planFwd.Execute(devInputCplx3D, devInterCplx3D);
            FFTPlan3D planRev = _fft.Plan3D(eFFTType.Complex2Complex, eDataType.Double, NX, NY, NZ, BATCH);
            planRev.Execute(devInterCplx3D, devOutputCplx3D, true);

            _devOutputCplx = _gpu.Cast(devOutputCplx3D, NX * NY * NZ * BATCH);

            _gpu.CopyFromDevice(_devOutputCplx, _hostOutputCplx);

            for (int i = 0; i < N * BATCH; i++)
            {
                Assert.AreEqual(_hostInputCplx[i].x, _hostOutputCplx[i].x / (double)N, 1, "Index {0} (x)", i);
                Assert.AreEqual(_hostInputCplx[i].y, _hostOutputCplx[i].y / (double)N, 1, "Index {0} (y)", i);
            }
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _gpu.FreeAll();
            _fft.RemoveAll();
        }

        [SetUp]
        public void TestSetUp()
        {
            ClearBuffers();
        }

        public void TestTearDown()
        {
            _fft.RemoveAll();
        }
    }
}
