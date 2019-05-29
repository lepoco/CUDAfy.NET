/* Added by Kichang Kim (kkc0923@hotmail.com) */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Reflection;
using Cudafy.Types;
using Cudafy.Host;
using Cudafy.Maths.SPARSE;
using Cudafy.UnitTests;
using NUnit.Framework;

namespace Cudafy.Maths.UnitTests
{
    [TestFixture]
    public class SPARSE1 : ICudafyUnitTest
    {
        
        private GPGPU _gpu;

        private GPGPUSPARSE _sparse;

        private const int N = 1024;
        private const int NNZRatio = 10; // NNZRatio % of values are non-zero.

        private float[] _hiVectorX;
        private float[] _hiValsX;
        private float[] _hiVectorY;
        private int[] _hiIndicesX;

        private float[] _hoValsX;
        private float[] _hoVectorY;

        private float[] _diValsX;
        private int[] _diIndicesX;
        private float[] _diVectorY;

        private float alpha = 4.0f;
        private float C = 3.0f;
        private float S = 4.0f;
        private int NNZ;

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.GetDevice();
            _sparse = GPGPUSPARSE.Create(_gpu);
                        
            _hiVectorX = new float[N];
            _hiVectorY = new float[N];
            _hoVectorY = new float[N];

            FillBufferSparse(_hiVectorX, out NNZ);
            FillBuffer(_hiVectorY);

            _hiIndicesX = new int[NNZ];
            _hoValsX = new float[NNZ];
            _hiValsX = new float[NNZ];

            GetSparseIndex(_hiVectorX, _hiValsX, _hiIndicesX);

            _diValsX = _gpu.Allocate(_hiValsX);
            _diIndicesX = _gpu.Allocate(_hiIndicesX);
            _diVectorY = _gpu.Allocate(_hiVectorY);

        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _sparse.Dispose();

            _gpu.FreeAll();
        }

        public void TestSetUp()
        {

        }

        public void TestTearDown()
        {

        }

        private void FillBuffer(float[] buffer)
        {
            Random rand = new Random(Environment.TickCount);

            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = (float)rand.Next(512);
            }

            System.Threading.Thread.Sleep(rand.Next(50));
        }

        private void FillBufferSparse(float[] buffer, out int nnz)
        {
            Random rand = new Random(Environment.TickCount);
            nnz = 0;

            for (int i = 0; i < buffer.Length; i++)
            {
                if (rand.Next(100) < NNZRatio)
                {
                    buffer[i] = (float)rand.Next(512);
                    nnz++;
                }
                else
                {
                    buffer[i] = 0;
                }
            }

            System.Threading.Thread.Sleep(rand.Next(50));
        }

        private void GetSparseIndex(float[] buffer, float[] vals, int[] indices)
        {
            int nnzCount = 0;

            for (int i = 0; i < buffer.Length; i++)
            {
                if (buffer[i] != 0)
                {
                    indices[nnzCount] = i;
                    vals[nnzCount] = buffer[i];
                    nnzCount++;
                }
            }
        }

        [Test]
        public void TestSparseVersion()
        {
            Console.WriteLine(_sparse.GetVersionInfo());
        }

        [Test]
        public void Test_SPARSE1_AXPY()
        {
            _gpu.CopyToDevice(_hiValsX, _diValsX);
            _gpu.CopyToDevice(_hiIndicesX, _diIndicesX);
            _gpu.CopyToDevice(_hiVectorY, _diVectorY);

            _sparse.AXPY(ref alpha, _diValsX, _diIndicesX, _diVectorY);

            _gpu.CopyFromDevice(_diVectorY, _hoVectorY);

            for (int i = 0; i < N; i++)
            {
                float cpuResult = alpha * _hiVectorX[i] + _hiVectorY[i];
                Assert.AreEqual(cpuResult, _hoVectorY[i]);
            }
        }

        [Test]
        public void Test_SPARSE1_DOT()
        {
            _gpu.CopyToDevice(_hiValsX, _diValsX);
            _gpu.CopyToDevice(_hiIndicesX, _diIndicesX);
            _gpu.CopyToDevice(_hiVectorY, _diVectorY);

            float gpuResult = _sparse.DOT(_diValsX, _diIndicesX, _diVectorY);

            float cpuResult = 0.0f;
            
            for (int i = 0; i < N; i++)
            {
                cpuResult += _hiVectorX[i] * _hiVectorY[i];
            }

            Assert.AreEqual(cpuResult, gpuResult);
        }

        [Test]
        public void Test_SPARSE1_GTHR()
        {
            _gpu.CopyToDevice(_hiValsX, _diValsX);
            _gpu.CopyToDevice(_hiIndicesX, _diIndicesX);
            _gpu.CopyToDevice(_hiVectorY, _diVectorY);

            _sparse.GTHR(_diVectorY, _diValsX, _diIndicesX);

            _gpu.CopyFromDevice(_diValsX, _hoValsX);

            for (int i = 0; i < NNZ; i++)
            {
                Assert.AreEqual(_hiVectorY[_hiIndicesX[i]], _hoValsX[i]);
            }
        }

        [Test]
        public void Test_SPARSE1_GTHRZ()
        {
            _gpu.CopyToDevice(_hiValsX, _diValsX);
            _gpu.CopyToDevice(_hiIndicesX, _diIndicesX);
            _gpu.CopyToDevice(_hiVectorY, _diVectorY);

            _sparse.GTHRZ(_diVectorY, _diValsX, _diIndicesX);

            _gpu.CopyFromDevice(_diValsX, _hoValsX);
            _gpu.CopyFromDevice(_diVectorY, _hoVectorY);

            for (int i = 0; i < NNZ; i++)
            {
                Assert.AreEqual(_hiVectorY[_hiIndicesX[i]], _hoValsX[i]);
                Assert.AreEqual(0.0f, _hoVectorY[_hiIndicesX[i]]);
            }
        }

       // [Test]
        public void Test_SPARSE1_ROT()
        {
            _gpu.CopyToDevice(_hiValsX, _diValsX);
            _gpu.CopyToDevice(_hiIndicesX, _diIndicesX);
            _gpu.CopyToDevice(_hiVectorY, _diVectorY);

            _sparse.ROT(_diValsX, _diIndicesX, _diVectorY, ref C, ref  S);

            _gpu.CopyFromDevice(_diValsX, _hoValsX);
            _gpu.CopyFromDevice(_diVectorY, _hoVectorY);

            for (int i = 0; i < NNZ; i++)
            {
                float cpuY = C * _hiVectorY[_hiIndicesX[i]] - S * _hiValsX[i];
                float cpuX = C * _hiValsX[i] + S * _hiVectorY[_hiIndicesX[i]];

                Assert.AreEqual(cpuY, _hoVectorY[_hiIndicesX[i]]);
                Assert.AreEqual(cpuX, _hoValsX[i]);
            }
        }

        [Test]
        public void Test_SPARSE1_SCTR()
        {
            _gpu.CopyToDevice(_hiValsX, _diValsX);
            _gpu.CopyToDevice(_hiIndicesX, _diIndicesX);
            _gpu.CopyToDevice(_hiVectorY, _diVectorY);

            _sparse.SCTR(_diValsX, _diIndicesX, _diVectorY);

            _gpu.CopyFromDevice(_diVectorY, _hoVectorY);

            for (int i = 0; i < NNZ; i++)
            {
                Assert.AreEqual(_hiValsX[i], _hoVectorY[_hiIndicesX[i]]);
            }
        }
    }
}
