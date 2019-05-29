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
    public class SPARSE_CONVERSION : ICudafyUnitTest
    {
        private const int M = 512;
        private const int N = 128;

        private double[] _hiMatrixMN;
        private double[] _hiMatrixMN2;
        private double[] _diMatrixMN;
        private double[] _diMatrixMN2;
        private double[] _hoMatrixMN;

        
        private int[] _diPerVector;
        private int[] _diPerVector2;
        private int[] _diCSRRows;
        private int[] _diCSRCols;
        private int[] _diCSCRows;
        private int[] _diCSCCols;
        private int[] _diCOORows;
        private double[] _diVals;
        private double[] _diVals2;
        private int[] _hoPerVector;
        private int[] _hoPerVector2;
        private int[] _hoCSRRows;
        private int[] _hoCSRRows2;
        private int[] _hoCSRCols;
        private int[] _hoCSCRows;
        private int[] _hoCSCCols;
        private int[] _hoCOORows;
        private int[] _hoCSCRows_r;
        private int[] _hoCSCCols_r;
        private double[] _hoVals;
        private double[] _hoVals2;
        private double[] _hoVals2_r;

        private int[] _hoCSRRowsCPU;
        private int[] _hoCSRColsCPU;
        private double[] _hoValsCPU;

        private GPGPU _gpu;

        private GPGPUSPARSE _sparse;

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.CreateDevice(CudafyModes.Target);
            _sparse = GPGPUSPARSE.Create(_gpu);

            _hiMatrixMN = new double[M * N];
            _hiMatrixMN2 = new double[M * N];
            _hoMatrixMN = new double[M * N];
            _hoPerVector = new int[M];
            _hoPerVector2 = new int[N];

            _diPerVector2 = _gpu.Allocate(_hoPerVector2);
            _diMatrixMN = _gpu.Allocate(_hiMatrixMN);
            _diMatrixMN2 = _gpu.Allocate(_hiMatrixMN2);
            _diPerVector = _gpu.Allocate(_hoPerVector);

        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _sparse.Dispose();

            
            _gpu.Free(_diMatrixMN);
            _gpu.Free(_diMatrixMN2);
            _gpu.Free(_diPerVector);
            _gpu.Free(_diPerVector2);
        }

        private void FillBuffer(double[] buffer)
        {
            Random rand = new Random(Environment.TickCount);

            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = rand.Next(32);
            }

            System.Threading.Thread.Sleep(rand.Next(50));
        }

        private void CreateDenseMatrixCSR(double[] buffer, int m, int n, out int[] nnzPerRow, out int nnz)
        {
            nnz = 0;
            nnzPerRow = new int[m];

            Random rand = new Random(Environment.TickCount);

            for (int i = 0; i < m; i++)
            {
                int nnzrow = 0;

                for (int j = 0; j < n; j++)
                {
                    double value = rand.Next(32);
                    buffer[_sparse.GetIndexColumnMajor(i, j, M)] = value;

                    if (value != 0)
                    {
                        nnzrow++;
                        nnz++;
                    }
                }

                nnzPerRow[i] = nnzrow;
            }

            System.Threading.Thread.Sleep(rand.Next(50));
        }

        private void CreateDenseMatrixCSC(double[] buffer, int m, int n, out int[] nnzPerCol, out int nnz)
        {
            nnz = 0;
            nnzPerCol = new int[m];

            Random rand = new Random(Environment.TickCount);

            for (int j = 0; j < n; j++)
            {
                int nnzcol = 0;
                for (int i = 0; i < m; i++)
                {
                    double value = rand.Next(32);
                    buffer[_sparse.GetIndexColumnMajor(i, j, M)] = value;

                    if (value != 0)
                    {
                        nnzcol++;
                        nnz++;
                    }
                }

                nnzPerCol[j] = nnzcol;
            }

            System.Threading.Thread.Sleep(rand.Next(50));
        }

        private void CPUDense2COO(double[] buffer, int m, int n, int nnz, out int[] rows, out int[] cols, out double[] vals)
        {
            vals = new double[nnz];
            cols = new int[nnz];
            rows = new int[nnz];

            int nnzCount = 0;

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double val = buffer[_sparse.GetIndexColumnMajor(i, j, m)];

                    if (val != 0)
                    {
                        vals[nnzCount] = val;
                        rows[nnzCount] = i;
                        cols[nnzCount] = j;

                        nnzCount++;
                    }
                }
            }
        }

        private void CPUDense2CSR(double[] buffer, int m, int n, int nnz, out int[] rows, out int[] cols, out double[] vals)
        {
            rows = new int[m + 1];
            cols = new int[nnz];
            vals = new double[nnz];

            int nnzCount = 0;

            for (int i = 0; i < m; i++)
            {
                bool flagFirst = true;

                for (int j = 0; j < n; j++)
                {
                    double val = buffer[_sparse.GetIndexColumnMajor(i, j, m)];

                    if (val != 0)
                    {
                        vals[nnzCount] = val;
                        cols[nnzCount] = j;

                        if (flagFirst == true)
                        {
                            rows[i] = nnzCount;
                            flagFirst = false;
                        }

                        nnzCount++;
                    }
                }
            }

            rows[m] = nnz + rows[0];
        }

        private void CPUDense2CSC(double[] buffer, int m, int n, int nnz, out int[] rows, out int[] cols, out double[] vals)
        {
            rows = new int[nnz];
            cols = new int[n + 1];
            vals = new double[nnz];

            int nnzCount = 0;

            for (int j = 0; j < n; j++)
            {
                bool flagFirst = true;
                for (int i = 0; i < m; i++)
                {
                    double val = buffer[_sparse.GetIndexColumnMajor(i, j, m)];

                    if (val != 0)
                    {
                        vals[nnzCount] = val;
                        rows[nnzCount] = i;

                        if (flagFirst == true)
                        {
                            cols[j] = nnzCount;
                            flagFirst = false;
                        }

                        nnzCount++;
                    }
                }
            }

            cols[n] = nnz + cols[0];
        }

        [Test]
        public void TestNNZ()
        {
            int[] cpuVector;
            int cpuNNZ;
            CreateDenseMatrixCSR(_hiMatrixMN, M, N, out cpuVector, out cpuNNZ);

            _gpu.CopyToDevice(_hiMatrixMN, _diMatrixMN);

            int nnz = _sparse.NNZ(M, N, _diMatrixMN, _diPerVector);

            _gpu.CopyFromDevice(_diPerVector, _hoPerVector);

            Assert.AreEqual(cpuNNZ, nnz);

            for (int i = 0; i < cpuVector.Length; i++)
            {
                Assert.AreEqual(cpuVector[i], _hoPerVector[i]);
            }
        }

        [Test]
        public void TestDENSE2CSR()
        {
            int[] cpunnzPerRow;
            int cpuNNZ;

            CreateDenseMatrixCSR(_hiMatrixMN, M, N, out cpunnzPerRow, out cpuNNZ);
            CPUDense2CSR(_hiMatrixMN, M, N, cpuNNZ, out _hoCSRRowsCPU, out _hoCSRColsCPU, out _hoValsCPU);

            _gpu.CopyToDevice(_hiMatrixMN, _diMatrixMN);

            int nnz = _sparse.NNZ(M, N, _diMatrixMN, _diPerVector);

            _hoVals = new double[nnz];
            _hoCSRRows = new int[M + 1];
            _hoCSRCols = new int[nnz];
            
            _diVals = _gpu.Allocate(_hoVals);
            _diCSRRows = _gpu.Allocate(_hoCSRRows);
            _diCSRCols = _gpu.Allocate(_hoCSRCols);

            _sparse.Dense2CSR(M, N, _diMatrixMN, _diPerVector, _diVals, _diCSRRows, _diCSRCols);

            _gpu.CopyFromDevice(_diVals, _hoVals);
            _gpu.CopyFromDevice(_diCSRRows, _hoCSRRows);
            _gpu.CopyFromDevice(_diCSRCols, _hoCSRCols);

            _gpu.Free(_diVals);
            _gpu.Free(_diCSRRows);
            _gpu.Free(_diCSRCols);

            for (int i = 0; i < M + 1; i++)
            {
                Assert.AreEqual(_hoCSRRowsCPU[i], _hoCSRRows[i]);
            }

            for (int i = 0; i < nnz; i++)
            {
                Assert.AreEqual(_hoValsCPU[i], _hoVals[i]);
                Assert.AreEqual(_hoCSRColsCPU[i], _hoCSRCols[i]);
            }
            
        }

        [Test]
        public void TestCSR2DENSE()
        {
            FillBuffer(_hiMatrixMN);

            _gpu.CopyToDevice(_hiMatrixMN, _diMatrixMN);

            int nnz = _sparse.NNZ(M, N, _diMatrixMN, _diPerVector);

            _hoVals = new double[nnz];
            _hoCSRRows = new int[M + 1];
            _hoCSRCols = new int[nnz];

            _diVals = _gpu.Allocate(_hoVals);
            _diCSRRows = _gpu.Allocate(_hoCSRRows);
            _diCSRCols = _gpu.Allocate(_hoCSRCols);

            _sparse.Dense2CSR(M, N, _diMatrixMN, _diPerVector, _diVals, _diCSRRows, _diCSRCols);

            _sparse.CSR2Dense(M, N, _diVals, _diCSRRows, _diCSRCols, _diMatrixMN2);

            _gpu.CopyFromDevice(_diVals, _hoVals);
            _gpu.CopyFromDevice(_diCSRRows, _hoCSRRows);
            _gpu.CopyFromDevice(_diCSRCols, _hoCSRCols);
            _gpu.CopyFromDevice(_diMatrixMN2, _hoMatrixMN);

            _gpu.Free(_diVals);
            _gpu.Free(_diCSRRows);
            _gpu.Free(_diCSRCols);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    Assert.AreEqual(_hiMatrixMN[_sparse.GetIndexColumnMajor(i, j, M)], _hoMatrixMN[_sparse.GetIndexColumnMajor(i, j, M)]);
                }
            }
        }

        [Test]
        public void TestDENSE2CSC()
        {
            int[] cpuVector;
            int cpuNNZ;

            CreateDenseMatrixCSC(_hiMatrixMN, M, N, out cpuVector, out cpuNNZ);
            CPUDense2CSC(_hiMatrixMN, M, N, cpuNNZ, out _hoCSRRowsCPU, out _hoCSRColsCPU, out _hoValsCPU);

            _gpu.CopyToDevice(_hiMatrixMN, _diMatrixMN);

            int nnz = _sparse.NNZ(M, N, _diMatrixMN, _diPerVector, cusparseDirection.Column);

            _hoVals = new double[nnz];
            _hoCSRRows = new int[nnz];
            _hoCSRCols = new int[N + 1];

            _diVals = _gpu.Allocate(_hoVals);
            _diCSRRows = _gpu.Allocate(_hoCSRRows);
            _diCSRCols = _gpu.Allocate(_hoCSRCols);

            _sparse.Dense2CSC(M, N, _diMatrixMN, _diPerVector, _diVals, _diCSRRows, _diCSRCols);

            _gpu.CopyFromDevice(_diVals, _hoVals);
            _gpu.CopyFromDevice(_diCSRRows, _hoCSRRows);
            _gpu.CopyFromDevice(_diCSRCols, _hoCSRCols);

            _gpu.Free(_diVals);
            _gpu.Free(_diCSRRows);
            _gpu.Free(_diCSRCols);

            for (int i = 0; i < N + 1; i++)
            {
                Assert.AreEqual(_hoCSRColsCPU[i], _hoCSRCols[i]);
            }

            for (int i = 0; i < nnz; i++)
            {
                Assert.AreEqual(_hoValsCPU[i], _hoVals[i]);
                Assert.AreEqual(_hoCSRRowsCPU[i], _hoCSRRows[i]);
            }
        }

        [Test]
        public void TestCSC2DENSE()
        {
            FillBuffer(_hiMatrixMN);

            _gpu.CopyToDevice(_hiMatrixMN, _diMatrixMN);

            int nnz = _sparse.NNZ(M, N, _diMatrixMN, _diPerVector, cusparseDirection.Column);

            _hoVals = new double[nnz];
            _hoCSRRows = new int[nnz];
            _hoCSRCols = new int[N + 1];

            _diVals = _gpu.Allocate(_hoVals);
            _diCSRRows = _gpu.Allocate(_hoCSRRows);
            _diCSRCols = _gpu.Allocate(_hoCSRCols);

            _sparse.Dense2CSC(M, N, _diMatrixMN, _diPerVector, _diVals, _diCSRRows, _diCSRCols);

            _sparse.CSC2Dense(M, N, _diVals, _diCSRRows, _diCSRCols, _diMatrixMN2);

            _gpu.CopyFromDevice(_diVals, _hoVals);
            _gpu.CopyFromDevice(_diCSRRows, _hoCSRRows);
            _gpu.CopyFromDevice(_diCSRCols, _hoCSRCols);
            _gpu.CopyFromDevice(_diMatrixMN2, _hoMatrixMN);

            _gpu.Free(_diVals);
            _gpu.Free(_diCSRRows);
            _gpu.Free(_diCSRCols);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    Assert.AreEqual(_hiMatrixMN[_sparse.GetIndexColumnMajor(i, j, M)], _hoMatrixMN[_sparse.GetIndexColumnMajor(i, j, M)]);
                }
            }
        }

        [Test]
        public void TestCSR2CSC()
        {
            FillBuffer(_hiMatrixMN);

            _gpu.CopyToDevice(_hiMatrixMN, _diMatrixMN);

            int nnz = _sparse.NNZ(M, N, _diMatrixMN, _diPerVector); // For CSR
            int nnz2 = _sparse.NNZ(M, N, _diMatrixMN, _diPerVector2, cusparseDirection.Column); // For CSC

            _hoVals = new double[nnz];
            _hoCSRRows = new int[M + 1];
            _hoCSRCols = new int[nnz];

            _diVals = _gpu.Allocate(_hoVals);
            _diCSRRows = _gpu.Allocate(_hoCSRRows);
            _diCSRCols = _gpu.Allocate(_hoCSRCols);

            _hoVals2 = new double[nnz2];
            _hoCSCRows = new int[nnz2];
            _hoCSCCols = new int[N + 1];
            _hoVals2_r = new double[nnz2];
            _hoCSCRows_r = new int[nnz2];
            _hoCSCCols_r = new int[N + 1];

            _diVals2 = _gpu.Allocate(_hoVals2);
            _diCSCRows = _gpu.Allocate(_hoCSCRows);
            _diCSCCols = _gpu.Allocate(_hoCSCCols);

            // Get CSC
            _sparse.Dense2CSC(M, N, _diMatrixMN, _diPerVector2, _diVals2, _diCSCRows, _diCSCCols);

            _gpu.CopyFromDevice(_diVals2, _hoVals2);
            _gpu.CopyFromDevice(_diCSCRows, _hoCSCRows);
            _gpu.CopyFromDevice(_diCSCCols, _hoCSCCols);

            // Get CSR and convert to CSC
            _sparse.Dense2CSR(M, N, _diMatrixMN, _diPerVector, _diVals, _diCSRRows, _diCSRCols);

            _sparse.CSR2CSC(M, N, nnz, _diVals, _diCSRRows, _diCSRCols, _diVals2, _diCSCRows, _diCSCCols);

            _gpu.CopyFromDevice(_diVals2, _hoVals2_r);
            _gpu.CopyFromDevice(_diCSCRows, _hoCSCRows_r);
            _gpu.CopyFromDevice(_diCSCCols, _hoCSCCols_r);

            _gpu.Free(_diVals);
            _gpu.Free(_diVals2);
            _gpu.Free(_diCSRRows);
            _gpu.Free(_diCSRCols);
            _gpu.Free(_diCSCRows);
            _gpu.Free(_diCSCCols);

            for (int i = 0; i < nnz2; i++)
            {
                Assert.AreEqual(_hoVals2[i], _hoVals2_r[i]);
                Assert.AreEqual(_hoCSCRows[i], _hoCSCRows_r[i]);
            }

            for (int i = 0; i < N + 1; i++)
            {
                Assert.AreEqual(_hoCSCCols[i], _hoCSCCols_r[i]);
            }
        }

        [Test]
        public void TestCOO2CSRAndCSR2COO()
        {
            FillBuffer(_hiMatrixMN);

            _gpu.CopyToDevice(_hiMatrixMN, _diMatrixMN);

            int nnz = _sparse.NNZ(M, N, _diMatrixMN, _diPerVector);

            int[] cooRowsCpu;
            int[] cooColsCpu;
            double[] cooValsCpu;

            CPUDense2COO(_hiMatrixMN, M, N, nnz, out cooRowsCpu, out cooColsCpu, out cooValsCpu);

            _hoVals = new double[nnz];
            _hoCSRRows = new int[M + 1];
            _hoCSRRows2 = new int[M + 1];
            _hoCSRCols = new int[nnz];
            _hoCOORows = new int[nnz];

            _diVals = _gpu.Allocate(_hoVals);
            _diCSRRows = _gpu.Allocate(_hoCSRRows);
            _diCSRCols = _gpu.Allocate(_hoCSRCols);
            _diCOORows = _gpu.Allocate(_hoCOORows);

            _sparse.Dense2CSR(M, N, _diMatrixMN, _diPerVector, _diVals, _diCSRRows, _diCSRCols);

            _gpu.CopyFromDevice(_diCSRRows, _hoCSRRows2);

            _sparse.CSR2COO(nnz, M, _diCSRRows, _diCOORows);

            _gpu.CopyFromDevice(_diCOORows, _hoCOORows);

            for (int i = 0; i < nnz; i++)
            {
                Assert.AreEqual(cooRowsCpu[i], _hoCOORows[i]);
            }

            _gpu.CopyToDevice(cooRowsCpu, _diCOORows);

            _sparse.COO2CSR(nnz, M, _diCOORows, _diCSRRows);

            _gpu.CopyFromDevice(_diCSRRows, _hoCSRRows);

            for (int i = 0; i < M + 1; i++)
            {
                Assert.AreEqual(_hoCSRRows2[i], _hoCSRRows[i]);
            }

            _gpu.Free(_diVals);
            _gpu.Free(_diCSRRows);
            _gpu.Free(_diCSRCols);
            _gpu.Free(_diCOORows);
        }

        public void TestSetUp()
        {

        }

        public void TestTearDown()
        {

        }
    }
}
