using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Reflection;
using Cudafy.Types;
using Cudafy.Host;
using Cudafy.Maths.BLAS;
using Cudafy.Maths.BLAS.Types;
using Cudafy.UnitTests;
using NUnit.Framework;

namespace Cudafy.Maths.UnitTests
{
    [TestFixture]
    public class BLAS3 : ICudafyUnitTest
    {
        GPGPU _gpu;

        GPGPUBLAS _blas;

        // Constants
        const int M = 128;
        const int K = 80;
        const int N = 64;

        const double Alpha = 6.0;
        const double Beta = 4.0;

        // Range of value for test matrix and buffer.
        // All value are in 1 <= value <= RamdomMax + 1
        private const int RandomMax = 255;

        // CPU Buffers
        double[] hiMatrixAMM;
        double[] hiMatrixANN;
        double[] hiMatrixAMK;
        double[] hiMatrixAKM;
        double[] hiMatrixBKN;
        double[] hiMatrixBNK;
        double[] hiMatrixBMN;
        double[] hiMatrixBMK;
        double[] hiMatrixBKM;
        double[] hiMatrixCMN;
        double[] hiMatrixCKN;
        double[] hiMatrixCMK;
        double[] hiMatrixCMM;
        double[] gpuResultMN;
        double[] gpuResultMM;

        double[] diMatrixA;
        double[] diMatrixB;
        double[] diMatrixC;

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.GetDevice(CudafyModes.Target);
            _blas = GPGPUBLAS.Create(_gpu);

            hiMatrixAMM = new double[M * M];
            hiMatrixANN = new double[N * N];
            hiMatrixAMK = new double[M * K];
            hiMatrixAKM = new double[K * M];
            hiMatrixBMN = new double[M * N];
            hiMatrixBKN = new double[K * N];
            hiMatrixBNK = new double[N * K];
            hiMatrixBMK = new double[M * K];
            hiMatrixBKM = new double[K * M];
            hiMatrixCMN = new double[M * N];
            hiMatrixCKN = new double[K * N];
            hiMatrixCMK = new double[M * K];
            hiMatrixCMM = new double[M * M];
            gpuResultMN = new double[M * N];
            gpuResultMM = new double[M * M];
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _blas.Dispose();

            _gpu.FreeAll();

            //_gpu.Dispose();
        }

        public void TestSetUp()
        {
        }

        public void TestTearDown()
        {
        }

        #region Utility Functions
        private int GetIndexColumnMajor(int i, int j, int m)
        {
            return i + j * m;
        }

        private void PrintMatrix(double[] buffer, int m, int n)
        {
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Console.Write("{0} ", buffer[GetIndexColumnMajor(i, j, m)]);
                }
                Console.WriteLine();
            }
        }

        private void FillBuffer(double[] buffer)
        {
            Random rand = new Random(Environment.TickCount);
            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = rand.Next(RandomMax) + 1;
            }
            System.Threading.Thread.Sleep(rand.Next(RandomMax));
        }

        private void ClearBuffer(double[] buffer)
        {
            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = 0.0;
            }
        }

        private void CreateBandedMatrix(double[] buffer, int m, int n, int kl, int ku)
        {
            Random rand = new Random(Environment.TickCount);
            int largestSize = m;
            if (largestSize < n)
            {
                largestSize = n;
            }

            for (int i = 0; i < largestSize; i++)
            {
                // Set main diagonal
                int index = _blas.GetIndexColumnMajor(i, i, m);
                if (i < m && i < n)
                {
                    buffer[index] = rand.Next(RandomMax) + 1;
                }

                // Set superdiagonal
                for (int si = 1; si <= ku; si++)
                {
                    int ni = i;
                    int nj = i + si;

                    if (ni >= 0 && ni < m && nj >= 0 && nj < n)
                    {
                        index = _blas.GetIndexColumnMajor(ni, nj, m);
                        buffer[index] = rand.Next(RandomMax) + 1; ;
                    }
                }

                // Set subdiagonal
                for (int si = 1; si <= kl; si++)
                {
                    int ni = i;
                    int nj = i - si;

                    if (ni >= 0 && ni < m && nj >= 0 && nj < n)
                    {
                        index = _blas.GetIndexColumnMajor(ni, nj, m);
                        buffer[index] = rand.Next(RandomMax) + 1; ;
                    }


                }
            }
            System.Threading.Thread.Sleep(rand.Next(50));
        }

        private void ConverToSymmetric(double[] buffer, int n)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    buffer[GetIndexColumnMajor(i, j, n)] = buffer[GetIndexColumnMajor(i, j, n)];
                    buffer[GetIndexColumnMajor(j, i, n)] = buffer[GetIndexColumnMajor(i, j, n)];
                }
            }
        }

        private void CreateMainDiagonalOnlyMatrix(double[] buffer, int n)
        {
            Random rand = new Random(Environment.TickCount);
            for (int i = 0; i < n; i++)
            {
                buffer[GetIndexColumnMajor(i, i, n)] = rand.Next(RandomMax) + 1;
            }
            System.Threading.Thread.Sleep(rand.Next(RandomMax));
        }
        #endregion

        [Test]
        public void Test_BLAS3_GEMM()
        {
            // A : No transpose, B : No transpose
            ClearBuffer(hiMatrixAMK);
            ClearBuffer(hiMatrixBKN);
            ClearBuffer(hiMatrixCMN);

            FillBuffer(hiMatrixAMK);
            FillBuffer(hiMatrixBKN);
            FillBuffer(hiMatrixCMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAMK);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBKN);
            diMatrixC = _gpu.CopyToDevice(hiMatrixCMN);

            _blas.GEMM(M, K, N, Alpha, diMatrixA, diMatrixB, Beta, diMatrixC);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for(int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += Alpha * hiMatrixAMK[GetIndexColumnMajor(i, k, M)] * hiMatrixBKN[GetIndexColumnMajor(k, j, K)];
                    }

                    cpuResult += Beta * hiMatrixCMN[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // A : Transpose, B : No transpose
            ClearBuffer(hiMatrixAKM);
            ClearBuffer(hiMatrixBKN);
            ClearBuffer(hiMatrixCMN);

            FillBuffer(hiMatrixAKM);
            FillBuffer(hiMatrixBKN);
            FillBuffer(hiMatrixCMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAKM);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBKN);
            diMatrixC = _gpu.CopyToDevice(hiMatrixCMN);

            _blas.GEMM(M, K, N, Alpha, diMatrixA, diMatrixB, Beta, diMatrixC, cublasOperation.T);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += Alpha * hiMatrixAKM[GetIndexColumnMajor(k, i, K)] * hiMatrixBKN[GetIndexColumnMajor(k, j, K)];
                    }

                    cpuResult += Beta * hiMatrixCMN[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // A : No transpose, B : Transpose
            ClearBuffer(hiMatrixAMK);
            ClearBuffer(hiMatrixBNK);
            ClearBuffer(hiMatrixCMN);

            FillBuffer(hiMatrixAMK);
            FillBuffer(hiMatrixBNK);
            FillBuffer(hiMatrixCMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAMK);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBNK);
            diMatrixC = _gpu.CopyToDevice(hiMatrixCMN);

            _blas.GEMM(M, K, N, Alpha, diMatrixA, diMatrixB, Beta, diMatrixC, cublasOperation.N, cublasOperation.T);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += Alpha * hiMatrixAMK[GetIndexColumnMajor(i, k, M)] * hiMatrixBNK[GetIndexColumnMajor(j, k, N)];
                    }

                    cpuResult += Beta * hiMatrixCMN[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // A : Transpose, B : Transpose
            ClearBuffer(hiMatrixAKM);
            ClearBuffer(hiMatrixBNK);
            ClearBuffer(hiMatrixCMN);

            FillBuffer(hiMatrixAKM);
            FillBuffer(hiMatrixBNK);
            FillBuffer(hiMatrixCMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAKM);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBNK);
            diMatrixC = _gpu.CopyToDevice(hiMatrixCMN);

            _blas.GEMM(M, K, N, Alpha, diMatrixA, diMatrixB, Beta, diMatrixC, cublasOperation.T, cublasOperation.T);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += Alpha * hiMatrixAKM[GetIndexColumnMajor(k, i, K)] * hiMatrixBNK[GetIndexColumnMajor(j, k, N)];
                    }

                    cpuResult += Beta * hiMatrixCMN[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS3_SYMM()
        {
            // Lower fill mode, Side left
            ClearBuffer(hiMatrixAMM);
            ClearBuffer(hiMatrixBMN);
            ClearBuffer(hiMatrixCMN);

            FillBuffer(hiMatrixAMM);
            ConverToSymmetric(hiMatrixAMM, M);
            FillBuffer(hiMatrixBMN);
            FillBuffer(hiMatrixCMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAMM);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);
            diMatrixC = _gpu.CopyToDevice(hiMatrixCMN);

            _blas.SYMM(M, N, Alpha, diMatrixA, diMatrixB, Beta, diMatrixC);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < M; k++)
                    {
                        cpuResult += Alpha * hiMatrixAMM[GetIndexColumnMajor(i, k, M)] * hiMatrixBMN[GetIndexColumnMajor(k, j, M)];
                    }

                    cpuResult += Beta * hiMatrixCMN[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Lower fill mode, Side right
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixBMN);
            ClearBuffer(hiMatrixCMN);

            FillBuffer(hiMatrixANN);
            ConverToSymmetric(hiMatrixANN, N);
            FillBuffer(hiMatrixBMN);
            FillBuffer(hiMatrixCMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixANN);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);
            diMatrixC = _gpu.CopyToDevice(hiMatrixCMN);

            _blas.SYMM(M, N, Alpha, diMatrixA, diMatrixB, Beta, diMatrixC, cublasSideMode.Right);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < N; k++)
                    {
                        cpuResult += Alpha * hiMatrixBMN[GetIndexColumnMajor(i, k, M)] * hiMatrixANN[GetIndexColumnMajor(k, j, N)];
                    }

                    cpuResult += Beta * hiMatrixCMN[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Upper fill mode, Side left
            ClearBuffer(hiMatrixAMM);
            ClearBuffer(hiMatrixBMN);
            ClearBuffer(hiMatrixCMN);

            FillBuffer(hiMatrixAMM);
            ConverToSymmetric(hiMatrixAMM, M);
            FillBuffer(hiMatrixBMN);
            FillBuffer(hiMatrixCMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAMM);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);
            diMatrixC = _gpu.CopyToDevice(hiMatrixCMN);

            _blas.SYMM(M, N, Alpha, diMatrixA, diMatrixB, Beta, diMatrixC, cublasSideMode.Left, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < M; k++)
                    {
                        cpuResult += Alpha * hiMatrixAMM[GetIndexColumnMajor(i, k, M)] * hiMatrixBMN[GetIndexColumnMajor(k, j, M)];
                    }

                    cpuResult += Beta * hiMatrixCMN[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Upper fill mode, Side right
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixBMN);
            ClearBuffer(hiMatrixCMN);

            FillBuffer(hiMatrixANN);
            ConverToSymmetric(hiMatrixANN, N);
            FillBuffer(hiMatrixBMN);
            FillBuffer(hiMatrixCMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixANN);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);
            diMatrixC = _gpu.CopyToDevice(hiMatrixCMN);

            _blas.SYMM(M, N, Alpha, diMatrixA, diMatrixB, Beta, diMatrixC, cublasSideMode.Right, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < N; k++)
                    {
                        cpuResult += Alpha * hiMatrixBMN[GetIndexColumnMajor(i, k, M)] * hiMatrixANN[GetIndexColumnMajor(k, j, N)];
                    }

                    cpuResult += Beta * hiMatrixCMN[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS3_SYRK()
        {
            // Lower fill mode, No transpose
            ClearBuffer(hiMatrixAMK);
            ClearBuffer(hiMatrixCMM);

            FillBuffer(hiMatrixAMK);
            FillBuffer(hiMatrixCMM);
            ConverToSymmetric(hiMatrixCMM, M);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAMK);
            diMatrixC = _gpu.CopyToDevice(hiMatrixCMM);

            _blas.SYRK(M, K, Alpha, diMatrixA, Beta, diMatrixC);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMM);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += Alpha * hiMatrixAMK[GetIndexColumnMajor(i, k, M)] * hiMatrixAMK[GetIndexColumnMajor(j, k, M)];
                    }

                    cpuResult += Beta * hiMatrixCMM[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMM[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Lower fill mode, Transpose
            ClearBuffer(hiMatrixAKM);
            ClearBuffer(hiMatrixCMM);

            FillBuffer(hiMatrixAKM);
            FillBuffer(hiMatrixCMM);
            ConverToSymmetric(hiMatrixCMM, M);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAKM);
            diMatrixC = _gpu.CopyToDevice(hiMatrixCMM);

            _blas.SYRK(M, K, Alpha, diMatrixA, Beta, diMatrixC, cublasOperation.T);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMM);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += Alpha * hiMatrixAKM[GetIndexColumnMajor(k, i, K)] * hiMatrixAKM[GetIndexColumnMajor(k, j, K)];
                    }

                    cpuResult += Beta * hiMatrixCMM[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMM[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Upper fill mode, No transpose
            ClearBuffer(hiMatrixAMK);
            ClearBuffer(hiMatrixCMM);

            FillBuffer(hiMatrixAMK);
            FillBuffer(hiMatrixCMM);
            ConverToSymmetric(hiMatrixCMM, M);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAMK);
            diMatrixC = _gpu.CopyToDevice(hiMatrixCMM);

            _blas.SYRK(M, K, Alpha, diMatrixA, Beta, diMatrixC, cublasOperation.N, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMM);

            for (int i = 0; i < M; i++)
            {
                for (int j = i; j < M; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += Alpha * hiMatrixAMK[GetIndexColumnMajor(i, k, M)] * hiMatrixAMK[GetIndexColumnMajor(j, k, M)];
                    }

                    cpuResult += Beta * hiMatrixCMM[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMM[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Upper fill mode, Transpose
            ClearBuffer(hiMatrixAKM);
            ClearBuffer(hiMatrixCMM);

            FillBuffer(hiMatrixAKM);
            FillBuffer(hiMatrixCMM);
            ConverToSymmetric(hiMatrixCMM, M);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAKM);
            diMatrixC = _gpu.CopyToDevice(hiMatrixCMM);

            _blas.SYRK(M, K, Alpha, diMatrixA, Beta, diMatrixC, cublasOperation.T, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMM);

            for (int i = 0; i < M; i++)
            {
                for (int j = i; j < M; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += Alpha * hiMatrixAKM[GetIndexColumnMajor(k, i, K)] * hiMatrixAKM[GetIndexColumnMajor(k, j, K)];
                    }

                    cpuResult += Beta * hiMatrixCMM[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMM[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS3_SYR2K()
        {
            // Lower fill mode, No transpose
            ClearBuffer(hiMatrixAMK);
            ClearBuffer(hiMatrixBMK);
            ClearBuffer(hiMatrixCMM);

            FillBuffer(hiMatrixAMK);
            FillBuffer(hiMatrixBMK);
            FillBuffer(hiMatrixCMM);
            ConverToSymmetric(hiMatrixCMM, M);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAMK);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMK);
            diMatrixC = _gpu.CopyToDevice(hiMatrixCMM);

            _blas.SYR2K(M, K, Alpha, diMatrixA, diMatrixB, Beta, diMatrixC);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMM);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += Alpha * (hiMatrixAMK[GetIndexColumnMajor(i, k, M)] * hiMatrixBMK[GetIndexColumnMajor(j, k, M)] +
                            hiMatrixBMK[GetIndexColumnMajor(i, k, M)] * hiMatrixAMK[GetIndexColumnMajor(j, k, M)]);
                    }

                    cpuResult += Beta * hiMatrixCMM[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMM[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Lower fill mode, With Transpose
            ClearBuffer(hiMatrixAKM);
            ClearBuffer(hiMatrixBKM);
            ClearBuffer(hiMatrixCMM);

            FillBuffer(hiMatrixAKM);
            FillBuffer(hiMatrixBKM);
            FillBuffer(hiMatrixCMM);
            ConverToSymmetric(hiMatrixCMM, M);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAKM);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBKM);
            diMatrixC = _gpu.CopyToDevice(hiMatrixCMM);

            _blas.SYR2K(M, K, Alpha, diMatrixA, diMatrixB, Beta, diMatrixC, cublasOperation.T);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMM);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += Alpha * (hiMatrixAKM[GetIndexColumnMajor(k, i, K)] * hiMatrixBKM[GetIndexColumnMajor(k, j, K)] +
                            hiMatrixBKM[GetIndexColumnMajor(k, i, K)] * hiMatrixAKM[GetIndexColumnMajor(k, j, K)]);
                    }

                    cpuResult += Beta * hiMatrixCMM[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMM[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Upper fill mode, No transpose
            ClearBuffer(hiMatrixAMK);
            ClearBuffer(hiMatrixBMK);
            ClearBuffer(hiMatrixCMM);

            FillBuffer(hiMatrixAMK);
            FillBuffer(hiMatrixBMK);
            FillBuffer(hiMatrixCMM);
            ConverToSymmetric(hiMatrixCMM, M);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAMK);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMK);
            diMatrixC = _gpu.CopyToDevice(hiMatrixCMM);

            _blas.SYR2K(M, K, Alpha, diMatrixA, diMatrixB, Beta, diMatrixC, cublasOperation.N, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMM);

            for (int i = 0; i < M; i++)
            {
                for (int j = i; j < M; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += Alpha * (hiMatrixAMK[GetIndexColumnMajor(i, k, M)] * hiMatrixBMK[GetIndexColumnMajor(j, k, M)] +
                            hiMatrixBMK[GetIndexColumnMajor(i, k, M)] * hiMatrixAMK[GetIndexColumnMajor(j, k, M)]);
                    }

                    cpuResult += Beta * hiMatrixCMM[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMM[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Lower fill mode, With Transpose
            ClearBuffer(hiMatrixAKM);
            ClearBuffer(hiMatrixBKM);
            ClearBuffer(hiMatrixCMM);

            FillBuffer(hiMatrixAKM);
            FillBuffer(hiMatrixBKM);
            FillBuffer(hiMatrixCMM);
            ConverToSymmetric(hiMatrixCMM, M);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAKM);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBKM);
            diMatrixC = _gpu.CopyToDevice(hiMatrixCMM);

            _blas.SYR2K(M, K, Alpha, diMatrixA, diMatrixB, Beta, diMatrixC, cublasOperation.T, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMM);

            for (int i = 0; i < M; i++)
            {
                for (int j = i; j < M; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += Alpha * (hiMatrixAKM[GetIndexColumnMajor(k, i, K)] * hiMatrixBKM[GetIndexColumnMajor(k, j, K)] +
                            hiMatrixBKM[GetIndexColumnMajor(k, i, K)] * hiMatrixAKM[GetIndexColumnMajor(k, j, K)]);
                    }

                    cpuResult += Beta * hiMatrixCMM[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMM[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS3_TRMM()
        {
            // Lower fill mode, Side left, No transpose
            ClearBuffer(hiMatrixAMM);
            ClearBuffer(hiMatrixBMN);

            CreateBandedMatrix(hiMatrixAMM, M, M, M - 1, 0);
            FillBuffer(hiMatrixBMN);
            FillBuffer(hiMatrixCMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAMM);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);
            diMatrixC = _gpu.Allocate(hiMatrixCMN);

            _blas.TRMM(M, N, Alpha, diMatrixA, diMatrixB, diMatrixC);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = i; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < M; k++)
                    {
                        cpuResult += Alpha * hiMatrixAMM[GetIndexColumnMajor(i, k, M)] * hiMatrixBMN[GetIndexColumnMajor(k, j, M)];
                    }

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Lower fill mode, Side left, Transpose
            ClearBuffer(hiMatrixAMM);
            ClearBuffer(hiMatrixBMN);

            CreateBandedMatrix(hiMatrixAMM, M, M, M - 1, 0);
            FillBuffer(hiMatrixBMN);
            FillBuffer(hiMatrixCMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAMM);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);
            diMatrixC = _gpu.Allocate(hiMatrixCMN);

            _blas.TRMM(M, N, Alpha, diMatrixA, diMatrixB, diMatrixC, cublasSideMode.Left, cublasOperation.T);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = i; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < M; k++)
                    {
                        cpuResult += Alpha * hiMatrixAMM[GetIndexColumnMajor(k, i, M)] * hiMatrixBMN[GetIndexColumnMajor(k, j, M)];
                    }

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Lower fill mode, Side right, No transpose
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixBMN);

            CreateBandedMatrix(hiMatrixANN, N, N, N - 1, 0);
            FillBuffer(hiMatrixBMN);
            FillBuffer(hiMatrixCMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixANN);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);
            diMatrixC = _gpu.Allocate(hiMatrixCMN);

            _blas.TRMM(M, N, Alpha, diMatrixA, diMatrixB, diMatrixC, cublasSideMode.Right);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = i; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < N; k++)
                    {
                        cpuResult += Alpha * hiMatrixBMN[GetIndexColumnMajor(i, k, M)] * hiMatrixANN[GetIndexColumnMajor(k, j, N)];
                    }

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Lower fill mode, Side right, Transpose
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixBMN);

            CreateBandedMatrix(hiMatrixANN, N, N, N - 1, 0);
            FillBuffer(hiMatrixBMN);
            FillBuffer(hiMatrixCMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixANN);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);
            diMatrixC = _gpu.Allocate(hiMatrixCMN);

            _blas.TRMM(M, N, Alpha, diMatrixA, diMatrixB, diMatrixC, cublasSideMode.Right, cublasOperation.T);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = i; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < N; k++)
                    {
                        cpuResult += Alpha * hiMatrixBMN[GetIndexColumnMajor(i, k, M)] * hiMatrixANN[GetIndexColumnMajor(j, k, N)];
                    }

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Upper fill mode, Side left, No transpose
            ClearBuffer(hiMatrixAMM);
            ClearBuffer(hiMatrixBMN);

            CreateBandedMatrix(hiMatrixAMM, M, M, 0, M - 1);
            FillBuffer(hiMatrixBMN);
            FillBuffer(hiMatrixCMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAMM);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);
            diMatrixC = _gpu.Allocate(hiMatrixCMN);

            _blas.TRMM(M, N, Alpha, diMatrixA, diMatrixB, diMatrixC, cublasSideMode.Left, cublasOperation.N, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = i; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < M; k++)
                    {
                        cpuResult += Alpha * hiMatrixAMM[GetIndexColumnMajor(i, k, M)] * hiMatrixBMN[GetIndexColumnMajor(k, j, M)];
                    }

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Upper fill mode, Side left, Transpose
            ClearBuffer(hiMatrixAMM);
            ClearBuffer(hiMatrixBMN);

            CreateBandedMatrix(hiMatrixAMM, M, M, 0, M - 1);
            FillBuffer(hiMatrixBMN);
            FillBuffer(hiMatrixCMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAMM);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);
            diMatrixC = _gpu.Allocate(hiMatrixCMN);

            _blas.TRMM(M, N, Alpha, diMatrixA, diMatrixB, diMatrixC, cublasSideMode.Left, cublasOperation.T, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = i; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < M; k++)
                    {
                        cpuResult += Alpha * hiMatrixAMM[GetIndexColumnMajor(k, i, M)] * hiMatrixBMN[GetIndexColumnMajor(k, j, M)];
                    }

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Upper fill mode, Side right, No transpose
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixBMN);

            CreateBandedMatrix(hiMatrixANN, N, N, 0, N - 1);
            FillBuffer(hiMatrixBMN);
            FillBuffer(hiMatrixCMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixANN);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);
            diMatrixC = _gpu.Allocate(hiMatrixCMN);

            _blas.TRMM(M, N, Alpha, diMatrixA, diMatrixB, diMatrixC, cublasSideMode.Right, cublasOperation.N, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = i; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < N; k++)
                    {
                        cpuResult += Alpha * hiMatrixBMN[GetIndexColumnMajor(i, k, M)] * hiMatrixANN[GetIndexColumnMajor(k, j, N)];
                    }

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Upper fill mode, Side right, Transpose
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixBMN);

            CreateBandedMatrix(hiMatrixANN, N, N, 0, N - 1);
            FillBuffer(hiMatrixBMN);
            FillBuffer(hiMatrixCMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixANN);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);
            diMatrixC = _gpu.Allocate(hiMatrixCMN);

            _blas.TRMM(M, N, Alpha, diMatrixA, diMatrixB, diMatrixC, cublasSideMode.Right, cublasOperation.T, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = i; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < N; k++)
                    {
                        cpuResult += Alpha * hiMatrixBMN[GetIndexColumnMajor(i, k, M)] * hiMatrixANN[GetIndexColumnMajor(j, k, N)];
                    }

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS3_TRSM()
        {
            // Solve AX = B
            double maxError;

            // Lower triangular, Side left, No transpose
            ClearBuffer(hiMatrixAMM);
            ClearBuffer(hiMatrixBMN);

            CreateMainDiagonalOnlyMatrix(hiMatrixAMM, M);
            FillBuffer(hiMatrixBMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAMM);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);

            _blas.TRSM(M, N, Alpha, diMatrixA, diMatrixB);

            _gpu.CopyFromDevice(diMatrixB, gpuResultMN);

            // Check AX - B
            maxError = 0.0;

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < M; k++)
                    {
                        cpuResult += hiMatrixAMM[GetIndexColumnMajor(i, k, M)] * gpuResultMN[GetIndexColumnMajor(k, j, M)];
                    }

                    double error = Math.Abs(cpuResult - Alpha * hiMatrixBMN[GetIndexColumnMajor(i, j, M)]);

                    if (maxError < error)
                    {
                        maxError = error;
                    }
                }
            }
            Console.WriteLine("Max error : {0} (Lower fill mode, Side left, No transpose)", maxError);

            _gpu.FreeAll();

            // Lower triangular, Side left, Transpose
            ClearBuffer(hiMatrixAMM);
            ClearBuffer(hiMatrixBMN);

            CreateMainDiagonalOnlyMatrix(hiMatrixAMM, M);
            FillBuffer(hiMatrixBMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAMM);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);

            _blas.TRSM(M, N, Alpha, diMatrixA, diMatrixB, cublasSideMode.Left, cublasOperation.T, cublasFillMode.Lower);

            _gpu.CopyFromDevice(diMatrixB, gpuResultMN);

            maxError = 0.0;

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < M; k++)
                    {
                        cpuResult += hiMatrixAMM[GetIndexColumnMajor(k, i, M)] * gpuResultMN[GetIndexColumnMajor(k, j, M)];
                    }

                    double error = Math.Abs(cpuResult - Alpha * hiMatrixBMN[GetIndexColumnMajor(i, j, M)]);

                    if (maxError < error)
                    {
                        maxError = error;
                    }
                }
            }
            Console.WriteLine("Max error : {0} (Lower fill mode, Side left, Transpose)", maxError);

            _gpu.FreeAll();

            // Lower triangular, Side right, No transpose
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixBMN);

            CreateMainDiagonalOnlyMatrix(hiMatrixANN, N);
            FillBuffer(hiMatrixBMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixANN);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);

            _blas.TRSM(M, N, Alpha, diMatrixA, diMatrixB, cublasSideMode.Right);

            _gpu.CopyFromDevice(diMatrixB, gpuResultMN);

            // Check AX - B
            maxError = 0.0;

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < N; k++)
                    {
                        cpuResult += gpuResultMN[GetIndexColumnMajor(i, k, M)] * hiMatrixANN[GetIndexColumnMajor(k, j, N)];
                    }

                    double error = Math.Abs(cpuResult - Alpha * hiMatrixBMN[GetIndexColumnMajor(i, j, M)]);

                    if (maxError < error)
                    {
                        maxError = error;
                    }
                }
            }
            Console.WriteLine("Max error : {0} (Lower fill mode, Side right, No transpose)", maxError);

            _gpu.FreeAll();

            // Lower triangular, Side right, Transpose
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixBMN);

            CreateMainDiagonalOnlyMatrix(hiMatrixANN, N);
            FillBuffer(hiMatrixBMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixANN);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);

            _blas.TRSM(M, N, Alpha, diMatrixA, diMatrixB, cublasSideMode.Right, cublasOperation.T);

            _gpu.CopyFromDevice(diMatrixB, gpuResultMN);

            // Check AX - B
            maxError = 0.0;

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < N; k++)
                    {
                        cpuResult += gpuResultMN[GetIndexColumnMajor(i, k, M)] * hiMatrixANN[GetIndexColumnMajor(j, k, N)];
                    }

                    double error = Math.Abs(cpuResult - Alpha * hiMatrixBMN[GetIndexColumnMajor(i, j, M)]);

                    if (maxError < error)
                    {
                        maxError = error;
                    }
                }
            }
            Console.WriteLine("Max error : {0} (Lower fill mode, Side right, Transpose)", maxError);

            _gpu.FreeAll();

            // Upper triangular, Side left, No transpose
            ClearBuffer(hiMatrixAMM);
            ClearBuffer(hiMatrixBMN);

            CreateMainDiagonalOnlyMatrix(hiMatrixAMM, M);
            FillBuffer(hiMatrixBMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAMM);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);

            _blas.TRSM(M, N, Alpha, diMatrixA, diMatrixB, cublasSideMode.Left, cublasOperation.N, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixB, gpuResultMN);

            // Check AX - B
            maxError = 0.0;

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < M; k++)
                    {
                        cpuResult += hiMatrixAMM[GetIndexColumnMajor(i, k, M)] * gpuResultMN[GetIndexColumnMajor(k, j, M)];
                    }

                    double error = Math.Abs(cpuResult - Alpha * hiMatrixBMN[GetIndexColumnMajor(i, j, M)]);

                    if (maxError < error)
                    {
                        maxError = error;
                    }
                }
            }
            Console.WriteLine("Max error : {0} (Upper fill mode, Side left, No transpose)", maxError);

            _gpu.FreeAll();

            // Upper triangular, Side left, Transpose
            ClearBuffer(hiMatrixAMM);
            ClearBuffer(hiMatrixBMN);

            CreateMainDiagonalOnlyMatrix(hiMatrixAMM, M);
            FillBuffer(hiMatrixBMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixAMM);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);

            _blas.TRSM(M, N, Alpha, diMatrixA, diMatrixB, cublasSideMode.Left, cublasOperation.T, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixB, gpuResultMN);

            maxError = 0.0;

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < M; k++)
                    {
                        cpuResult += hiMatrixAMM[GetIndexColumnMajor(k, i, M)] * gpuResultMN[GetIndexColumnMajor(k, j, M)];
                    }

                    double error = Math.Abs(cpuResult - Alpha * hiMatrixBMN[GetIndexColumnMajor(i, j, M)]);

                    if (maxError < error)
                    {
                        maxError = error;
                    }
                }
            }
            Console.WriteLine("Max error : {0} (Upper fill mode, Side left, Transpose)", maxError);

            _gpu.FreeAll();

            // Upper triangular, Side right, No transpose
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixBMN);

            CreateMainDiagonalOnlyMatrix(hiMatrixANN, N);
            FillBuffer(hiMatrixBMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixANN);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);

            _blas.TRSM(M, N, Alpha, diMatrixA, diMatrixB, cublasSideMode.Right, cublasOperation.N, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixB, gpuResultMN);

            // Check AX - B
            maxError = 0.0;

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < N; k++)
                    {
                        cpuResult += gpuResultMN[GetIndexColumnMajor(i, k, M)] * hiMatrixANN[GetIndexColumnMajor(k, j, N)];
                    }

                    double error = Math.Abs(cpuResult - Alpha * hiMatrixBMN[GetIndexColumnMajor(i, j, M)]);

                    if (maxError < error)
                    {
                        maxError = error;
                    }
                }
            }
            Console.WriteLine("Max error : {0} (Upper fill mode, Side right, No transpose)", maxError);

            _gpu.FreeAll();

            // Upper triangular, Side right, Transpose
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixBMN);

            CreateMainDiagonalOnlyMatrix(hiMatrixANN, N);
            FillBuffer(hiMatrixBMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixANN);
            diMatrixB = _gpu.CopyToDevice(hiMatrixBMN);

            _blas.TRSM(M, N, Alpha, diMatrixA, diMatrixB, cublasSideMode.Right, cublasOperation.T, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixB, gpuResultMN);

            // Check AX - B
            maxError = 0.0;

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < N; k++)
                    {
                        cpuResult += gpuResultMN[GetIndexColumnMajor(i, k, M)] * hiMatrixANN[GetIndexColumnMajor(j, k, N)];
                    }

                    double error = Math.Abs(cpuResult - Alpha * hiMatrixBMN[GetIndexColumnMajor(i, j, M)]);

                    if (maxError < error)
                    {
                        maxError = error;
                    }
                }
            }
            Console.WriteLine("Max error : {0} (Upper fill mode, Side right, Transpose)", maxError);

            _gpu.FreeAll();
        }
    }
}
