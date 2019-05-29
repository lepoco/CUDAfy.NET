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
    public class BLAS2 : ICudafyUnitTest
    {
        GPGPU _gpu;
        GPGPUBLAS _blas;

        // Define constants
        const int M = 2; // Row size of matrix A
        const int N = 2; // Column size of matrix A
        const int KL = 1; // number of subdiagonals of matrix A
        const int KU = 1; // number of superdiagonals of matrix A
        const int K = 1; // number of superdiagonals and subdiagonals of symmetric matrix A
        const double Alpha = 3.0;
        const double Beta = 4.0;

        // Range of value for test matrix and buffer.
        // All value are in 1 <= value <= RamdomMax + 1
        private const int RandomMax = 255;

        // CPU Buffers
        double[] hiMatrixA; // General matrix (M x N)
        double[] hiMatrixANN; // Symmetric matrix (N x N)
        double[] hiMatrixACBC; // General banded matrix CBC format. (KL + KU + 1) x N
        double[] hiMatrixASCBC; // Symmetric banded matrix CBC format. (K + 1) x N
        double[] hiMatrixAPS; // Symmetric matrix stored in Packed format. It has (n x (n + 1)) / 2 elements.
        double[] hiVectorXM;
        double[] hiVectorXN;
        double[] hiVectorYM;
        double[] hiVectorYN;
        double[] diMatrixA;
        double[] diVectorXM;
        double[] diVectorXN;
        double[] diVectorYM;
        double[] diVectorYN;
        double[] gpuResultM;
        double[] gpuResultN;
        double[] gpuResultMN;
        double[] gpuResultNN;
        double[] gpuResultP; // Symmetric matrix stored in Packed format.

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.GetDevice(CudafyModes.Target);
            _blas = GPGPUBLAS.Create(_gpu);
            Console.Write("BLAS Version={0}", _blas.GetVersion());
            // Initialize CPU Buffer
            hiMatrixA = new double[M * N];
            hiMatrixANN = new double[N * N];
            hiMatrixACBC = new double[(KL + KU + 1) * N];
            hiMatrixASCBC = new double[(K + 1) * N];
            hiMatrixAPS = new double[(N * (N + 1)) / 2];
            hiVectorXM = new double[M];
            hiVectorXN = new double[N];
            hiVectorYM = new double[M];
            hiVectorYN = new double[N];
            gpuResultM = new double[M];
            gpuResultN = new double[N];
            gpuResultMN = new double[M * N];
            gpuResultNN = new double[N * N];
            gpuResultP = new double[(N * (N + 1)) / 2];
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

        private int GetIndexPackedSymmetric(int i, int j, int n, cublasFillMode fillMode)
        {
            if (fillMode == cublasFillMode.Lower)
            {
                if (i < j)
                {
                    throw new ArgumentOutOfRangeException("Please set i >= j in Lower fill mode.");
                }

                return i + ((2 * n - j - 1) * j) / 2;
            }
            else
            {
                if (i > j)
                {
                    throw new ArgumentOutOfRangeException("Please set i <= j in Upper fill mode.");
                }
                return i + (j * (j + 1)) / 2;
            }
        }

        private void PrintMatrix(double[] buffer, int m, int n)
        {
            for (int i = 0; i < m; i++)
            {
                for(int j = 0; j < n; j++)
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

        private void CompressBandedMatrixToCBC(double[] A, double[] CBC, int m, int n, int kl, int ku)
        {
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if ((i - kl <= j) && (j <= i + ku))
                    {
                        int packedIndex = GetIndexColumnMajor(ku + i - j, j, kl + ku + 1);
                        int index = GetIndexColumnMajor(i, j, m);
                        if (index >= 0 && index < A.Length)
                        {
                            CBC[packedIndex] = A[index];
                        }
                    }
                }
            }
        }

        private void CompressSymmetricBandedMatrixToCBC(double[] A, double[] CBC, int n, int k, cublasFillMode fillMode)
        {
            if (fillMode == cublasFillMode.Lower)
            {
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        if ((i >= j) && (i <= j + k))
                        {
                            int packedIndex = GetIndexColumnMajor(i - j, j, k + 1);
                            int index = GetIndexColumnMajor(i, j, n);
                            CBC[packedIndex] = A[index];
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        if ((i <= j) && (i >= j - k))
                        {
                            int packedIndex = GetIndexColumnMajor(k + i - j, j, k + 1);
                            int index = GetIndexColumnMajor(i, j, n);
                            CBC[packedIndex] = A[index];
                        }
                    }
                }
            }
        }

        private void PackSymmetricMatrix(double[] matrixSymmetric, double[] buffer, int n, cublasFillMode fillMode)
        {
            if (fillMode == cublasFillMode.Lower)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int i = j; i < n; i++)
                    {
                        buffer[GetIndexPackedSymmetric(i, j, n, fillMode)] = matrixSymmetric[GetIndexColumnMajor(i, j, n)];
                    }
                }
            }
            else
            {
                for (int j = 0; j < n; j++)
                {
                    for (int i = 0; i <= j; i++)
                    {
                        buffer[GetIndexPackedSymmetric(i, j, n, fillMode)] = matrixSymmetric[GetIndexColumnMajor(i, j, n)];
                    }
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
        public void Test_BLAS2_GBMV()
        {
            ClearBuffer(hiMatrixA);
            ClearBuffer(hiMatrixACBC);
            ClearBuffer(hiVectorXM);
            ClearBuffer(hiVectorXN);
            ClearBuffer(hiVectorYM);
            ClearBuffer(hiVectorYN);

            CreateBandedMatrix(hiMatrixA, M, N, KL, KU);
            CompressBandedMatrixToCBC(hiMatrixA, hiMatrixACBC, M, N, KL, KU);
            FillBuffer(hiVectorXM);
            FillBuffer(hiVectorXN);
            FillBuffer(hiVectorYM);
            FillBuffer(hiVectorYN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixACBC);
            diVectorXM = _gpu.CopyToDevice(hiVectorXM);
            diVectorXN = _gpu.CopyToDevice(hiVectorXN);


            // Test without transpose
            diVectorYM = _gpu.CopyToDevice(hiVectorYM);
            _blas.GBMV(M, N, KL, KU, Alpha, diMatrixA, diVectorXN, Beta, diVectorYM);
            _gpu.CopyFromDevice(diVectorYM, gpuResultM);

            for (int i = 0; i < M; i++)
            {
                double cpuResult = 0.0;

                for (int j = 0; j < N; j++)
                {
                    cpuResult += Alpha * hiMatrixA[GetIndexColumnMajor(i, j, M)] * hiVectorXN[j];
                }

                cpuResult += Beta * hiVectorYM[i];

                Assert.AreEqual(cpuResult, gpuResultM[i]);
            }

            // Test with transpose
            diVectorYN = _gpu.CopyToDevice(hiVectorYN);
            _blas.GBMV(M, N, KL, KU, Alpha, diMatrixA, diVectorXM, Beta, diVectorYN, cublasOperation.T);
            _gpu.CopyFromDevice(diVectorYN, gpuResultN);

            for (int j = 0; j < N; j++)
            {
                double cpuResult = 0.0;

                for (int i = 0; i < M; i++)
                {
                    cpuResult += Alpha * hiMatrixA[GetIndexColumnMajor(i, j, M)] * hiVectorXM[i];
                }

                cpuResult += Beta * hiVectorYN[j];

                Assert.AreEqual(cpuResult, gpuResultN[j]);
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS2_GEMV()
        {
            ClearBuffer(hiMatrixA);
            ClearBuffer(hiVectorXM);
            ClearBuffer(hiVectorXN);
            ClearBuffer(hiVectorYM);
            ClearBuffer(hiVectorYN);

            FillBuffer(hiMatrixA);
            FillBuffer(hiVectorXM);
            FillBuffer(hiVectorXN);
            FillBuffer(hiVectorYM);
            FillBuffer(hiVectorYN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixA);
            diVectorXM = _gpu.CopyToDevice(hiVectorXM);
            diVectorXN = _gpu.CopyToDevice(hiVectorXN);


            // Test without transpose
            diVectorYM = _gpu.CopyToDevice(hiVectorYM);
            _blas.GEMV(M, N, Alpha, diMatrixA, diVectorXN, Beta, diVectorYM);
            _gpu.CopyFromDevice(diVectorYM, gpuResultM);

            for (int i = 0; i < M; i++)
            {
                double cpuResult = 0.0;

                for (int j = 0; j < N; j++)
                {
                    cpuResult += Alpha * hiMatrixA[GetIndexColumnMajor(i, j, M)] * hiVectorXN[j];
                }

                cpuResult += Beta * hiVectorYM[i];

                Assert.AreEqual(cpuResult, gpuResultM[i]);
            }

            // Test with transpose
            diVectorYN = _gpu.CopyToDevice(hiVectorYN);
            _blas.GEMV(M, N, Alpha, diMatrixA, diVectorXM, Beta, diVectorYN, cublasOperation.T);
            _gpu.CopyFromDevice(diVectorYN, gpuResultN);

            for (int j = 0; j < N; j++)
            {
                double cpuResult = 0.0;

                for (int i = 0; i < M; i++)
                {
                    cpuResult += Alpha * hiMatrixA[GetIndexColumnMajor(i, j, M)] * hiVectorXM[i];
                }

                cpuResult += Beta * hiVectorYN[j];

                Assert.AreEqual(cpuResult, gpuResultN[j]);
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS2_GER()
        {
            ClearBuffer(hiMatrixA);
            ClearBuffer(hiVectorXM);
            ClearBuffer(hiVectorYN);

            FillBuffer(hiMatrixA);
            FillBuffer(hiVectorXM);
            FillBuffer(hiVectorYN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixA);
            diVectorXM = _gpu.CopyToDevice(hiVectorXM);
            diVectorYN = _gpu.CopyToDevice(hiVectorYN);

            _blas.GER(M, N, Alpha, diVectorXM, diVectorYN, diMatrixA);

            _gpu.CopyFromDevice(diMatrixA, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = Alpha * hiVectorXM[i] * hiVectorYN[j] + hiMatrixA[GetIndexColumnMajor(i, j, M)];
                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
                
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS2_SBMV()
        {
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixASCBC);
            ClearBuffer(hiVectorXN);
            ClearBuffer(hiVectorYN);

            CreateBandedMatrix(hiMatrixANN, N, N, K, K);
            ConverToSymmetric(hiMatrixANN, N);
            FillBuffer(hiVectorXN);
            FillBuffer(hiVectorYN);
            diMatrixA = _gpu.Allocate(hiMatrixASCBC);
            diVectorYN = _gpu.Allocate(hiVectorYN);
            diVectorXN = _gpu.CopyToDevice(hiVectorXN);

            // Lower fill mode
            CompressSymmetricBandedMatrixToCBC(hiMatrixANN, hiMatrixASCBC, N, K, cublasFillMode.Lower);
            _gpu.CopyToDevice(hiMatrixASCBC, diMatrixA);

            _gpu.CopyToDevice(hiVectorYN, diVectorYN);

            _blas.SBMV(N, K, Alpha, diMatrixA, diVectorXN, Beta, diVectorYN);

            _gpu.CopyFromDevice(diVectorYN, gpuResultN);

            for (int i = 0; i < N; i++)
            {
                double cpuResult = 0.0;

                for (int j = 0; j < N; j++)
                {
                    cpuResult += Alpha * hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[j];
                }

                cpuResult += Beta * hiVectorYN[i];

                Assert.AreEqual(cpuResult, gpuResultN[i]);
            }

            // Upper fill mode
            CompressSymmetricBandedMatrixToCBC(hiMatrixANN, hiMatrixASCBC, N, K, cublasFillMode.Upper);
            _gpu.CopyToDevice(hiMatrixASCBC, diMatrixA);

            _gpu.CopyToDevice(hiVectorYN, diVectorYN);

            _blas.SBMV(N, K, Alpha, diMatrixA, diVectorXN, Beta, diVectorYN, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diVectorYN, gpuResultN);

            for (int i = 0; i < N; i++)
            {
                double cpuResult = 0.0;

                for (int j = 0; j < N; j++)
                {
                    cpuResult += Alpha * hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[j];
                }

                cpuResult += Beta * hiVectorYN[i];

                Assert.AreEqual(cpuResult, gpuResultN[i]);
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS2_SPMV()
        {
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixAPS);
            ClearBuffer(hiVectorXN);
            ClearBuffer(hiVectorYN);

            FillBuffer(hiMatrixANN);
            ConverToSymmetric(hiMatrixANN, N);
            FillBuffer(hiVectorXN);
            FillBuffer(hiVectorYN);
            diMatrixA = _gpu.Allocate(hiMatrixAPS);
            diVectorYN = _gpu.Allocate(hiVectorYN);
            diVectorXN = _gpu.CopyToDevice(hiVectorXN);

            // Lower fill mode
            PackSymmetricMatrix(hiMatrixANN, hiMatrixAPS, N, cublasFillMode.Lower);
            _gpu.CopyToDevice(hiMatrixAPS, diMatrixA);

            _gpu.CopyToDevice(hiVectorYN, diVectorYN);

            _blas.SPMV(N, Alpha, diMatrixA, diVectorXN, Beta, diVectorYN);

            _gpu.CopyFromDevice(diVectorYN, gpuResultN);

            for (int i = 0; i < N; i++)
            {
                double cpuResult = 0.0;

                for (int j = 0; j < N; j++)
                {
                    cpuResult += Alpha * hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[j];
                }

                cpuResult += Beta * hiVectorYN[i];

                Assert.AreEqual(cpuResult, gpuResultN[i]);
            }

            // Upper fill mode
            PackSymmetricMatrix(hiMatrixANN, hiMatrixAPS, N, cublasFillMode.Upper);
            _gpu.CopyToDevice(hiMatrixAPS, diMatrixA);

            _gpu.CopyToDevice(hiVectorYN, diVectorYN);

            _blas.SPMV(N, Alpha, diMatrixA, diVectorXN, Beta, diVectorYN, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diVectorYN, gpuResultN);

            for (int i = 0; i < N; i++)
            {
                double cpuResult = 0.0;

                for (int j = 0; j < N; j++)
                {
                    cpuResult += Alpha * hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[j];
                }

                cpuResult += Beta * hiVectorYN[i];

                Assert.AreEqual(cpuResult, gpuResultN[i]);
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS2_SPR()
        {
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixAPS);
            ClearBuffer(hiVectorXN);

            FillBuffer(hiMatrixANN);
            ConverToSymmetric(hiMatrixANN, N);
            FillBuffer(hiVectorXN);
            diMatrixA = _gpu.Allocate(hiMatrixAPS);
            diVectorXN = _gpu.CopyToDevice(hiVectorXN);

            // Lower fill mode
            PackSymmetricMatrix(hiMatrixANN, hiMatrixAPS, N, cublasFillMode.Lower);
            _gpu.CopyToDevice(hiMatrixAPS, diMatrixA);

            _blas.SPR(N, Alpha, diVectorXN, diMatrixA);

            _gpu.CopyFromDevice(diMatrixA, gpuResultP);

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double cpuResult = Alpha * hiVectorXN[i] * hiVectorXN[j] + hiMatrixANN[GetIndexColumnMajor(i, j, N)];
                    Assert.AreEqual(cpuResult, gpuResultP[GetIndexPackedSymmetric(i, j, N, cublasFillMode.Lower)]);
                }
            }

            // Upper fill mode
            PackSymmetricMatrix(hiMatrixANN, hiMatrixAPS, N, cublasFillMode.Upper);
            _gpu.CopyToDevice(hiMatrixAPS, diMatrixA);

            _blas.SPR(N, Alpha, diVectorXN, diMatrixA, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixA, gpuResultP);

            for (int i = 0; i < N; i++)
            {
                for (int j = i; j < N; j++)
                {
                    double cpuResult = Alpha * hiVectorXN[i] * hiVectorXN[j] + hiMatrixANN[GetIndexColumnMajor(i, j, N)];
                    Assert.AreEqual(cpuResult, gpuResultP[GetIndexPackedSymmetric(i, j, N, cublasFillMode.Upper)]);
                }
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS2_SPR2()
        {
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixAPS);
            ClearBuffer(hiVectorXN);
            ClearBuffer(hiVectorYN);

            FillBuffer(hiMatrixANN);
            ConverToSymmetric(hiMatrixANN, N);
            FillBuffer(hiVectorXN);
            FillBuffer(hiVectorYN);
            diMatrixA = _gpu.Allocate(hiMatrixAPS);
            diVectorXN = _gpu.CopyToDevice(hiVectorXN);
            diVectorYN = _gpu.CopyToDevice(hiVectorYN);

            // Lower fill mode
            PackSymmetricMatrix(hiMatrixANN, hiMatrixAPS, N, cublasFillMode.Lower);
            _gpu.CopyToDevice(hiMatrixAPS, diMatrixA);

            _blas.SPR2(N, Alpha, diVectorXN, diVectorYN, diMatrixA);

            _gpu.CopyFromDevice(diMatrixA, gpuResultP);

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double cpuResult = Alpha * (hiVectorXN[i] * hiVectorYN[j] + hiVectorYN[i] * hiVectorXN[j]) + hiMatrixANN[GetIndexColumnMajor(i, j, N)];
                    Assert.AreEqual(cpuResult, gpuResultP[GetIndexPackedSymmetric(i, j, N, cublasFillMode.Lower)]);
                }
            }

            // Upper fill mode
            PackSymmetricMatrix(hiMatrixANN, hiMatrixAPS, N, cublasFillMode.Upper);
            _gpu.CopyToDevice(hiMatrixAPS, diMatrixA);

            _blas.SPR2(N, Alpha, diVectorXN, diVectorYN, diMatrixA, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixA, gpuResultP);

            for (int i = 0; i < N; i++)
            {
                for (int j = i; j < N; j++)
                {
                    double cpuResult = Alpha * (hiVectorXN[i] * hiVectorYN[j] + hiVectorYN[i] * hiVectorXN[j]) + hiMatrixANN[GetIndexColumnMajor(i, j, N)];
                    Assert.AreEqual(cpuResult, gpuResultP[GetIndexPackedSymmetric(i, j, N, cublasFillMode.Upper)]);
                }
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS2_SYR()
        {
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiVectorXN);

            FillBuffer(hiMatrixANN);
            ConverToSymmetric(hiMatrixANN, N);
            FillBuffer(hiVectorXN);

            diMatrixA = _gpu.Allocate(hiMatrixANN);
            diVectorXN = _gpu.CopyToDevice(hiVectorXN);

            // Lower fill mode
            _gpu.CopyToDevice(hiMatrixANN, diMatrixA);

            _blas.SYR(N, Alpha, diVectorXN, diMatrixA);

            _gpu.CopyFromDevice(diMatrixA, gpuResultNN);

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double cpuResult = Alpha * hiVectorXN[i] * hiVectorXN[j] + hiMatrixANN[GetIndexColumnMajor(i, j, N)];
                    Assert.AreEqual(cpuResult, gpuResultNN[GetIndexColumnMajor(i, j, N)]);
                }
            }

            // Upper fill mode
            _gpu.CopyToDevice(hiMatrixANN, diMatrixA);

            _blas.SYR(N, Alpha, diVectorXN, diMatrixA, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixA, gpuResultNN);

            for (int i = 0; i < N; i++)
            {
                for (int j = i; j < N; j++)
                {
                    double cpuResult = Alpha * hiVectorXN[i] * hiVectorXN[j] + hiMatrixANN[GetIndexColumnMajor(i, j, N)];
                    Assert.AreEqual(cpuResult, gpuResultNN[GetIndexColumnMajor(i, j, N)]);
                }
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS2_SYR2()
        {
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiVectorXN);
            ClearBuffer(hiVectorYN);

            FillBuffer(hiMatrixANN);
            ConverToSymmetric(hiMatrixANN, N);
            FillBuffer(hiVectorXN);
            FillBuffer(hiVectorYN);

            diMatrixA = _gpu.Allocate(hiMatrixANN);
            diVectorXN = _gpu.CopyToDevice(hiVectorXN);
            diVectorYN = _gpu.CopyToDevice(hiVectorYN);

            // Lower fill mode
            _gpu.CopyToDevice(hiMatrixANN, diMatrixA);

            _blas.SYR2(N, Alpha, diVectorXN,diVectorYN, diMatrixA);

            _gpu.CopyFromDevice(diMatrixA, gpuResultNN);

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double cpuResult = Alpha * (hiVectorXN[i] * hiVectorYN[j] + hiVectorYN[i] * hiVectorXN[j]) + hiMatrixANN[GetIndexColumnMajor(i, j, N)];
                    Assert.AreEqual(cpuResult, gpuResultNN[GetIndexColumnMajor(i, j, N)]);
                }
            }

            // Upper fill mode
            _gpu.CopyToDevice(hiMatrixANN, diMatrixA);

            _blas.SYR2(N, Alpha, diVectorXN, diVectorYN, diMatrixA, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diMatrixA, gpuResultNN);

            for (int i = 0; i < N; i++)
            {
                for (int j = i; j < N; j++)
                {
                    double cpuResult = Alpha * (hiVectorXN[i] * hiVectorYN[j] + hiVectorYN[i] * hiVectorXN[j]) + hiMatrixANN[GetIndexColumnMajor(i, j, N)];
                    Assert.AreEqual(cpuResult, gpuResultNN[GetIndexColumnMajor(i, j, N)]);
                }
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS2_TBMV()
        {
            ClearBuffer(hiMatrixASCBC);
            ClearBuffer(hiVectorXN);

            FillBuffer(hiVectorXN);

            diMatrixA = _gpu.Allocate(hiMatrixASCBC);
            diVectorXN = _gpu.Allocate(hiVectorXN);

            // Lower triangular banded matrix
            ClearBuffer(hiMatrixANN);
            CreateBandedMatrix(hiMatrixANN, N, N, K, 0);
            CompressSymmetricBandedMatrixToCBC(hiMatrixANN, hiMatrixASCBC, N, K, cublasFillMode.Lower);
            _gpu.CopyToDevice(hiMatrixASCBC, diMatrixA);
            
            // Test without transpose
            _gpu.CopyToDevice(hiVectorXN, diVectorXN);
            _blas.TBMV(N, K, diMatrixA, diVectorXN);
            _gpu.CopyFromDevice(diVectorXN, gpuResultN);

            for (int i = 0; i < N; i++)
            {
                double cpuResult = 0.0;

                for (int j = 0; j < N; j++)
                {
                    cpuResult += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[j];
                }

                Assert.AreEqual(cpuResult, gpuResultN[i]);
            }

            // Test with transpose
            _gpu.CopyToDevice(hiVectorXN, diVectorXN);
            _blas.TBMV(N, K, diMatrixA, diVectorXN, cublasOperation.T);
            _gpu.CopyFromDevice(diVectorXN, gpuResultN);

            for (int j = 0; j < N; j++)
            {
                double cpuResult = 0.0;

                for (int i = 0; i < N; i++)
                {
                    cpuResult += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[i];
                }

                Assert.AreEqual(cpuResult, gpuResultN[j]);
            }

            // Upper triangular banded matrix
            ClearBuffer(hiMatrixANN);
            CreateBandedMatrix(hiMatrixANN, N, N, 0, K);
            CompressSymmetricBandedMatrixToCBC(hiMatrixANN, hiMatrixASCBC, N, K, cublasFillMode.Upper);
            _gpu.CopyToDevice(hiMatrixASCBC, diMatrixA);

            // Test without transpose
            _gpu.CopyToDevice(hiVectorXN, diVectorXN);
            _blas.TBMV(N, K, diMatrixA, diVectorXN, cublasOperation.N, cublasFillMode.Upper);
            _gpu.CopyFromDevice(diVectorXN, gpuResultN);

            for (int i = 0; i < N; i++)
            {
                double cpuResult = 0.0;

                for (int j = 0; j < N; j++)
                {
                    cpuResult += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[j];
                }

                Assert.AreEqual(cpuResult, gpuResultN[i]);
            }

            // Test with transpose
            _gpu.CopyToDevice(hiVectorXN, diVectorXN);
            _blas.TBMV(N, K, diMatrixA, diVectorXN, cublasOperation.T, cublasFillMode.Upper);
            _gpu.CopyFromDevice(diVectorXN, gpuResultN);

            for (int j = 0; j < N; j++)
            {
                double cpuResult = 0.0;

                for (int i = 0; i < N; i++)
                {
                    cpuResult += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[i];
                }

                Assert.AreEqual(cpuResult, gpuResultN[j]);
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS2_TBSV()
        {
            // Solve Ax = y, and x overwrites y.

            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixASCBC);
            ClearBuffer(hiVectorXN);
            ClearBuffer(hiVectorYN);

            diMatrixA = _gpu.Allocate(hiMatrixASCBC);
            diVectorYN = _gpu.Allocate(hiVectorYN);

            // Because of there is no singular test, set matrix A to simple maindiagonal only matrix.
            CreateMainDiagonalOnlyMatrix(hiMatrixANN, N);
            FillBuffer(hiVectorYN);

            // Lower fill mode
            CompressSymmetricBandedMatrixToCBC(hiMatrixANN, hiMatrixASCBC, N, 0, cublasFillMode.Lower);
            _gpu.CopyToDevice(hiMatrixASCBC, diMatrixA);

            // Test without transpose
            _gpu.CopyToDevice(hiVectorYN, diVectorYN);

            _blas.TBSV(N, 0, diMatrixA, diVectorYN);

            _gpu.CopyFromDevice(diVectorYN, hiVectorXN);

            // Check solution error : y - Ax by elemental wise.
            double maxError = 0.0;

            for (int i = 0; i < N; i++)
            {
                double ax = 0.0;

                for (int j = 0; j < N; j++)
                {
                    ax += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[j];
                }

                double error = Math.Abs(ax - hiVectorYN[i]);

                if (maxError < error)
                {
                    maxError = error;
                }
            }

            Console.WriteLine("max error : {0} (lower fill mode, without transpose)", maxError);

            // Test with transpose
            _gpu.CopyToDevice(hiVectorYN, diVectorYN);

            _blas.TBSV(N, 0, diMatrixA, diVectorYN, cublasOperation.T);

            _gpu.CopyFromDevice(diVectorYN, hiVectorXN);

            // Check solution error : y - Ax by elemental wise.
            maxError = 0.0;

            for (int j = 0; j < N; j++)
            {
                double ax = 0.0;

                for (int i = 0; i < N; i++)
                {
                    ax += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[i];
                }

                double error = Math.Abs(ax - hiVectorYN[j]);

                if (maxError < error)
                {
                    maxError = error;
                }
            }

            Console.WriteLine("max error : {0} (lower fill mode, with transpose)", maxError);

            // Upper fill mode
            CompressSymmetricBandedMatrixToCBC(hiMatrixANN, hiMatrixASCBC, N, 0, cublasFillMode.Upper);
            _gpu.CopyToDevice(hiMatrixASCBC, diMatrixA);

            // Test without transpose
            _gpu.CopyToDevice(hiVectorYN, diVectorYN);

            _blas.TBSV(N, 0, diMatrixA, diVectorYN, cublasOperation.N, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diVectorYN, hiVectorXN);

            // Check solution error : y - Ax by elemental wise.
            maxError = 0.0;

            for (int i = 0; i < N; i++)
            {
                double ax = 0.0;

                for (int j = 0; j < N; j++)
                {
                    ax += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[j];
                }

                double error = Math.Abs(ax - hiVectorYN[i]);

                if (maxError < error)
                {
                    maxError = error;
                }
            }

            Console.WriteLine("max error : {0} (upper fill mode, without transpose)", maxError);

            // Test with transpose
            _gpu.CopyToDevice(hiVectorYN, diVectorYN);

            _blas.TBSV(N, 0, diMatrixA, diVectorYN, cublasOperation.T, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diVectorYN, hiVectorXN);

            // Check solution error : y - Ax by elemental wise.
            maxError = 0.0;

            for (int j = 0; j < N; j++)
            {
                double ax = 0.0;

                for (int i = 0; i < N; i++)
                {
                    ax += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[i];
                }

                double error = Math.Abs(ax - hiVectorYN[j]);

                if (maxError < error)
                {
                    maxError = error;
                }
            }

            Console.WriteLine("max error : {0} (upper fill mode, with transpose)", maxError);

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS2_TPMV()
        {
            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixAPS);
            ClearBuffer(hiVectorXN);

            FillBuffer(hiVectorXN);

            diMatrixA = _gpu.Allocate(hiMatrixAPS);
            diVectorXN = _gpu.Allocate(hiVectorXN);

            // Lower triangular banded matrix
            CreateBandedMatrix(hiMatrixANN, N, N, N - 1, 0);
            PackSymmetricMatrix(hiMatrixANN, hiMatrixAPS, N, cublasFillMode.Lower);
            _gpu.CopyToDevice(hiMatrixAPS, diMatrixA);

            // Test without transpose
            _gpu.CopyToDevice(hiVectorXN, diVectorXN);
            _blas.TPMV(N, diMatrixA, diVectorXN);
            _gpu.CopyFromDevice(diVectorXN, gpuResultN);

            for (int i = 0; i < N; i++)
            {
                double cpuResult = 0.0;

                for (int j = 0; j < N; j++)
                {
                    cpuResult += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[j];
                }

                Assert.AreEqual(cpuResult, gpuResultN[i]);
            }

            // Test with transpose
            _gpu.CopyToDevice(hiVectorXN, diVectorXN);
            _blas.TPMV(N, diMatrixA, diVectorXN, cublasOperation.T);
            _gpu.CopyFromDevice(diVectorXN, gpuResultN);

            for (int j = 0; j < N; j++)
            {
                double cpuResult = 0.0;

                for (int i = 0; i < N; i++)
                {
                    cpuResult += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[i];
                }

                Assert.AreEqual(cpuResult, gpuResultN[j]);
            }

            // Upper triangular banded matrix
            ClearBuffer(hiMatrixANN);
            CreateBandedMatrix(hiMatrixANN, N, N, 0, N - 1);
            PackSymmetricMatrix(hiMatrixANN, hiMatrixAPS, N, cublasFillMode.Upper);
            _gpu.CopyToDevice(hiMatrixAPS, diMatrixA);

            // Test without transpose
            _gpu.CopyToDevice(hiVectorXN, diVectorXN);
            _blas.TPMV(N, diMatrixA, diVectorXN, cublasOperation.N, cublasFillMode.Upper);
            _gpu.CopyFromDevice(diVectorXN, gpuResultN);

            for (int i = 0; i < N; i++)
            {
                double cpuResult = 0.0;

                for (int j = 0; j < N; j++)
                {
                    cpuResult += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[j];
                }

                Assert.AreEqual(cpuResult, gpuResultN[i]);
            }

            // Test with transpose
            _gpu.CopyToDevice(hiVectorXN, diVectorXN);
            _blas.TPMV(N, diMatrixA, diVectorXN, cublasOperation.T, cublasFillMode.Upper);
            _gpu.CopyFromDevice(diVectorXN, gpuResultN);

            for (int j = 0; j < N; j++)
            {
                double cpuResult = 0.0;

                for (int i = 0; i < N; i++)
                {
                    cpuResult += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[i];
                }

                Assert.AreEqual(cpuResult, gpuResultN[j]);
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS2_TPSV()
        {
            // Solve Ax = y, and x overwrites y.

            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiMatrixAPS);
            ClearBuffer(hiVectorXN);
            ClearBuffer(hiVectorYN);

            diMatrixA = _gpu.Allocate(hiMatrixAPS);
            diVectorYN = _gpu.Allocate(hiVectorYN);

            // Because of there is no singular test, set matrix A to simple maindiagonal only matrix.
            CreateMainDiagonalOnlyMatrix(hiMatrixANN, N);
            FillBuffer(hiVectorYN);

            // Lower fill mode
            PackSymmetricMatrix(hiMatrixANN, hiMatrixAPS, N, cublasFillMode.Lower);
            _gpu.CopyToDevice(hiMatrixAPS, diMatrixA);

            // Test without transpose
            _gpu.CopyToDevice(hiVectorYN, diVectorYN);

            _blas.TPSV(N, diMatrixA, diVectorYN);

            _gpu.CopyFromDevice(diVectorYN, hiVectorXN);

            // Check solution error : y - Ax by elemental wise.
            double maxError = 0.0;

            for (int i = 0; i < N; i++)
            {
                double ax = 0.0;

                for (int j = 0; j < N; j++)
                {
                    ax += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[j];
                }

                double error = Math.Abs(ax - hiVectorYN[i]);

                if (maxError < error)
                {
                    maxError = error;
                }
            }

            Console.WriteLine("max error : {0} (lower fill mode, without transpose)", maxError);

            // Test with transpose
            _gpu.CopyToDevice(hiVectorYN, diVectorYN);

            _blas.TPSV(N, diMatrixA, diVectorYN, cublasOperation.T);

            _gpu.CopyFromDevice(diVectorYN, hiVectorXN);

            // Check solution error : y - Ax by elemental wise.
            maxError = 0.0;

            for (int j = 0; j < N; j++)
            {
                double ax = 0.0;

                for (int i = 0; i < N; i++)
                {
                    ax += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[i];
                }

                double error = Math.Abs(ax - hiVectorYN[j]);

                if (maxError < error)
                {
                    maxError = error;
                }
            }

            Console.WriteLine("max error : {0} (lower fill mode, with transpose)", maxError);

            // Upper fill mode
            PackSymmetricMatrix(hiMatrixANN, hiMatrixAPS, N, cublasFillMode.Upper);
            _gpu.CopyToDevice(hiMatrixAPS, diMatrixA);

            // Test without transpose
            _gpu.CopyToDevice(hiVectorYN, diVectorYN);

            _blas.TPSV(N, diMatrixA, diVectorYN, cublasOperation.N, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diVectorYN, hiVectorXN);

            // Check solution error : y - Ax by elemental wise.
            maxError = 0.0;

            for (int i = 0; i < N; i++)
            {
                double ax = 0.0;

                for (int j = 0; j < N; j++)
                {
                    ax += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[j];
                }

                double error = Math.Abs(ax - hiVectorYN[i]);

                if (maxError < error)
                {
                    maxError = error;
                }
            }

            Console.WriteLine("max error : {0} (upper fill mode, without transpose)", maxError);

            // Test with transpose
            _gpu.CopyToDevice(hiVectorYN, diVectorYN);

            _blas.TPSV(N, diMatrixA, diVectorYN, cublasOperation.T, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diVectorYN, hiVectorXN);

            // Check solution error : y - Ax by elemental wise.
            maxError = 0.0;

            for (int j = 0; j < N; j++)
            {
                double ax = 0.0;

                for (int i = 0; i < N; i++)
                {
                    ax += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[i];
                }

                double error = Math.Abs(ax - hiVectorYN[j]);

                if (maxError < error)
                {
                    maxError = error;
                }
            }

            Console.WriteLine("max error : {0} (upper fill mode, with transpose)", maxError);

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS2_TRMV()
        {
            ClearBuffer(hiVectorXN);

            FillBuffer(hiVectorXN);
            diVectorXN = _gpu.Allocate(hiVectorXN);
            diMatrixA = _gpu.Allocate(hiMatrixANN);

            // Lower triangle matrix
            ClearBuffer(hiMatrixANN);
            CreateBandedMatrix(hiMatrixANN, N, N, N - 1, 0);
            _gpu.CopyToDevice(hiMatrixANN, diMatrixA);

            // Test without transpose
            _gpu.CopyToDevice(hiVectorXN, diVectorXN);

            _blas.TRMV(N, diMatrixA, diVectorXN);

            _gpu.CopyFromDevice(diVectorXN, gpuResultN);

            for (int i = 0; i < N; i++)
            {
                double cpuResult = 0.0;

                for (int j = 0; j < N; j++)
                {
                    cpuResult += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[j];
                }

                Assert.AreEqual(cpuResult, gpuResultN[i]);
            }

            // Test with transpose
            _gpu.CopyToDevice(hiVectorXN, diVectorXN);

            _blas.TRMV(N, diMatrixA, diVectorXN, cublasOperation.T);

            _gpu.CopyFromDevice(diVectorXN, gpuResultN);

            for (int j = 0; j < N; j++)
            {
                double cpuResult = 0.0;

                for (int i = 0; i < N; i++)
                {
                    cpuResult += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[i];
                }

                Assert.AreEqual(cpuResult, gpuResultN[j]);
            }

            // Upper triangle matrix
            ClearBuffer(hiMatrixANN);
            CreateBandedMatrix(hiMatrixANN, N, N, 0, N - 1);
            _gpu.CopyToDevice(hiMatrixANN, diMatrixA);

            // Test without transpose
            _gpu.CopyToDevice(hiVectorXN, diVectorXN);

            _blas.TRMV(N, diMatrixA, diVectorXN, cublasOperation.N, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diVectorXN, gpuResultN);

            for (int i = 0; i < N; i++)
            {
                double cpuResult = 0.0;

                for (int j = 0; j < N; j++)
                {
                    cpuResult += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[j];
                }

                Assert.AreEqual(cpuResult, gpuResultN[i]);
            }

            // Test with transpose
            _gpu.CopyToDevice(hiVectorXN, diVectorXN);

            _blas.TRMV(N, diMatrixA, diVectorXN, cublasOperation.T, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diVectorXN, gpuResultN);

            for (int j = 0; j < N; j++)
            {
                double cpuResult = 0.0;

                for (int i = 0; i < N; i++)
                {
                    cpuResult += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[i];
                }

                Assert.AreEqual(cpuResult, gpuResultN[j]);
            }

            _gpu.FreeAll();
        }

        [Test]
        public void Test_BLAS2_TRSV()
        {
            // Solve Ax = y, and x overwrites y.

            ClearBuffer(hiMatrixANN);
            ClearBuffer(hiVectorXN);
            ClearBuffer(hiVectorYN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixANN);
            diVectorYN = _gpu.Allocate(hiVectorYN);

            // Because of there is no singular test, set matrix A to simple maindiagonal only matrix.
            CreateMainDiagonalOnlyMatrix(hiMatrixANN, N);
            FillBuffer(hiVectorYN);

            // Lower fill mode
            // Test without transpose
            _gpu.CopyToDevice(hiVectorYN, diVectorYN);

            _blas.TRSV(N, diMatrixA, diVectorYN);

            _gpu.CopyFromDevice(diVectorYN, hiVectorXN);

            // Check solution error : y - Ax by elemental wise.
            double maxError = 0.0;

            for (int i = 0; i < N; i++)
            {
                double ax = 0.0;

                for (int j = 0; j < N; j++)
                {
                    ax += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[j];
                }

                double error = Math.Abs(ax - hiVectorYN[i]);

                if (maxError < error)
                {
                    maxError = error;
                }
            }

            Console.WriteLine("max error : {0} (lower fill mode, without transpose)", maxError);

            // Test with transpose
            _gpu.CopyToDevice(hiVectorYN, diVectorYN);

            _blas.TRSV(N, diMatrixA, diVectorYN, cublasOperation.T);

            _gpu.CopyFromDevice(diVectorYN, hiVectorXN);

            // Check solution error : y - Ax by elemental wise.
            maxError = 0.0;

            for (int j = 0; j < N; j++)
            {
                double ax = 0.0;

                for (int i = 0; i < N; i++)
                {
                    ax += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[i];
                }

                double error = Math.Abs(ax - hiVectorYN[j]);

                if (maxError < error)
                {
                    maxError = error;
                }
            }

            Console.WriteLine("max error : {0} (lower fill mode, with transpose)", maxError);

            // Upper fill mode

            // Test without transpose
            _gpu.CopyToDevice(hiVectorYN, diVectorYN);

            _blas.TRSV(N, diMatrixA, diVectorYN, cublasOperation.N, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diVectorYN, hiVectorXN);

            // Check solution error : y - Ax by elemental wise.
            maxError = 0.0;

            for (int i = 0; i < N; i++)
            {
                double ax = 0.0;

                for (int j = 0; j < N; j++)
                {
                    ax += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[j];
                }

                double error = Math.Abs(ax - hiVectorYN[i]);

                if (maxError < error)
                {
                    maxError = error;
                }
            }

            Console.WriteLine("max error : {0} (upper fill mode, without transpose)", maxError);

            // Test with transpose
            _gpu.CopyToDevice(hiVectorYN, diVectorYN);

            _blas.TRSV(N, diMatrixA, diVectorYN, cublasOperation.T, cublasFillMode.Upper);

            _gpu.CopyFromDevice(diVectorYN, hiVectorXN);

            // Check solution error : y - Ax by elemental wise.
            maxError = 0.0;

            for (int j = 0; j < N; j++)
            {
                double ax = 0.0;

                for (int i = 0; i < N; i++)
                {
                    ax += hiMatrixANN[GetIndexColumnMajor(i, j, N)] * hiVectorXN[i];
                }

                double error = Math.Abs(ax - hiVectorYN[j]);

                if (maxError < error)
                {
                    maxError = error;
                }
            }

            Console.WriteLine("max error : {0} (upper fill mode, with transpose)", maxError);

            _gpu.FreeAll();
        }
    }
}
