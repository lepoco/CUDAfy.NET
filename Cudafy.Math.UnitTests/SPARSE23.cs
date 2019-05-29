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
    public class SPARSE23 : ICudafyUnitTest
    {
        private GPGPU _gpu;

        private GPGPUSPARSE _sparse;

        // Constants
        const int M = 256;
        const int K = 128;
        const int N = 128;

        const int NNZRATIO = 10; // non-zero elements ratio. %
        const int RandomMax = 255; // all of elements are 1 <= value <= RandomMax + 1

        double Alpha = 3.0;
        double Beta = 4.0;

        // CPU Buffers
        double[] hiMatrixMN;
        double[] hiMatrixMK;
        double[] hiMatrixKN;
        double[] hiMatrixKM;
        double[] hiMatrixNN;
        double[] hiVectorXM;
        double[] hiVectorXN;
        double[] hiVectorYM;
        double[] hiVectorYN;
        double[] gpuResultM;
        double[] gpuResultN;
        double[] gpuResultMN;

        double[] diMatrixA;
        double[] diMatrixB;
        double[] diMatrixC;
        int[] diNNZRows;
        double[] diVals;
        int[] diRows;
        int[] diCols;
        double[] diVectorXM;
        double[] diVectorXN;
        double[] diVectorYM;
        double[] diVectorYN;

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.GetDevice();

            _sparse = GPGPUSPARSE.Create(_gpu);

            hiMatrixMN = new double[M * N];
            hiMatrixMK = new double[M * K];
            hiMatrixKM = new double[K * M];
            hiMatrixKN = new double[K * N];
            hiMatrixNN = new double[N * N];
            hiVectorXM = new double[M];
            hiVectorXN = new double[N];
            hiVectorYM = new double[M];
            hiVectorYN = new double[N];
            gpuResultM = new double[M];
            gpuResultN = new double[N];
            gpuResultMN = new double[M * N];
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

        private int GetIndexColumnMajor(int i, int j, int m)
        {
            return i + j * m;
        }

        private void FillBuffer(double[] buffer)
        {
            Random rand = new Random(Environment.TickCount);
            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = rand.Next(RandomMax) + 1;
            }

            System.Threading.Thread.Sleep(rand.Next(50));
        }

        private void FillBufferSparse(double[] buffer)
        {
            Random rand = new Random(Environment.TickCount);

            bool allZero = true;

            for (int i = 0; i < buffer.Length; i++)
            {
                if (rand.Next(99) + 1 < NNZRATIO)
                {
                    buffer[i] = rand.Next(RandomMax) + 1;
                    allZero = false;
                }
                else
                {
                    buffer[i] = 0.0f;
                }
            }

            if (allZero == true)
            {
                buffer[0] = 1.0;
            }

            System.Threading.Thread.Sleep(rand.Next(50));
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

        private void ClearBuffer(double[] buffer)
        {
            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = 0.0;
            }
        }

        [Test]
        public void Test_SPARSE2_CSRMV()
        {
            int nnz;

            // No transpose
            ClearBuffer(hiMatrixMN);
            ClearBuffer(hiVectorXN);
            ClearBuffer(hiVectorYM);

            FillBufferSparse(hiMatrixMN);
            FillBuffer(hiVectorXN);
            FillBuffer(hiVectorYM);

            diMatrixA = _gpu.CopyToDevice(hiMatrixMN);
            diVectorXN = _gpu.CopyToDevice(hiVectorXN);
            diVectorYM = _gpu.CopyToDevice(hiVectorYM);

            diNNZRows = _gpu.Allocate<int>(M);
            nnz = _sparse.NNZ(M, N, diMatrixA, diNNZRows);
            diVals = _gpu.Allocate<double>(nnz);
            diRows = _gpu.Allocate<int>(M + 1);
            diCols = _gpu.Allocate<int>(nnz);

            _sparse.Dense2CSR(M, N, diMatrixA, diNNZRows, diVals, diRows, diCols);

            _sparse.CSRMV(M, N, nnz, ref Alpha, diVals, diRows, diCols, diVectorXN, ref Beta, diVectorYM);

            _gpu.CopyFromDevice(diVectorYM, gpuResultM);

            for (int i = 0; i < M; i++)
            {
                double cpuResult = 0.0;
                for (int j = 0; j < N; j++)
                {
                    cpuResult += Alpha * hiMatrixMN[GetIndexColumnMajor(i, j, M)] * hiVectorXN[j];
                }

                cpuResult += Beta * hiVectorYM[i];

                Assert.AreEqual(cpuResult, gpuResultM[i]);
            }

            _gpu.FreeAll();

            // Transpose
            ClearBuffer(hiMatrixMN);
            ClearBuffer(hiVectorXM);
            ClearBuffer(hiVectorYN);

            FillBufferSparse(hiMatrixMN);
            FillBuffer(hiVectorXM);
            FillBuffer(hiVectorYN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixMN);
            diVectorXM = _gpu.CopyToDevice(hiVectorXM);
            diVectorYN = _gpu.CopyToDevice(hiVectorYN);

            diNNZRows = _gpu.Allocate<int>(M);
            nnz = _sparse.NNZ(M, N, diMatrixA, diNNZRows);
            diVals = _gpu.Allocate<double>(nnz);
            diRows = _gpu.Allocate<int>(M + 1);
            diCols = _gpu.Allocate<int>(nnz);

            _sparse.Dense2CSR(M, N, diMatrixA, diNNZRows, diVals, diRows, diCols);

            _sparse.CSRMV(M, N, nnz, ref Alpha, diVals, diRows, diCols, diVectorXM, ref Beta, diVectorYN, SPARSE.cusparseOperation.Transpose);

            _gpu.CopyFromDevice(diVectorYN, gpuResultN);

            for (int j = 0; j < N; j++)
            {
                double cpuResult = 0.0;
                for (int i = 0; i < M; i++)
                {
                    cpuResult += Alpha * hiMatrixMN[GetIndexColumnMajor(i, j, M)] * hiVectorXM[i];
                }

                cpuResult += Beta * hiVectorYN[j];

                Assert.AreEqual(cpuResult, gpuResultN[j]);
            }

            _gpu.FreeAll();
        }

        //[Test]
        //public void Test_SPARSE2_CSRSV()
        //{
        //    int nnz;
        //    double maxError;

        //    cusparseSolveAnalysisInfo info = new cusparseSolveAnalysisInfo();
        //    cusparseSolveAnalysisInfo infoTrans = new cusparseSolveAnalysisInfo();
        //    _sparse.CreateSolveAnalysisInfo(ref info);
        //    _sparse.CreateSolveAnalysisInfo(ref infoTrans);
        //    cusparseMatDescr desc = new cusparseMatDescr();
        //    desc.MatrixType = cusparseMatrixType.Triangular;

        //    // No transpose
        //    ClearBuffer(hiMatrixNN);
        //    ClearBuffer(hiVectorXN);
        //    ClearBuffer(hiVectorYN);

        //    CreateMainDiagonalOnlyMatrix(hiMatrixNN, N);
        //    FillBuffer(hiVectorXN);
        //    FillBuffer(hiVectorYN);

        //    diMatrixA = _gpu.CopyToDevice(hiMatrixNN);
        //    diVectorXN = _gpu.CopyToDevice(hiVectorXN);
        //    diVectorYN = _gpu.CopyToDevice(hiVectorYN);

        //    diNNZRows = _gpu.Allocate<int>(N);
        //    nnz = _sparse.NNZ(N, N, diMatrixA, diNNZRows);
        //    diVals = _gpu.Allocate<double>(nnz);
        //    diRows = _gpu.Allocate<int>(N + 1);
        //    diCols = _gpu.Allocate<int>(nnz);

        //    _sparse.Dense2CSR(N, N, diMatrixA, diNNZRows, diVals, diRows, diCols);

        //    _sparse.CSRSV_ANALYSIS(N, nnz, diVals, diRows, diCols, cusparseOperation.NonTranspose, info, desc);

        //    _sparse.CSRSV_SOLVE(N, ref Alpha, diVals, diRows, diCols, diVectorXN, diVectorYN, cusparseOperation.NonTranspose, info, desc);

        //    _gpu.CopyFromDevice(diVectorYN, gpuResultN);

        //    maxError = 0.0;

        //    for (int i = 0; i < N; i++)
        //    {
        //        double cpuResult = 0.0;

        //        for (int j = 0; j < N; j++)
        //        {
        //            cpuResult += hiMatrixNN[GetIndexColumnMajor(i, j, N)] * gpuResultN[j];
        //        }

        //        double error = Math.Abs(cpuResult - Alpha * hiVectorXN[i]);

        //        if (maxError < error)
        //        {
        //            maxError = error;
        //        }
        //    }

        //    Console.WriteLine("max error : {0} (No transpose)", maxError);

        //    _gpu.FreeAll();

        //    // Transpose
        //    ClearBuffer(hiMatrixNN);
        //    ClearBuffer(hiVectorXN);
        //    ClearBuffer(hiVectorYN);

        //    CreateMainDiagonalOnlyMatrix(hiMatrixNN, N);
        //    FillBuffer(hiVectorXN);
        //    FillBuffer(hiVectorYN);

        //    diMatrixA = _gpu.CopyToDevice(hiMatrixNN);
        //    diVectorXN = _gpu.CopyToDevice(hiVectorXN);
        //    diVectorYN = _gpu.CopyToDevice(hiVectorYN);

        //    diNNZRows = _gpu.Allocate<int>(N);
        //    nnz = _sparse.NNZ(N, N, diMatrixA, diNNZRows);
        //    diVals = _gpu.Allocate<double>(nnz);
        //    diRows = _gpu.Allocate<int>(N + 1);
        //    diCols = _gpu.Allocate<int>(nnz);

        //    _sparse.Dense2CSR(N, N, diMatrixA, diNNZRows, diVals, diRows, diCols);

        //    _sparse.CSRSV_ANALYSIS(N, nnz, diVals, diRows, diCols, cusparseOperation.Transpose, infoTrans, desc);

        //    _sparse.CSRSV_SOLVE(N, ref Alpha, diVals, diRows, diCols, diVectorXN, diVectorYN, cusparseOperation.Transpose, infoTrans, desc);

        //    _gpu.CopyFromDevice(diVectorYN, gpuResultN);

        //    maxError = 0.0;

        //    for (int i = 0; i < N; i++)
        //    {
        //        double cpuResult = 0.0;

        //        for (int j = 0; j < N; j++)
        //        {
        //            cpuResult += hiMatrixNN[GetIndexColumnMajor(j, i, N)] * gpuResultN[j];
        //        }

        //        double error = Math.Abs(cpuResult - Alpha * hiVectorXN[i]);

        //        if (maxError < error)
        //        {
        //            maxError = error;
        //        }
        //    }

        //    Console.WriteLine("max error : {0} (Transpose)", maxError);

        //    _gpu.FreeAll();
        //}

        [Test]
        public void Test_SPARSE3_CSRMM()
        {
            int nnz;

            // No transpose
            ClearBuffer(hiMatrixMK);
            ClearBuffer(hiMatrixKN);
            ClearBuffer(hiMatrixMN);

            FillBufferSparse(hiMatrixMK);
            FillBuffer(hiMatrixKN);
            FillBuffer(hiMatrixMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixMK);
            diMatrixB = _gpu.CopyToDevice(hiMatrixKN);
            diMatrixC = _gpu.CopyToDevice(hiMatrixMN);

            diNNZRows = _gpu.Allocate<int>(M);
            nnz = _sparse.NNZ(M, K, diMatrixA, diNNZRows);
            diVals = _gpu.Allocate<double>(nnz);
            diRows = _gpu.Allocate<int>(M + 1);
            diCols = _gpu.Allocate<int>(nnz);

            _sparse.Dense2CSR(M, K, diMatrixA, diNNZRows, diVals, diRows, diCols);

            _sparse.CSRMM(M, K, N, nnz, ref Alpha, diVals, diRows, diCols, diMatrixB, ref Beta, diMatrixC);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += Alpha * hiMatrixMK[GetIndexColumnMajor(i, k, M)] * hiMatrixKN[GetIndexColumnMajor(k, j, K)];
                    }

                    cpuResult += Beta * hiMatrixMN[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();

            // Transpose
            ClearBuffer(hiMatrixKM);
            ClearBuffer(hiMatrixKN);
            ClearBuffer(hiMatrixMN);

            FillBufferSparse(hiMatrixKM);
            FillBuffer(hiMatrixKN);
            FillBuffer(hiMatrixMN);

            diMatrixA = _gpu.CopyToDevice(hiMatrixKM);
            diMatrixB = _gpu.CopyToDevice(hiMatrixKN);
            diMatrixC = _gpu.CopyToDevice(hiMatrixMN);

            diNNZRows = _gpu.Allocate<int>(K);
            nnz = _sparse.NNZ(K, M, diMatrixA, diNNZRows);
            diVals = _gpu.Allocate<double>(nnz);
            diRows = _gpu.Allocate<int>(K + 1);
            diCols = _gpu.Allocate<int>(nnz);

            _sparse.Dense2CSR(K, M, diMatrixA, diNNZRows, diVals, diRows, diCols);

            _sparse.CSRMM(K, M, N, nnz, ref Alpha, diVals, diRows, diCols, diMatrixB, ref Beta, diMatrixC, cusparseOperation.Transpose);

            _gpu.CopyFromDevice(diMatrixC, gpuResultMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double cpuResult = 0.0;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += Alpha * hiMatrixKM[GetIndexColumnMajor(k, i, K)] * hiMatrixKN[GetIndexColumnMajor(k, j, K)];
                    }

                    cpuResult += Beta * hiMatrixMN[GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, gpuResultMN[GetIndexColumnMajor(i, j, M)]);
                }
            }

            _gpu.FreeAll();
        }
    }
}
