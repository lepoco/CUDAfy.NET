/* Added by Kichang Kim (kkc0923@hotmail.com) */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Diagnostics;
using Cudafy.Types;
using Cudafy.Host;

using GASS.CUDA.Types;
using GASS.CUDA;

namespace Cudafy.Maths.SPARSE
{
    internal class CudaSPARSE : GPGPUSPARSE
    {
        private GPGPU _gpu;
        private cusparseHandle _sparse;
        private ICUSPARSEDriver _driver;
        private CUSPARSEStatus _status;

        private CUSPARSEStatus LastStatus
        {
            get { return _status; }
            set
            {
                _status = value;
                if (_status != CUSPARSEStatus.Success)
                    throw new CudafyMathException("SPARSE Error : {0}", _status.ToString());
            }
        }

        internal CudaSPARSE(GPGPU gpu)
            : base()
        {
            if (IntPtr.Size == 8)
            {
                _driver = new CUSPARSEDriver64();
            }
            else
            {
                throw new NotSupportedException();
            }

            LastStatus = _driver.CusparseCreate(ref _sparse);
            _gpu = gpu;
        }

        protected override void Shutdown()
        {
            try
            {
                LastStatus = _driver.CusparseDestroy(_sparse);
            }
            catch(DllNotFoundException ex)
            {
                Debug.WriteLine(ex.Message);
            }
        }

        public override string GetVersionInfo()
        {
            int version = 0;
            _driver.CusparseGetVersion(_sparse, ref version);
            return string.Format("CUSPARSE Version : {0}", version);
        }

        private static eDataType GetDataType<T>()
        {
            eDataType type;
            Type t = typeof(T);
            if (t == typeof(Double))
                type = eDataType.D;
            else if (t == typeof(Single))
                type = eDataType.S;
            else if (t == typeof(ComplexD))
                type = eDataType.Z;
            else if (t == typeof(ComplexF))
                type = eDataType.C;
            else
                throw new CudafyMathException(CudafyHostException.csX_NOT_SUPPORTED, typeof(T).Name);
            return type;
        }

        private CUdeviceptr GetDeviceMemory(object vector, ref int n)
        {
            CUDevicePtrEx ptrEx = _gpu.GetDeviceMemory(vector) as CUDevicePtrEx;
            n = (n == 0 ? ptrEx.TotalSize : n);

            return ptrEx.DevPtr;
        }

        private CUdeviceptr GetDeviceMemory(object vector)
        {
            CUDevicePtrEx ptrEx = _gpu.GetDeviceMemory(vector) as CUDevicePtrEx;
            return ptrEx.DevPtr;
        }

        #region CUSPARSE Helper Functions
        public override void CreateSolveAnalysisInfo(ref cusparseSolveAnalysisInfo info)
        {
            LastStatus = _driver.CusparseCreateSolveAnalysisInfo(ref info);
        }

        public override void DestroySolveAnalysisInfo(cusparseSolveAnalysisInfo info)
        {
            LastStatus = _driver.CusparseDestroySolveAnalysisInfo(info);
        }
        #endregion

        #region Formation Conversion Functions
        #region NNZ
        public override int NNZ(int m, int n, float[] A, int[] vector, cusparseMatDescr descrA, cusparseDirection dirA = cusparseDirection.Row, int lda = 0)
        {
            lda = (lda == 0 ? m : lda);

            CUdeviceptr ptra = GetDeviceMemory(A);
            CUdeviceptr ptrv = GetDeviceMemory(vector);
            
            int nzz = 0;

            LastStatus = _driver.CusparseSnnz(_sparse, dirA, m, n, descrA, ptra.Pointer, lda, ptrv.Pointer, ref nzz);

            return nzz;
        }

        public override int NNZ(int m, int n, double[] A, int[] vector, cusparseMatDescr descrA, cusparseDirection dirA = cusparseDirection.Row, int lda = 0)
        {
            lda = (lda == 0 ? m : lda);

            CUdeviceptr ptra = GetDeviceMemory(A);
            CUdeviceptr ptrv = GetDeviceMemory(vector);

            int nzz = 0;

            LastStatus = _driver.CusparseDnnz(_sparse, dirA, m, n, descrA, ptra.Pointer, lda, ptrv.Pointer, ref nzz);

            return nzz;
        }
        #endregion

        #region DENSE2CSR
        public override void Dense2CSR(int m, int n, float[] A, int[] nnzPerRow, float[] csrValA, int[] csrRow, int[] csrColIndA, cusparseMatDescr descrA, int lda = 0)
        {
            lda = (lda == 0 ? m : lda);

            CUdeviceptr ptra = GetDeviceMemory(A);
            CUdeviceptr ptrn = GetDeviceMemory(nnzPerRow);
            CUdeviceptr ptrval = GetDeviceMemory(csrValA);
            CUdeviceptr ptrrow = GetDeviceMemory(csrRow);
            CUdeviceptr ptrcol = GetDeviceMemory(csrColIndA);

            LastStatus = _driver.CusparseSdense2csr(_sparse, m, n, descrA, ptra.Pointer, lda, ptrn.Pointer, ptrval.Pointer, ptrrow.Pointer, ptrcol.Pointer);
        }

        public override void Dense2CSR(int m, int n, double[] A, int[] nnzPerRow, double[] csrValA, int[] csrRow, int[] csrColIndA, cusparseMatDescr descrA, int lda = 0)
        {
            lda = (lda == 0 ? m : lda);

            CUdeviceptr ptra = GetDeviceMemory(A);
            CUdeviceptr ptrn = GetDeviceMemory(nnzPerRow);
            CUdeviceptr ptrval = GetDeviceMemory(csrValA);
            CUdeviceptr ptrrow = GetDeviceMemory(csrRow);
            CUdeviceptr ptrcol = GetDeviceMemory(csrColIndA);

            LastStatus = _driver.CusparseDdense2csr(_sparse, m, n, descrA, ptra.Pointer, lda, ptrn.Pointer, ptrval.Pointer, ptrrow.Pointer, ptrcol.Pointer);
        }
        #endregion

        #region CSR2DENSE
        public override void CSR2Dense(int m, int n, float[] csrValA, int[] csrRowA, int[] csrColA, float[] A, cusparseMatDescr descrA, int lda = 0)
        {
            lda = (lda == 0 ? m : lda);

            CUdeviceptr ptra = GetDeviceMemory(A);
            CUdeviceptr ptrval = GetDeviceMemory(csrValA);
            CUdeviceptr ptrrow = GetDeviceMemory(csrRowA);
            CUdeviceptr ptrcol = GetDeviceMemory(csrColA);

            LastStatus = _driver.CusparseScsr2dense(_sparse, m, n, descrA, ptrval.Pointer, ptrrow.Pointer, ptrcol.Pointer, ptra.Pointer, lda);
        }

        public override void CSR2Dense(int m, int n, double[] csrValA, int[] csrRowA, int[] csrColA, double[] A, cusparseMatDescr descrA, int lda = 0)
        {
            lda = (lda == 0 ? m : lda);

            CUdeviceptr ptra = GetDeviceMemory(A);
            CUdeviceptr ptrval = GetDeviceMemory(csrValA);
            CUdeviceptr ptrrow = GetDeviceMemory(csrRowA);
            CUdeviceptr ptrcol = GetDeviceMemory(csrColA);

            LastStatus = _driver.CusparseDcsr2dense(_sparse, m, n, descrA, ptrval.Pointer, ptrrow.Pointer, ptrcol.Pointer, ptra.Pointer, lda);
        }
        #endregion

        #region DENSE2CSC
        public override void Dense2CSC(int m, int n, float[] A, int[] nnzPerCol, float[] cscValA, int[] cscRowIndA, int[] cscColA, cusparseMatDescr descrA, int lda = 0)
        {
            lda = (lda == 0 ? m : lda);

            CUdeviceptr ptra = GetDeviceMemory(A);
            CUdeviceptr ptrn = GetDeviceMemory(nnzPerCol);
            CUdeviceptr ptrval = GetDeviceMemory(cscValA);
            CUdeviceptr ptrrow = GetDeviceMemory(cscRowIndA);
            CUdeviceptr ptrcol = GetDeviceMemory(cscColA);

            LastStatus = _driver.CusparseSdense2csc(_sparse, m, n, descrA, ptra.Pointer, lda, ptrn.Pointer, ptrval.Pointer, ptrrow.Pointer, ptrcol.Pointer);
        }

        public override void Dense2CSC(int m, int n, double[] A, int[] nnzPerCol, double[] cscValA, int[] cscRowIndA, int[] cscColA, cusparseMatDescr descrA, int lda = 0)
        {
            lda = (lda == 0 ? m : lda);

            CUdeviceptr ptra = GetDeviceMemory(A);
            CUdeviceptr ptrn = GetDeviceMemory(nnzPerCol);
            CUdeviceptr ptrval = GetDeviceMemory(cscValA);
            CUdeviceptr ptrrow = GetDeviceMemory(cscRowIndA);
            CUdeviceptr ptrcol = GetDeviceMemory(cscColA);

            LastStatus = _driver.CusparseDdense2csc(_sparse, m, n, descrA, ptra.Pointer, lda, ptrn.Pointer, ptrval.Pointer, ptrrow.Pointer, ptrcol.Pointer);
        }
        #endregion

        #region CSC2DENSE
        public override void CSC2Dense(int m, int n, float[] cscValA, int[] cscRowA, int[] cscColA, float[] A, cusparseMatDescr descrA, int lda = 0)
        {
            lda = (lda == 0 ? m : lda);

            CUdeviceptr ptra = GetDeviceMemory(A);
            CUdeviceptr ptrval = GetDeviceMemory(cscValA);
            CUdeviceptr ptrrow = GetDeviceMemory(cscRowA);
            CUdeviceptr ptrcol = GetDeviceMemory(cscColA);

            LastStatus = _driver.CusparseScsc2dense(_sparse, m, n, descrA, ptrval.Pointer, ptrrow.Pointer, ptrcol.Pointer, ptra.Pointer, lda);
        }

        public override void CSC2Dense(int m, int n, double[] cscValA, int[] cscRowA, int[] cscColA, double[] A, cusparseMatDescr descrA, int lda = 0)
        {
            lda = (lda == 0 ? m : lda);

            CUdeviceptr ptra = GetDeviceMemory(A);
            CUdeviceptr ptrval = GetDeviceMemory(cscValA);
            CUdeviceptr ptrrow = GetDeviceMemory(cscRowA);
            CUdeviceptr ptrcol = GetDeviceMemory(cscColA);

            LastStatus = _driver.CusparseDcsc2dense(_sparse, m, n, descrA, ptrval.Pointer, ptrrow.Pointer, ptrcol.Pointer, ptra.Pointer, lda);
        }
        #endregion

        #region CSR2CSC
        public override void CSR2CSC(int m, int n, int nnz, float[] csrVal, int[] csrRow, int[] csrCol, float[] cscVal, int[] cscRow, int[] cscCol, cusparseAction copyValues = cusparseAction.Numeric, cusparseIndexBase bs = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrcsrv = GetDeviceMemory(csrVal);
            CUdeviceptr ptrcsrr = GetDeviceMemory(csrRow);
            CUdeviceptr ptrcsrc = GetDeviceMemory(csrCol);

            CUdeviceptr ptrcscv = GetDeviceMemory(cscVal);
            CUdeviceptr ptrcscr = GetDeviceMemory(cscRow);
            CUdeviceptr ptrcscc = GetDeviceMemory(cscCol);

            LastStatus = _driver.CusparseScsr2csc(_sparse, m, n,nnz, ptrcsrv.Pointer, ptrcsrr.Pointer, ptrcsrc.Pointer, ptrcscv.Pointer, ptrcscr.Pointer, ptrcscc.Pointer, copyValues, bs);
        }

        public override void CSR2CSC(int m, int n, int nnz, double[] csrVal, int[] csrRow, int[] csrCol, double[] cscVal, int[] cscRow, int[] cscCol, cusparseAction copyValues = cusparseAction.Numeric, cusparseIndexBase bs = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrcsrv = GetDeviceMemory(csrVal);
            CUdeviceptr ptrcsrr = GetDeviceMemory(csrRow);
            CUdeviceptr ptrcsrc = GetDeviceMemory(csrCol);

            CUdeviceptr ptrcscv = GetDeviceMemory(cscVal);
            CUdeviceptr ptrcscr = GetDeviceMemory(cscRow);
            CUdeviceptr ptrcscc = GetDeviceMemory(cscCol);

            LastStatus = _driver.CusparseDcsr2csc(_sparse, m, n, nnz, ptrcsrv.Pointer, ptrcsrr.Pointer, ptrcsrc.Pointer, ptrcscv.Pointer, ptrcscr.Pointer, ptrcscc.Pointer, copyValues, bs);
        }
        #endregion

        #region COO2CSR
        public override void COO2CSR(int nnz, int m, int[] cooRow, int[] csrRow, cusparseIndexBase idxBase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrcsr = GetDeviceMemory(csrRow);
            CUdeviceptr ptrcoo = GetDeviceMemory(cooRow);

            LastStatus = _driver.CusparseXcoo2csr(_sparse, ptrcoo.Pointer, nnz, m, ptrcsr.Pointer, idxBase);
        }
        #endregion

        #region CSR2COO
        public override void CSR2COO(int nnz, int m, int[] csrRow, int[] cooRow, cusparseIndexBase idxBase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrcsr = GetDeviceMemory(csrRow);
            CUdeviceptr ptrcoo = GetDeviceMemory(cooRow);

            LastStatus = _driver.CusparseXcsr2coo(_sparse, ptrcsr.Pointer, nnz, m, ptrcoo.Pointer, idxBase);
        }
        #endregion
        #endregion

        #region SPARSE Level 1

        #region AXPY
        public override void AXPY(ref float alpha, float[] vectorx, int[] indexx, float[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseSaxpyi(_sparse, n, ref alpha, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, ibase);
        }
        public override void AXPY(ref double alpha, double[] vectorx, int[] indexx, double[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseDaxpyi(_sparse, n, ref alpha, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, ibase);
        }
        #endregion

        #region DOT
        public override float DOT(float[] vectorx, int[] indexx, float[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            float result = 0;

            LastStatus = _driver.CusparseSdoti(_sparse, n, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, ref result, ibase);
            return result;
        }
        public override double DOT(double[] vectorx, int[] indexx, double[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            double result = 0;

            LastStatus = _driver.CusparseDdoti(_sparse, n, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, ref result, ibase);
            return result;
        }
        #endregion

        #region GTHR
        public override void GTHR(float[] vectory, float[] vectorx, int[] indexx, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseSgthr(_sparse, n, ptry.Pointer, ptrx.Pointer, ptrix.Pointer, ibase);
        }
        public override void GTHR(double[] vectory, double[] vectorx, int[] indexx, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseDgthr(_sparse, n, ptry.Pointer, ptrx.Pointer, ptrix.Pointer, ibase);
        }
        #endregion

        #region GTHRZ
        public override void GTHRZ(float[] vectory, float[] vectorx, int[] indexx, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseSgthrz(_sparse, n, ptry.Pointer, ptrx.Pointer, ptrix.Pointer, ibase);
        }
        public override void GTHRZ(double[] vectory, double[] vectorx, int[] indexx, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseDgthrz(_sparse, n, ptry.Pointer, ptrx.Pointer, ptrix.Pointer, ibase);
        }
        #endregion

        #region ROT
        public override void ROT(float[] vectorx, int[] indexx, float[] vectory, ref float c, ref float s, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseSroti(_sparse, n, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, ref c, ref s, ibase);
        }
        public override void ROT(double[] vectorx, int[] indexx, double[] vectory, ref double c, ref double s, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseDroti(_sparse, n, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, ref c, ref s, ibase);
        }
        #endregion

        #region SCTR
        public override void SCTR(float[] vectorx, int[] indexx, float[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseSsctr(_sparse, n, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, ibase);
        }
        public override void SCTR(double[] vectorx, int[] indexx, double[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseDsctr(_sparse, n, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, ibase);
        }
        #endregion
        #endregion

        #region SPARSE Level 2
        #region CSRMV
        public override void CSRMV(int m, int n, int nnz, ref float alpha, float[] csrValA, int[] csrRowA, int[] csrColA, float[] x, ref float beta, float[] y, cusparseMatDescr descrA, cusparseOperation op = cusparseOperation.NonTranspose)
        {
            CUdeviceptr ptrcsrv = GetDeviceMemory(csrValA);
            CUdeviceptr ptrcsrr = GetDeviceMemory(csrRowA);
            CUdeviceptr ptrcsrc = GetDeviceMemory(csrColA);
            CUdeviceptr ptrx = GetDeviceMemory(x);
            CUdeviceptr ptry = GetDeviceMemory(y);

            LastStatus = _driver.CusparseScsrmv(_sparse, op, m, n, nnz, ref alpha, descrA, ptrcsrv.Pointer, ptrcsrr.Pointer, ptrcsrc.Pointer, ptrx.Pointer, ref beta, ptry.Pointer);
        }

        public override void CSRMV(int m, int n, int nnz, ref double alpha, double[] csrValA, int[] csrRowA, int[] csrColA, double[] x, ref double beta, double[] y, cusparseMatDescr descrA, cusparseOperation op = cusparseOperation.NonTranspose)
        {
            CUdeviceptr ptrcsrv = GetDeviceMemory(csrValA);
            CUdeviceptr ptrcsrr = GetDeviceMemory(csrRowA);
            CUdeviceptr ptrcsrc = GetDeviceMemory(csrColA);
            CUdeviceptr ptrx = GetDeviceMemory(x);
            CUdeviceptr ptry = GetDeviceMemory(y);

            LastStatus = _driver.CusparseDcsrmv(_sparse, op, m, n, nnz, ref alpha, descrA, ptrcsrv.Pointer, ptrcsrr.Pointer, ptrcsrc.Pointer, ptrx.Pointer, ref beta, ptry.Pointer);
        }
        #endregion

        #region CSRSV_ANALYSIS
        public override void CSRSV_ANALYSIS(int m, int nnz, float[] csrValA, int[] csrRowA, int[] csrColA, cusparseOperation op, cusparseSolveAnalysisInfo info, cusparseMatDescr descrA)
        {
            CUdeviceptr ptrcsrv = GetDeviceMemory(csrValA);
            CUdeviceptr ptrcsrr = GetDeviceMemory(csrRowA);
            CUdeviceptr ptrcsrc = GetDeviceMemory(csrColA);

            LastStatus = _driver.CusparseScsrsv_analysis(_sparse, op, m, nnz, descrA, ptrcsrv.Pointer, ptrcsrr.Pointer, ptrcsrc.Pointer, info);
        }

        public override void CSRSV_ANALYSIS(int m, int nnz, double[] csrValA, int[] csrRowA, int[] csrColA, cusparseOperation op, cusparseSolveAnalysisInfo info, cusparseMatDescr descrA)
        {
            CUdeviceptr ptrcsrv = GetDeviceMemory(csrValA);
            CUdeviceptr ptrcsrr = GetDeviceMemory(csrRowA);
            CUdeviceptr ptrcsrc = GetDeviceMemory(csrColA);

            LastStatus = _driver.CusparseDcsrsv_analysis(_sparse, op, m, nnz, descrA, ptrcsrv.Pointer, ptrcsrr.Pointer, ptrcsrc.Pointer, info);
        }
        #endregion

        #region CSRSV_SOLVE
        public override void CSRSV_SOLVE(int m, ref float alpha, float[] csrValA, int[] csrRowA, int[] csrColA, float[] x, float[] y, cusparseOperation op, cusparseSolveAnalysisInfo info, cusparseMatDescr descrA)
        {
            CUdeviceptr ptrcsrv = GetDeviceMemory(csrValA);
            CUdeviceptr ptrcsrr = GetDeviceMemory(csrRowA);
            CUdeviceptr ptrcsrc = GetDeviceMemory(csrColA);
            CUdeviceptr ptrx = GetDeviceMemory(x);
            CUdeviceptr ptry = GetDeviceMemory(y);

            LastStatus = _driver.CusparseScsrsv_solve(_sparse, op, m, ref alpha, descrA, ptrcsrv.Pointer, ptrcsrr.Pointer, ptrcsrc.Pointer, info, ptrx.Pointer, ptry.Pointer);
        }

        public override void CSRSV_SOLVE(int m, ref double alpha, double[] csrValA, int[] csrRowA, int[] csrColA, double[] x, double[] y, cusparseOperation op, cusparseSolveAnalysisInfo info, cusparseMatDescr descrA)
        {
            CUdeviceptr ptrcsrv = GetDeviceMemory(csrValA);
            CUdeviceptr ptrcsrr = GetDeviceMemory(csrRowA);
            CUdeviceptr ptrcsrc = GetDeviceMemory(csrColA);
            CUdeviceptr ptrx = GetDeviceMemory(x);
            CUdeviceptr ptry = GetDeviceMemory(y);

            LastStatus = _driver.CusparseDcsrsv_solve(_sparse, op, m, ref alpha, descrA, ptrcsrv.Pointer, ptrcsrr.Pointer, ptrcsrc.Pointer, info, ptrx.Pointer, ptry.Pointer);
        }
        #endregion
        #endregion

        #region SPARSE Level 3
        #region CSRMM
        public override void CSRMM(int m, int k, int n, int nnz, ref float alpha, float[] csrValA, int[] csrRowA, int[] csrColA, float[] B, ref float beta, float[] C, cusparseMatDescr descrA, cusparseOperation op = cusparseOperation.NonTranspose, int ldb = 0, int ldc = 0)
        {
            if (op == cusparseOperation.NonTranspose)
            {
                ldb = (ldb == 0 ? k : ldb);
                ldc = (ldc == 0 ? m : ldc);
            }
            else
            {
                ldb = (ldb == 0 ? m : ldb);
                ldc = (ldc == 0 ? k : ldc);
            }

            CUdeviceptr ptrcsrv = GetDeviceMemory(csrValA);
            CUdeviceptr ptrcsrr = GetDeviceMemory(csrRowA);
            CUdeviceptr ptrcsrc = GetDeviceMemory(csrColA);

            CUdeviceptr ptrb = GetDeviceMemory(B);
            CUdeviceptr ptrc = GetDeviceMemory(C);

            LastStatus = _driver.CusparseScsrmm(_sparse, op, m, n, k, nnz, ref alpha, descrA, ptrcsrv.Pointer, ptrcsrr.Pointer, ptrcsrc.Pointer, ptrb.Pointer, ldb, ref beta, ptrc.Pointer, ldc); 
        }

        public override void CSRMM(int m, int k, int n, int nnz, ref double alpha, double[] csrValA, int[] csrRowA, int[] csrColA, double[] B, ref double beta, double[] C, cusparseMatDescr descrA, cusparseOperation op = cusparseOperation.NonTranspose, int ldb = 0, int ldc = 0)
        {
            if (op == cusparseOperation.NonTranspose)
            {
                ldb = (ldb == 0 ? k : ldb);
                ldc = (ldc == 0 ? m : ldc);
            }
            else
            {
                ldb = (ldb == 0 ? m : ldb);
                ldc = (ldc == 0 ? k : ldc);
            }

            CUdeviceptr ptrcsrv = GetDeviceMemory(csrValA);
            CUdeviceptr ptrcsrr = GetDeviceMemory(csrRowA);
            CUdeviceptr ptrcsrc = GetDeviceMemory(csrColA);

            CUdeviceptr ptrb = GetDeviceMemory(B);
            CUdeviceptr ptrc = GetDeviceMemory(C);

            LastStatus = _driver.CusparseDcsrmm(_sparse, op, m, n, k, nnz, ref alpha, descrA, ptrcsrv.Pointer, ptrcsrr.Pointer, ptrcsrc.Pointer, ptrb.Pointer, ldb, ref beta, ptrc.Pointer, ldc); 
        }
        #endregion
        #endregion
    }
}
