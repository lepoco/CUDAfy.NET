/* Added by Kichang Kim (kkc0923@hotmail.com) */
namespace Cudafy.Maths.SPARSE
{
    using GASS.CUDA;
    using GASS.CUDA.Types;
    using System;
    using System.Runtime.InteropServices;

    public class CUSPARSEDriver64 : ICUSPARSEDriver
    {
#if LINUX
        internal const string CUSPARSE_DLL_NAME = "libcusparse";
#else
        internal const string CUSPARSE_DLL_NAME = "cusparse64_70";
#endif

        //[DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        //public static extern IntPtr LoadLibrary(string lpFileName);

        #region Native Functions : Help Functions
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseCreate(ref cusparseHandle handle);

        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDestroy(cusparseHandle handle);

        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseGetVersion(cusparseHandle handle, ref int version);

        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseSetStream(cusparseHandle handle, cudaStream streamId);

        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseCreateMatDescr(ref cusparseMatDescr descrA);

        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDestroyMatDescr(cusparseMatDescr descrA);

        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseSetMatType(cusparseMatDescr descrA, cusparseMatrixType type);

        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern cusparseMatrixType cusparseGetMatType(cusparseMatDescr descrA);

        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseSetMatFillMode(cusparseMatDescr descrA, cusparseFillMode fillMode);

        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern cusparseFillMode cusparseGetMatFillMode(cusparseMatDescr descrA);

        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseSetMatDiagType(cusparseMatDescr descrA, cusparseDiagType diagType);

        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern cusparseDiagType cusparseGetMatDiagType(cusparseMatDescr descrA);

        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseSetMatIndexBase(cusparseMatDescr descrA, cusparseIndexBase ibase);

        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern cusparseIndexBase cusparseGetMatIndexBase(cusparseMatDescr descrA);

        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseCreateSolveAnalysisInfo(ref cusparseSolveAnalysisInfo info);

        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo info);
        #endregion

        #region Native Functions : Format Conversion Functions
        #region NNZ
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseSnnz(cusparseHandle handle, cusparseDirection dirA, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzperVector, ref int nnzHostPtr);
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDnnz(cusparseHandle handle, cusparseDirection dirA, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzperVector, ref int nnzHostPtr);
        #endregion

        #region DENSE2CSR
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseSdense2csr(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzPerRow, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA);
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDdense2csr(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzPerRow, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA);
        #endregion

        #region CSR2DENSE
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseScsr2dense(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr A, int lda);
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDcsr2dense(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr A, int lda);
        #endregion

        #region DENSE2CSC
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseSdense2csc(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzPerCol, IntPtr cscValA, IntPtr cscRowIndA, IntPtr cscColPtrA);
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDdense2csc(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzPerCol, IntPtr cscValA, IntPtr cscRowIndA, IntPtr cscColPtrA);
        #endregion

        #region CSC2DENSE
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseScsc2dense(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr cscValA, IntPtr cscRowIndA, IntPtr cscColPtrA, IntPtr A, int lda);
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDcsc2dense(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr cscValA, IntPtr cscRowIndA, IntPtr cscColPtrA, IntPtr A, int lda);
        #endregion

        #region CSR2CSC
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseScsr2csc_v2(cusparseHandle handle, int m, int n, int nnz, IntPtr csrVal, IntPtr csrRowPtr, IntPtr csrColInd, IntPtr cscVal, IntPtr cscRowInd, IntPtr cscColPtr, cusparseAction copyvalues, cusparseIndexBase bs);
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDcsr2csc_v2(cusparseHandle handle, int m, int n, int nnz, IntPtr csrVal, IntPtr csrRowPtr, IntPtr csrColInd, IntPtr cscVal, IntPtr cscRowInd, IntPtr cscColPtr, cusparseAction copyvalues, cusparseIndexBase bs);
        #endregion

        #region COO2CSR
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseXcoo2csr(cusparseHandle handle, IntPtr cooRowInd, int nnz, int m, IntPtr csrRowPtr, cusparseIndexBase idxBase);
        #endregion

        #region CSR2COO
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseXcsr2coo(cusparseHandle handle, IntPtr csrRowPtr, int nnz, int m, IntPtr cooRowInd, cusparseIndexBase idxBase);
        #endregion
        #endregion

        #region Native Functions : SPARSE Level 1
        #region AXPY
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseSaxpyi_v2(cusparseHandle handle, int nnz, ref float alpha, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase);
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDaxpyi_v2(cusparseHandle handle, int nnz, ref double alpha, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase);
        #endregion

        #region DOT
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseSdoti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref float resultHost, cusparseIndexBase idxBase);
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDdoti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref double resultHost, cusparseIndexBase idxBase);
        #endregion

        #region GTHR
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseSgthr(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase ibase);
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDgthr(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase ibase);
        #endregion

        #region GTHRZ
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseSgthrz(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase ibase);
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDgthrz(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase ibase);
        #endregion

        #region ROT
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseSroti_v2(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref float c, ref float s, cusparseIndexBase idxBase);
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDroti_v2(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref double c, ref double s, cusparseIndexBase idxBase);
        #endregion

        #region SCTR
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseSsctr(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase ibase);
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDsctr(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase ibase);
        #endregion

        #endregion

        #region Native Functions : SPARSE Level 2
        #region CSRMV
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseScsrmv_v2(cusparseHandle handle, cusparseOperation transA, int m, int n, int nnz, ref float alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr x, ref float beta, IntPtr y);
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDcsrmv_v2(cusparseHandle handle, cusparseOperation transA, int m, int n, int nnz, ref double alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr x, ref double beta, IntPtr y);
        #endregion

        #region CSRSV_ANALYSIS
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseScsrsv_analysis_v2(cusparseHandle handle, cusparseOperation transA, int m, int nnz, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, cusparseSolveAnalysisInfo info);
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDcsrsv_analysis_v2(cusparseHandle handle, cusparseOperation transA, int m, int nnz, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, cusparseSolveAnalysisInfo info);
        #endregion

        #region CSRSV_SOLVE
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseScsrsv_solve_v2(cusparseHandle handle, cusparseOperation transA, int m,ref  float alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, cusparseSolveAnalysisInfo info, IntPtr x, IntPtr y);
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDcsrsv_solve_v2(cusparseHandle handle, cusparseOperation transA, int m, ref double alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, cusparseSolveAnalysisInfo info, IntPtr x, IntPtr y);
        #endregion
        #endregion

        #region Native Functions : SPARSE Level 3
        #region CSRMM
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseScsrmm_v2(cusparseHandle handle, cusparseOperation transA, int m, int n, int k, int nnz, ref float alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc);
        [DllImport(CUSPARSE_DLL_NAME)]
        private static extern CUSPARSEStatus cusparseDcsrmm_v2(cusparseHandle handle, cusparseOperation transA, int m, int n, int k, int nnz, ref double alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc);
        #endregion
        #endregion

        public string GetDllName()
        {
            return CUSPARSE_DLL_NAME;
        }

        #region Wrapper Functions : HelperFunctions
        public CUSPARSEStatus CusparseCreate(ref cusparseHandle handle)
        {
            return cusparseCreate(ref handle);
        }

        public CUSPARSEStatus CusparseDestroy(cusparseHandle handle)
        {
            return cusparseDestroy(handle);
        }

        public CUSPARSEStatus CusparseGetVersion(cusparseHandle handle, ref int version)
        {
            return cusparseGetVersion(handle, ref version);
        }

        public CUSPARSEStatus CusparseSetStream(cusparseHandle handle, GASS.CUDA.cudaStream streamId)
        {
            return cusparseSetStream(handle, streamId);
        }

        public CUSPARSEStatus CusparseCreateMatDescr(ref cusparseMatDescr descrA)
        {
            return cusparseCreateMatDescr(ref descrA);
        }

        public CUSPARSEStatus CusparseDestroyMatDescr(cusparseMatDescr descrA)
        {
            return cusparseDestroyMatDescr(descrA);
        }

        public CUSPARSEStatus CusparseSetMatType(cusparseMatDescr descrA, cusparseMatrixType type)
        {
            return cusparseSetMatType(descrA, type);
        }

        public cusparseMatrixType CusparseGetMatType(cusparseMatDescr descrA)
        {
            return cusparseGetMatType(descrA);
        }

        public CUSPARSEStatus CusparseSetMatFillMode(cusparseMatDescr descrA, cusparseFillMode fillMode)
        {
            return cusparseSetMatFillMode(descrA, fillMode);
        }

        public cusparseFillMode CusparseGetMatFillMode(cusparseMatDescr descrA)
        {
            return cusparseGetMatFillMode(descrA);
        }

        public CUSPARSEStatus CusparseSetMatDiagType(cusparseMatDescr descrA, cusparseDiagType diagType)
        {
            return cusparseSetMatDiagType(descrA, diagType);
        }

        public cusparseDiagType CusparseGetMatDiagType(cusparseMatDescr descrA)
        {
            return cusparseGetMatDiagType(descrA);
        }

        public CUSPARSEStatus CusparseSetMatIndexBase(cusparseMatDescr descrA, cusparseIndexBase ibase)
        {
            return cusparseSetMatIndexBase(descrA, ibase);
        }

        public cusparseIndexBase CusparseGetMatIndexBase(cusparseMatDescr descrA)
        {
            return cusparseGetMatIndexBase(descrA);
        }

        public CUSPARSEStatus CusparseCreateSolveAnalysisInfo(ref cusparseSolveAnalysisInfo info)
        {
            return cusparseCreateSolveAnalysisInfo(ref info);
        }

        public CUSPARSEStatus CusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo info)
        {
            return cusparseDestroySolveAnalysisInfo(info);
        }

        #endregion

        #region Wrapper Functions : Format Conversion Functions
        #region NNZ
        public CUSPARSEStatus CusparseSnnz(cusparseHandle handle, cusparseDirection dirA, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzPerVector, ref int nnzHostPtr)
        {
            return cusparseSnnz(handle, dirA, m, n, descrA, A, lda, nnzPerVector, ref nnzHostPtr);
        }
        public CUSPARSEStatus CusparseDnnz(cusparseHandle handle, cusparseDirection dirA, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzPerVector, ref int nnzHostPtr)
        {
            return cusparseDnnz(handle, dirA, m, n, descrA, A, lda, nnzPerVector, ref nnzHostPtr);
        }
        #endregion

        #region DENSE2CSR
        public CUSPARSEStatus CusparseSdense2csr(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzPerRow, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA)
        {
            return cusparseSdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA);
        }
        public CUSPARSEStatus CusparseDdense2csr(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzPerRow, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA)
        {
            return cusparseDdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA);
        }
        #endregion

        #region CSR2DENSE
        public CUSPARSEStatus CusparseScsr2dense(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr A, int lda)
        {
            return cusparseScsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
        }
        public CUSPARSEStatus CusparseDcsr2dense(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr A, int lda)
        {
            return cusparseDcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
        }
        #endregion

        #region DENSE2CSC
        public CUSPARSEStatus CusparseSdense2csc(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzPerCol, IntPtr cscValA, IntPtr cscRowIndA, IntPtr cscColPtrA)
        {
            return cusparseSdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA);
        }
        public CUSPARSEStatus CusparseDdense2csc(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzPerCol, IntPtr cscValA, IntPtr cscRowIndA, IntPtr cscColPtrA)
        {
            return cusparseDdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA);
        }
        #endregion

        #region CSC2DENSE
        public CUSPARSEStatus CusparseScsc2dense(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr cscValA, IntPtr cscRowIndA, IntPtr cscColPtrA, IntPtr A, int lda)
        {
            return cusparseScsc2dense(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda);
        }
        public CUSPARSEStatus CusparseDcsc2dense(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr cscValA, IntPtr cscRowIndA, IntPtr cscColPtrA, IntPtr A, int lda)
        {
            return cusparseDcsc2dense(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda);
        }
        #endregion

        #region CSR2CSC
        public CUSPARSEStatus CusparseScsr2csc(cusparseHandle handle, int m, int n, int nnz, IntPtr csrVal, IntPtr csrRowPtr, IntPtr csrColInd, IntPtr cscVal, IntPtr cscRowInd, IntPtr cscColPtr, cusparseAction copyvalues, cusparseIndexBase bs)
        {
            return cusparseScsr2csc_v2(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyvalues, bs);
        }
        public CUSPARSEStatus CusparseDcsr2csc(cusparseHandle handle, int m, int n, int nnz, IntPtr csrVal, IntPtr csrRowPtr, IntPtr csrColInd, IntPtr cscVal, IntPtr cscRowInd, IntPtr cscColPtr, cusparseAction copyvalues, cusparseIndexBase bs)
        {
            return cusparseDcsr2csc_v2(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyvalues, bs);
        }
        #endregion

        #region COO2CSR
        public CUSPARSEStatus CusparseXcoo2csr(cusparseHandle handle, IntPtr cooRowInd, int nnz, int m, IntPtr csrRowPtr, cusparseIndexBase idxBase)
        {
            return cusparseXcoo2csr(handle, cooRowInd, nnz, m, csrRowPtr, idxBase);
        }
        #endregion

        #region CSR2COO
        public CUSPARSEStatus CusparseXcsr2coo(cusparseHandle handle, IntPtr csrRowPtr, int nnz, int m, IntPtr cooRowInd, cusparseIndexBase idxBase)
        {
            return cusparseXcsr2coo(handle, csrRowPtr, nnz, m, cooRowInd, idxBase);
        }
        #endregion
        #endregion

        #region Wrapper Functions : SPARSE Level 1
        #region AXPY
        public CUSPARSEStatus CusparseSaxpyi(cusparseHandle handle, int nnz, ref float alpha, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase)
        {
            return cusparseSaxpyi_v2(handle, nnz, ref alpha, xVal, xInd, y, idxBase);
        }
        public CUSPARSEStatus CusparseDaxpyi(cusparseHandle handle, int nnz, ref double alpha, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase)
        {
            return cusparseDaxpyi_v2(handle, nnz, ref alpha, xVal, xInd, y, idxBase);
        }
        #endregion

        #region DOT
        public CUSPARSEStatus CusparseSdoti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref float resultHost, cusparseIndexBase idxBase)
        {
            return cusparseSdoti(handle, nnz, xVal, xInd, y, ref resultHost, idxBase);
        }
        public CUSPARSEStatus CusparseDdoti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref double resultHost, cusparseIndexBase idxBase)
        {
            return cusparseDdoti(handle, nnz, xVal, xInd, y, ref resultHost, idxBase);
        }
        #endregion

        #region GTHR
        public CUSPARSEStatus CusparseSgthr(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase ibase)
        {
            return cusparseSgthr(handle, nnz, y, xVal, xInd, ibase);
        }
        public CUSPARSEStatus CusparseDgthr(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase ibase)
        {
            return cusparseDgthr(handle, nnz, y, xVal, xInd, ibase);
        }
        #endregion

        #region GTHRZ
        public CUSPARSEStatus CusparseSgthrz(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase ibase)
        {
            return cusparseSgthrz(handle, nnz, y, xVal, xInd, ibase);
        }
        public CUSPARSEStatus CusparseDgthrz(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase ibase)
        {
            return cusparseDgthrz(handle, nnz, y, xVal, xInd, ibase);
        }
        #endregion

        #region ROT
        public CUSPARSEStatus CusparseSroti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref float c, ref float s, cusparseIndexBase idxBase)
        {
            return cusparseSroti_v2(handle, nnz, xVal, xInd, y, ref c, ref s, idxBase);
        }
        public CUSPARSEStatus CusparseDroti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref double c, ref double s, cusparseIndexBase idxBase)
        {
            return cusparseDroti_v2(handle, nnz, xVal, xInd, y, ref c, ref s, idxBase);
        }
        #endregion

        #region SCTR
        public CUSPARSEStatus CusparseSsctr(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase ibase)
        {
            return cusparseSsctr(handle, nnz, xVal, xInd, y, ibase);
        }
        public CUSPARSEStatus CusparseDsctr(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase ibase)
        {
            return cusparseDsctr(handle, nnz, xVal, xInd, y, ibase);
        }
        #endregion

        #endregion

        #region Wrapper Functions : SPARSE Level 2
        #region CSRMV
        public CUSPARSEStatus CusparseScsrmv(cusparseHandle handle, cusparseOperation transA, int m, int n, int nnz, ref float alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr x, ref float beta, IntPtr y)
        {
            return cusparseScsrmv_v2(handle, transA, m, n, nnz, ref alpha, descrA, csrValA, csrRowPtrA ,csrColIndA, x, ref beta, y);
        }
        public CUSPARSEStatus CusparseDcsrmv(cusparseHandle handle, cusparseOperation transA, int m, int n, int nnz, ref double alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr x, ref double beta, IntPtr y)
        {
            return cusparseDcsrmv_v2(handle, transA, m, n, nnz, ref alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, ref beta, y);
        }
        #endregion

        #region CSRSV_ANALYSIS
        public CUSPARSEStatus CusparseScsrsv_analysis(cusparseHandle handle, cusparseOperation transA, int m, int nnz, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, cusparseSolveAnalysisInfo info)
        {
            return cusparseScsrsv_analysis_v2(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info);
        }
        public CUSPARSEStatus CusparseDcsrsv_analysis(cusparseHandle handle, cusparseOperation transA, int m, int nnz, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, cusparseSolveAnalysisInfo info)
        {
            return cusparseDcsrsv_analysis_v2(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info); ;
        }
        #endregion

        #region CSRSV_SOLVE
        public CUSPARSEStatus CusparseScsrsv_solve(cusparseHandle handle, cusparseOperation transA, int m, ref float alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, cusparseSolveAnalysisInfo info, IntPtr x, IntPtr y)
        {
            return cusparseScsrsv_solve_v2(handle, transA, m, ref alpha, descrA, csrValA, csrRowPtrA, csrColIndA, info, x, y);
        }
        public CUSPARSEStatus CusparseDcsrsv_solve(cusparseHandle handle, cusparseOperation transA, int m, ref double alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, cusparseSolveAnalysisInfo info, IntPtr x, IntPtr y)
        {
            return cusparseDcsrsv_solve_v2(handle, transA, m, ref alpha, descrA, csrValA, csrRowPtrA, csrColIndA, info, x, y);
        }
        #endregion
        #endregion

        #region Wrapper Functions : SPARSE Level 3
        #region CSRMM
        public CUSPARSEStatus CusparseScsrmm(cusparseHandle handle, cusparseOperation transA, int m, int n, int k, int nnz, ref float alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc)
        {
            return cusparseScsrmm_v2(handle, transA, m, n, k, nnz, ref alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, ref beta, C, ldc);
        }
        public CUSPARSEStatus CusparseDcsrmm(cusparseHandle handle, cusparseOperation transA, int m, int n, int k, int nnz, ref double alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc)
        {
            return cusparseDcsrmm_v2(handle, transA, m, n, k, nnz, ref alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, ref beta, C, ldc);
        }
        #endregion
        #endregion

    }
}
