/* Added by Kichang Kim (kkc0923@hotmail.com) */
namespace Cudafy.Maths.SPARSE
{
    using GASS.CUDA;
    using GASS.CUDA.Types;
    using System;
    using System.Runtime.InteropServices;

    public interface ICUSPARSEDriver
    {
        string GetDllName();

        #region Helper Functions
        CUSPARSEStatus CusparseCreate(ref cusparseHandle handle);
        CUSPARSEStatus CusparseDestroy(cusparseHandle handle);
        CUSPARSEStatus CusparseGetVersion(cusparseHandle handle, ref int version);
        CUSPARSEStatus CusparseSetStream(cusparseHandle handle, cudaStream streamId);
        CUSPARSEStatus CusparseCreateMatDescr(ref cusparseMatDescr descrA);
        CUSPARSEStatus CusparseDestroyMatDescr(cusparseMatDescr descrA);
        CUSPARSEStatus CusparseSetMatType(cusparseMatDescr descrA, cusparseMatrixType type);
        cusparseMatrixType CusparseGetMatType(cusparseMatDescr descrA);
        CUSPARSEStatus CusparseSetMatFillMode(cusparseMatDescr descrA, cusparseFillMode fillMode);
        cusparseFillMode CusparseGetMatFillMode(cusparseMatDescr descrA);
        CUSPARSEStatus CusparseSetMatDiagType(cusparseMatDescr descrA, cusparseDiagType diagType);
        cusparseDiagType CusparseGetMatDiagType(cusparseMatDescr descrA);
        CUSPARSEStatus CusparseSetMatIndexBase(cusparseMatDescr descrA, cusparseIndexBase ibase);
        cusparseIndexBase CusparseGetMatIndexBase(cusparseMatDescr descrA);
        CUSPARSEStatus CusparseCreateSolveAnalysisInfo(ref cusparseSolveAnalysisInfo info);
        CUSPARSEStatus CusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo info);
        #endregion

        #region Format Conversoin Functions
        #region NNZ
        CUSPARSEStatus CusparseSnnz(cusparseHandle handle, cusparseDirection dirA, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzPerVector, ref int nnzHostPtr);
        CUSPARSEStatus CusparseDnnz(cusparseHandle handle, cusparseDirection dirA, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzPerVector, ref int nnzHostPtr);
        #endregion

        #region DENSE2CSR
        CUSPARSEStatus CusparseSdense2csr(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzPerRow, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA);
        CUSPARSEStatus CusparseDdense2csr(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzPerRow, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA);
        #endregion

        #region CSR2DENSE
        CUSPARSEStatus CusparseScsr2dense(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr A, int lda);
        CUSPARSEStatus CusparseDcsr2dense(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr A, int lda);
        #endregion

        #region DENSE2CSC
        CUSPARSEStatus CusparseSdense2csc(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzPerCol, IntPtr cscValA, IntPtr cscRowIndA, IntPtr cscColPtrA);
        CUSPARSEStatus CusparseDdense2csc(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr A, int lda, IntPtr nnzPerCol, IntPtr cscValA, IntPtr cscRowIndA, IntPtr cscColPtrA);
        #endregion

        #region CSC2DENSE
        CUSPARSEStatus CusparseScsc2dense(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr cscValA, IntPtr cscRowIndA, IntPtr cscColPtrA, IntPtr A, int lda);
        CUSPARSEStatus CusparseDcsc2dense(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, IntPtr cscValA, IntPtr cscRowIndA, IntPtr cscColPtrA, IntPtr A, int lda);
        #endregion

        #region CSR2CSC
        CUSPARSEStatus CusparseScsr2csc(cusparseHandle handle, int m, int n, int nnz, IntPtr csrVal, IntPtr csrRowPtr, IntPtr csrColInd, IntPtr cscVal, IntPtr cscRowInd, IntPtr cscColPtr, cusparseAction copyvalues, cusparseIndexBase bs);
        CUSPARSEStatus CusparseDcsr2csc(cusparseHandle handle, int m, int n, int nnz, IntPtr csrVal, IntPtr csrRowPtr, IntPtr csrColInd, IntPtr cscVal, IntPtr cscRowInd, IntPtr cscColPtr, cusparseAction copyvalues, cusparseIndexBase bs);
        #endregion

        #region COO2CSR
        CUSPARSEStatus CusparseXcoo2csr(cusparseHandle handle, IntPtr cooRowInd, int nnz, int m, IntPtr csrRowPtr, cusparseIndexBase idxBase);
        #endregion

        #region CSR2COO
        CUSPARSEStatus CusparseXcsr2coo(cusparseHandle handle, IntPtr csrRowPtr, int nnz, int m, IntPtr cooRowInd, cusparseIndexBase idxBase);
        #endregion
        #endregion

        #region SPARSE Level 1 Functions

        #region AXPY
        CUSPARSEStatus CusparseSaxpyi(cusparseHandle handle, int nnz, ref float alpha, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase);
        CUSPARSEStatus CusparseDaxpyi(cusparseHandle handle, int nnz, ref double alpha, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase);
        #endregion

        #region DOT
        CUSPARSEStatus CusparseSdoti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref float resultHost, cusparseIndexBase idxBase);
        CUSPARSEStatus CusparseDdoti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref double resultHost, cusparseIndexBase idxBase);
        #endregion

        #region GTHR
        CUSPARSEStatus CusparseSgthr(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase idxBase);
        CUSPARSEStatus CusparseDgthr(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase idxBase);
        #endregion

        #region GTHRZ
        CUSPARSEStatus CusparseSgthrz(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase idxBase);
        CUSPARSEStatus CusparseDgthrz(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase idxBase);
        #endregion

        #region ROT
        CUSPARSEStatus CusparseSroti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref float c, ref float s, cusparseIndexBase idxBase);
        CUSPARSEStatus CusparseDroti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref double c, ref double s, cusparseIndexBase idxBase);
        #endregion

        #region SCTR
        CUSPARSEStatus CusparseSsctr(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase);
        CUSPARSEStatus CusparseDsctr(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase);
        #endregion
        #endregion

        #region SPARSE Level 2 Functions
        #region CSRMV
        CUSPARSEStatus CusparseScsrmv(cusparseHandle handle, cusparseOperation transA, int m, int n, int nnz, ref float alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr x, ref float beta, IntPtr y);
        CUSPARSEStatus CusparseDcsrmv(cusparseHandle handle, cusparseOperation transA, int m, int n, int nnz, ref double alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr x, ref double beta, IntPtr y);
        #endregion

        #region CSRSV_ANALYSIS
        CUSPARSEStatus CusparseScsrsv_analysis(cusparseHandle handle, cusparseOperation transA, int m, int nnz, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, cusparseSolveAnalysisInfo info);
        CUSPARSEStatus CusparseDcsrsv_analysis(cusparseHandle handle, cusparseOperation transA, int m, int nnz, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, cusparseSolveAnalysisInfo info);
        #endregion

        #region CSRSV_SOLVE
        CUSPARSEStatus CusparseScsrsv_solve(cusparseHandle handle, cusparseOperation transA, int m, ref float alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, cusparseSolveAnalysisInfo info, IntPtr x, IntPtr y);
        CUSPARSEStatus CusparseDcsrsv_solve(cusparseHandle handle, cusparseOperation transA, int m, ref double alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, cusparseSolveAnalysisInfo info, IntPtr x, IntPtr y);
        #endregion
        #endregion

        #region SPARSE Level 3 Functions
        #region CSRMM
        CUSPARSEStatus CusparseScsrmm(cusparseHandle handle, cusparseOperation transA, int m, int n, int k, int nnz, ref  float alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc);
        CUSPARSEStatus CusparseDcsrmm(cusparseHandle handle, cusparseOperation transA, int m, int n, int k, int nnz, ref double alpha, cusparseMatDescr descrA, IntPtr csrValA, IntPtr csrRowPtrA, IntPtr csrColIndA, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc);
        #endregion
        #endregion
    }
}
