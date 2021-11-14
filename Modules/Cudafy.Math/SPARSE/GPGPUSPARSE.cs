/* Added by Kichang Kim (kkc0923@hotmail.com) */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;

using Cudafy.Host;
using Cudafy.Types;

namespace Cudafy.Maths.SPARSE
{
    internal enum eDataType { S, C, D, Z };

    /// <summary>
    /// Abstract base class for devices supporting SPARSE matrices.
    /// Warning: This code is alpha and incomplete.
    /// </summary>
    public abstract class GPGPUSPARSE : IDisposable
    {
        private object _lock;
        private bool _disposed = false;
        protected cusparseMatDescr defaultMatDescr = new cusparseMatDescr();
        /// <summary>
        /// Gets a value indicating whether this instance is disposed.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if this instance is disposed; otherwise, <c>false</c>.
        /// </value>
        public bool IsDisposed
        {
            get{lock(_lock){return _disposed;}}
        }

        /// <summary>
        /// Creates a SPARSE wrapper based on the specified gpu. Note only CudaGPU is supported.
        /// </summary>
        /// <param name="gpu">The gpu.</param>
        /// <returns></returns>
        public static GPGPUSPARSE Create(GPGPU gpu)
        {
            if (gpu is CudaGPU)
                return new CudaSPARSE(gpu);
            else
                throw new NotImplementedException(gpu.ToString());
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="GPGPUSPARSE"/> class.
        /// </summary>
        protected GPGPUSPARSE()
        {
            _lock = new object();
        }

        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="GPGPUSPARSE"/> is reclaimed by garbage collection.
        /// </summary>
        ~GPGPUSPARSE()
        {
            Dispose(false);
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Shutdowns this instance.
        /// </summary>
        protected abstract void Shutdown();

        /// <summary>
        /// Gets the version info.
        /// </summary>
        /// <returns></returns>
        public abstract string GetVersionInfo();

        /// <summary>
        /// Releases unmanaged and - optionally - managed resources
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            lock (_lock)
            {
                Debug.WriteLine(string.Format("GPGPUSPARSE::Dispose({0})", disposing));
                // Check to see if Dispose has already been called.
                if (!this._disposed)
                {
                    Debug.WriteLine("Disposing GPGPUSPARSE");
                    if (disposing)
                    {
                    }

                    Shutdown();
                    _disposed = true;

                }
                else
                    Debug.WriteLine("Already disposed");
            }
        }
        
        #region CUSPARSE Helper Function
        public abstract void CreateSolveAnalysisInfo(ref cusparseSolveAnalysisInfo info);
        public abstract void DestroySolveAnalysisInfo(cusparseSolveAnalysisInfo info);
        #endregion

        #region Matrix Helper
        public int GetIndexColumnMajor(int i, int j, int m)
        {
            return i + j * m;
        }
        #endregion

        #region Format Conversion Functions
        #region NNZ
        /// <summary>
        /// Computes the number of non-zero elements per row or column and the total number of non-zero elements.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="vector">array of size m or n containing the number of non-zero elements per row or column, respectively.</param>
        /// <param name="dirA">indicates whether to count the number of non-zero elements per row or per column, respectively.</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        /// <returns>total number of non-zero elements.</returns>
        public int NNZ(int m, int n, float[] A, int[] vector, cusparseDirection dirA = cusparseDirection.Row, int lda = 0)
        {
            return NNZ(m, n, A, vector, defaultMatDescr, dirA, lda);
        }
        /// <summary>
        /// Computes the number of non-zero elements per row or column and the total number of non-zero elements.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="vector">array of size m or n containing the number of non-zero elements per row or column, respectively.</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        /// <param name="dirA">indicates whether to count the number of non-zero elements per row or per column, respectively.</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        /// <returns>total number of non-zero elements.</returns>
        public abstract int NNZ(int m, int n, float[] A, int[] vector, cusparseMatDescr descrA, cusparseDirection dirA = cusparseDirection.Row, int lda = 0);

        /// <summary>
        /// Computes the number of non-zero elements per row or column and the total number of non-zero elements.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="vector">array of size m or n containing the number of non-zero elements per row or column, respectively.</param>
        /// <param name="dirA">indicates whether to count the number of non-zero elements per row or per column, respectively.</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        /// <returns>total number of non-zero elements.</returns>
        public int NNZ(int m, int n, double[] A, int[] vector, cusparseDirection dirA = cusparseDirection.Row, int lda = 0)
        {
            return NNZ(m, n, A, vector, defaultMatDescr, dirA, lda);
        }

        /// <summary>
        /// Computes the number of non-zero elements per row or column and the total number of non-zero elements.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="vector">array of size m or n containing the number of non-zero elements per row or column, respectively.</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        /// <param name="dirA">indicates whether to count the number of non-zero elements per row or per column, respectively.</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        /// <returns>total number of non-zero elements.</returns>
        public abstract int NNZ(int m, int n, double[] A, int[] vector, cusparseMatDescr descrA, cusparseDirection dirA = cusparseDirection.Row, int lda = 0);
        #endregion

        #region DENSE2CSR
        /// <summary>
        /// Converts the matrix A in dense format into a matrix in CSR format. All the parameters are pre-allocated by the user, and the arrays are filled in based on nnzPerRow.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="nnzPerRow">array of size m containing the number of non-zero elements per row.</param>
        /// <param name="csrValA">array of nnz elements to be filled.</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColIndA">array of nnz column indices, corresponding to the non-zero elements in the matrix.</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        public abstract void Dense2CSR(int m, int n, float[] A, int[] nnzPerRow, float[] csrValA, int[] csrRowA, int[] csrColIndA, cusparseMatDescr descrA, int lda = 0);

        /// <summary>
        /// Converts the matrix A in dense format into a matrix in CSR format. All the parameters are pre-allocated by the user, and the arrays are filled in based on nnzPerRow.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="nnzPerRow">array of size m containing the number of non-zero elements per row.</param>
        /// <param name="csrValA">array of nnz elements to be filled.</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColIndA">array of nnz column indices, corresponding to the non-zero elements in the matrix.</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        public void Dense2CSR(int m, int n, float[] A, int[] nnzPerRow, float[] csrValA, int[] csrRowA, int[] csrColIndA, int lda = 0)
        {
            Dense2CSR(m, n, A, nnzPerRow, csrValA, csrRowA, csrColIndA, defaultMatDescr, lda);
        }

        /// <summary>
        /// Converts the matrix A in dense format into a matrix in CSR format. All the parameters are pre-allocated by the user, and the arrays are filled in based on nnzPerRow.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="nnzPerRow">array of size m containing the number of non-zero elements per row.</param>
        /// <param name="csrValA">array of nnz elements to be filled.</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColIndA">array of nnz column indices, corresponding to the non-zero elements in the matrix.</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        public abstract void Dense2CSR(int m, int n, double[] A, int[] nnzPerRow, double[] csrValA, int[] csrRowA, int[] csrColIndA, cusparseMatDescr descrA, int lda = 0);

        /// <summary>
        /// Converts the matrix A in dense format into a matrix in CSR format. All the parameters are pre-allocated by the user, and the arrays are filled in based on nnzPerRow.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="nnzPerRow">array of size m containing the number of non-zero elements per row.</param>
        /// <param name="csrValA">array of nnz elements to be filled.</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColIndA">array of nnz column indices, corresponding to the non-zero elements in the matrix.</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        public void Dense2CSR(int m, int n, double[] A, int[] nnzPerRow, double[] csrValA, int[] csrRowA, int[] csrColIndA, int lda = 0)
        {
            Dense2CSR(m, n, A, nnzPerRow, csrValA, csrRowA, csrColIndA, defaultMatDescr, lda);
        }
        #endregion

        #region CSR2DENSE
        /// <summary>
        /// Converts the matrix in CSR format defined by the three arrays csrValA, csrRowA and csrColA into a matrix A in dense format.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRowA[m] - csrRowA[0].</param>
        /// <param name="csrRowsA">array of m+1 index elements.</param>
        /// <param name="csrColsA">array of nnz column indices.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        public abstract void CSR2Dense(int m, int n, float[] csrValA, int[] csrRowA, int[] csrColA, float[] A, cusparseMatDescr descrA, int lda = 0);

        /// <summary>
        /// Converts the matrix in CSR format defined by the three arrays csrValA, csrRowA and csrColA into a matrix A in dense format.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRowA[m] - csrRowA[0].</param>
        /// <param name="csrRowsA">array of m+1 index elements.</param>
        /// <param name="csrColsA">array of nnz column indices.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        public void CSR2Dense(int m, int n, float[] csrValA, int[] csrRowA, int[] csrColA, float[] A, int lda = 0)
        {
            CSR2Dense(m, n, csrValA, csrRowA, csrColA, A, defaultMatDescr, lda);
        }

        /// <summary>
        /// Converts the matrix in CSR format defined by the three arrays csrValA, csrRowA and csrColA into a matrix A in dense format.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRowA[m] - csrRowA[0].</param>
        /// <param name="csrRowsA">array of m+1 index elements.</param>
        /// <param name="csrColsa">array of nnz column indices.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        public abstract void CSR2Dense(int m, int n, double[] csrValA, int[] csrRowA, int[] csrColA, double[] A, cusparseMatDescr descrA, int lda = 0);

        /// <summary>
        /// Converts the matrix in CSR format defined by the three arrays csrValA, csrRowA and csrColA into a matrix A in dense format.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRowA[m] - csrRowA[0].</param>
        /// <param name="csrRowsA">array of m+1 index elements.</param>
        /// <param name="csrColsA">array of nnz column indices.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        public void CSR2Dense(int m, int n, double[] csrValA, int[] csrRowA, int[] csrColA, double[] A, int lda = 0)
        {
            CSR2Dense(m, n, csrValA, csrRowA, csrColA, A, defaultMatDescr, lda);
        }
        #endregion

        #region DENSE2CSC
        /// <summary>
        /// Converts the matrix A in dense format into a matrix in CSC format. All the parameters are pre-allocated by the user, and the arrays are filled in based nnzPerCol.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="nnzPerCol">>array of size m containing the number of non-zero elements per column.</param>
        /// <param name="cscValA">array of nnz elements to be filled.</param>
        /// <param name="cscRowIndA">array of nnz row indices, corresponding to the non-zero elements in the matrix.</param>
        /// <param name="cscColA">array of n+1 index elements.</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        public abstract void Dense2CSC(int m, int n, float[] A, int[] nnzPerCol, float[] cscValA, int[] cscRowIndA, int[] cscColA, cusparseMatDescr descrA, int lda = 0);

        /// <summary>
        /// Converts the matrix A in dense format into a matrix in CSC format. All the parameters are pre-allocated by the user, and the arrays are filled in based nnzPerCol.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="nnzPerCol">>array of size m containing the number of non-zero elements per column.</param>
        /// <param name="cscValA">array of nnz elements to be filled.</param>
        /// <param name="cscRowIndA">array of nnz row indices, corresponding to the non-zero elements in the matrix.</param>
        /// <param name="cscColA">array of n+1 index elements.</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        public void Dense2CSC(int m, int n, float[] A, int[] nnzPerCol, float[] cscValA, int[] cscRowIndA, int[] cscColA, int lda = 0)
        {
            Dense2CSC(m, n, A, nnzPerCol, cscValA, cscRowIndA, cscColA, defaultMatDescr, lda);
        }

        /// <summary>
        /// Converts the matrix A in dense format into a matrix in CSC format. All the parameters are pre-allocated by the user, and the arrays are filled in based nnzPerCol.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="nnzPerCol">>array of size m containing the number of non-zero elements per column.</param>
        /// <param name="cscValA">array of nnz elements to be filled.</param>
        /// <param name="cscRowIndA">array of nnz row indices, corresponding to the non-zero elements in the matrix.</param>
        /// <param name="cscColA">array of n+1 index elements.</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        public abstract void Dense2CSC(int m, int n, double[] A, int[] nnzPerCol, double[] cscValA, int[] cscRowIndA, int[] cscColA, cusparseMatDescr descrA, int lda = 0);

        /// <summary>
        /// Converts the matrix A in dense format into a matrix in CSC format. All the parameters are pre-allocated by the user, and the arrays are filled in based nnzPerCol.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="nnzPerCol">>array of size m containing the number of non-zero elements per column.</param>
        /// <param name="cscValA">array of nnz elements to be filled.</param>
        /// <param name="cscRowIndA">array of nnz row indices, corresponding to the non-zero elements in the matrix.</param>
        /// <param name="cscColA">array of n+1 index elements.</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        public void Dense2CSC(int m, int n, double[] A, int[] nnzPerCol, double[] cscValA, int[] cscRowIndA, int[] cscColA, int lda = 0)
        {
            Dense2CSC(m, n, A, nnzPerCol, cscValA, cscRowIndA, cscColA, defaultMatDescr, lda);
        }
        #endregion

        #region CSC2DENSE
        /// <summary>
        /// Converts the matrix in CSC format defined by the three arrays cscValA, cscColA and cscRowA into matrix A in dense format. The dense matrix A is filled in with the values of the sparse matrix and with zeros elsewhere.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="cscValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrColA[m] - csrColA[0].</param>
        /// <param name="cscRowA">array of nnz row indices.</param>
        /// <param name="cscColA">array of n+1 index elements.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        public abstract void CSC2Dense(int m, int n, float[] cscValA, int[] cscRowA, int[] cscColA, float[] A, cusparseMatDescr descrA, int lda = 0);

        /// <summary>
        /// Converts the matrix in CSC format defined by the three arrays cscValA, cscColA and cscRowA into matrix A in dense format. The dense matrix A is filled in with the values of the sparse matrix and with zeros elsewhere.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="cscValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrColA[m] - csrColA[0].</param>
        /// <param name="cscRowA">array of nnz row indices.</param>
        /// <param name="cscColA">array of n+1 index elements.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        public void CSC2Dense(int m, int n, float[] cscValA, int[] cscRowA, int[] cscColA, float[] A, int lda = 0)
        {
            CSC2Dense(m, n, cscValA, cscRowA, cscColA, A, defaultMatDescr, lda);
        }

        /// <summary>
        /// Converts the matrix in CSC format defined by the three arrays cscValA, cscColA and cscRowA into matrix A in dense format. The dense matrix A is filled in with the values of the sparse matrix and with zeros elsewhere.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="cscValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrColA[m] - csrColA[0].</param>
        /// <param name="cscRowA">array of nnz row indices.</param>
        /// <param name="cscColA">array of n+1 index elements.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        public abstract void CSC2Dense(int m, int n, double[] cscValA, int[] cscRowA, int[] cscColA, double[] A, cusparseMatDescr descrA, int lda = 0);

        /// <summary>
        /// Converts the matrix in CSC format defined by the three arrays cscValA, cscColA and cscRowA into matrix A in dense format. The dense matrix A is filled in with the values of the sparse matrix and with zeros elsewhere.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="cscValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrColA[m] - csrColA[0].</param>
        /// <param name="cscRowA">array of nnz row indices.</param>
        /// <param name="cscColA">array of n+1 index elements.</param>
        /// <param name="A">array of dimension (lda, n)</param>
        /// <param name="lda">leading dimension of A. If lda is 0, automatically be m.</param>
        public void CSC2Dense(int m, int n, double[] cscValA, int[] cscRowA, int[] cscColA, double[] A, int lda = 0)
        {
            CSC2Dense(m, n, cscValA, cscRowA, cscColA, A, defaultMatDescr, lda);
        }
        #endregion

        #region CSR2CSC
        /// <summary>
        /// Converts the matrix in CSR format defined with the three arrays csrVal, csrRow and csrCol into matrix A in CSC format defined by array cscVal, cscRow, cscCol.
        /// The resultng matrix can also be seen as the transpose of the original sparse matrix. This routine can also be used to convert a matrix in CSC format into a matrix in CSR format.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="nnz">number of non-zero elements of matrix A.</param>
        /// <param name="csrVal">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRow[m] - csrRow[0].</param>
        /// <param name="csrRow">array of m+1 indices.</param>
        /// <param name="csrCol">array of nnz column indices.</param>
        /// <param name="cscVal">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrCol[n] - csrCol[0]. if copyValues is non-zero, updated array.</param>
        /// <param name="cscRow">updated array of nnz row indices.</param>
        /// <param name="cscCol">updated array of n+1 index elements.</param>
        /// <param name="copyValues">if Symbloic, cscVal array is not filled.</param>
        /// <param name="bs">base index.</param>
        public abstract void CSR2CSC(int m, int n, int nnz, float[] csrVal, int[] csrRow, int[] csrCol, float[] cscVal, int[] cscRow, int[] cscCol, cusparseAction copyValues = cusparseAction.Numeric, cusparseIndexBase bs = cusparseIndexBase.Zero);

        /// <summary>
        /// Converts the matrix in CSR format defined with the three arrays csrVal, csrRow and csrCol into matrix A in CSC format defined by array cscVal, cscRow, cscCol.
        /// The resultng matrix can also be seen as the transpose of the original sparse matrix. This routine can also be used to convert a matrix in CSC format into a matrix in CSR format.
        /// </summary>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="n">number of columns of the matrix A; n must be at least zero.</param>
        /// <param name="nnz">number of non-zero elements of matrix A.</param>
        /// <param name="csrVal">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRow[m] - csrRow[0].</param>
        /// <param name="csrRow">array of m+1 indices.</param>
        /// <param name="csrCol">array of nnz column indices.</param>
        /// <param name="cscVal">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrCol[n] - csrCol[0]. if copyValues is non-zero, updated array.</param>
        /// <param name="cscRow">updated array of nnz row indices.</param>
        /// <param name="cscCol">updated array of n+1 index elements.</param>
        /// <param name="copyValues">if Symbloic, cscVal array is not filled.</param>
        /// <param name="bs">base index.</param>
        public abstract void CSR2CSC(int m, int n, int nnz, double[] csrVal, int[] csrRow, int[] csrCol, double[] cscVal, int[] cscRow, int[] cscCol, cusparseAction copyValues = cusparseAction.Numeric, cusparseIndexBase bs = cusparseIndexBase.Zero);
        #endregion

        #region COO2CSR
        /// <summary>
        /// Converts the array containing the uncompressed row indices (corresponding to COO format) into an array of compressed row pointers (corresponding to CSR format).
        /// It can also be used to convert the array containing the uncompressed column indices (corresponding to COO format) into an array of column pointers (corresponding to CSC format).
        /// </summary>
        /// <param name="nnz">number of non-zeros of the matrix in COO format; this is also the length of array cooRow.</param>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="cooRow">array of row indices.</param>
        /// <param name="csrRow">array of row pointers.</param>
        /// <param name="idxBase">base index.</param>
        public abstract void COO2CSR(int nnz, int m, int[] cooRow, int[] csrRow, cusparseIndexBase idxBase = cusparseIndexBase.Zero);
        #endregion

        #region CSR2COO
        /// <summary>
        /// Converts the array containing the compressed row pointers (corresponding to CSR format) into an array of uncompressed row indices ( corresponding to COO format).
        /// It can also be used to convert the array containing the compressed column pointers (corresponding to CSC format) into an array of uncompressed column indices (corresponding to COO format).
        /// </summary>
        /// <param name="nnz">number of non-zeros of the matrix in COO format; this is also the length of array cooRow</param>
        /// <param name="m">number of rows of the matrix A; m must be at least zero.</param>
        /// <param name="csrRow">array of compressed row pointers.</param>
        /// <param name="cooRow">array of umcompressed row indices.</param>
        /// <param name="idxBase">base index.</param>
        public abstract void CSR2COO(int nnz, int m, int[] csrRow, int[] cooRow, cusparseIndexBase idxBase = cusparseIndexBase.Zero);
        #endregion
        #endregion

        #region SPARSE Level 1

        #region AXPY
        /// <summary>
        /// Multiplies the vector x in sparse format by the constant alpha and adds
        /// the result to the vector y in dense format.
        /// y = alpha * x + y
        /// </summary>
        /// <param name="alpha">constant multiplier.</param>
        /// <param name="vectorx">non-zero values of vector x.</param>
        /// <param name="indexx">indices corresponding to non‐zero values of vector x.</param>
        /// <param name="vectory">initial vector in dense format.</param>
        /// <param name="nnz">number of elements of the vector x (set to 0 for all elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void AXPY(ref float alpha, float[] vectorx, int[] indexx, float[] vectory, int nnz = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        /// <summary>
        /// Multiplies the vector x in sparse format by the constant alpha and adds
        /// the result to the vector y in dense format.
        /// y = alpha * x + y
        /// </summary>
        /// <param name="alpha">constant multiplier.</param>
        /// <param name="vectorx">non-zero values of vector x.</param>
        /// <param name="indexx">indices corresponding to non‐zero values of vector x.</param>
        /// <param name="vectory">initial vector in dense format.</param>
        /// <param name="nnz">number of elements of the vector x (set to 0 for all elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void AXPY(ref double alpha, double[] vectorx, int[] indexx, double[] vectory, int nnz = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        #endregion

        #region DOT
        /// <summary>
        /// Returns the dot product of a vector x in sparse format and vector y in dense format.
        /// For i = 0 to n-1
        ///     result += x[i] * y[i]
        /// </summary>
        /// <param name="vectorx">non-zero values of vector x.</param>
        /// <param name="indexx">indices corresponding to non-zero values of vector x.</param>
        /// <param name="vectory">vector in dense format.</param>
        /// <param name="n">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        /// <returns>result.</returns>
        public abstract float DOT(float[] vectorx, int[] indexx, float[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        /// <summary>
        /// Returns the dot product of a vector x in sparse format and vector y in dense format.
        /// For i = 0 to n-1
        ///     result += x[i] * y[i]
        /// </summary>
        /// <param name="vectorx">non-zero values of vector x.</param>
        /// <param name="indexx">indices corresponding to non-zero values of vector x.</param>
        /// <param name="vectory">vector in dense format.</param>
        /// <param name="n">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        /// <returns>result.</returns>
        public abstract double DOT(double[] vectorx, int[] indexx, double[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        #endregion

        #region GTHR
        /// <summary>
        /// Gathers the elements of the vector y listed by the index array indexx into the array vectorx.
        /// x[i] = y[i]
        /// </summary>
        /// <param name="vectory">vector in dense format, of size greater than or equal to max(indexx)-idxBase+1</param>
        /// <param name="vectorx">pre-allocated array in device memory of size greater than or equal to nnz</param>
        /// <param name="indexx">indices corresponding to non-zero values of vector x.</param>
        /// <param name="nnz">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void GTHR(float[] vectory, float[] vectorx, int[] indexx, int nnz = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        /// <summary>
        /// Gathers the elements of the vector y listed by the index array indexx into the array vectorx.
        /// x[i] = y[i]
        /// </summary>
        /// <param name="vectory">vector in dense format, of size greater than or equal to max(indexx)-idxBase+1</param>
        /// <param name="vectorx">pre-allocated array in device memory of size greater than or equal to nnz</param>
        /// <param name="indexx">indices corresponding to non-zero values of vector x.</param>
        /// <param name="nnz">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void GTHR(double[] vectory, double[] vectorx, int[] indexx, int nnz = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        #endregion

        #region GTHRZ
        /// <summary>
        /// Gathers the elements of the vector y listed by the index array indexx into the array vectorx, and zeroes those elements in the vector y.
        /// x[i] = y[i]
        /// y[i] = 0
        /// </summary>
        /// <param name="vectory">vector in dense format, of size greater than or equal to max(indexx)-idxBase+1.</param>
        /// <param name="vectorx">pre-allocated array in device memory of size greater than or equal to nnz.</param>
        /// <param name="indexx">indices corresponding to non-zero values of vector x.</param>
        /// <param name="nnz">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void GTHRZ(float[] vectory, float[] vectorx, int[] indexx, int nnz = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        /// <summary>
        /// Gathers the elements of the vector y listed by the index array indexx into the array vectorx, and zeroes those elements in the vector y.
        /// x[i] = y[i]
        /// y[i] = 0
        /// </summary>
        /// <param name="vectory">vector in dense format, of size greater than or equal to max(indexx)-idxBase+1.</param>
        /// <param name="vectorx">pre-allocated array in device memory of size greater than or equal to nnz.</param>
        /// <param name="indexx">indices corresponding to non-zero values of vector x.</param>
        /// <param name="nnz">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void GTHRZ(double[] vectory, double[] vectorx, int[] indexx, int nnz = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        #endregion

        #region ROT
        /// <summary>
        /// Applies givens rotation, defined by values c and s, to vectors x in sparse and y in dense format.
        /// x[i] = c * x[i] + s * y[i];
        /// y[i] = c * y[i] - s * x[i];
        /// </summary>
        /// <param name="vectorx">non-zero values of the vector x.</param>
        /// <param name="indexx">indices correspoding to non-zero values of vector x.</param>
        /// <param name="vectory">vector in dense format.</param>
        /// <param name="c">scalar</param>
        /// <param name="s">scalar</param>
        /// <param name="nnz">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void ROT(float[] vectorx, int[] indexx, float[] vectory, ref float c, ref float s, int nnz = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        /// <summary>
        /// Applies givens rotation, defined by values c and s, to vectors x in sparse and y in dense format.
        /// x[i] = c * x[i] + s * y[i];
        /// y[i] = c * y[i] - s * x[i];
        /// </summary>
        /// <param name="vectorx">non-zero values of the vector x.</param>
        /// <param name="indexx">indices correspoding to non-zero values of vector x.</param>
        /// <param name="vectory">vector in dense format.</param>
        /// <param name="c">scalar</param>
        /// <param name="s">scalar</param>
        /// <param name="nnz">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void ROT(double[] vectorx, int[] indexx, double[] vectory, ref double c, ref double s, int nnz = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        #endregion

        #region SCTR
        /// <summary>
        /// Scatters the vector x in sparse format into the vector y in dense format.
        /// It modifies only the lements of y whose indices are listed in the array indexx.
        /// y[i] = x[i]
        /// </summary>
        /// <param name="vectorx">non-zero values of vector x.</param>
        /// <param name="indexx">indices correspoding to non-zero values of vector x.</param>
        /// <param name="vectory">pre-allocated vector in dense format, of size greater than or equal to max(indexx)-ibase+1.</param>
        /// <param name="nnz">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void SCTR(float[] vectorx, int[] indexx, float[] vectory, int nnz = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        /// <summary>
        /// Scatters the vector x in sparse format into the vector y in dense format.
        /// It modifies only the lements of y whose indices are listed in the array indexx.
        /// y[i] = x[i]
        /// </summary>
        /// <param name="vectorx">non-zero values of vector x.</param>
        /// <param name="indexx">indices correspoding to non-zero values of vector x.</param>
        /// <param name="vectory">pre-allocated vector in dense format, of size greater than or equal to max(indexx)-ibase+1.</param>
        /// <param name="nnz">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void SCTR(double[] vectorx, int[] indexx, double[] vectory, int nnz = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        #endregion



        #endregion

        #region SPARSE Level 2
        #region CSRMV
        /// <summary>
        /// Performs one of the matrix-vector operations.
        /// y = alpha * op(A) * x + beta * y
        /// </summary>
        /// <param name="m">specifies the number of rows of matrix A; m mmust be at least zero.</param>
        /// <param name="n">specifies the number of columns of matrix A; n mmust be at least zero.</param>
        /// <param name="nnz">number of non-zero elements of matrix A.</param>
        /// <param name="alpha">scalar multiplier applied to op(A) * x.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be ontained from csrRow[m] - csrRow[0].</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColA">array of nnz column indices.</param>
        /// <param name="x">vector of n elements if op(A) = A, and m elements if op(A) = transpose(A).</param>
        /// <param name="beta">scalar multiplier applied to y. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">vector of m elements if op(A) = A, and n elements if op(A) = transpose(A).</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        /// <param name="op">specifies op(A).</param>
        public abstract void CSRMV(int m, int n, int nnz, ref float alpha, float[] csrValA, int[] csrRowA, int[] csrColA, float[] x, ref float beta, float[] y, cusparseMatDescr descrA, cusparseOperation op = cusparseOperation.NonTranspose);

        /// <summary>
        /// Performs one of the matrix-vector operations.
        /// y = alpha * op(A) * x + beta * y
        /// </summary>
        /// <param name="m">specifies the number of rows of matrix A; m mmust be at least zero.</param>
        /// <param name="n">specifies the number of columns of matrix A; n mmust be at least zero.</param>
        /// <param name="nnz">number of non-zero elements of matrix A.</param>
        /// <param name="alpha">scalar multiplier applied to op(A) * x.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be ontained from csrRow[m] - csrRow[0].</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColA">array of nnz column indices.</param>
        /// <param name="x">vector of n elements if op(A) = A, and m elements if op(A) = transpose(A).</param>
        /// <param name="beta">scalar multiplier applied to y. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">vector of m elements if op(A) = A, and n elements if op(A) = transpose(A).</param>
        /// <param name="op">specifies op(A).</param>
        public void CSRMV(int m, int n, int nnz, ref float alpha, float[] csrValA, int[] csrRowA, int[] csrColA, float[] x, ref float beta, float[] y, cusparseOperation op = cusparseOperation.NonTranspose)
        {
            CSRMV(m, n, nnz, ref alpha, csrValA, csrRowA, csrColA, x, ref beta, y, defaultMatDescr, op);
        }

        /// <summary>
        /// Performs one of the matrix-vector operations.
        /// y = alpha * op(A) * x + beta * y
        /// </summary>
        /// <param name="m">specifies the number of rows of matrix A; m mmust be at least zero.</param>
        /// <param name="n">specifies the number of columns of matrix A; n mmust be at least zero.</param>
        /// <param name="nnz">number of non-zero elements of matrix A.</param>
        /// <param name="alpha">scalar multiplier applied to op(A) * x.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be ontained from csrRow[m] - csrRow[0].</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColA">array of nnz column indices.</param>
        /// <param name="x">vector of n elements if op(A) = A, and m elements if op(A) = transpose(A).</param>
        /// <param name="beta">scalar multiplier applied to y. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">vector of m elements if op(A) = A, and n elements if op(A) = transpose(A).</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        /// <param name="op">specifies op(A).</param>
        public abstract void CSRMV(int m, int n, int nnz, ref double alpha, double[] csrValA, int[] csrRowA, int[] csrColA, double[] x, ref double beta, double[] y, cusparseMatDescr descrA, cusparseOperation op = cusparseOperation.NonTranspose);

        /// <summary>
        /// Performs one of the matrix-vector operations.
        /// y = alpha * op(A) * x + beta * y
        /// </summary>
        /// <param name="m">specifies the number of rows of matrix A; m mmust be at least zero.</param>
        /// <param name="n">specifies the number of columns of matrix A; n mmust be at least zero.</param>
        /// <param name="nnz">number of non-zero elements of matrix A.</param>
        /// <param name="alpha">scalar multiplier applied to op(A) * x.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be ontained from csrRow[m] - csrRow[0].</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColA">array of nnz column indices.</param>
        /// <param name="x">vector of n elements if op(A) = A, and m elements if op(A) = transpose(A).</param>
        /// <param name="beta">scalar multiplier applied to y. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">vector of m elements if op(A) = A, and n elements if op(A) = transpose(A).</param>
        /// <param name="op">specifies op(A).</param>
        public void CSRMV(int m, int n, int nnz, ref double alpha, double[] csrValA, int[] csrRowA, int[] csrColA, double[] x, ref double beta, double[] y, cusparseOperation op = cusparseOperation.NonTranspose)
        {
            CSRMV(m, n, nnz, ref alpha, csrValA, csrRowA, csrColA, x, ref beta, y, defaultMatDescr, op);
        }
       
        #endregion

        #region CSRSV_ANALYSIS
        /// <summary>
        /// Performs the analysis phase of the solution of a sparse triangular linear system.
        /// op(A) * y = alpha * x
        /// </summary>
        /// <param name="m">specifies the number of rows and columns of matrix A; m must be at least zero.</param>
        /// <param name="nnz">number of non-zero elements of matrix A.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRow[m] - csrRow[0].</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColA">array of nnz column indices.</param>
        /// <param name="op">specifies op(A).</param>
        /// <param name="info">structure that stores the information collected during the analysis phase. It should be passed to the solve phase unchanged.</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        public abstract void CSRSV_ANALYSIS(int m, int nnz, float[] csrValA, int[] csrRowA, int[] csrColA, cusparseOperation op, cusparseSolveAnalysisInfo info, cusparseMatDescr descrA);

        /// <summary>
        /// Performs the analysis phase of the solution of a sparse triangular linear system.
        /// op(A) * y = alpha * x
        /// </summary>
        /// <param name="m">specifies the number of rows and columns of matrix A; m must be at least zero.</param>
        /// <param name="nnz">number of non-zero elements of matrix A.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRow[m] - csrRow[0].</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColA">array of nnz column indices.</param>
        /// <param name="op">specifies op(A).</param>
        /// <param name="info">structure that stores the information collected during the analysis phase. It should be passed to the solve phase unchanged.</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        public abstract void CSRSV_ANALYSIS(int m, int nnz, double[] csrValA, int[] csrRowA, int[] csrColA, cusparseOperation op, cusparseSolveAnalysisInfo info, cusparseMatDescr descrA);
        #endregion

        #region CSRSV_SOLVE
        /// <summary>
        /// Performs the solve phase of the solution of a sparse triangular linear system.
        /// op(A) * y = alpha * x
        /// </summary>
        /// <param name="m">specifies the number of rows and columns of matrix A; m must be at least zero.</param>
        /// <param name="alpha">scalar multiplier applied to x.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRow[m] - csrRow[0].</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColA">array of nnz column indices.</param>
        /// <param name="x">vector of m elements.</param>
        /// <param name="y">vector of m elements. updated according to op(A) * y = alpha * x</param>
        /// <param name="op">specifies op(A).</param>
        /// <param name="info">structure that stores the information collected during the analysis phase. It should be passed to the solve phase unchanged.</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        public abstract void CSRSV_SOLVE(int m, ref float alpha, float[] csrValA, int[] csrRowA, int[] csrColA, float[] x, float[] y, cusparseOperation op, cusparseSolveAnalysisInfo info, cusparseMatDescr descrA);

        /// <summary>
        /// Performs the solve phase of the solution of a sparse triangular linear system.
        /// op(A) * y = alpha * x
        /// </summary>
        /// <param name="m">specifies the number of rows and columns of matrix A; m must be at least zero.</param>
        /// <param name="alpha">scalar multiplier applied to x.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRow[m] - csrRow[0].</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColA">array of nnz column indices.</param>
        /// <param name="x">vector of m elements.</param>
        /// <param name="y">vector of m elements. updated according to op(A) * y = alpha * x</param>
        /// <param name="op">specifies op(A).</param>
        /// <param name="info">structure that stores the information collected during the analysis phase. It should be passed to the solve phase unchanged.</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        public abstract void CSRSV_SOLVE(int m, ref double alpha, double[] csrValA, int[] csrRowA, int[] csrColA, double[] x, double[] y, cusparseOperation op, cusparseSolveAnalysisInfo info, cusparseMatDescr descrA);
        #endregion
        #endregion

        #region SPARSE Level 3
        /// <summary>
        /// Performs matrix-matrix operations. A is CSR format matrix and B, C is dense format.
        /// C = alpha * op(A) * B + beta * C
        /// </summary>
        /// <param name="m">number of rows of matrix A; m must be at least zero.</param>
        /// <param name="k">number of columns of matrix A; k must be at least zero.</param>
        /// <param name="n">number of columns of matrices B and C; n must be at least zero.</param>
        /// <param name="nnz">number of non-zero elements of matrix A.</param>
        /// <param name="alpha">scalar multiplier applied to op(A) * B.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRowA[m] - csrRowA[0].</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColA">array of nnz column indices.</param>
        /// <param name="B">array of dimension (ldb, n).</param>
        /// <param name="beta">scalar multiplier applied to C. If beta is zero, C does not have to be a valid input.</param>
        /// <param name="C">array of dimension (ldc, n).</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        /// <param name="op">specifies op(A).</param>
        /// <param name="ldb">leading dimension of B.</param>
        /// <param name="ldc">leading dimension of C.</param>
        public abstract void CSRMM(int m, int k, int n, int nnz, ref float alpha, float[] csrValA, int[] csrRowA, int[] csrColA, float[] B, ref float beta, float[] C, cusparseMatDescr descrA, cusparseOperation op = cusparseOperation.NonTranspose, int ldb = 0, int ldc = 0);

        /// <summary>
        /// Performs matrix-matrix operations. A is CSR format matrix and B, C is dense format.
        /// C = alpha * op(A) * B + beta * C
        /// </summary>
        /// <param name="m">number of rows of matrix A; m must be at least zero.</param>
        /// <param name="k">number of columns of matrix A; k must be at least zero.</param>
        /// <param name="n">number of columns of matrices B and C; n must be at least zero.</param>
        /// <param name="nnz">number of non-zero elements of matrix A.</param>
        /// <param name="alpha">scalar multiplier applied to op(A) * B.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRowA[m] - csrRowA[0].</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColA">array of nnz column indices.</param>
        /// <param name="B">array of dimension (ldb, n).</param>
        /// <param name="beta">scalar multiplier applied to C. If beta is zero, C does not have to be a valid input.</param>
        /// <param name="C">array of dimension (ldc, n).</param>
        /// <param name="op">specifies op(A).</param>
        /// <param name="ldb">leading dimension of B.</param>
        /// <param name="ldc">leading dimension of C.</param>
        public void CSRMM(int m, int k, int n, int nnz, ref float alpha, float[] csrValA, int[] csrRowA, int[] csrColA, float[] B, ref float beta, float[] C, cusparseOperation op = cusparseOperation.NonTranspose, int ldb = 0, int ldc = 0)
        {
            CSRMM(m, k, n, nnz, ref alpha, csrValA, csrRowA, csrColA, B, ref beta, C, defaultMatDescr, op, ldb, ldc);
        }

        /// <summary>
        /// Performs matrix-matrix operations. A is CSR format matrix and B, C is dense format.
        /// C = alpha * op(A) * B + beta * C
        /// </summary>
        /// <param name="m">number of rows of matrix A; m must be at least zero.</param>
        /// <param name="k">number of columns of matrix A; k must be at least zero.</param>
        /// <param name="n">number of columns of matrices B and C; n must be at least zero.</param>
        /// <param name="nnz">number of non-zero elements of matrix A.</param>
        /// <param name="alpha">scalar multiplier applied to op(A) * B.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRowA[m] - csrRowA[0].</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColA">array of nnz column indices.</param>
        /// <param name="B">array of dimension (ldb, n).</param>
        /// <param name="beta">scalar multiplier applied to C. If beta is zero, C does not have to be a valid input.</param>
        /// <param name="C">array of dimension (ldc, n).</param>
        /// <param name="descrA">descriptor of matrix A.</param>
        /// <param name="op">specifies op(A).</param>
        /// <param name="ldb">leading dimension of B.</param>
        /// <param name="ldc">leading dimension of C.</param>
        public abstract void CSRMM(int m, int k, int n, int nnz, ref double alpha, double[] csrValA, int[] csrRowA, int[] csrColA, double[] B, ref double beta, double[] C, cusparseMatDescr descrA, cusparseOperation op = cusparseOperation.NonTranspose, int ldb = 0, int ldc = 0);

        /// <summary>
        /// Performs matrix-matrix operations. A is CSR format matrix and B, C is dense format.
        /// C = alpha * op(A) * B + beta * C
        /// </summary>
        /// <param name="m">number of rows of matrix A; m must be at least zero.</param>
        /// <param name="k">number of columns of matrix A; k must be at least zero.</param>
        /// <param name="n">number of columns of matrices B and C; n must be at least zero.</param>
        /// <param name="nnz">number of non-zero elements of matrix A.</param>
        /// <param name="alpha">scalar multiplier applied to op(A) * B.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRowA[m] - csrRowA[0].</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColA">array of nnz column indices.</param>
        /// <param name="B">array of dimension (ldb, n).</param>
        /// <param name="beta">scalar multiplier applied to C. If beta is zero, C does not have to be a valid input.</param>
        /// <param name="C">array of dimension (ldc, n).</param>
        /// <param name="op">specifies op(A).</param>
        /// <param name="ldb">leading dimension of B.</param>
        /// <param name="ldc">leading dimension of C.</param>
        public void CSRMM(int m, int k, int n, int nnz, ref double alpha, double[] csrValA, int[] csrRowA, int[] csrColA, double[] B, ref double beta, double[] C, cusparseOperation op = cusparseOperation.NonTranspose, int ldb = 0, int ldc = 0)
        {
            CSRMM(m, k, n, nnz, ref alpha, csrValA, csrRowA, csrColA, B, ref beta, C, defaultMatDescr, op, ldb, ldc);
        }
        #endregion
    }
}
