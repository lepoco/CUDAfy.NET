/*
Linear system solver. Conjugate Gradient, 
Working ..., not completed.

I referred NVIDIA conjugate gradient solver sample code.
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using Cudafy.Maths.BLAS;
using Cudafy.Maths.SPARSE;

namespace Cudafy.Maths.LA
{
    /// <summary>
    /// Linear solver class. (Not implemented. Do not use yet.)
    /// </summary>
    public class Solver
    {
        GPGPU gpu;
        GPGPUBLAS blas;
        GPGPUSPARSE sparse;

        public Solver(GPGPU gpu, GPGPUBLAS blas, GPGPUSPARSE sparse)
        {
            this.gpu = gpu;
            this.blas = blas;
            this.sparse = sparse;

            var km = CudafyModule.TryDeserialize();
            if (km == null || !km.TryVerifyChecksums())
            {
                km = CudafyTranslator.Cudafy();
                km.TrySerialize();
            }

            gpu.LoadModule(km);
        }

        #region CUDAfy Functions
        [Cudafy]
        public static void SetValueGPUSingle(GThread thread, int n, float[] vector, float value)
        {
            int tid = thread.blockIdx.x;

            if (tid < n)
            {
                vector[tid] = value;
            }
        }

        [Cudafy]
        public static void SetValueGPUDouble(GThread thread, int n, double[] vector, double value)
        {
            int tid = thread.blockIdx.x;

            if (tid < n)
            {
                vector[tid] = value;
            }
        }



        [Cudafy]
        public static void DefineLower(GThread thread, int n, int[] rowsICP, int[] colsICP)
        {
            rowsICP[0] = 0;
            colsICP[0] = 0;

            int inz = 1;

            for (int k = 1; k < n; k++)
            {
                rowsICP[k] = inz;
                for (int j = k - 1; j <= k; j++)
                {
                    colsICP[inz] = j;
                    inz++;
                }
            }

            rowsICP[n] = inz;
        }

        [Cudafy]
        public static void CopyAIntoH(GThread thread, int n, float[] vals, int[] rows, float[] valsICP, int[] rowsICP)
        {
            int tid = thread.blockIdx.x;

            if (tid == 0)
            {
                valsICP[0] = vals[0];
            }
            else if (tid < n)
            {
                valsICP[rowsICP[tid]] = vals[rows[tid]];
                valsICP[rowsICP[tid] + 1] = vals[rows[tid] + 1];
            }
        }

        [Cudafy]
        public static void ConstructH(GThread thread, int n, float[] valsICP, int[] rowsICP)
        {
            int tid = thread.blockIdx.x;

            if (tid < n)
            {
                valsICP[rowsICP[tid + 1] - 1] = (float)Math.Sqrt(valsICP[rowsICP[tid + 1] - 1]);

                if (tid < n - 1)
                {
                    valsICP[rowsICP[tid + 1]] /= valsICP[rowsICP[tid + 1] - 1];
                    valsICP[rowsICP[tid + 1] + 1] -= valsICP[rowsICP[tid + 1]] * valsICP[rowsICP[tid + 1]];
                }
            }
        }

        #endregion

        #region help Functions
        public void SetValue(int n, float[] x, float val)
        {
            gpu.Launch(n, 1).SetValueGPUSingle(n, x, val);
        }

        public void SetValue(int n, double[] x, double val)
        {
            gpu.Launch(n, 1).SetValueGPUDouble(n, x, val);
        }
        #endregion

        #region Conjugate gradient solver (CG)
        /// <summary>
        /// Solves symmetric linear system with conjugate gradient solver.
        /// A * x = b
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRowA[m] - csrRowA[0].</param>
        /// <param name="csrRowA">array of n+1 index elements.</param>
        /// <param name="csrColA">array of nnz column indices.</param>
        /// <param name="dx">vector of n elements.</param>
        /// <param name="db">vector of n elements.</param>
        /// <param name="dp">vector of n elements. (temporary vector)</param>
        /// <param name="dAx">vector of n elements. (temporary vector)</param>
        /// <param name="tolerence">iterate tolerence of conjugate gradient solver.</param>
        /// <param name="maxIterate">max iterate count of conjugate gradient solver.</param>
        /// <returns>if A has singulrarity or failure in max iterate count, returns false. return true otherwise.</returns>
        public SolveResult CG(
            int n, int nnz, float[] csrValA, int[] csrRowA, int[] csrColA,
            float[] dx, float[] db, float[] dp, float[] dAx, float tolerence = 0.00001f, int maxIterate = 300)
        {
            SolveResult result = new SolveResult();
            int k; // Iterate count.
            float a, b, r0, r1;
            float zero = 0.0f;
            float one = 1.0f;

            if (blas.DOT(db, db) == 0)
            {
                SetValue(n, dx, 0);
                result.IsSuccess = true;

                return result;
            }

            sparse.CSRMV(n, n, nnz, ref one, csrValA, csrRowA, csrColA, dx, ref zero, dAx);
            blas.AXPY(-1.0f, dAx, db);

            r1 = blas.DOT(db, db);

            k = 1;
            r0 = 0;

            while (true)
            {
                if (k > 1)
                {
                    b = r1 / r0;
                    blas.SCAL(b, dp);
                    blas.AXPY(1.0f, db, dp);
                }
                else
                {
                    blas.COPY(db, dp);
                }

                sparse.CSRMV(n, n, nnz, ref one, csrValA, csrRowA, csrColA, dp, ref zero, dAx);
                a = r1 / blas.DOT(dp, dAx);
                blas.AXPY(a, dp, dx);
                blas.AXPY(-a, dAx, db);

                r0 = r1;
                r1 = blas.DOT(db, db);

                k++;

                if (r1 <= tolerence * tolerence)
                {
                    result.IsSuccess = true;
                    result.LastError = r1;
                    result.IterateCount = k;
                    break;
                }

                if (k > maxIterate)
                {
                    result.IsSuccess = false;
                    result.LastError = r1;
                    result.IterateCount = k;
                    break;
                }
            }

            return result;
        }

        public SolveResult CG(
            int n, int nnz, double[] csrValA, int[] csrRowA, int[] csrColA,
            double[] dx, double[] db, double[] dp, double[] dAx, double tolerence = 0.00001f, int maxIterate = 300)
        {
            SolveResult result = new SolveResult();
            int k; // Iterate count.
            double a, b, r0, r1;
            double zero = 0.0;
            double one = 1.0;

            if (blas.DOT(db, db) == 0.0)
            {
                SetValue(n, dx, 0);
                result.IsSuccess = true;

                return result;
            }

            sparse.CSRMV(n, n, nnz, ref one, csrValA, csrRowA, csrColA, dx, ref zero, dAx);
            blas.AXPY(-1.0f, dAx, db);

            r1 = blas.DOT(db, db);

            k = 1;
            r0 = 0;

            while (true)
            {
                if (k > 1)
                {
                    b = r1 / r0;
                    blas.SCAL(b, dp);
                    blas.AXPY(1.0f, db, dp);
                }
                else
                {
                    blas.COPY(db, dp);
                }

                sparse.CSRMV(n, n, nnz, ref one, csrValA, csrRowA, csrColA, dp, ref zero, dAx);
                a = r1 / blas.DOT(dp, dAx);
                blas.AXPY(a, dp, dx);
                blas.AXPY(-a, dAx, db);

                r0 = r1;
                r1 = blas.DOT(db, db);

                k++;

                if (r1 <= tolerence * tolerence)
                {
                    result.IsSuccess = true;
                    result.LastError = r1;
                    result.IterateCount = k;
                    break;
                }

                if (k > maxIterate)
                {
                    result.IsSuccess = false;
                    result.LastError = r1;
                    result.IterateCount = k;
                    break;
                }
            }

            return result;
        }
        #endregion

        #region Preconditioned conjugate gradient solver (Preconditioned CG) (Not inplemented yet)
        public SolveResult CGPreconditioned(
            int n, int nnz, float[] csrValA, int[] csrRowA, int[] csrColA, float[] dx, float[] db,
            float[] csrValICP, int[] csrRowICP, int[] csrColICP,
            float[] dy, float[] dp, float[] domega, float[] zm1, float[] zm2, float[] rm2, float tolerence = 0.0001f, int maxIterate = 300)
        {
            SolveResult result = new SolveResult();

            // Make Incomplete Cholesky Preconditioner.
            gpu.Launch().DefineLower(n, csrRowICP, csrColICP);
            gpu.Launch(n, 1).CopyAIntoH(n, csrValA, csrRowA, csrValICP, csrRowICP);
            gpu.Launch(n, 1).ConstructH(n, csrValICP, csrRowICP);

            cusparseMatDescr descrM = new cusparseMatDescr();
            descrM.MatrixType = cusparseMatrixType.Triangular;
            descrM.FillMode = cusparseFillMode.Lower;
            descrM.IndexBase = cusparseIndexBase.Zero;
            descrM.DiagType = cusparseDiagType.NonUnit;

            cusparseSolveAnalysisInfo info = new cusparseSolveAnalysisInfo();
            sparse.CreateSolveAnalysisInfo(ref info);
            cusparseSolveAnalysisInfo infoTrans = new cusparseSolveAnalysisInfo();
            sparse.CreateSolveAnalysisInfo(ref infoTrans);

            sparse.CSRSV_ANALYSIS(n, nnz, csrValICP, csrRowICP, csrColICP, cusparseOperation.NonTranspose, info, descrM);
            sparse.CSRSV_ANALYSIS(n, nnz, csrValICP, csrRowICP, csrColICP, cusparseOperation.Transpose, infoTrans, descrM);

            int k = 0;
            float r1 = blas.DOT(db, db);
            float alpha, beta;

            float identityFloat = 1.0f;
            float zeroFloat = 0.0f;

            while (true)
            {
                sparse.CSRSV_SOLVE(n, ref identityFloat, csrValICP, csrRowICP, csrColICP, db, dy, cusparseOperation.NonTranspose, info, descrM);
                sparse.CSRSV_SOLVE(n, ref identityFloat, csrValICP, csrRowICP, csrColICP, dy, zm1, cusparseOperation.Transpose, infoTrans, descrM);

                k++;

                if (k == 1)
                {
                    blas.COPY(zm1, dp);
                }
                else
                {
                    beta = blas.DOT(db, zm1) / blas.DOT(rm2, zm2);
                    blas.SCAL(beta, dp);
                    blas.AXPY(1.0f, zm1, dp);
                }

                sparse.CSRMV(n, n, nnz, ref identityFloat, csrValA, csrRowA, csrColA, dp, ref zeroFloat, domega);
                alpha = blas.DOT(db, zm1) / blas.DOT(dp, domega);

                blas.AXPY(alpha, dp, dx);
                blas.COPY(db, rm2);
                blas.COPY(zm1, zm2);
                blas.AXPY(-alpha, domega, db);

                r1 = blas.DOT(db, db);

                if (r1 <= tolerence * tolerence)
                {
                    result.IsSuccess = true;
                    result.IterateCount = k;
                    result.LastError = r1;
                    break;
                }
                if (k > maxIterate)
                {
                    result.IsSuccess = false;
                    result.IterateCount = k;
                    result.LastError = r1;
                    break;
                }
            }

            return result;
        }

        #endregion

        #region Biconjugate gradient stabilized method (BiCGSTAB) Alpha version (Reducing memory required.)
        /// <summary>
        /// Solve linear system with Biconjugate gradient stabilized method (BiCGSTAB).
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRowA[m] - csrRowA[0].</param>
        /// <param name="csrRowA">array of n+1 index elements.</param>
        /// <param name="csrColA">array of nnz column indices.</param>
        /// <param name="x">vector of n elements. (updated after solving.)</param>
        /// <param name="b">vector of n elements.</param>
        /// <param name="ax">temporary memory for BiCGSTAB.</param>
        /// <param name="r0">temporary memory for BiCGSTAB.</param>
        /// <param name="r">temporary memory for BiCGSTAB.</param>
        /// <param name="v">temporary memory for BiCGSTAB.</param>
        /// <param name="p">temporary memory for BiCGSTAB.</param>
        /// <param name="s">temporary memory for BiCGSTAB.</param>
        /// <param name="t">temporary memory for BiCGSTAB.</param>
        /// <param name="threshold">iterate tolerence of BiCGSTAB solver.</param>
        /// <param name="maxIterate">max iterate count of BiCGSTAB solver.</param>
        /// <returns></returns>
        public SolveResult BiCGSTAB(int n, int nnz, double[] csrValA, int[] csrRowA, int[] csrColA, double[] x, double[] b, double[] ax, double[] r0, double[] r, double[] v, double[] p, double[] s, double[] t, double threshold = 1e-10, int maxIterate = 1000)
        {
            SolveResult result = new SolveResult();
            double l0 = 1.0, alpha = 1.0, w0 = 1.0;
            double l1, beta, w1;
            double bn = blas.NRM2(b);
            int k = 1;

            double minusOne = -1.0;
            double one = 1.0;
            double zero = 0.0;

            blas.COPY(b, r0);
            sparse.CSRMV(n, n, nnz, ref minusOne, csrValA, csrRowA, csrColA, x, ref one, r0);
            blas.COPY(r0, r);

            SetValue(n, v, 0.0);
            SetValue(n, p, 0.0);

            double residual = 0.0;

            while (true)
            {
                l1 = blas.DOT(r0, r);
                beta = (l1 / l0) * (alpha / w0);

                // Update p
                blas.AXPY(-w0, v, p);
                blas.SCAL(beta, p);
                blas.AXPY(1.0, r, p);

                sparse.CSRMV(n, n, nnz, ref one, csrValA, csrRowA, csrColA, p, ref zero, v);

                // Update v
                alpha = l1 / blas.DOT(r0, v);
                blas.COPY(r, s);
                blas.AXPY(-alpha, v, s);
                sparse.CSRMV(n, n, nnz, ref one, csrValA, csrRowA, csrColA, s, ref zero, t);
                w1 = blas.DOT(t, s) / blas.DOT(t, t);

                // Update x
                blas.AXPY(alpha, p, x);
                blas.AXPY(w1, s, x);

                // Update r
                blas.COPY(s, r);
                blas.AXPY(-w1, t, r);

                //reidual = blas.NRM2(r) / bn;
                residual = blas.NRM2(s);

                if (k > maxIterate)
                {
                    result.IterateCount = k;
                    result.IsSuccess = false;
                    result.LastError = residual;
                    return result;
                }

                if (residual <= threshold)
                {
                    result.IterateCount = k;
                    result.IsSuccess = true;
                    result.LastError = residual;
                    return result;
                }

                k++;

                w0 = w1;
                l0 = l1;
            }
        }

        public SolveResult BiCGSTAB(int n, int nnz, float[] csrValA, int[] csrRowA, int[] csrColA, float[] x, float[] b, float[] ax, float[] r0, float[] r, float[] v, float[] p, float[] s, float[] t, float threshold = 0.000001f, int maxIterate = 1000)
        {
            SolveResult result = new SolveResult();
            float l0 = 1.0f, alpha = 1.0f, w0 = 1.0f;
            float l1, beta, w1;
            float bn = blas.NRM2(b);
            int k = 1;

            float minusOne = 1.0f;
            float one = 1.0f;
            float zero = 0.0f;

            blas.COPY(b, r0);
            sparse.CSRMV(n, n, nnz, ref minusOne, csrValA, csrRowA, csrColA, x, ref one, r0);
            blas.COPY(r0, r);

            SetValue(n, v, 0.0f);
            SetValue(n, p, 0.0f);

            double residual = 0.0;

            while (true)
            {
                l1 = blas.DOT(r0, r);
                beta = (l1 / l0) * (alpha / w0);

                // Update p
                blas.AXPY(-w0, v, p);
                blas.SCAL(beta, p);
                blas.AXPY(1.0f, r, p);

                sparse.CSRMV(n, n, nnz, ref one, csrValA, csrRowA, csrColA, p, ref zero, v);

                // Update v
                alpha = l1 / blas.DOT(r0, v);
                blas.COPY(r, s);
                blas.AXPY(-alpha, v, s);
                sparse.CSRMV(n, n, nnz, ref one, csrValA, csrRowA, csrColA, s, ref zero, t);
                w1 = blas.DOT(t, s) / blas.DOT(t, t);

                // Update x
                blas.AXPY(alpha, p, x);
                blas.AXPY(w1, s, x);

                // Update r
                blas.COPY(s, r);
                blas.AXPY(-w1, t, r);

                //reidual = blas.NRM2(r) / bn;
                residual = blas.NRM2(s);

                if (k > maxIterate)
                {
                    result.IterateCount = k;
                    result.IsSuccess = false;
                    result.LastError = residual;
                    return result;
                }

                if (residual <= threshold)
                {
                    result.IterateCount = k;
                    result.IsSuccess = true;
                    result.LastError = residual;
                    return result;
                }

                k++;

                w0 = w1;
                l0 = l1;
            }
        }
        #endregion
    }
}
