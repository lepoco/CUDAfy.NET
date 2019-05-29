/* Added by Kichang Kim (kkc0923@hotmail.com) */
namespace Cudafy.Maths.SPARSE
{
    using System;
    using System.Runtime.InteropServices;

    /// <summary>
    /// The structure is used to describe the shape and properties of a matrix.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseMatDescr
    {
        public cusparseMatrixType MatrixType;
        public cusparseFillMode FillMode;
        public cusparseDiagType DiagType;
        public cusparseIndexBase IndexBase;

        public static cusparseMatDescr DefaultTriangular()
        {
            cusparseMatDescr descr = new cusparseMatDescr();
            descr.MatrixType = cusparseMatrixType.Triangular;
            descr.FillMode = cusparseFillMode.Lower;
            descr.DiagType = cusparseDiagType.NonUnit;
            descr.IndexBase = cusparseIndexBase.Zero;

            return descr;
        }
    }
}
