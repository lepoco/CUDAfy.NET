/* Added by Kichang Kim (kkc0923@hotmail.com) */
namespace Cudafy.Maths.SPARSE
{
    /// <summary>
    /// This type indicates the type of matrix stored in sparse storage. Notice that for symmetric, Hermitian and triangular matrices only their lower or upper part is assumed to be stored.
    /// </summary>
    public enum cusparseMatrixType
    {
        /// <summary>
        /// The matrix is general.
        /// </summary>
        General = 0,

        /// <summary>
        /// The matrix is symmetric.
        /// </summary>
        Symmetric = 1,

        /// <summary>
        /// The matrix is Hermitian.
        /// </summary>
        Hermitian = 2,

        /// <summary>
        /// The matrix is triangular.
        /// </summary>
        Triangular = 3
    }
}
