/* Added by Kichang Kim (kkc0923@hotmail.com) */
namespace Cudafy.Maths.SPARSE
{
    /// <summary>
    /// This type indicates which operations need to be performed with sparse matrix.
    /// </summary>
    public enum cusparseOperation
    {
        /// <summary>
        /// The non-transpose operation is selected.
        /// </summary>
        NonTranspose = 0,
        
        /// <summary>
        /// The transpose operation is selected.
        /// </summary>
        Transpose = 1,

        /// <summary>
        /// The conjugate transpose operation is selected.
        /// </summary>
        ConjugateTranspose = 2
    }
}
