/* Added by Kichang Kim (kkc0923@hotmail.com) */
namespace Cudafy.Maths.SPARSE
{
    /// <summary>
    /// This type indicates whether the elements of a dense matrix should be parsed by rows or by columns.
    /// </summary>
    public enum cusparseDirection
    {
        /// <summary>
        /// The matrix should be parsed by rows.
        /// </summary>
        Row = 0,
        /// <summary>
        /// The matrix should be parsed by columns.
        /// </summary>
        Column = 1
    }
}
