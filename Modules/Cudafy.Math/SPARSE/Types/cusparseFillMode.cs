/* Added by Kichang Kim (kkc0923@hotmail.com) */
namespace Cudafy.Maths.SPARSE
{
    /// <summary>
    /// This type indicates if the lower or upper part of a matrix is stored in sparse storage.
    /// </summary>
    public enum cusparseFillMode
    {
        /// <summary>
        /// The lower triangular part is stored.
        /// </summary>
        Lower = 0,

        /// <summary>
        /// The upper triangular part is stored.
        /// </summary>
        Upper = 1
    }
}
