/* Added by Kichang Kim (kkc0923@hotmail.com) */
namespace Cudafy.Maths.SPARSE
{
    /// <summary>
    /// This type indicates if the matrix diagonal entries are unity.
    /// </summary>
    public enum cusparseDiagType
    {
        /// <summary>
        /// The matrix diagonal has non-unit elements.
        /// </summary>
        NonUnit = 0,
        /// <summary>
        /// The matrix diagonal has unit elements.
        /// </summary>
        Unit = 1,
    }
}
