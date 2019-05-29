namespace Cudafy.Maths.SPARSE
{
    /// <summary>
    /// This type indicates whether the operation is performed only on indices or on data and indices.
    /// </summary>
    public enum cusparseAction
    {
        /// <summary>
        /// The operation is performed only on indices.
        /// </summary>
        Symbolic = 0,
        /// <summary>
        /// The operation is performed on data and indices.
        /// </summary>
        Numeric = 1
    }
}
