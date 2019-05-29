namespace Cudafy.Maths.SPARSE
{
    /// <summary>
    /// This type indicates whether the scalar values are passed by reference on the host or device.
    /// </summary>
    public enum cusparsePointerMode
    {
        /// <summary>
        /// The scalars are passed by reference on the host.
        /// </summary>
        Host = 0,

        /// <summary>
        /// The scalars are passed by reference on the device.
        /// </summary>
        Device = 1
    }
}
