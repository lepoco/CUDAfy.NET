/* Added by Kichang Kim (kkc0923@hotmail.com) */
namespace Cudafy.Maths.SPARSE
{
    using System;

    /// <summary>
    /// This is a status type returned by the library functions.
    /// </summary>
    public enum CUSPARSEStatus
    {
        /// <summary>
        /// The operation completed successfully.
        /// </summary>
        Success = 0,

        /// <summary>
        /// The CUSPARSE library was not initialized.
        /// </summary>
        NotInitialized = 1,
        AllocFailed = 2,
        InvalidValue = 3,
        ArchMismatch = 4,
        MappingError = 5,
        ExecutionFailed = 6,
        InternalError = 7,
        MatrixTypeNotSupported = 8
    }
}

