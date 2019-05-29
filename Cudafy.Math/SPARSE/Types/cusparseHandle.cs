/* Added by Kichang Kim (kkc0923@hotmail.com) */
namespace Cudafy.Maths.SPARSE
{
    using System;
    using System.Runtime.InteropServices;

    /// <summary>
    /// This is a pointer type to an opaque CUSPARSE context, which the user must initialize by calling cusparseCreate() prior to calling any other library function.
    /// The Handle created and retruned by cusparseCreate() must be passed to every CUSPARSE function.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseHandle
    {
        public ulong handle;
    }
}
