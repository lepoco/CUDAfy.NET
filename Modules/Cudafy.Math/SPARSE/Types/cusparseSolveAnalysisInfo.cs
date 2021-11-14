/* Added by Kichang Kim (kkc0923@hotmail.com) */
namespace Cudafy.Maths.SPARSE
{
    using System;
    using System.Runtime.InteropServices;

    /// <summary>
    /// This is a pointer type to an opaque structure holding the information collected in the analysis phase of the solution of the sparse triangular linear system.
    /// It is expected to be passed unchanged to the solution phase of the sparse triangular linear system.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseSolveAnalysisInfo
    {
        public uint ptr;
    }
}
