namespace GASS.CUDA.BLAS
{
    using System;

    public enum CUBLASStatusv2
    {
        Success = 0,
        NotInitialized = 1,
        AllocFailed = 3,
        InvalidValue = 7,
        ArchMismatch = 8,
        MappingError = 11,
        ExecutionFailed = 13,
        InternalError = 14
    }
}


