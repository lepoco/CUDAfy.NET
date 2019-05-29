namespace GASS.CUDA.BLAS
{
    using System;

    public enum CUBLASStatus
    {
        AllocFailed = 3,
        ArchMismatch = 8,
        ExecutionFailed = 13,
        InternalError = 14,
        InvalidValue = 7,
        MappingError = 11,
        NotInitialized = 1,
        Success = 0
    }
}

