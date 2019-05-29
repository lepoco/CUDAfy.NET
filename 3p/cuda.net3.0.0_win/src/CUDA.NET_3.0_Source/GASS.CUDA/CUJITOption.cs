namespace GASS.CUDA
{
    using System;

    public enum CUJITOption
    {
        MaxRegisters,
        ThreadsPerBlock,
        WallTime,
        InfoLogBuffer,
        InfoLogBufferSizeBytes,
        ErrorLogBuffer,
        ErrorLogBufferSizeBytes,
        OptimizationLevel,
        TargetFromContext,
        Target,
        FallbackStrategy
    }
}

