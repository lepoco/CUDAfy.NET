namespace GASS.CUDA.FFT
{
    using System;

    public enum CUFFTResult
    {
        Success,
        InvalidPlan,
        AllocFailed,
        InvalidType,
        InvalidValue,
        InternalError,
        ExecFailed,
        SetupFailed,
        InvalidSize,

        UnalignedData
    }
}

