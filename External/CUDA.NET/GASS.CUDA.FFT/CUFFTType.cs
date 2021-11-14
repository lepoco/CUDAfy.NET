namespace GASS.CUDA.FFT
{
    using System;

    public enum CUFFTType
    {
        C2C = 0x29,
        C2R = 0x2c,
        D2Z = 0x6a,
        R2C = 0x2a,
        Z2D = 0x6c,
        Z2Z = 0x69
    }
}

