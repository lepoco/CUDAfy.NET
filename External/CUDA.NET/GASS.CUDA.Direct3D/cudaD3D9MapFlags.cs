namespace GASS.CUDA.Direct3D
{
    using System;

    [Flags]
    public enum cudaD3D9MapFlags : uint
    {
        None = 0,
        ReadOnly = 1,
        WriteDiscard = 2
    }
}

