namespace GASS.CUDA
{
    using System;

    [Flags]
    public enum CUMemHostAllocFlags
    {
        DeviceMap = 2,
        Portable = 1,
        WriteCombined = 4
    }
}

