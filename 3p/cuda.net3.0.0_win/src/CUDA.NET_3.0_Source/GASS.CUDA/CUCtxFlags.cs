namespace GASS.CUDA
{
    using System;

    public enum CUCtxFlags
    {
        BlockingSync = 4,
        FlagsMask = 0x1f,
        LMemResizeToMax = 0x10,
        MapHost = 8,
        SchedAuto = 0,
        SchedMask = 3,
        SchedSpin = 1,
        SchedYield = 2
    }
}

