namespace GASS.CUDA
{
    using System;

    public enum CUFunctionAttribute
    {
        MaxThreadsPerBlock,
        SharedSizeBytes,
        ConstSizeBytes,
        LocalSizeBytes,
        NumRegs,
        PTXVersion,
        BinaryVersion,
        Max
    }
}

