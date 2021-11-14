namespace GASS.CUDA
{
    using System;

    public enum cudaMemcpyKind
    {
        cudaMemcpyHostToHost,
        cudaMemcpyHostToDevice,
        cudaMemcpyDeviceToHost,
        cudaMemcpyDeviceToDevice
    }
}

