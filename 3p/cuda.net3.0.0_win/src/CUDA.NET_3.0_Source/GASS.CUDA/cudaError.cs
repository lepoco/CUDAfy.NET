namespace GASS.CUDA
{
    using System;

    public enum cudaError
    {
        cudaErrorAddressOfConstant = 0x16,
        cudaErrorApiFailureBase = 0x2710,
        cudaErrorCudartUnloading = 0x1d,
        cudaErrorInitializationError = 3,
        cudaErrorInsufficientDriver = 0x23,
        cudaErrorInvalidChannelDescriptor = 20,
        cudaErrorInvalidConfiguration = 9,
        cudaErrorInvalidDevice = 10,
        cudaErrorInvalidDeviceFunction = 8,
        cudaErrorInvalidDevicePointer = 0x11,
        cudaErrorInvalidFilterSetting = 0x1a,
        cudaErrorInvalidHostPointer = 0x10,
        cudaErrorInvalidMemcpyDirection = 0x15,
        cudaErrorInvalidNormSetting = 0x1b,
        cudaErrorInvalidPitchValue = 12,
        cudaErrorInvalidResourceHandle = 0x21,
        cudaErrorInvalidSymbol = 13,
        cudaErrorInvalidTexture = 0x12,
        cudaErrorInvalidTextureBinding = 0x13,
        cudaErrorInvalidValue = 11,
        cudaErrorLaunchFailure = 4,
        cudaErrorLaunchOutOfResources = 7,
        cudaErrorLaunchTimeout = 6,
        cudaErrorMapBufferObjectFailed = 14,
        cudaErrorMemoryAllocation = 2,
        cudaErrorMemoryValueTooLarge = 0x20,
        cudaErrorMissingConfiguration = 1,
        cudaErrorMixedDeviceExecution = 0x1c,
        cudaErrorNoDevice = 0x25,
        cudaErrorNotReady = 0x22,
        cudaErrorNotYetImplemented = 0x1f,
        cudaErrorPriorLaunchFailure = 5,
        cudaErrorSetOnActiveProcess = 0x24,
        cudaErrorStartupFailure = 0x7f,
        cudaErrorSynchronizationError = 0x19,
        cudaErrorTextureFetchFailed = 0x17,
        cudaErrorTextureNotBound = 0x18,
        cudaErrorUnknown = 30,
        cudaErrorUnmapBufferObjectFailed = 15,
        cudaErrorIncompatibleDriverContext = 49,// NK210211
        cudaSuccess = 0
    }
}

