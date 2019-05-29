namespace GASS.CUDA
{
    using System;

    public enum CUResult
    {
        ECCUncorrectable = 0xd6,
        ErrorAlreadyAcquired = 210,
        ErrorAlreadyMapped = 0xd0,
        ErrorArrayIsMapped = 0xcf,
        ErrorContextAlreadyCurrent = 0xca,
        ErrorDeinitialized = 4,
        ErrorFileNotFound = 0x12d,
        ErrorInvalidContext = 0xc9,
        ErrorInvalidDevice = 0x65,
        ErrorInvalidHandle = 400,
        ErrorInvalidImage = 200,
        ErrorInvalidSource = 300,
        ErrorInvalidValue = 1,
        ErrorLaunchFailed = 700,
        ErrorLaunchIncompatibleTexturing = 0x2bf,
        ErrorLaunchOutOfResources = 0x2bd,
        ErrorLaunchTimeout = 0x2be,
        ErrorMapFailed = 0xcd,
        ErrorNoBinaryForGPU = 0xd1,
        ErrorNoDevice = 100,
        ErrorNotFound = 500,
        ErrorNotInitialized = 3,
        ErrorNotMapped = 0xd3,
        ErrorNotReady = 600,
        ErrorOutOfMemory = 2,
        ErrorUnknown = 0x3e7,
        ErrorUnmapFailed = 0xce,
        NotMappedAsArray = 0xd4,
        NotMappedAsPointer = 0xd5,
        PointerIs64Bit = 800,
        SizeIs64Bit = 0x321,
        Success = 0,
        ErrorLaunchTimeOut = 702,
        ErrorPeerAccessNotEnabled = 705,
        ErrorPeerAccessAlreadyEnabled = 704,
        ErrorPrimaryContextActive = 708,
        ErrorContextIsDestroyed = 709,
        ErrorAssert = 710,
        ErrorTooManyPeers = 711,
        ErrorHostMemoryAlreadyInitialized = 712,
        ErrorHostMemoryNotRegistered = 713

    }
}

