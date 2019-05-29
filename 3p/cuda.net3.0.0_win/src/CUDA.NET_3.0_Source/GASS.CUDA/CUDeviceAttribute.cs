namespace GASS.CUDA
{
    using System;

    public enum CUDeviceAttributeOld
    {
        MaxThreadsPerBlock = 1,        /**< Maximum number of threads per block */
        MaxBlockDimX = 2,              /**< Maximum block dimension X */
        MaxBlockDimY = 3,              /**< Maximum block dimension Y */
        MaxBlockDimZ = 4,              /**< Maximum block dimension Z */
        MaxGridDimX = 5,               /**< Maximum grid dimension X */
        MaxGridDimY = 6,               /**< Maximum grid dimension Y */
        MaxGridDimZ = 7,               /**< Maximum grid dimension Z */

    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,        /**< Maximum shared memory available per block in bytes */
    CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,            /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK */
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,              /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,                         /**< Warp size in threads */
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,                         /**< Maximum pitch in bytes allowed by memory copies */
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,           /**< Maximum number of 32-bit registers available per block */
    CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,               /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK */
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,                        /**< Peak clock frequency in kilohertz */
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,                 /**< Alignment requirement for textures */
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,                       /**< Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT. */
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,              /**< Number of multiprocessors on device */
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,               /**< Specifies whether there is a run time limit on kernels */
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,                        /**< Device is integrated with host memory */
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,               /**< Device can map host memory into CUDA address space */
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,                      /**< Compute mode (See ::CUcomputemode for details) */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,           /**< Maximum 1D texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,           /**< Maximum 2D texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,          /**< Maximum 2D texture height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,           /**< Maximum 3D texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,          /**< Maximum 3D texture height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,           /**< Maximum 3D texture depth */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,   /**< Maximum 2D layered texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,  /**< Maximum 2D layered texture height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,  /**< Maximum layers in a 2D layered texture */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,     /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,    /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29, /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS */
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,                 /**< Alignment requirement for surfaces */
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,                /**< Device can possibly execute multiple kernels concurrently */
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,                       /**< Device has ECC support enabled */
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,                        /**< PCI bus ID of the device */
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,                     /**< PCI device ID of the device */
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,                        /**< Device is using TCC driver model */
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,                 /**< Peak memory clock frequency in kilohertz */
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,           /**< Global memory bus width in bits */
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,                     /**< Size of L2 cache in bytes */
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,    /**< Maximum resident threads per multiprocessor */
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,                /**< Number of asynchronous engines */
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,                /**< Device shares a unified address space with the host */    
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,   /**< Maximum 1D layered texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,  /**< Maximum layers in a 1D layered texture */
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,                  /**< Deprecated, do not use. */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,    /**< Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,   /**< Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47, /**< Alternate maximum 3D texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,/**< Alternate maximum 3D texture height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49, /**< Alternate maximum 3D texture depth */
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,                     /**< PCI domain ID of the device */
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,           /**< Pitch alignment requirement for textures */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,      /**< Maximum cubemap texture width/height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,  /**< Maximum cubemap layered texture width/height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54, /**< Maximum layers in a cubemap layered texture */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,           /**< Maximum 1D surface width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,           /**< Maximum 2D surface width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,          /**< Maximum 2D surface height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,           /**< Maximum 3D surface width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,          /**< Maximum 3D surface height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,           /**< Maximum 3D surface depth */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,   /**< Maximum 1D layered surface width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,  /**< Maximum layers in a 1D layered surface */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,   /**< Maximum 2D layered surface width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,  /**< Maximum 2D layered surface height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,  /**< Maximum layers in a 2D layered surface */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,      /**< Maximum cubemap surface width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,  /**< Maximum cubemap layered surface width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68, /**< Maximum layers in a cubemap layered surface */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,    /**< Maximum 1D linear texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,    /**< Maximum 2D linear texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,   /**< Maximum 2D linear texture height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,    /**< Maximum 2D linear texture pitch in bytes */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73, /**< Maximum mipmapped 2D texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,/**< Maximum mipmapped 2D texture height */
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,          /**< Major compute capability version number */     
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,          /**< Minor compute capability version number */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77, /**< Maximum mipmapped 1D texture width */
    CU_DEVICE_ATTRIBUTE_MAX
} 

    public enum CUDeviceAttribute
    {
        MaxThreadsPerBlock = 1,
        MaxBlockDimX = 2,
        MaxBlockDimY = 3,
        MaxBlockDimZ = 4,
        MaxGridDimX = 5,
        MaxGridDimY = 6,
        MaxGridDimZ = 7,
        MaxSharedMemoryPerBlock = 8,
        [Obsolete("Use MaxSharedMemoryPerBlock")]
        SharedMemoryPerBlock = 8,
        TotalConstantMemory = 9,
        WarpSize = 10,
        MaxPitch = 11,
        [Obsolete("Use MaxRegistersPerBlock")]
        RegistersPerBlock = 12,
        MaxRegistersPerBlock = 12,
        ClockRate = 13,
        TextureAlignment = 14,
        [Obsolete("Use AsyncEngineCount")]
        GPUOverlap = 15,
        MultiProcessorCount = 16,
        KernelExecTimeout = 17,
        Integrated = 18,
        CanMapHostMemory = 19,
        ComputeMode = 20,
        MaximumTexture1DWidth = 21,
        MaximumTexture2DWidth = 22,
        MaximumTexture2DHeight = 23,
        MaximumTexture3DWidth = 24,
        MaximumTexture3DHeight = 25,
        MaximumTexture3DDepth = 26,

        [Obsolete("Use MaximumTexture2DArrayLayeredWidth")]
        MaximumTexture2DArrayWidth = 27,
        [Obsolete("Use MaximumTexture2DArrayLayeredHeight")]
        MaximumTexture2DArrayHeight = 28,
        [Obsolete("Use MaximumTexture2DArrayLayeredLayers")]
        MaximumTexture2DArrayNumSlices = 29,
        MaximumTexture2DArrayLayeredWidth = 27,   /**< Maximum 2D layered texture width */
        MaximumTexture2DArrayLayeredHeight = 28,  /**< Maximum 2D layered texture height */
        MaximumTexture2DArrayLayeredLayers = 29,  /**< Maximum layers in a 2D layered texture */

        SurfaceAlignment = 30,                 /**< Pitch requirement for surfaces */
        ConcurrentKernels = 31,                /**< Device can possibly execute multiple kernels concurrently */
        ECCEnabled = 32,                       /**< Device has ECC support enabled */
        PCIBusId = 33,                        /**< PCI bus ID of the device */
        PCIDeviceID = 34,                     /**< PCI device ID of the device */
        TCCDriver = 35,                        /**< Device is using TCC driver model */
        MemoryClockRate = 36,                 /**< Peak memory clock frequency in kilohertz */
        GlobalMemoryBusWidth = 37,           /**< Global memory bus width in bits */
        L2CacheSize = 38,                     /**< Size of L2 cache in bytes */
        MaxThreadsPerMultiprocessor = 39,    /**< Maximum resident threads per multiprocessor */
        AsyncEngineCount = 40,                /**< Number of asynchronous engines */
        UnifiedAddressing = 41,                /**< Device shares a unified address space with the host */
        MaximumTexture1DLayeredWidth = 42,   /**< Maximum 1D layered texture width */
        MaximumTexture1DLayeredLayers = 43,  /**< Maximum layers in a 1D layered texture */
        //CANTEX2DGather = 44,                  /**< Deprecated, do not use. */
        MaximumTexture2DGatherWidth = 45,    /**< Maximum 2D texture width if CUDAARRAY3DTextureGather is set */
        MaximumTexture2DGatherHeight = 46,   /**< Maximum 2D texture height if CUDAARRAY3DTextureGather is set */
        MaximumTexture3DWidthAlternate = 47, /**< Alternate maximum 3D texture width */
        MaximumTexture3DHeightAlternate = 48,/**< Alternate maximum 3D texture height */
        MaximumTexture3DDepthAlternate = 49, /**< Alternate maximum 3D texture depth */
        PCIDomainID = 50,                     /**< PCI domain ID of the device */
        TexturePitchAlignment = 51,           /**< Pitch alignment requirement for textures */
        MaximumTextureCubeMapWidth = 52,      /**< Maximum cubemap texture width/height */
        MaximumTextureCubeMapLayeredWidth = 53,  /**< Maximum cubemap layered texture width/height */
        MaximumTextureCubeMapLayeredLayers = 54, /**< Maximum layers in a cubemap layered texture */
        MaximumSurface1DWidth = 55,           /**< Maximum 1D surface width */
        MaximumSurface2DWidth = 56,           /**< Maximum 2D surface width */
        MaximumSurface2DHeight = 57,          /**< Maximum 2D surface height */
        MaximumSurface3DWidth = 58,           /**< Maximum 3D surface width */
        MaximumSurface3DHeight = 59,          /**< Maximum 3D surface height */
        MaximumSurface3DDepth = 60,           /**< Maximum 3D surface depth */
        MaximumSurface1DLayeredWidth = 61,   /**< Maximum 1D layered surface width */
        MaximumSurface1DLayeredLayers = 62,  /**< Maximum layers in a 1D layered surface */
        MaximumSurface2DLayeredWidth = 63,   /**< Maximum 2D layered surface width */
        MaximumSurface2DLayeredHeight = 64,  /**< Maximum 2D layered surface height */
        MaximumSurface2DLayeredLayers = 65,  /**< Maximum layers in a 2D layered surface */
        MaximumSurfaceCubeMapWidth = 66,      /**< Maximum cubemap surface width */
        MaximumSurfaceCubeMapLayeredWidth = 67,  /**< Maximum cubemap layered surface width */
        MaximumSurfaceCubeMapLayeredLayers = 68, /**< Maximum layers in a cubemap layered surface */
        MaximumTexture1DLinearWidth = 69,    /**< Maximum 1D linear texture width */
        MaximumTexture2DLinearWidth = 70,    /**< Maximum 2D linear texture width */
        MaximumTexture2DLinearHeight = 71,   /**< Maximum 2D linear texture height */
        MaximumTexture2DLinearPitch = 72,    /**< Maximum 2D linear texture pitch in bytes */
        MaximumTexture2DMipmappedWidth = 73, /**< Maximum mipmapped 2D texture width */
        MaximumTexture2DMipmappedHeight = 74,/**< Maximum mipmapped 2D texture height */
        ComputeCapabilityMajor = 75,          /**< Major compute capability version number */
        ComputeCapabilityMinor = 76,          /**< Minor compute capability version number */
        MaximumTexture1DMipmappedWidth = 77, /**< Maximum mipmapped 1D texture width */
        MAX   
        
        
    }
}

