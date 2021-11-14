namespace GASS.CUDA
{
    using GASS.Types;
    using System;
    using System.Runtime.InteropServices;
    using System.Text;


    //struct cudaDeviceProp {
    //    char name[256];
    //    size_t totalGlobalMem;
    //    size_t sharedMemPerBlock;
    //    int regsPerBlock;
    //    int warpSize;
    //    size_t memPitch;
    //    int maxThreadsPerBlock;
    //    int maxThreadsDim[3];
    //    int maxGridSize[3];
    //    int clockRate;
    //    size_t totalConstMem;
    //    int major;
    //    int minor;
    //    size_t textureAlignment;
    //    size_t texturePitchAlignment;
    //    int deviceOverlap;
    //    int multiProcessorCount;
    //    int kernelExecTimeoutEnabled;
    //    int integrated;
    //    int canMapHostMemory;
    //    int computeMode;
    //    int maxTexture1D;
    //    int maxTexture1DMipmap;
    //    int maxTexture1DLinear;
    //    int maxTexture2D[2];
    //    int maxTexture2DMipmap[2];
    //    int maxTexture2DLinear[3];
    //    int maxTexture2DGather[2];
    //    int maxTexture3D[3];
    //    int maxTextureCubemap;
    //    int maxTexture1DLayered[2];
    //    int maxTexture2DLayered[3];
    //    int maxTextureCubemapLayered[2];
    //    int maxSurface1D;
    //    int maxSurface2D[2];
    //    int maxSurface3D[3];
    //    int maxSurface1DLayered[2];
    //    int maxSurface2DLayered[3];
    //    int maxSurfaceCubemap;
    //    int maxSurfaceCubemapLayered[2];
    //    size_t surfaceAlignment;
    //    int concurrentKernels;
    //    int ECCEnabled;
    //    int pciBusID;
    //    int pciDeviceID;
    //    int pciDomainID;
    //    int tccDriver;
    //    int asyncEngineCount;
    //    int unifiedAddressing;
    //    int memoryClockRate;
    //    int memoryBusWidth;
    //    int l2CacheSize;
    //    int maxThreadsPerMultiProcessor;
    //}

    // 02-11-2011
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct cudaDeviceProp
    {
        public string name
        {
            get { return (new string(nameChar)).Trim('\0'); }
        }

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
        public char[] nameChar;
        public SizeT totalGlobalMem;
        public SizeT sharedMemPerBlock;
        public int regsPerBlock;
        public int warpSize;
        public SizeT memPitch;
        public int maxThreadsPerBlock;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxThreadsDim;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxGridSize;

        public int clockRate;

        public SizeT totalConstMem;
        public int major;
        public int minor;
        
        public SizeT textureAlignment;
        public SizeT texturePitchAlignment;/*4.2*/
        public int deviceOverlap;
        public int multiProcessorCount;
        public int kernelExecTimeoutEnabled;
        public int integrated;
        public int canMapHostMemory;
        public int computeMode;

        public int maxTexture1D;

        public int maxTexture1DMipmap;//021112

        public int maxTexture1DLinear;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture2D;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture2DMipmap;//021112
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxTexture2DLinear;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture2DGather;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxTexture3D;
        public int maxTextureCubemap;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture1DLayered;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxTexture2DLayered;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTextureCubemapLayered;

        public int    maxSurface1D;               /**< Maximum 1D surface size */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[]    maxSurface2D;            /**< Maximum 2D surface dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[]    maxSurface3D;            /**< Maximum 3D surface dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[]    maxSurface1DLayered;     /**< Maximum 1D layered surface dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[]    maxSurface2DLayered;     /**< Maximum 2D layered surface dimensions */
        public int    maxSurfaceCubemap;          /**< Maximum Cubemap surface dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[]    maxSurfaceCubemapLayered;/**< Maximum Cubemap layered surface dimensions */


        public SizeT surfaceAlignment;
        public int concurrentKernels;
        public int ECCEnabled;
        public int pciBusID;
        public int pciDeviceID;
        public int pciDomainID;
        public int tccDriver;
        public int asyncEngineCount;
        public int unifiedAddressing;
        public int memoryClockRate;
        public int memoryBusWidth;
        public int l2CacheSize;
        public int maxThreadsPerMultiProcessor;
    }
}

