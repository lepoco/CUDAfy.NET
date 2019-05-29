namespace GASS.CUDA
{
    using GASS.CUDA.Types;
    using GASS.Types;
    using System;
    using System.Runtime.InteropServices;

    public sealed class CUDADriver
    {
        public const uint CU_MEMHOSTALLOC_DEVICEMAP = 2;
        public const uint CU_MEMHOSTALLOC_PORTABLE = 1;
        public const uint CU_MEMHOSTALLOC_WRITECOMBINED = 4;
        public const int CU_PARAM_TR_DEFAULT = -1;
        public const int CU_TRSA_OVERRIDE_FORMAT = 1;
        public const int CU_TRSF_NORMALIZED_COORDINATES = 2;
        public const int CU_TRSF_READ_AS_INTEGER = 1;
        public const uint CUDA_ARRAY3D_2DARRAY = 1;
        public const uint CU_MEMHOSTREGISTER_PORTABLE = 1;
        public const uint CU_MEMHOSTREGISTER_DEVICEMAP = 2;
#if LINUX
        internal const string CUDA_DLL_NAME = "libcuda";
#else
        internal const string CUDA_DLL_NAME = "nvcuda";
#endif
        


        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuMemHostRegister(IntPtr hostPtr, SizeT bytes, uint flags);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuMemHostUnregister(IntPtr hostPtr);
        
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuDeviceCanAccessPeer(ref int canAccessPeer, CUdevice dev, CUdevice peerDev);

        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuCtxEnablePeerAccess(CUcontext peerContext, uint Flags);

        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuCtxDisablePeerAccess(CUcontext peerContext);

        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuMemcpyPeer (CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, SizeT ByteCount, CUstream hStream);

        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuPointerGetAttribute(ref IntPtr data, CUPointerAttribute attribute, CUdeviceptr ptr);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuPointerGetAttribute(ref CUcontext ctx, CUPointerAttribute attribute, CUdeviceptr ptr);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuPointerGetAttribute(ref CUMemoryType data, CUPointerAttribute attribute, CUdeviceptr ptr);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuPointerGetAttribute(ref CUP2PTokens data, CUPointerAttribute attribute, CUdeviceptr ptr);


        [DllImport(CUDA_DLL_NAME, EntryPoint="cuArray3DCreate_v2")]
        public static extern CUResult cuArray3DCreate(ref CUarray pHandle, ref CUDAArray3DDescriptor pAllocateArray);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuArray3DGetDescriptor_v2")]
        public static extern CUResult cuArray3DGetDescriptor(ref CUDAArray3DDescriptor pArrayDescriptor, CUarray hArray);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuArrayCreate_v2")]
        public static extern CUResult cuArrayCreate(ref CUarray pHandle, ref CUDAArrayDescriptor pAllocateArray);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuArrayDestroy(CUarray hArray);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuArrayGetDescriptor_v2")]
        public static extern CUResult cuArrayGetDescriptor(ref CUDAArrayDescriptor pArrayDescriptor, CUarray hArray);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuCtxAttach(ref CUcontext pctx, uint flags);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuCtxCreate_v2")]
        public static extern CUResult cuCtxCreate(ref CUcontext pctx, uint flags, CUdevice dev);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuCtxDestroy(CUcontext ctx);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuCtxDestroy_v2(CUcontext ctx);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuCtxDetach(CUcontext ctx);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuCtxGetDevice(ref CUdevice device);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuCtxGetCurrent(ref CUcontext pctx);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuCtxSetCurrent(CUcontext pctx);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuCtxPopCurrent(ref CUcontext pctx);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuCtxPopCurrent_v2(ref CUcontext pctx);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuCtxPushCurrent(CUcontext ctx);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuCtxPushCurrent_v2(CUcontext ctx);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuCtxSynchronize();
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuDeviceComputeCapability(ref int major, ref int minor, CUdevice dev);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuDeviceGet(ref CUdevice device, int ordinal);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuDeviceGetAttribute(ref int pi, CUDeviceAttribute attrib, CUdevice dev);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuDeviceGetCount(ref int count);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuDeviceGetName([Out] byte[] name, int len, CUdevice dev);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuDeviceGetProperties(ref CUDeviceProperties prop, CUdevice dev);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuDeviceTotalMem_v2")]
        public static extern CUResult cuDeviceTotalMem(ref SizeT bytes, CUdevice dev);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuDriverGetVersion(ref int driverVersion);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuEventCreate(ref CUevent phEvent, uint Flags);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuEventDestroy(CUevent hEvent);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuEventDestroy_v2(CUevent hEvent);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuEventElapsedTime(ref float pMilliseconds, CUevent hStart, CUevent hEnd);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuEventQuery(CUevent hEvent);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuEventRecord(CUevent hEvent, CUstream hStream);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuEventSynchronize(CUevent hEvent);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuFuncGetAttribute(ref int pi, CUFunctionAttribute attrib, CUfunction hfunc);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuFuncSetCacheConfig(CUfunction hfunc, CUFunctionCache config);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuFuncSetSharedSize(CUfunction hfunc, uint bytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuGetExportTable([In, Out] IntPtr[] ppExportTable, ref CUuuid pExportTableId);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuGraphicsMapResources(uint count, [In] CUgraphicsResource[] resources, CUstream hStream);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuGraphicsResourceGetMappedPointer_v2")]
        public static extern CUResult cuGraphicsResourceGetMappedPointer(ref CUdeviceptr pDevPtr, ref SizeT pSize, CUgraphicsResource resource);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, uint flags);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuGraphicsSubResourceGetMappedArray(ref CUarray pArray, CUgraphicsResource resource, uint arrayIndex, uint mipLevel);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuGraphicsUnmapResources(uint count, [In] CUgraphicsResource[] resources, CUstream hStream);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuGraphicsUnregisterResource(CUgraphicsResource resource);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuInit(uint Flags);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuLaunch(CUfunction f);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuLaunchGrid(CUfunction f, int grid_width, int grid_height);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemAlloc_v2")]
        public static extern CUResult cuMemAlloc(ref CUdeviceptr dptr, SizeT bytesize);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuMemAllocHost_v2")]
        public static extern CUResult cuMemAllocHost(ref IntPtr pp, SizeT bytesize);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuMemAllocPitch_v2")]
        public static extern CUResult cuMemAllocPitch(ref CUdeviceptr dptr, ref SizeT pPitch, SizeT WidthInBytes, SizeT Height, uint ElementSizeBytes);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpy2D_v2")]
        public static extern CUResult cuMemcpy2D(ref CUDAMemCpy2D pCopy);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpy2DAsync_v2")]
        public static extern CUResult cuMemcpy2DAsync(ref CUDAMemCpy2D pCopy, CUstream hStream);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpy2DUnaligned_v2")]
        public static extern CUResult cuMemcpy2DUnaligned(ref CUDAMemCpy2D pCopy);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpy3D_v2")]
        public static extern CUResult cuMemcpy3D(ref CUDAMemCpy3D pCopy);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpy3DAsync_v2")]
        public static extern CUResult cuMemcpy3DAsync(ref CUDAMemCpy3D pCopy, CUstream hStream);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuMemcpyAtoA_v2")]
        public static extern CUResult cuMemcpyAtoA(CUarray dstArray, SizeT dstIndex, CUarray srcArray, SizeT srcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuMemcpyAtoD_v2")]
        public static extern CUResult cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray hSrc, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Char1[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Char2[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Char3[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Char4[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] cuDoubleComplex[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] cuDoubleReal[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] cuFloatComplex[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] cuFloatReal[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Double1[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Double2[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Float1[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Float2[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Float3[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Float4[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Int1[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Int2[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Int3[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Int4[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Long1[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Long2[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Long3[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Long4[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Short1[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Short2[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Short3[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] Short4[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] UChar1[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] UChar2[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] UChar3[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] UChar4[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] UInt1[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] UInt2[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] UInt3[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] UInt4[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] ULong1[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] ULong2[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] ULong3[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] ULong4[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] UShort1[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] UShort2[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] UShort3[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] UShort4[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] byte[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] double[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] short[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuMemcpyAtoH([Out] int[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH(IntPtr dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] long[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] sbyte[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] float[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] ushort[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] uint[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2")]
        public static extern CUResult cuMemcpyAtoH([Out] ulong[] dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuMemcpyAtoHAsync_v2")]
        public static extern CUResult cuMemcpyAtoHAsync(IntPtr dstHost, CUarray srcArray, SizeT SrcIndex, SizeT ByteCount, CUstream hStream);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuMemcpyDtoA_v2")]
        public static extern CUResult cuMemcpyDtoA(CUarray dstArray, SizeT dstIndex, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuMemcpyDtoD_v2")]
        public static extern CUResult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoDAsync_v2")]
        public static extern CUResult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, SizeT ByteCount, CUstream hStream);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Char1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Char2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Char3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Char4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] cuDoubleComplex[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] cuDoubleReal[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] cuFloatComplex[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] cuFloatReal[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Double1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Double2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Float1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Float2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Float3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Float4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Int1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Int2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Int3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Int4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Long1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Long2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Long3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Long4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Short1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Short2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Short3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] Short4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] UChar1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] UChar2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] UChar3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] UChar4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] UInt1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] UInt2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] UInt3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] UInt4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] ULong1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] ULong2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] ULong3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] ULong4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] UShort1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] UShort2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH(IntPtr dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] UShort3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] UShort4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] byte[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] double[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] short[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] int[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] long[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] sbyte[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] float[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] ushort[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] uint[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern CUResult cuMemcpyDtoH([Out] ulong[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuMemcpyDtoHAsync_v2")]
        public static extern CUResult cuMemcpyDtoHAsync(IntPtr dstHost, CUdeviceptr srcDevice, SizeT ByteCount, CUstream hStream);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Char1[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Char2[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Char3[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Char4[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] cuDoubleComplex[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] cuDoubleReal[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, IntPtr pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] cuFloatComplex[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] cuFloatReal[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Double1[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Double2[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Float1[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Float2[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Float3[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Float4[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Int1[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Int2[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Int3[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Int4[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Long1[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Long2[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Long3[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Long4[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Short1[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Short2[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Short3[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] Short4[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] UChar1[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] UChar2[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] UChar3[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] UChar4[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] UInt1[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] UInt2[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] UInt3[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] UInt4[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] ULong1[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] ULong2[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] ULong3[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] ULong4[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] UShort1[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] UShort2[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] UShort3[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] UShort4[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] byte[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] double[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] short[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] int[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] long[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] sbyte[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] float[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] ushort[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] uint[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2")]
        public static extern CUResult cuMemcpyHtoA(CUarray dstArray, SizeT dstIndex, [In] ulong[] pSrc, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuMemcpyHtoAAsync_v2")]
        public static extern CUResult cuMemcpyHtoAAsync(CUarray dstArray, SizeT dstIndex, IntPtr pSrc, SizeT ByteCount, CUstream hStream);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Char1[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Char2[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Char3[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Char4[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] cuDoubleComplex[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] cuDoubleReal[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] cuFloatComplex[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] cuFloatReal[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Double1[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Double2[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Float1[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Float2[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Float3[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Float4[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Int1[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Int2[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Int3[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Int4[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Long1[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Long2[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Long3[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Long4[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Short1[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Short2[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Short3[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] Short4[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] UChar1[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] UChar2[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] UChar3[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] UChar4[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] UInt1[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] UInt2[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] UInt3[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] UInt4[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] ULong1[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] ULong2[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] ULong3[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] ULong4[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] UShort1[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] UShort2[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] UShort3[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] UShort4[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] byte[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] double[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] short[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] int[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] long[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] sbyte[] srcHost, SizeT ByteCount);//)
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, IntPtr srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] float[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] ushort[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] uint[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern CUResult cuMemcpyHtoD(CUdeviceptr dstDevice, [In] ulong[] srcHost, SizeT ByteCount);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuMemcpyHtoDAsync_v2")]
        public static extern CUResult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, IntPtr srcHost, SizeT ByteCount, CUstream hStream);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemFree_v2")]
        public static extern CUResult cuMemFree(CUdeviceptr dptr);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuMemFreeHost(IntPtr p);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuMemGetAddressRange_v2")]
        public static extern CUResult cuMemGetAddressRange(ref CUdeviceptr pbase, ref SizeT psize, CUdeviceptr dptr);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuMemGetInfo_v2")]
        public static extern CUResult cuMemGetInfo(ref SizeT free, ref SizeT total);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuMemHostAlloc(ref IntPtr pp, SizeT bytesize, uint Flags);
        //[DllImport(CUDA_DLL_NAME)]
        //public static extern CUResult cuMemHostAlloc(ref IntPtr pp, uint bytesize, uint Flags);
        //[DllImport(CUDA_DLL_NAME)]
        //public static extern CUResult cuMemHostAlloc(ref IntPtr pp, ulong bytesize, uint Flags);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuMemHostGetDevicePointer_v2")]
        public static extern CUResult cuMemHostGetDevicePointer(ref CUdeviceptr pdptr, IntPtr p, uint Flags);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuMemHostGetFlags(ref uint pFlags, ref IntPtr p);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemsetD16_v2")]
        public static extern CUResult cuMemsetD16(CUdeviceptr dstDevice, ushort us, SizeT N);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemsetD2D16_v2")]
        public static extern CUResult cuMemsetD2D16(CUdeviceptr dstDevice, uint dstPitch, ushort us, SizeT Width, SizeT Height);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemsetD2D32_v2")]
        public static extern CUResult cuMemsetD2D32(CUdeviceptr dstDevice, uint dstPitch, uint ui, SizeT Width, SizeT Height);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemsetD2D8_v2")]
        public static extern CUResult cuMemsetD2D8(CUdeviceptr dstDevice, uint dstPitch, byte uc, SizeT Width, SizeT Height);
        [DllImport(CUDA_DLL_NAME, EntryPoint = "cuMemsetD32_v2")]
        public static extern CUResult cuMemsetD32(CUdeviceptr dstDevice, uint ui, SizeT N);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuMemsetD8_v2")]
        public static extern CUResult cuMemsetD8(CUdeviceptr dstDevice, byte uc, SizeT N);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuModuleGetFunction(ref CUfunction hfunc, CUmodule hmod, string name);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuModuleGetGlobal_v2")]
        public static extern CUResult cuModuleGetGlobal(ref CUdeviceptr dptr, ref SizeT bytes, CUmodule hmod, string name);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuModuleGetTexRef(ref CUtexref pTexRef, CUmodule hmod, string name);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuModuleLoad(ref CUmodule module, string fname);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuModuleLoadData(ref CUmodule module, [In] byte[] image);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuModuleLoadDataEx(ref CUmodule module, [In] byte[] image, uint numOptions, [In] CUJITOption[] options, [In] IntPtr[] optionValues);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuModuleLoadFatBinary(ref CUmodule module, [In] byte[] fatCubin);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuModuleUnload(CUmodule hmod);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetf(CUfunction hfunc, int offset, float value);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSeti(CUfunction hfunc, int offset, uint value);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetSize(CUfunction hfunc, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref long value, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, IntPtr ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Char1[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Char2[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Char3[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Char4[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] UChar1[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] UChar2[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] double[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] int[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] long[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Char1 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] uint[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Char2 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] float[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Char3 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] ushort[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Char4 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] ulong[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Double1 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] short[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Float1 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] UChar3[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Double2 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] sbyte[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Float2 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] UChar4[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Float3 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] byte[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Float4 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] UShort1[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Int1 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Float2[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Short1 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Float3[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Short3 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Float1[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Short4 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Int4[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref UChar4 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Int3[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref UShort1 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Int1[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref UShort2 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Int2[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref UInt1 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] UInt1[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Int2 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Short4[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Long2 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] UInt2[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Int3 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] UShort4[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Int4 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] UInt3[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Long1 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] UShort3[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Long4 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Long1[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref UInt3 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Long2[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref UChar1 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Long3[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref UInt2 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] UInt4[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref UChar2 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] ULong1[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Long3 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] ULong2[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref Short2 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] ULong3[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref UChar3 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Long4[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref UInt4 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] ULong4[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref ULong1 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Short3[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref ULong2 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] UShort2[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref ULong3 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Short2[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref ULong4 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Float4[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref UShort3 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Double1[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref UShort4 ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Double2[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, ref double ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] Short1[] ptr, uint numbytes);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuStreamCreate(ref CUstream phStream, uint Flags);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuStreamDestroy(CUstream hStream);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuStreamDestroy_v2(CUstream hStream);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuStreamQuery(CUstream hStream);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuStreamSynchronize(CUstream hStream);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuTexRefCreate(ref CUtexref pTexRef);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuTexRefDestroy(CUtexref hTexRef);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuTexRefGetAddress_v2")]
        public static extern CUResult cuTexRefGetAddress(ref CUdeviceptr pdptr, CUtexref hTexRef);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuTexRefGetAddressMode(ref CUAddressMode pam, CUtexref hTexRef, int dim);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuTexRefGetArray(ref CUarray phArray, CUtexref hTexRef);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuTexRefGetFilterMode(ref CUFilterMode pfm, CUtexref hTexRef);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuTexRefGetFlags(ref uint pFlags, CUtexref hTexRef);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuTexRefGetFormat(ref CUArrayFormat pFormat, ref int pNumChannels, CUtexref hTexRef);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuTexRefSetAddress_v2")]
        public static extern CUResult cuTexRefSetAddress(ref uint ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, SizeT bytes);
        [DllImport(CUDA_DLL_NAME, EntryPoint="cuTexRefSetAddress2D_v2")]
        public static extern CUResult cuTexRefSetAddress2D(CUtexref hTexRef, CUDAArrayDescriptor desc, CUdeviceptr dptr, SizeT Pitch);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuTexRefSetAddress2D_v3(CUtexref hTexRef, CUDAArrayDescriptor desc, CUdeviceptr dptr, SizeT Pitch);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUAddressMode am);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, uint Flags);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuTexRefSetFilterMode(CUtexref hTexRef, CUFilterMode fm);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuTexRefSetFlags(CUtexref hTexRef, uint Flags);
        [DllImport(CUDA_DLL_NAME)]
        public static extern CUResult cuTexRefSetFormat(CUtexref hTexRef, CUArrayFormat fmt, int NumPackedComponents);
        //[DllImport(CUDA_DLL_NAME)]
        //public static extern CUResult cuDriverGetVersion(ref int version);

        //public static extern CUResult cuDeviceGetAttribute(ref int pi, CUDeviceAttribute attrib, CUdevice dev);

        public System.Version Version
        {
            get
            {
                return new System.Version(4, 1);
            }
        }
    }
}

