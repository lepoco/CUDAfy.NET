namespace GASS.CUDA
{
    using GASS.CUDA.Types;
    using GASS.Types;
    using System;
    using System.Runtime.InteropServices;

    public interface ICUDARuntime
    {
        cudaError GetDeviceProperties(ref cudaDeviceProp prop, int device);
    }

    public class CUDARuntime64 : ICUDARuntime
    {

#if LINUX
        internal const string DLL_NAME = "libcudart";
#else
        internal const string DLL_NAME = "cudart64_70";
        internal const string DLL_NAME_PREV = "cudart64_65";
#endif
        [DllImport(DLL_NAME)]
        public static extern cudaError cudaGetDeviceProperties(ref cudaDeviceProp prop, int device);

        [DllImport(DLL_NAME_PREV, EntryPoint = "cudaGetDeviceProperties")]
        public static extern cudaError cudaGetDevicePropertiesPrev(ref cudaDeviceProp prop, int device);

        public cudaError GetDeviceProperties(ref cudaDeviceProp prop, int device)
        {
            try
            {
                return cudaGetDeviceProperties(ref prop, device);
            }
            catch (DllNotFoundException)
            {
                return cudaGetDevicePropertiesPrev(ref prop, device);
            }
        }
    }

    public class CUDARuntime32 : ICUDARuntime
    {

#if LINUX
        internal const string DLL_NAME = "libcudart";
#else
        internal const string DLL_NAME = "cudart32_70";
        internal const string DLL_NAME_PREV = "cudart32_65";
#endif
        [DllImport(DLL_NAME)]
        public static extern cudaError cudaGetDeviceProperties(ref cudaDeviceProp prop, int device);

        [DllImport(DLL_NAME_PREV, EntryPoint = "cudaGetDeviceProperties")]
        public static extern cudaError cudaGetDevicePropertiesPrev(ref cudaDeviceProp prop, int device);

        public cudaError GetDeviceProperties(ref cudaDeviceProp prop, int device)
        {
            try
            {
                return cudaGetDeviceProperties(ref prop, device);
            }
            catch (DllNotFoundException)
            {
                return cudaGetDevicePropertiesPrev(ref prop, device);
            }
        }
    }

    public class CUDARuntime
    {
        public const int cudaDeviceBlockingSync = 4;
        public const int cudaDeviceMapHost = 8;
        public const int cudaDeviceMask = 15;
        public const int cudaDeviceScheduleAuto = 0;
        public const int cudaDeviceScheduleSpin = 1;
        public const int cudaDeviceScheduleYield = 2;
        public const int cudaEventBlockingSync = 1;
        public const int cudaEventDefault = 0;
        public const int cudaHostAllocDefault = 0;
        public const int cudaHostAllocMapped = 2;
        public const int cudaHostAllocPortable = 1;
        public const int cudaHostAllocWriteCombined = 4;

#if LINUX
        internal const string CUDART_DLL_NAME = "libcudart";
#else
        internal const string CUDART_DLL_NAME = "cudart64_65";
#endif
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaBindTexture(ref SizeT offset, ref textureReference texref, CUdeviceptr devPtr, ref cudaChannelFormatDesc desc, SizeT size);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaBindTexture(ref uint offset, ref textureReference texref, CUdeviceptr devPtr, ref cudaChannelFormatDesc desc, uint size);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaBindTexture(ref ulong offset, ref textureReference texref, CUdeviceptr devPtr, ref cudaChannelFormatDesc desc, ulong size);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaBindTexture2D(ref SizeT offset, textureReference texref, CUdeviceptr devPtr, cudaChannelFormatDesc desc, SizeT width, SizeT height, SizeT pitch);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaBindTexture2D(ref uint offset, textureReference texref, CUdeviceptr devPtr, cudaChannelFormatDesc desc, uint width, uint height, uint pitch);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaBindTexture2D(ref ulong offset, textureReference texref, CUdeviceptr devPtr, cudaChannelFormatDesc desc, ulong width, ulong height, ulong pitch);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaBindTextureToArray(ref textureReference texref, ref cudaArray array, ref cudaChannelFormatDesc desc);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaChooseDevice(ref int device, ref cudaDeviceProp prop);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaConfigureCall(Int3 gridDim, Int3 blockDim, SizeT sharedMem, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaConfigureCall(Int3 gridDim, Int3 blockDim, uint sharedMem, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaConfigureCall(Int3 gridDim, Int3 blockDim, ulong sharedMem, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaDriverGetVersion(ref int driverVersion);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaEventCreate(ref cudaEvent e);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaEventCreateWithFlags(ref cudaEvent e, int flags);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaEventDestroy(cudaEvent e);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaEventElapsedTime(ref float ms, cudaEvent start, cudaEvent end);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaEventQuery(cudaEvent e);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaEventRecord(cudaEvent e, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaEventSynchronize(cudaEvent e);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaFree(CUdeviceptr devPtr);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaFreeArray(ref cudaArray array);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaFreeHost(IntPtr ptr);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaFuncGetAttributes(ref cudaFuncAttributes attr, string func);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGetChannelDesc(ref cudaChannelFormatDesc desc, ref cudaArray array);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGetDevice(ref int device);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGetDeviceCount(ref int count);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGetDeviceProperties(ref cudaDeviceProp prop, int device);
        [DllImport(CUDART_DLL_NAME, CharSet=CharSet.Ansi)]
        public static extern string cudaGetErrorString(cudaError error);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGetLastError();
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGetSymbolAddress(ref CUdeviceptr devPtr, string symbol);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGetSymbolSize(ref SizeT size, string symbol);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGetSymbolSize(ref uint size, string symbol);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGetSymbolSize(ref ulong size, string symbol);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGetTextureAlignmentOffset(ref SizeT offset, ref textureReference texref);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGetTextureAlignmentOffset(ref uint offset, ref textureReference texref);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGetTextureAlignmentOffset(ref ulong offset, ref textureReference texref);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGetTextureReference(ref textureReference texref, string symbol);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGraphicsMapResources(int count, [In] cudaGraphicsResource[] resources, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGraphicsResourceGetMappedPointer(ref CUdeviceptr devPtr, SizeT size, ref cudaGraphicsResource resource);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGraphicsResourceSetMapFlags(ref cudaGraphicsResource resource, uint flags);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGraphicsSubResourceGetMappedArray(ref cudaArray arrayPtr, ref cudaGraphicsResource resource, uint arrayIndex, uint mipLevel);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGraphicsUnmapResources(int count, [In] cudaGraphicsResource[] resources, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaGraphicsUnregisterResource(ref cudaGraphicsResource resource);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaHostAlloc(ref IntPtr pHost, SizeT bytes, uint flags);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaHostAlloc(ref IntPtr pHost, uint bytes, uint flags);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaHostAlloc(ref IntPtr pHost, ulong bytes, uint flags);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaHostGetDevicePointer(ref CUdeviceptr pDevice, IntPtr pHost, uint flags);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaHostGetFlags(ref uint flags, IntPtr pHost);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaLaunch(string symbol);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMalloc(ref CUdeviceptr devPtr, SizeT size);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMalloc(ref CUdeviceptr devPtr, uint size);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMalloc(ref CUdeviceptr devPtr, ulong size);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMalloc3D(ref cudaPitchedPtr pitchDevPtr, cudaExtent extent);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMalloc3DArray(ref cudaArray arrayPtr, ref cudaChannelFormatDesc desc, cudaExtent extent);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMallocArray(ref cudaArray array, ref cudaChannelFormatDesc desc, SizeT width, SizeT height);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMallocArray(ref cudaArray array, ref cudaChannelFormatDesc desc, uint width, uint height);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMallocArray(ref cudaArray array, ref cudaChannelFormatDesc desc, ulong width, ulong height);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMallocHost(ref IntPtr ptr, SizeT size);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMallocHost(ref IntPtr ptr, uint size);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMallocHost(ref IntPtr ptr, ulong size);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMallocPitch(ref CUdeviceptr devPtr, ref SizeT pitch, SizeT width, SizeT height);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMallocPitch(ref CUdeviceptr devPtr, ref uint pitch, uint width, uint height);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMallocPitch(ref CUdeviceptr devPtr, ref ulong pitch, ulong width, ulong height);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Char1[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Char1[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Char2[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Char2[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Char3[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Char3[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Char4[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Char4[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] cuDoubleComplex[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] cuDoubleComplex[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] cuDoubleComplex[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] cuDoubleReal[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] cuDoubleReal[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] cuDoubleReal[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] cuFloatComplex[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] cuFloatComplex[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] cuFloatComplex[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] cuFloatReal[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] cuFloatReal[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] cuFloatReal[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Double1[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Double1[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Double1[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Double2[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Double2[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Double2[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Float1[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Float1[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Float1[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Float2[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Float2[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Float2[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Float3[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Float3[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Float3[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Float4[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Float4[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Float4[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Int1[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Int1[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Int2[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Int2[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Int2[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Int3[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Int3[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Int3[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Int4[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Int4[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Int4[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Long1[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Long1[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Long1[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Long2[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Long2[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Long2[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Long3[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Long3[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Long3[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Long4[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Long4[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Long4[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Short1[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Short1[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Short2[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Short2[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Short3[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Short3[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Short4[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Short4[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UChar1[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UChar1[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UChar2[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UChar2[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UChar3[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UChar3[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UChar4[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UChar4[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UInt1[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UInt1[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UInt2[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UInt2[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UInt2[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UInt3[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UInt3[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UInt3[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UInt4[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UInt4[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UInt4[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ULong1[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ULong1[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ULong1[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ULong2[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ULong2[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ULong2[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ULong3[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ULong3[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ULong3[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ULong4[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ULong4[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ULong4[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UShort1[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UShort1[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UShort2[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UShort2[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UShort3[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UShort3[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UShort4[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UShort4[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] byte[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] double[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] double[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] short[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] int[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] long[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] sbyte[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] float[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] float[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ushort[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] uint[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ulong[] dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, IntPtr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] byte[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] short[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] int[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] long[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] sbyte[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ushort[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] uint[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ulong[] dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Char1[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Char1[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Char1[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Char2[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Char2[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Char2[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Char3[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Char3[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Char3[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Char4[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Char4[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Char4[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] cuDoubleComplex[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] cuDoubleComplex[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] cuDoubleComplex[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] cuDoubleReal[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] cuDoubleReal[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] cuDoubleReal[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] cuFloatComplex[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] cuFloatComplex[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] cuFloatComplex[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] cuFloatReal[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] cuFloatReal[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] cuFloatReal[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Double1[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Double1[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Double1[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Double2[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Double2[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Double2[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Float1[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Float1[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Float1[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Float2[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Float2[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Float2[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Float3[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Float3[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Float3[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Float4[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Float4[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Float4[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Int1[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Int1[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, IntPtr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Int1[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Int2[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Int2[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Int2[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Int3[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Int3[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Int3[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Int4[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Int4[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Int4[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Long1[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Long1[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Long1[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Long2[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Long2[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Long2[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Long3[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Long3[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Long3[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Long4[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Long4[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Long4[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Short1[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Short1[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Short1[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Short2[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Short2[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Short2[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Short3[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Short3[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Short3[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Short4[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Short4[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] Short4[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UInt1[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UChar1[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UChar1[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UChar1[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UChar2[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UChar2[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UChar2[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UChar3[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UChar3[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UChar3[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UChar4[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UChar4[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UChar4[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UInt1[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UInt1[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UInt1[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UInt2[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UInt2[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, IntPtr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UInt2[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UInt3[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UInt3[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UInt3[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UInt4[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UInt4[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UInt4[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ULong1[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ULong1[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ULong1[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ULong2[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ULong2[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ULong2[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ULong3[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ULong3[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ULong3[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ULong4[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ULong4[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ULong4[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UShort1[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UShort1[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UShort1[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UShort2[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UShort2[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UShort2[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UShort3[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UShort3[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UShort3[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UShort4[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UShort4[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] UShort4[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] byte[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] byte[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] byte[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] double[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] double[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] double[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] short[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] short[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] short[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] int[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] int[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] int[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] long[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] long[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] long[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] sbyte[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] sbyte[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] sbyte[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] float[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] float[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ulong[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] float[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ushort[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ushort[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ushort[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] uint[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] uint[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ulong[] src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] ulong[] src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(IntPtr dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(IntPtr dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(IntPtr dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Char1[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Char2[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Char3[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Char4[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Int1[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Short1[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Short2[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Short3[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy(CUdeviceptr dst, [In] uint[] src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] Short4[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UChar1[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UChar2[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UChar3[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UChar4[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UShort1[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UShort2[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UShort3[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] UShort4[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] byte[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] double[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] short[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] int[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] long[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] sbyte[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] float[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ushort[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] uint[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy([Out] ulong[] dst, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2D(CUdeviceptr dst, SizeT dpitch, CUdeviceptr src, SizeT spitch, SizeT width, SizeT height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2D(CUdeviceptr dst, SizeT dpitch, IntPtr src, SizeT spitch, SizeT width, SizeT height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2D(CUdeviceptr dst, int dpitch, CUdeviceptr src, int spitch, int width, int height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2D(CUdeviceptr dst, int dpitch, IntPtr src, int spitch, int width, int height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2D(CUdeviceptr dst, uint dpitch, CUdeviceptr src, uint spitch, uint width, uint height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2D(CUdeviceptr dst, uint dpitch, IntPtr src, uint spitch, uint width, uint height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2D(CUdeviceptr dst, ulong dpitch, CUdeviceptr src, ulong spitch, ulong width, ulong height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2D(CUdeviceptr dst, ulong dpitch, IntPtr src, ulong spitch, ulong width, ulong height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2D(IntPtr dst, SizeT dpitch, CUdeviceptr src, SizeT spitch, SizeT width, SizeT height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2D(IntPtr dst, int dpitch, CUdeviceptr src, int spitch, int width, int height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2D(IntPtr dst, uint dpitch, CUdeviceptr src, uint spitch, uint width, uint height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2D(IntPtr dst, ulong dpitch, CUdeviceptr src, ulong spitch, ulong width, ulong height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DArrayToArray(ref cudaArray dst, SizeT wOffsetDst, SizeT hOffsetDst, ref cudaArray src, SizeT wOffsetSrc, SizeT hOffsetSrc, SizeT width, SizeT height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DArrayToArray(ref cudaArray dst, int wOffsetDst, int hOffsetDst, ref cudaArray src, int wOffsetSrc, int hOffsetSrc, int width, int height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DArrayToArray(ref cudaArray dst, uint wOffsetDst, uint hOffsetDst, ref cudaArray src, uint wOffsetSrc, uint hOffsetSrc, uint width, uint height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DArrayToArray(ref cudaArray dst, ulong wOffsetDst, ulong hOffsetDst, ref cudaArray src, ulong wOffsetSrc, ulong hOffsetSrc, ulong width, ulong height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DAsync(CUdeviceptr dst, SizeT dpitch, CUdeviceptr src, SizeT spitch, SizeT width, SizeT height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DAsync(CUdeviceptr dst, SizeT dpitch, IntPtr src, SizeT spitch, SizeT width, SizeT height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DAsync(CUdeviceptr dst, int dpitch, CUdeviceptr src, int spitch, int width, int height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DAsync(CUdeviceptr dst, int dpitch, IntPtr src, int spitch, int width, int height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DAsync(CUdeviceptr dst, uint dpitch, CUdeviceptr src, uint spitch, uint width, uint height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DAsync(CUdeviceptr dst, uint dpitch, IntPtr src, uint spitch, uint width, uint height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DAsync(CUdeviceptr dst, ulong dpitch, CUdeviceptr src, ulong spitch, ulong width, ulong height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DAsync(CUdeviceptr dst, ulong dpitch, IntPtr src, ulong spitch, ulong width, ulong height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DAsync(IntPtr dst, SizeT dpitch, CUdeviceptr src, SizeT spitch, SizeT width, SizeT height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DAsync(IntPtr dst, int dpitch, CUdeviceptr src, int spitch, int width, int height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DAsync(IntPtr dst, uint dpitch, CUdeviceptr src, uint spitch, uint width, uint height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DAsync(IntPtr dst, ulong dpitch, CUdeviceptr src, ulong spitch, ulong width, ulong height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DFromArray(CUdeviceptr dst, SizeT dpitch, ref cudaArray src, SizeT wOffset, SizeT hOffset, SizeT width, SizeT height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DFromArray(CUdeviceptr dst, int dpitch, ref cudaArray src, int wOffset, int hOffset, int width, int height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DFromArray(CUdeviceptr dst, uint dpitch, ref cudaArray src, uint wOffset, uint hOffset, uint width, uint height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DFromArray(CUdeviceptr dst, ulong dpitch, ref cudaArray src, ulong wOffset, ulong hOffset, ulong width, ulong height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DFromArray(IntPtr dst, SizeT dpitch, ref cudaArray src, SizeT wOffset, SizeT hOffset, SizeT width, SizeT height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DFromArray(IntPtr dst, int dpitch, ref cudaArray src, int wOffset, int hOffset, int width, int height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DFromArray(IntPtr dst, uint dpitch, ref cudaArray src, uint wOffset, uint hOffset, uint width, uint height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DFromArray(IntPtr dst, ulong dpitch, ref cudaArray src, ulong wOffset, ulong hOffset, ulong width, ulong height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DFromArrayAsync(CUdeviceptr dst, SizeT dpitch, ref cudaArray src, SizeT wOffset, SizeT hOffset, SizeT width, SizeT height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DFromArrayAsync(CUdeviceptr dst, int dpitch, ref cudaArray src, int wOffset, int hOffset, int width, int height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DFromArrayAsync(CUdeviceptr dst, uint dpitch, ref cudaArray src, uint wOffset, uint hOffset, uint width, uint height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DFromArrayAsync(CUdeviceptr dst, ulong dpitch, ref cudaArray src, ulong wOffset, ulong hOffset, ulong width, ulong height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DFromArrayAsync(IntPtr dst, SizeT dpitch, ref cudaArray src, SizeT wOffset, SizeT hOffset, SizeT width, SizeT height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DFromArrayAsync(IntPtr dst, int dpitch, ref cudaArray src, int wOffset, int hOffset, int width, int height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DFromArrayAsync(IntPtr dst, uint dpitch, ref cudaArray src, uint wOffset, uint hOffset, uint width, uint height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DFromArrayAsync(IntPtr dst, ulong dpitch, ref cudaArray src, ulong wOffset, ulong hOffset, ulong width, ulong height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DToArray(ref cudaArray dst, SizeT wOffset, SizeT hOffset, CUdeviceptr src, SizeT spitch, SizeT width, SizeT height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DToArray(ref cudaArray dst, SizeT wOffset, SizeT hOffset, IntPtr src, SizeT spitch, SizeT width, SizeT height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DToArray(ref cudaArray dst, int wOffset, int hOffset, CUdeviceptr src, int spitch, int width, int height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DToArray(ref cudaArray dst, int wOffset, int hOffset, IntPtr src, int spitch, int width, int height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DToArray(ref cudaArray dst, uint wOffset, uint hOffset, CUdeviceptr src, uint spitch, uint width, uint height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DToArray(ref cudaArray dst, uint wOffset, uint hOffset, IntPtr src, uint spitch, uint width, uint height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DToArray(ref cudaArray dst, ulong wOffset, ulong hOffset, CUdeviceptr src, ulong spitch, ulong width, ulong height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DToArray(ref cudaArray dst, ulong wOffset, ulong hOffset, IntPtr src, ulong spitch, ulong width, ulong height, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DToArrayAsync(ref cudaArray dst, SizeT wOffset, SizeT hOffset, CUdeviceptr src, SizeT spitch, SizeT width, SizeT height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DToArrayAsync(ref cudaArray dst, SizeT wOffset, SizeT hOffset, IntPtr src, SizeT spitch, SizeT width, SizeT height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DToArrayAsync(ref cudaArray dst, int wOffset, int hOffset, CUdeviceptr src, int spitch, int width, int height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DToArrayAsync(ref cudaArray dst, int wOffset, int hOffset, IntPtr src, int spitch, int width, int height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DToArrayAsync(ref cudaArray dst, uint wOffset, uint hOffset, CUdeviceptr src, uint spitch, uint width, uint height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DToArrayAsync(ref cudaArray dst, uint wOffset, uint hOffset, IntPtr src, uint spitch, uint width, uint height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DToArrayAsync(ref cudaArray dst, ulong wOffset, ulong hOffset, CUdeviceptr src, ulong spitch, ulong width, ulong height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy2DToArrayAsync(ref cudaArray dst, ulong wOffset, ulong hOffset, IntPtr src, ulong spitch, ulong width, ulong height, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy3D(ref cudaMemcpy3DParms p);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpy3DAsync(ref cudaMemcpy3DParms p, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyArrayToArray(ref cudaArray dst, SizeT wOffsetDst, SizeT hOffsetDst, ref cudaArray src, SizeT wOffsetSrc, SizeT hOffsetSrc, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyArrayToArray(ref cudaArray dst, int wOffsetDst, int hOffsetDst, ref cudaArray src, int wOffsetSrc, int hOffsetSrc, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyArrayToArray(ref cudaArray dst, uint wOffsetDst, uint hOffsetDst, ref cudaArray src, uint wOffsetSrc, uint hOffsetSrc, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyArrayToArray(ref cudaArray dst, ulong wOffsetDst, ulong hOffsetDst, ref cudaArray src, ulong wOffsetSrc, ulong hOffsetSrc, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, uint count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyAsync(CUdeviceptr dst, IntPtr src, SizeT count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyAsync(CUdeviceptr dst, IntPtr src, uint count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyAsync(CUdeviceptr dst, IntPtr src, ulong count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyAsync(IntPtr dst, CUdeviceptr src, SizeT count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyAsync(IntPtr dst, CUdeviceptr src, uint count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyAsync(IntPtr dst, CUdeviceptr src, ulong count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromArray(CUdeviceptr dst, ref cudaArray src, SizeT wOffset, SizeT hOffset, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromArray(CUdeviceptr dst, ref cudaArray src, int wOffset, int hOffset, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromArray(CUdeviceptr dst, ref cudaArray src, uint wOffset, uint hOffset, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromArray(CUdeviceptr dst, ref cudaArray src, ulong wOffset, ulong hOffset, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromArray(IntPtr dst, ref cudaArray src, SizeT wOffset, SizeT hOffset, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromArray(IntPtr dst, ref cudaArray src, int wOffset, int hOffset, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromArray(IntPtr dst, ref cudaArray src, uint wOffset, uint hOffset, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromArray(IntPtr dst, ref cudaArray src, ulong wOffset, ulong hOffset, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromArrayAsync(CUdeviceptr dst, ref cudaArray src, SizeT wOffset, SizeT hOffset, SizeT count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromArrayAsync(CUdeviceptr dst, ref cudaArray src, int wOffset, int hOffset, uint count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromArrayAsync(CUdeviceptr dst, ref cudaArray src, uint wOffset, uint hOffset, uint count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromArrayAsync(CUdeviceptr dst, ref cudaArray src, ulong wOffset, ulong hOffset, ulong count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromArrayAsync(IntPtr dst, ref cudaArray src, SizeT wOffset, SizeT hOffset, SizeT count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromArrayAsync(IntPtr dst, ref cudaArray src, int wOffset, int hOffset, uint count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromArrayAsync(IntPtr dst, ref cudaArray src, uint wOffset, uint hOffset, uint count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromArrayAsync(IntPtr dst, ref cudaArray src, ulong wOffset, ulong hOffset, ulong count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromSymbol(CUdeviceptr dst, string symbol, SizeT count, SizeT offset, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromSymbol(CUdeviceptr dst, string symbol, uint count, int offset, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromSymbol(CUdeviceptr dst, string symbol, uint count, uint offset, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromSymbol(CUdeviceptr dst, string symbol, ulong count, ulong offset, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromSymbol(IntPtr dst, string symbol, SizeT count, SizeT offset, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromSymbol(IntPtr dst, string symbol, uint count, int offset, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromSymbol(IntPtr dst, string symbol, uint count, uint offset, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromSymbol(IntPtr dst, string symbol, ulong count, ulong offset, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromSymbolAsync(CUdeviceptr dst, string symbol, SizeT count, SizeT offset, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromSymbolAsync(CUdeviceptr dst, string symbol, uint count, int offset, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromSymbolAsync(CUdeviceptr dst, string symbol, uint count, uint offset, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromSymbolAsync(CUdeviceptr dst, string symbol, ulong count, ulong offset, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromSymbolAsync(IntPtr dst, string symbol, SizeT count, SizeT offset, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromSymbolAsync(IntPtr dst, string symbol, uint count, int offset, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromSymbolAsync(IntPtr dst, string symbol, uint count, uint offset, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyFromSymbolAsync(IntPtr dst, string symbol, ulong count, ulong offset, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToArray(ref cudaArray dst, SizeT wOffset, SizeT hOffset, CUdeviceptr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToArray(ref cudaArray dst, SizeT wOffset, SizeT hOffset, IntPtr src, SizeT count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToArray(ref cudaArray dst, int wOffset, int hOffset, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToArray(ref cudaArray dst, int wOffset, int hOffset, IntPtr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToArray(ref cudaArray dst, uint wOffset, uint hOffset, CUdeviceptr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToArray(ref cudaArray dst, uint wOffset, uint hOffset, IntPtr src, uint count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToArray(ref cudaArray dst, ulong wOffset, ulong hOffset, CUdeviceptr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToArray(ref cudaArray dst, ulong wOffset, ulong hOffset, IntPtr src, ulong count, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToArrayAsync(ref cudaArray dst, SizeT wOffset, SizeT hOffset, CUdeviceptr src, SizeT count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToArrayAsync(ref cudaArray dst, SizeT wOffset, SizeT hOffset, IntPtr src, SizeT count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToArrayAsync(ref cudaArray dst, int wOffset, int hOffset, CUdeviceptr src, uint count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToArrayAsync(ref cudaArray dst, int wOffset, int hOffset, IntPtr src, uint count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToArrayAsync(ref cudaArray dst, uint wOffset, uint hOffset, CUdeviceptr src, uint count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToArrayAsync(ref cudaArray dst, uint wOffset, uint hOffset, IntPtr src, uint count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToArrayAsync(ref cudaArray dst, ulong wOffset, ulong hOffset, CUdeviceptr src, ulong count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToArrayAsync(ref cudaArray dst, ulong wOffset, ulong hOffset, IntPtr src, ulong count, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToSymbol(string symbol, CUdeviceptr src, SizeT count, SizeT offset, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToSymbol(string symbol, CUdeviceptr src, uint count, int offset, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToSymbol(string symbol, CUdeviceptr src, uint count, uint offset, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToSymbol(string symbol, CUdeviceptr src, ulong count, ulong offset, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToSymbol(string symbol, IntPtr src, SizeT count, SizeT offset, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToSymbol(string symbol, IntPtr src, uint count, int offset, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToSymbol(string symbol, IntPtr src, uint count, uint offset, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToSymbol(string symbol, IntPtr src, ulong count, ulong offset, cudaMemcpyKind kind);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToSymbolAsync(string symbol, CUdeviceptr src, SizeT count, SizeT offset, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToSymbolAsync(string symbol, CUdeviceptr src, uint count, int offset, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToSymbolAsync(string symbol, CUdeviceptr src, uint count, uint offset, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToSymbolAsync(string symbol, CUdeviceptr src, ulong count, ulong offset, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToSymbolAsync(string symbol, IntPtr src, SizeT count, SizeT offset, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToSymbolAsync(string symbol, IntPtr src, uint count, int offset, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToSymbolAsync(string symbol, IntPtr src, uint count, uint offset, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemcpyToSymbolAsync(string symbol, IntPtr src, ulong count, ulong offset, cudaMemcpyKind kind, cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemset(CUdeviceptr mem, int c, SizeT count);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemset(CUdeviceptr mem, int c, uint count);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemset(CUdeviceptr mem, int c, ulong count);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemset2D(CUdeviceptr mem, SizeT pitch, int value, SizeT width, SizeT height);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemset2D(CUdeviceptr mem, uint pitch, int value, int width, uint height);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemset2D(CUdeviceptr mem, uint pitch, int value, uint width, uint height);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemset2D(CUdeviceptr mem, ulong pitch, int value, ulong width, ulong height);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaRuntimeGetVersion(ref int runtimeVersion);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaSetDevice(int device);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaSetDeviceFlags(int flags);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaSetDoubleForDevice(double[] d);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaSetDoubleForHost(double[] d);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaSetupArgument(CUdeviceptr arg, SizeT size, SizeT offset);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaSetupArgument(CUdeviceptr arg, uint size, uint offset);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaSetupArgument(CUdeviceptr arg, ulong size, ulong offset);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaSetupArgument(IntPtr arg, int size, int offset);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaSetValidDevices([In] int[] device_arr, int len);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaStreamCreate(ref cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaStreamDestroy(cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaStreamQuery(cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaStreamSynchronize(cudaStream stream);
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaThreadExit();
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaThreadSynchronize();
        [DllImport(CUDART_DLL_NAME)]
        public static extern cudaError cudaUnbindTexture(ref textureReference texref);

        public System.Version Version
        {
            get
            {
                return new System.Version(3, 0);
            }
        }
    }
}

