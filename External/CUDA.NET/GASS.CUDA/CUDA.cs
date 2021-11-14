namespace GASS.CUDA
{
    using GASS.CUDA.Types;
    using GASS.Types;
    using System;
    using System.Collections.Generic;
    using System.Runtime.InteropServices;
    using System.Text;
    using System.Diagnostics;

    public class CUDA : IDisposable
    {
        private CUcontext curCtx;
        private Device curDev;
        private CUfunction curFunc;
        private CUmodule curMod;
        private List<Device> devices;
        private CUResult lastError;
        private bool useRuntimeExceptions;

        public CUDA() : this(false)
        {
        }

        public CUDA(bool initialize) : this(initialize, InitializationFlags.None)
        {
        }

        public CUDA(int ordinal) : this(false)
        {
        }

        public CUDA(bool initialize, InitializationFlags flags)
        {
            this.useRuntimeExceptions = true;
            if (initialize)
            {
                _version = GetDriverVersion();
                CUDADriver.cuInit((uint) flags);
            }
        }

        public CUDA(int ordinal, bool initialize) : this(initialize)
        {
            //this.CurrentContext = this.CreateContext(ordinal);//, CUCtxFlags.MapHost);
            curCtx = this.CreateContext(ordinal);
            SetCurrentContext(curCtx);
        }

        public CUcontext DeviceContext
        {
            get { return curCtx;  }
        }


        private int _version = -1;

        public CUdeviceptr Allocate<T>(T[] array)
        {
            return this.Allocate(this.GetSize<T>(array));
        }

        public CUdeviceptr Allocate(uint bytes)
        {
            CUdeviceptr dptr = new CUdeviceptr();
            this.LastError = CUDADriver.cuMemAlloc(ref dptr, bytes);
            return dptr;
        }

        public IntPtr AllocateHost(uint bytes)
        {
            IntPtr pp = new IntPtr();
            this.LastError = CUDADriver.cuMemAllocHost(ref pp, bytes);
            return pp;
        }

        public IntPtr AllocateHost<T>(T[] array)
        {
            return this.AllocateHost(this.GetSize<T>(array));
        }

        [Obsolete]
        public void AttachContext(CUcontext ctx)
        {
            this.AttachContext(ctx, CUCtxFlags.SchedAuto);
        }
        [Obsolete]
        public void AttachContext(CUcontext ctx, CUCtxFlags flags)
        {
            //this.LastError = CUDADriver.cuCtxAttach(ref this.curCtx, (uint) flags);
            this.LastError = CUDADriver.cuCtxAttach(ref ctx, (uint)flags);
        }

        public void Copy2D(CUDAMemCpy2D desc)
        {
            this.LastError = CUDADriver.cuMemcpy2D(ref desc);
        }

        public void Copy2DAsync(CUDAMemCpy2D desc, CUstream stream)
        {
            this.LastError = CUDADriver.cuMemcpy2DAsync(ref desc, stream);
        }

        public void Copy2DUnaligned(CUDAMemCpy2D desc)
        {
            this.LastError = CUDADriver.cuMemcpy2DUnaligned(ref desc);
        }

        public void Copy3D(CUDAMemCpy3D desc)
        {
            this.LastError = CUDADriver.cuMemcpy3D(ref desc);
        }

        public void Copy3DAsync(CUDAMemCpy3D desc, CUstream stream)
        {
            this.LastError = CUDADriver.cuMemcpy3DAsync(ref desc, stream);
        }

        public void CopyArrayToArray(CUarray src, uint srcIndex, CUarray dst, uint dstIndex, uint bytes)
        {
            this.LastError = CUDADriver.cuMemcpyAtoA(dst, dstIndex, src, srcIndex, bytes);
        }

        public void CopyArrayToHost<T>(CUarray devArray, T[] data, uint index)
        {
            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            this.LastError = CUDADriver.cuMemcpyAtoH(handle.AddrOfPinnedObject(), devArray, index, this.GetSize<T>(data));
            handle.Free();
        }

        public void CopyArrayToHostAsync(CUarray devArray, IntPtr buffer, uint index, uint size, CUstream stream)
        {
            this.LastError = CUDADriver.cuMemcpyAtoHAsync(buffer, devArray, index, size, stream);
        }

        public void CopyDeviceToDevice(CUdeviceptr src, CUdeviceptr dst, uint bytes)
        {
            this.LastError = CUDADriver.cuMemcpyDtoD(dst, src, bytes);
        }

        public void CopyDeviceToDeviceAsync(CUdeviceptr src, CUdeviceptr dst, uint bytes, CUstream stream)
        {
            this.LastError = CUDADriver.cuMemcpyDtoDAsync(dst, src, bytes, stream);
        }

        public void CopyPeerToPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, SizeT ByteCount)
        {
            this.LastError = CUDADriver.cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
        }

        public void CopyPeerToPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, SizeT ByteCount, CUstream stream)
        {
            this.LastError = CUDADriver.cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, stream);
        }

        public void CopyDeviceToHost<T>(CUdeviceptr devPtr, T[] data)
        {
            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            this.LastError = CUDADriver.cuMemcpyDtoH(handle.AddrOfPinnedObject(), devPtr, this.GetSize<T>(data));
            handle.Free();
        }

        public void CopyDeviceToHost(CUdeviceptr devPtr, IntPtr data, uint size)
        {
            this.LastError = CUDADriver.cuMemcpyDtoH(data, devPtr, size);
        }

        public void CopyDeviceToHostAsync(CUdeviceptr devPtr, IntPtr buffer, uint size, CUstream stream)
        {
            this.LastError = CUDADriver.cuMemcpyDtoHAsync(buffer, devPtr, size, stream);
        }

        public CUarray CopyHostToArray<T>(T[] data)
        {
            return this.CopyHostToArray<T>(data, 0);
        }

        public CUarray CopyHostToArray<T>(T[] data, uint index)
        {
            CUarray devArray = this.CreateArray(data);
            this.CopyHostToArray<T>(devArray, data, index);
            return devArray;
        }

        public void CopyHostToArray<T>(CUarray devArray, T[] data, uint index)
        {
            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            this.LastError = CUDADriver.cuMemcpyHtoA(devArray, index, handle.AddrOfPinnedObject(), this.GetSize<T>(data));
            handle.Free();
        }

        public void CopyHostToArrayAsync(CUarray devArray, IntPtr buffer, uint size, CUstream stream)
        {
            this.CopyHostToArrayAsync(devArray, 0, buffer, size, stream);
        }

        public void CopyHostToArrayAsync(CUarray devArray, uint index, IntPtr buffer, uint size, CUstream stream)
        {
            this.LastError = CUDADriver.cuMemcpyHtoAAsync(devArray, index, buffer, size, stream);
        }

        public CUdeviceptr CopyHostToDevice<T>(T[] data)
        {
            CUdeviceptr devPtr = this.Allocate<T>(data);
            this.CopyHostToDevice<T>(devPtr, data);
            return devPtr;
        }

        public void CopyHostToDevice<T>(CUdeviceptr devPtr, T[] data)
        {
            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            this.LastError = CUDADriver.cuMemcpyHtoD(devPtr, handle.AddrOfPinnedObject(), this.GetSize<T>(data));
            handle.Free();
        }

        public CUdeviceptr CopyHostToDevice(IntPtr buffer, uint size)
        {
            CUdeviceptr devPtr = this.Allocate(size);
            this.CopyHostToDevice(devPtr, buffer, size);
            return devPtr;
        }

        public void CopyHostToDevice(CUdeviceptr devPtr, IntPtr buffer, uint size)
        {
            this.LastError = CUDADriver.cuMemcpyHtoD(devPtr, buffer, size);
        }

        public CUdeviceptr CopyHostToDeviceAsync(IntPtr buffer, uint size, CUstream stream)
        {
            CUdeviceptr devPtr = this.Allocate(size);
            this.CopyHostToDeviceAsync(devPtr, buffer, size, stream);
            return devPtr;
        }

        public void CopyHostToDeviceAsync(CUdeviceptr devPtr, IntPtr buffer, uint size, CUstream stream)
        {
            this.LastError = CUDADriver.cuMemcpyHtoDAsync(devPtr, buffer, size, stream);
        }

        public CUarray CreateArray(CUDAArray3DDescriptor desc)
        {
            CUarray pHandle = new CUarray();
            this.LastError = CUDADriver.cuArray3DCreate(ref pHandle, ref desc);
            return pHandle;
        }

        public CUarray CreateArray(CUDAArrayDescriptor desc)
        {
            CUarray pHandle = new CUarray();
            this.LastError = CUDADriver.cuArrayCreate(ref pHandle, ref desc);
            return pHandle;
        }

        public CUarray CreateArray(Array arr)
        {
            CUDAArrayDescriptor desc = new CUDAArrayDescriptor {
                Width = (uint) arr.GetLongLength(0),
                Height = 0
            };
            if (arr.Rank > 1)
            {
                desc.Height = (uint) arr.GetLongLength(1);
            }
            Type type = arr.GetValue(0).GetType();
            desc.Format = this.GetTypeFormat(type);
            desc.NumChannels = (uint) this.GetTypeComponents(type);
            return this.CreateArray(desc);
        }

        public CUarray CreateArray(CUArrayFormat format, uint width, uint height)
        {
            return this.CreateArray(format, 1, width, height);
        }

        public CUarray CreateArray(CUArrayFormat format, uint channels, uint width, uint height)
        {
            CUDAArrayDescriptor desc = new CUDAArrayDescriptor {
                Format = format,
                Width = width,
                Height = height,
                NumChannels = channels
            };
            return this.CreateArray(desc);
        }

        public CUarray CreateArray(Array array, uint width, uint height, uint depth)
        {
            CUDAArray3DDescriptor desc = new CUDAArray3DDescriptor {
                Width = width,
                Height = height,
                Depth = depth,
                Flags = 0
            };
            Type type = array.GetValue(0).GetType();
            desc.Format = this.GetTypeFormat(type);
            desc.NumChannels = (uint) this.GetTypeComponents(type);
            return this.CreateArray(desc);
        }

        public CUarray CreateArray(CUArrayFormat format, uint numChannels, uint width, uint height, uint depth)
        {
            CUDAArray3DDescriptor desc = new CUDAArray3DDescriptor {
                Width = width,
                Height = height,
                Depth = depth,
                Format = format,
                Flags = 0,
                NumChannels = numChannels
            };
            return this.CreateArray(desc);
        }

        public CUcontext CreateContext(int ordinal)
        {
            return this.CreateContext(ordinal, CUCtxFlags.SchedAuto);
        }

        public CUcontext CreateContext(int ordinal, CUCtxFlags flags)
        {
            this.curCtx = new CUcontext();
            this.LastError = CUDADriver.cuCtxCreate(ref this.curCtx, (uint) flags, this.Devices[ordinal].Handle);
            this.CurrentDevice = this.Devices[ordinal];
            return this.curCtx;
        }

        public CUevent CreateEvent()
        {
            return this.CreateEvent(CUEventFlags.Default);
        }

        public CUevent CreateEvent(CUEventFlags flags)
        {
            CUevent phEvent = new CUevent();
            this.LastError = CUDADriver.cuEventCreate(ref phEvent, (uint) flags);
            return phEvent;
        }

        public CUstream CreateStream()
        {
            return this.CreateStream(StreamFlags.None);
        }

        public CUstream CreateStream(StreamFlags flags)
        {
            CUstream phStream = new CUstream();
            this.LastError = CUDADriver.cuStreamCreate(ref phStream, (uint) flags);
            return phStream;
        }

        public CUtexref CreateTexture()
        {
            CUtexref pTexRef = new CUtexref();
            this.LastError = CUDADriver.cuTexRefCreate(ref pTexRef);
            return pTexRef;
        }

        public void DestroyArray(CUarray devArr)
        {
            this.LastError = CUDADriver.cuArrayDestroy(devArr);
        }

        public void DestroyContext()
        {
            if(_version >= 4000)
                this.LastError = CUDADriver.cuCtxDestroy_v2(this.CurrentContext);
            else
                this.LastError = CUDADriver.cuCtxDestroy(this.CurrentContext);
        }

        public void DestroyContext(CUcontext ctx)
        {
            if (_version >= 4000)
                this.LastError = CUDADriver.cuCtxDestroy_v2(ctx);
            else
                this.LastError = CUDADriver.cuCtxDestroy(ctx);
        }

        public void DestroyEvent(CUevent e)
        {
            if (_version >= 4000)
                this.LastError = CUDADriver.cuEventDestroy_v2(e);
            else
                this.LastError = CUDADriver.cuEventDestroy(e);
        }

        public void DestroyStream(CUstream stream)
        {
            if(_version >= 4000)
                this.LastError = CUDADriver.cuStreamDestroy_v2(stream);
            else
                this.LastError = CUDADriver.cuStreamDestroy(stream);
        }

        public void DestroyTexture(CUtexref tex)
        {
            this.LastError = CUDADriver.cuTexRefDestroy(tex);
        }
        [Obsolete]
        public void DetachContext()
        {
            this.DetachContext(this.curCtx);
        }
        [Obsolete]
        public void DetachContext(CUcontext ctx)
        {
            this.LastError = CUDADriver.cuCtxDetach(ctx);
        }

        public void Dispose()
        {
        }

        public float ElapsedTime(CUevent start, CUevent end)
        {
            float pMilliseconds = 0f;
            this.LastError = CUDADriver.cuEventElapsedTime(ref pMilliseconds, start, end);
            return pMilliseconds;
        }

        ~CUDA()
        {
            this.Dispose();
        }

        public void Free(CUdeviceptr ptr)
        {
            this.LastError = CUDADriver.cuMemFree(ptr);
        }

        public void FreeHost(IntPtr pointer)
        {
            this.LastError = CUDADriver.cuMemFreeHost(pointer);
        }

        public CUDAArray3DDescriptor GetArray3DDescriptor(CUarray devArr)
        {
            CUDAArray3DDescriptor pArrayDescriptor = new CUDAArray3DDescriptor();
            this.LastError = CUDADriver.cuArray3DGetDescriptor(ref pArrayDescriptor, devArr);
            return pArrayDescriptor;
        }

        public CUDAArrayDescriptor GetArrayDescriptor(CUarray devArr)
        {
            CUDAArrayDescriptor pArrayDescriptor = new CUDAArrayDescriptor();
            this.LastError = CUDADriver.cuArrayGetDescriptor(ref pArrayDescriptor, devArr);
            return pArrayDescriptor;
        }

        public int GetDriverVersion()
        {
            int version = -1;
            CUResult res;
            try
            {
                res = CUDADriver.cuDriverGetVersion(ref version);
                this.LastError = res;
            }
            catch (EntryPointNotFoundException ex)
            {
                Debug.WriteLine("GetDriverVersion(): " + ex.Message);
            }
           
            return version;
        }

        public CUdevice GetContextDevice()
        {
            CUdevice device = new CUdevice();
            this.LastError = CUDADriver.cuCtxGetDevice(ref device);
            return device;
        }

        public CUdevice GetDevice(int ordinal)
        {
            CUdevice device = new CUdevice();
            this.LastError = CUDADriver.cuDeviceGet(ref device, ordinal);
            return device;
        }

        public int GetDeviceAttribute(CUDeviceAttribute attrib)
        {
            return this.GetDeviceAttribute(attrib, this.curDev.Handle);
        }

        public int GetDeviceAttribute(CUDeviceAttribute attrib, CUdevice dev)
        {
            int pi = 0;
            this.LastError = CUDADriver.cuDeviceGetAttribute(ref pi, attrib, dev);
            return pi;
        }

        public int GetDeviceCount()
        {
            int count = 0;
            this.LastError = CUDADriver.cuDeviceGetCount(ref count);
            return count;
        }

        public string GetDeviceName()
        {
            return this.GetDeviceName(this.curDev.Handle);
        }

        public string GetDeviceName(CUdevice dev)
        {
            byte[] name = new byte[0x100];
            this.LastError = CUDADriver.cuDeviceGetName(name, name.Length, dev);
            return Encoding.ASCII.GetString(name);
        }

        public string GetDeviceName(int ordinal)
        {
            return this.GetDeviceName(this.GetDevice(ordinal));
        }

        public int GetFunctionAttribute(CUFunctionAttribute attrib)
        {
            return this.GetFunctionAttribute(this.curFunc, attrib);
        }

        public int GetFunctionAttribute(CUfunction func, CUFunctionAttribute attrib)
        {
            int pi = 0;
            this.LastError = CUDADriver.cuFuncGetAttribute(ref pi, attrib, func);
            return pi;
        }

        public CUdeviceptr GetGraphicsResourceMappedPointer(CUgraphicsResource resource)
        {
            uint num;
            return this.GetGraphicsResourceMappedPointer(resource, out num);
        }

        public CUdeviceptr GetGraphicsResourceMappedPointer(CUgraphicsResource resource, out uint size)
        {
            CUdeviceptr pDevPtr = new CUdeviceptr();
            SizeT pSize = 0;
            this.LastError = CUDADriver.cuGraphicsResourceGetMappedPointer(ref pDevPtr, ref pSize, resource);
            size = pSize;
            return pDevPtr;
        }

        public ulong GetGraphicsResourceMappedPointerSize(CUgraphicsResource resource)
        {
            CUdeviceptr pDevPtr = new CUdeviceptr();
            SizeT pSize = 0;
            this.LastError = CUDADriver.cuGraphicsResourceGetMappedPointer(ref pDevPtr, ref pSize, resource);
            return pSize;
        }

        public CUarray GetGraphicsSubResourceMappedArray(CUgraphicsResource resource, uint arrIndex, uint mipLevel)
        {
            CUarray pArray = new CUarray();
            this.LastError = CUDADriver.cuGraphicsSubResourceGetMappedArray(ref pArray, resource, arrIndex, mipLevel);
            return pArray;
        }

        public CUdeviceptr GetHostDevicePointer(IntPtr hostPtr, uint flags)
        {
            CUdeviceptr pdptr = new CUdeviceptr();
            this.LastError = CUDADriver.cuMemHostGetDevicePointer(ref pdptr, hostPtr, flags);
            return pdptr;
        }

        public CUfunction GetModuleFunction(string funcName)
        {
            return this.GetModuleFunction(this.curMod, funcName);
        }

        public CUfunction GetModuleFunction(CUmodule mod, string funcName)
        {
            CUfunction hfunc = new CUfunction();
            this.LastError = CUDADriver.cuModuleGetFunction(ref hfunc, mod, funcName);
            return hfunc;
        }

        public CUdeviceptr GetModuleGlobal(string globalName)
        {
            return this.GetModuleGlobal(this.curMod, globalName);
        }

        public CUdeviceptr GetModuleGlobal(CUmodule mod, string globalName)
        {
            CUdeviceptr dptr = new CUdeviceptr();
            SizeT bytes = 0;
            this.LastError = CUDADriver.cuModuleGetGlobal(ref dptr, ref bytes, mod, globalName);
            return dptr;
        }

        public uint GetModuleGlobalBytes(string globalName)
        {
            return this.GetModuleGlobalBytes(this.curMod, globalName);
        }

        public uint GetModuleGlobalBytes(CUmodule mod, string globalName)
        {
            CUdeviceptr dptr = new CUdeviceptr();
            SizeT bytes = 0;
            this.LastError = CUDADriver.cuModuleGetGlobal(ref dptr, ref bytes, mod, globalName);
            return bytes;
        }

        public CUtexref GetModuleTexture(string textureName)
        {
            return this.GetModuleTexture(this.curMod, textureName);
        }

        public CUtexref GetModuleTexture(CUmodule mod, string textureName)
        {
            CUtexref pTexRef = new CUtexref();
            this.LastError = CUDADriver.cuModuleGetTexRef(ref pTexRef, mod, textureName);
            return pTexRef;
        }

        private uint GetSize<T>(T[] data)
        {
            return (uint)(CUDA.MSizeOf(typeof(T)) * data.Length);
        }

        public CUdeviceptr GetTextureAddress(CUtexref tex)
        {
            CUdeviceptr pdptr = new CUdeviceptr();
            this.LastError = CUDADriver.cuTexRefGetAddress(ref pdptr, tex);
            return pdptr;
        }

        public CUAddressMode GetTextureAddressMode(CUtexref tex, int dimension)
        {
            CUAddressMode wrap = CUAddressMode.Wrap;
            this.LastError = CUDADriver.cuTexRefGetAddressMode(ref wrap, tex, dimension);
            return wrap;
        }

        public CUarray GetTextureArray(CUtexref tex)
        {
            CUarray phArray = new CUarray();
            this.LastError = CUDADriver.cuTexRefGetArray(ref phArray, tex);
            return phArray;
        }

        public int GetTextureChannels(CUtexref tex)
        {
            CUArrayFormat pFormat = (CUArrayFormat) 0;
            int pNumChannels = 0;
            this.LastError = CUDADriver.cuTexRefGetFormat(ref pFormat, ref pNumChannels, tex);
            return pNumChannels;
        }

        public CUFilterMode GetTextureFilterMode(CUtexref tex)
        {
            CUFilterMode point = CUFilterMode.Point;
            this.LastError = CUDADriver.cuTexRefGetFilterMode(ref point, tex);
            return point;
        }

        public uint GetTextureFlags(CUtexref tex)
        {
            uint pFlags = 0;
            this.LastError = CUDADriver.cuTexRefGetFlags(ref pFlags, tex);
            return pFlags;
        }

        public CUArrayFormat GetTextureFormat(CUtexref tex)
        {
            CUArrayFormat pFormat = (CUArrayFormat) 0;
            int pNumChannels = 0;
            this.LastError = CUDADriver.cuTexRefGetFormat(ref pFormat, ref pNumChannels, tex);
            return pFormat;
        }

        private int GetTypeComponents(Type type)
        {
            if ((((type == typeof(byte)) || (type == typeof(sbyte))) || ((type == typeof(short)) || (type == typeof(ushort)))) || ((((type == typeof(int)) || (type == typeof(uint))) || ((type == typeof(long)) || (type == typeof(ulong)))) || ((type == typeof(float)) || (type == typeof(double)))))
            {
                return 1;
            }
            if (((((type == typeof(Char1)) || (type == typeof(UChar1))) || ((type == typeof(Short1)) || (type == typeof(UShort1)))) || (((type == typeof(Int1)) || (type == typeof(UInt1))) || ((type == typeof(Long1)) || (type == typeof(ULong1))))) || (((type == typeof(Float1)) || (type == typeof(Double1))) || ((type == typeof(cuFloatReal)) || (type == typeof(cuDoubleReal)))))
            {
                return 1;
            }
            if (((((type == typeof(Char2)) || (type == typeof(UChar2))) || ((type == typeof(Short2)) || (type == typeof(UShort2)))) || (((type == typeof(Int2)) || (type == typeof(UInt2))) || ((type == typeof(Long2)) || (type == typeof(ULong2))))) || (((type == typeof(Float2)) || (type == typeof(Double2))) || ((type == typeof(cuFloatComplex)) || (type == typeof(cuDoubleComplex)))))
            {
                return 2;
            }
            if ((((type == typeof(Char3)) || (type == typeof(UChar3))) || ((type == typeof(Short3)) || (type == typeof(UShort3)))) || (((type == typeof(Int3)) || (type == typeof(UInt3))) || (((type == typeof(Long3)) || (type == typeof(ULong3))) || (type == typeof(Float3)))))
            {
                return 3;
            }
            if ((((type != typeof(Char4)) && (type != typeof(UChar4))) && ((type != typeof(Short4)) && (type != typeof(UShort4)))) && (((type != typeof(Int4)) && (type != typeof(UInt4))) && (((type != typeof(Long4)) && (type != typeof(ULong4))) && (type != typeof(Float4)))))
            {
                return -1;
            }
            return 4;
        }

        private CUArrayFormat GetTypeFormat(Type type)
        {
            if (((type == typeof(sbyte)) || (type == typeof(Char1))) || (((type == typeof(Char2)) || (type == typeof(Char3))) || (type == typeof(Char4))))
            {
                return CUArrayFormat.SignedInt8;
            }
            if (((type == typeof(byte)) || (type == typeof(UChar1))) || (((type == typeof(UChar2)) || (type == typeof(UChar3))) || (type == typeof(UChar4))))
            {
                return CUArrayFormat.UnsignedInt8;
            }
            if (((type == typeof(short)) || (type == typeof(Short1))) || (((type == typeof(Short2)) || (type == typeof(Short3))) || (type == typeof(Short4))))
            {
                return CUArrayFormat.SignedInt16;
            }
            if (((type == typeof(ushort)) || (type == typeof(UShort1))) || (((type == typeof(UShort2)) || (type == typeof(UShort3))) || (type == typeof(UShort4))))
            {
                return CUArrayFormat.UnsignedInt16;
            }
            if (((type == typeof(int)) || (type == typeof(Int1))) || (((type == typeof(Int2)) || (type == typeof(Int3))) || (type == typeof(Int4))))
            {
                return CUArrayFormat.SignedInt32;
            }
            if (((type == typeof(uint)) || (type == typeof(UInt1))) || (((type == typeof(UInt2)) || (type == typeof(UInt3))) || (type == typeof(UInt4))))
            {
                return CUArrayFormat.UnsignedInt32;
            }
            if ((((type != typeof(float)) && (type != typeof(Float1))) && ((type != typeof(Float2)) && (type != typeof(Float3)))) && (((type != typeof(Float4)) && (type != typeof(cuFloatReal))) && (type != typeof(cuFloatComplex))))
            {
                return (CUArrayFormat) (-1);
            }
            return CUArrayFormat.Float;
        }

        public IntPtr HostAllocate(uint size, uint flags)
        {
            IntPtr pp = new IntPtr();
            this.LastError = CUDADriver.cuMemHostAlloc(ref pp, size, flags);
            return pp;
        }

        public void Init()
        {
            this.Init(InitializationFlags.None);
        }

        public void Init(InitializationFlags initializationFlags)
        {
            this.LastError = CUDADriver.cuInit((uint) initializationFlags);
        }

        private bool IsPrimitive(Type type)
        {
            if ((((type != typeof(byte)) && (type != typeof(sbyte))) && ((type != typeof(short)) && (type != typeof(ushort)))) && ((((type != typeof(int)) && (type != typeof(uint))) && ((type != typeof(long)) && (type != typeof(ulong)))) && ((type != typeof(float)) && (type != typeof(double)))))
            {
                return false;
            }
            return true;
        }

        private bool IsVector(Type type)
        {
            if ((((((type != typeof(Char1)) && (type != typeof(UChar1))) && ((type != typeof(Char2)) && (type != typeof(UChar2)))) && (((type != typeof(Char3)) && (type != typeof(UChar3))) && ((type != typeof(Char4)) && (type != typeof(UChar4))))) && ((((type != typeof(Short1)) && (type != typeof(UShort1))) && ((type != typeof(Short2)) && (type != typeof(UShort2)))) && (((type != typeof(Short3)) && (type != typeof(UShort3))) && ((type != typeof(Short4)) && (type != typeof(UShort4)))))) && ((((((type != typeof(Int1)) && (type != typeof(UInt1))) && ((type != typeof(Int2)) && (type != typeof(UInt2)))) && (((type != typeof(Int3)) && (type != typeof(UInt3))) && ((type != typeof(Int4)) && (type != typeof(UInt4))))) && ((((type != typeof(Long1)) && (type != typeof(ULong1))) && ((type != typeof(Long2)) && (type != typeof(ULong2)))) && (((type != typeof(Long3)) && (type != typeof(ULong3))) && ((type != typeof(Long4)) && (type != typeof(ULong4)))))) && ((((type != typeof(Float1)) && (type != typeof(Float2))) && ((type != typeof(Float3)) && (type != typeof(Float4)))) && ((((type != typeof(Double1)) && (type != typeof(Double2))) && ((type != typeof(cuFloatReal)) && (type != typeof(cuDoubleReal)))) && ((type != typeof(cuFloatComplex)) && (type != typeof(cuDoubleComplex)))))))
            {
                return false;
            }
            return true;
        }

        public void Launch(CUfunction func)
        {
            this.LastError = CUDADriver.cuLaunch(func);
        }

        public void Launch(CUfunction func, int gridWidth, int gridHeight)
        {
            this.LastError = CUDADriver.cuLaunchGrid(func, gridWidth, gridHeight);
        }

        public void LaunchAsync(CUfunction func, int gridWidth, int gridHeight, CUstream stream)
        {
            this.LastError = CUDADriver.cuLaunchGridAsync(func, gridWidth, gridHeight, stream);
        }

        [Obsolete("NVIDIA driver doesn't support this method yet.")]
        public CUmodule LoadFatModule(byte[] fatBin)
        {
            this.curMod = new CUmodule();
            this.LastError = CUDADriver.cuModuleLoadFatBinary(ref this.curMod, fatBin);
            return this.curMod;
        }

        public CUmodule LoadModule(string filename)
        {
            this.curMod = new CUmodule();
            this.LastError = CUDADriver.cuModuleLoad(ref this.curMod, filename);
            return this.curMod;
        }

        public CUmodule LoadModule(byte[] binaryImage)
        {
            this.curMod = new CUmodule();
            this.LastError = CUDADriver.cuModuleLoadData(ref this.curMod, binaryImage);
            return this.curMod;
        }

        public void MapGraphicsResources(CUgraphicsResource[] resources)
        {
            this.MapGraphicsResources(resources, new CUstream());
        }

        public void MapGraphicsResources(CUgraphicsResource[] resources, CUstream stream)
        {
            this.LastError = CUDADriver.cuGraphicsMapResources((uint) resources.Length, resources, stream);
        }

        public void Memset(CUdeviceptr ptr, byte value, uint count)
        {
            this.LastError = CUDADriver.cuMemsetD8(ptr, value, count);
        }

        public void Memset(CUdeviceptr ptr, ushort value, uint count)
        {
            this.LastError = CUDADriver.cuMemsetD16(ptr, value, count);
        }

        public void Memset(CUdeviceptr ptr, uint value, uint count)
        {
            this.LastError = CUDADriver.cuMemsetD32(ptr, value, count);
        }

        public void Memset(CUdeviceptr ptr, uint pitch, byte value, uint width, uint height)
        {
            this.LastError = CUDADriver.cuMemsetD2D8(ptr, pitch, value, width, height);
        }

        public void Memset(CUdeviceptr ptr, uint pitch, ushort value, uint width, uint height)
        {
            this.LastError = CUDADriver.cuMemsetD2D16(ptr, pitch, value, width, height);
        }

        public void Memset(CUdeviceptr ptr, uint pitch, uint value, uint width, uint height)
        {
            this.LastError = CUDADriver.cuMemsetD2D32(ptr, pitch, value, width, height);
        }

        public void SetCurrentContext(CUcontext ctx)
        {
            this.LastError = CUDADriver.cuCtxSetCurrent(ctx);
        }

        public CUcontext GetCurrentContextV1()
        {
            CUcontext ctx = new CUcontext();
            this.LastError = CUDADriver.cuCtxGetCurrent(ref ctx);
            return ctx;
        }

        public static CUcontext GetCurrentContext()
        {
            CUcontext ctx = new CUcontext();
            CUResult res = CUDADriver.cuCtxGetCurrent(ref ctx);
            if(res != CUResult.Success)
                throw new CUDAException(res);
            return ctx;
        }

        public static CUcontext? TryGetCurrentContext()
        {
            CUcontext ctx = new CUcontext();
            CUResult res = CUDADriver.cuCtxGetCurrent(ref ctx);
            if (res != CUResult.Success)
                return null;
            return ctx;
        }

        public CUcontext PopCurrentContext()
        {
            CUcontext pctx = new CUcontext();
            if(_version >= 4000)
                this.LastError = CUDADriver.cuCtxPopCurrent_v2(ref pctx);
            else
                this.LastError = CUDADriver.cuCtxPopCurrent(ref pctx);
            this.curCtx = pctx;
            return pctx;
        }

        public void PushCurrentContext()
        {
            this.PushCurrentContext(this.curCtx);
        }

        public void PushCurrentContext(CUcontext ctx)
        {
            if (_version >= 4000)
                this.LastError = CUDADriver.cuCtxPushCurrent_v2(ctx);
            else
                this.LastError = CUDADriver.cuCtxPushCurrent(ctx);
        }

        public void RecordEvent(CUevent e)
        {
            this.RecordEvent(e, new CUstream());
        }

        public void RecordEvent(CUevent e, CUstream stream)
        {
            this.LastError = CUDADriver.cuEventRecord(e, stream);
        }

        public void SetFunctionBlockShape(CUfunction func, int x, int y, int z)
        {
            this.LastError = CUDADriver.cuFuncSetBlockShape(func, x, y, z);
        }

        public void SetFunctionSharedSize(CUfunction func, uint sharedSize)
        {
            this.LastError = CUDADriver.cuFuncSetSharedSize(func, sharedSize);
        }

        public void SetGraphicsResourceMapFlags(CUgraphicsResource resource, CUGraphicsMapResourceFlags flags)
        {
            this.LastError = CUDADriver.cuGraphicsResourceSetMapFlags(resource, (uint) flags);
        }

        public void SetParameter(CUfunction func, CUtexref tex)
        {
            this.LastError = CUDADriver.cuParamSetTexRef(func, -1, tex);
        }

        public void SetParameter(CUfunction func, int offset, float value)
        {
            this.LastError = CUDADriver.cuParamSetf(func, offset, value);
        }

        public void SetParameter(CUfunction func, int offset, long value)
        {
            this.LastError = CUDADriver.cuParamSetv(func, offset, ref value, 8);
        }

        public void SetParameter<T>(CUfunction func, int offset, T vector)
        {
            GCHandle handle = GCHandle.Alloc(vector, GCHandleType.Pinned);
            int size = Marshal.SizeOf(vector);
            this.LastError = CUDADriver.cuParamSetv(func, offset, handle.AddrOfPinnedObject(), (uint)size );
            handle.Free();
        }

        public void SetParameter<T>(CUfunction func, int offset, T[] array)
        {
            GCHandle handle = GCHandle.Alloc(array, GCHandleType.Pinned);
            this.LastError = CUDADriver.cuParamSetv(func, offset, handle.AddrOfPinnedObject(), this.GetSize<T>(array));
            handle.Free();
        }

        public void SetParameter(CUfunction func, int offset, uint value)
        {
            this.LastError = CUDADriver.cuParamSeti(func, offset, value);
        }

        public void SetParameterSize(CUfunction func, uint bytes)
        {
            this.LastError = CUDADriver.cuParamSetSize(func, bytes);
        }

        public uint SetTextureAddress(CUtexref tex, CUdeviceptr dptr, uint bytes)
        {
            uint byteOffset = 0;
            this.LastError = CUDADriver.cuTexRefSetAddress(ref byteOffset, tex, dptr, bytes);
            return byteOffset;
        }

        public void SetTextureAddress(CUtexref tex, CUDAArrayDescriptor desc, CUdeviceptr dptr, uint pitch)
        {
            if (_version >= 4010)
                this.LastError = CUDADriver.cuTexRefSetAddress2D_v3(tex, desc, dptr, pitch);
            else
                this.LastError = CUDADriver.cuTexRefSetAddress2D(tex, desc, dptr, pitch);
        }

        public void SetTextureAddressMode(CUtexref tex, int dimension, CUAddressMode addressMode)
        {
            this.LastError = CUDADriver.cuTexRefSetAddressMode(tex, dimension, addressMode);
        }

        public void SetTextureArray(CUtexref tex, CUarray array)
        {
            this.SetTextureArray(tex, array, 1);
        }

        public void SetTextureArray(CUtexref tex, CUarray array, uint flags)
        {
            this.LastError = CUDADriver.cuTexRefSetArray(tex, array, flags);
        }

        public void SetTextureFilterMode(CUtexref tex, CUFilterMode filterMode)
        {
            this.LastError = CUDADriver.cuTexRefSetFilterMode(tex, filterMode);
        }

        public void SetTextureFlags(CUtexref tex, uint flags)
        {
            this.LastError = CUDADriver.cuTexRefSetFlags(tex, flags);
        }

        public void SetTextureFormat(CUtexref tex, CUArrayFormat format, int numComponents)
        {
            this.LastError = CUDADriver.cuTexRefSetFormat(tex, format, numComponents);
        }

        public void SynchronizeContext()
        {
            this.LastError = CUDADriver.cuCtxSynchronize();
        }

        public void SynchronizeEvent(CUevent e)
        {
            this.LastError = CUDADriver.cuEventSynchronize(e);
        }

        public void SynchronizeStream(CUstream stream)
        {
            this.LastError = CUDADriver.cuStreamSynchronize(stream);
        }

        public void UnloadModule()
        {
            this.UnloadModule(this.curMod);
        }

        public void UnloadModule(CUmodule mod)
        {
            this.LastError = CUDADriver.cuModuleUnload(mod);
        }

        public void UnmapGraphicsResources(CUgraphicsResource[] resources)
        {
            this.UnmapGraphicsResources(resources, new CUstream());
        }

        public void UnmapGraphicsResources(CUgraphicsResource[] resources, CUstream stream)
        {
            this.LastError = CUDADriver.cuGraphicsUnmapResources((uint) resources.Length, resources, stream);
        }

        public void UnregisterGraphicsResource(CUgraphicsResource resource)
        {
            this.LastError = CUDADriver.cuGraphicsUnregisterResource(resource);
        }

        public CUcontext CurrentContext
        {
            get
            {
                return this.curCtx;
            }
            internal set
            {
                this.curCtx = value;
            }
            //get
            //{
            //    var ctx = new CUcontext();
            //    this.LastError = CUDADriver.cuCtxGetCurrent(ref ctx);
            //    return ctx;
            //}
            //internal set
            //{
            //    //this.curCtx = value;
            //    this.LastError = CUDADriver.cuCtxSetCurrent(value);
            //}
        }

        public Device CurrentDevice
        {
            get
            {
                return this.curDev;
            }
            internal set
            {
                this.curDev = value;
            }
        }

        public CUfunction CurrentFunction
        {
            get
            {
                return this.curFunc;
            }
            internal set
            {
                this.curFunc = value;
            }
        }

        public CUmodule CurrentModule
        {
            get
            {
                return this.curMod;
            }
            internal set
            {
                this.curMod = value;
            }
        }

        public Device[] Devices
        {
            get
            {
                if (this.devices == null)
                {
                    this.devices = new List<Device>();
                    int count = 0;
                    this.LastError = CUDADriver.cuDeviceGetCount(ref count);
                    if (count > 0)
                    {
                        for (int i = 0; i < count; i++)
                        {
                            CUdevice udevice = new CUdevice();
                            this.LastError = CUDADriver.cuDeviceGet(ref udevice, i);
                            Device item = new Device {
                                Ordinal = i
                            };
                            byte[] name = new byte[0x100];
                            this.LastError = CUDADriver.cuDeviceGetName(name, name.Length, udevice);
                            item.Name = Encoding.ASCII.GetString(name);
                            item.Name = item.Name.Substring(0, item.Name.IndexOf('\0'));
                            int major = 0;
                            int minor = 0;
                            this.LastError = CUDADriver.cuDeviceComputeCapability(ref major, ref minor, udevice);
                            item.ComputeCapability = new System.Version(major, minor);
                            item.Handle = udevice;
                            CUDeviceProperties prop = new CUDeviceProperties();
                            this.LastError = CUDADriver.cuDeviceGetProperties(ref prop, udevice);
                            DeviceProperties properties2 = new DeviceProperties {
                                ClockRate = prop.clockRate,
                                MaxGridSize = prop.maxGridSize,
                                MaxThreadsDim = prop.maxThreadsDim,
                                MaxThreadsPerBlock = prop.maxThreadsPerBlock,
                                MemoryPitch = prop.memPitch,
                                RegistersPerBlock = prop.regsPerBlock,
                                SharedMemoryPerBlock = prop.sharedMemPerBlock,
                                SIMDWidth = prop.SIMDWidth,
                                TextureAlign = prop.textureAlign,
                                TotalConstantMemory = prop.totalConstantMemory
                            };
                            item.Properties = properties2;
                            SizeT bytes = 0;
                            this.LastError = CUDADriver.cuDeviceTotalMem(ref bytes, udevice);
                            item.TotalMemory = bytes;
                            this.devices.Add(item);
                        }
                    }
                }
                return this.devices.ToArray();
            }
        }

        public ulong FreeMemory
        {
            get
            {
                SizeT free = 0;
                SizeT total = 0;
                this.LastError = CUDADriver.cuMemGetInfo(ref free, ref total);
                return free;
            }
        }

        public CUResult LastError
        {
            get
            {
                return this.lastError;
            }
            private set
            {
                this.lastError = value;
                if (this.useRuntimeExceptions && (this.lastError != CUResult.Success))
                {
                    throw new CUDAException(this.lastError);
                }
            }
        }

        public ulong TotalMemory
        {
            get
            {
                SizeT free = 0;
                SizeT total = 0;
                this.LastError = CUDADriver.cuMemGetInfo(ref free, ref total);
                return total;
            }
        }

        public bool UseRuntimeExceptions
        {
            get
            {
                return this.useRuntimeExceptions;
            }
            set
            {
                this.useRuntimeExceptions = value;
            }
        }

        public System.Version Version
        {
            get
            {
                return new System.Version(3, 2);
            }
        }

        public static int MSizeOf(Type t)
        {
            if (t == typeof(char))
                return 2;
            else
                return Marshal.SizeOf(t);
        }

        public void HostRegister(IntPtr buffer, int bytes, uint flags)
        {
            CUDADriver.cuMemHostRegister(buffer, bytes, flags);
        }

        public void HostUnregister(IntPtr buffer)
        {
            CUDADriver.cuMemHostUnregister(buffer);
        }

        public bool DeviceCanAccessPeer(CUdevice device, CUdevice peerDevice)
        {
            int res = 0;
            this.LastError = CUDADriver.cuDeviceCanAccessPeer(ref res, device, peerDevice);
            return res != 0;
        }

        public void EnablePeerAccess(CUcontext peerContext, uint flags)
        {
            this.LastError = CUDADriver.cuCtxEnablePeerAccess(peerContext, flags);
        }

        public void DisablePeerAccess(CUcontext peerContext)
        {
            this.LastError = CUDADriver.cuCtxDisablePeerAccess(peerContext);
        }

        public IntPtr GetPointerAttribute(CUPointerAttribute attribute, CUdeviceptr ptr)
        {
            IntPtr data = new IntPtr();
            this.LastError = CUDADriver.cuPointerGetAttribute(ref data, attribute, ptr);
            return data;
        }

        public CUMemoryType GetPointerMemoryType(CUPointerAttribute attribute, CUdeviceptr ptr)
        {
            CUMemoryType data = new CUMemoryType();
            this.LastError = CUDADriver.cuPointerGetAttribute(ref data, attribute, ptr);
            return data;
        }

        public CUcontext GetPointerContext(CUdeviceptr ptr)
        {
            CUcontext ctx = new CUcontext();
            this.LastError = CUDADriver.cuPointerGetAttribute(ref ctx, CUPointerAttribute.Context, ptr);
            return ctx;
        }

        public CUP2PTokens GetP2PTokens(CUdeviceptr ptr)
        {
            CUP2PTokens tokens = new CUP2PTokens();
            this.LastError = CUDADriver.cuPointerGetAttribute(ref tokens, CUPointerAttribute.P2PTokens, ptr);
            return tokens;
        }
    }
}

