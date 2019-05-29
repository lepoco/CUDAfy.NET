namespace GASS.CUDA.BLAS
{
    using GASS.CUDA;
    using GASS.CUDA.Types;
    using System;
    using System.Runtime.InteropServices;

    public class CUBLAS
    {
        private GASS.CUDA.CUDA cuda;
        private CUBLASStatus lastError;
        private bool useRuntimeExceptions = true;

        public CUBLAS(GASS.CUDA.CUDA cuda)
        {
            this.cuda = cuda;
        }

        public CUdeviceptr Allocate<T>(T[] array)
        {
            CUdeviceptr devicePtr = new CUdeviceptr();
            this.LastError = CUBLASDriver.cublasAlloc(array.Length, CUDA.MSizeOf(typeof(T)), ref devicePtr);
            return devicePtr;
        }

        public CUdeviceptr Allocate(int numOfElements, int elementSize)
        {
            CUdeviceptr devicePtr = new CUdeviceptr();
            this.LastError = CUBLASDriver.cublasAlloc(numOfElements, elementSize, ref devicePtr);
            return devicePtr;
        }

        //public void Free(CUdeviceptr ptr)
        //{
        //    this.LastError = CUBLASDriver.cublasFree(ptr);
        //}

        public CUBLASStatus GetError()
        {
            this.LastError = CUBLASDriver.cublasGetError();
            return this.LastError;
        }

        public void GetMatrix<T>(int rows, int cols, CUdeviceptr ptr, int lda, T[] data, int ldb)
        {
            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            this.LastError = CUBLASDriver.cublasGetMatrix(rows, cols, CUDA.MSizeOf(typeof(T)), ptr, lda, handle.AddrOfPinnedObject(), ldb);
            handle.Free();
        }

        public void GetVector<T>(CUdeviceptr ptr, T[] data)
        {
            this.GetVector<T>(ptr, 1, data, 1);
        }

        public void GetVector<T>(CUdeviceptr ptr, int incx, T[] data, int incy)
        {
            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            this.LastError = CUBLASDriver.cublasGetVector(data.Length, CUDA.MSizeOf(typeof(T)), ptr, incx, handle.AddrOfPinnedObject(), incy);
            handle.Free();
        }

        public void Init()
        {
            this.LastError = CUBLASDriver.cublasInit();
        }

        //public void Create()
        //{
        //    this.LastError = CUBLASDriver.cublasCreate_v2();
        //}

        public void SetMatrix<T>(int rows, int cols, T[] data, int lda, CUdeviceptr ptr, int ldb)
        {
            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            this.LastError = CUBLASDriver.cublasSetMatrix(rows, cols, CUDA.MSizeOf(typeof(T)), handle.AddrOfPinnedObject(), lda, ptr, ldb);
            handle.Free();
        }

        public void SetVector<T>(T[] data, CUdeviceptr ptr)
        {
            this.SetVector<T>(data, 1, ptr, 1);
        }

        public void SetVector<T>(T[] data, int incx, CUdeviceptr ptr, int incy)
        {
            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            this.LastError = CUBLASDriver.cublasSetVector(data.Length, CUDA.MSizeOf(typeof(T)), handle.AddrOfPinnedObject(), incx, ptr, incy);
            handle.Free();
        }

        public void Shutdown()
        {
            this.LastError = CUBLASDriver.cublasShutdown();
        }

        public CUBLASStatus LastError
        {
            get
            {
                return this.lastError;
            }
            private set
            {
                this.lastError = value;
                if (this.useRuntimeExceptions && (this.lastError != CUBLASStatus.Success))
                {
                    throw new CUBLASException(this.lastError);
                }
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
    }
}

