namespace GASS.CUDA.Tools
{
    using GASS.CUDA;
    using GASS.CUDA.Types;
    using System;
    using System.Threading;

    public class CUDAContextSynchronizer
    {
        private CUcontext ctx;
        private CUResult res;
        private object sync = new object();
        private CUcontext tempCtx = new CUcontext();

        public CUDAContextSynchronizer(CUcontext ctx)
        {
            this.ctx = ctx;
        }

        public void Lock()
        {
            Monitor.Enter(this.sync);
            this.LastError = CUDADriver.cuCtxPushCurrent(this.ctx);
            if (this.LastError != CUResult.Success)
            {
                throw new CUDAException(this.res);
            }
            _isLocked = true;
        }

        //private CUcontext _pctx;

        public CUcontext MakeFloating()
        {
            CUcontext pctx = new CUcontext();
            this.LastError = CUDADriver.cuCtxPopCurrent(ref pctx);
            return pctx;
        }

        public void StopFloating(CUcontext pctx)
        {
            this.LastError = CUDADriver.cuCtxPushCurrent(pctx);
        }

        public void Unlock()
        {
            this.LastError = CUDADriver.cuCtxPopCurrent(ref this.tempCtx);
            if (this.LastError != CUResult.Success)
            {
                throw new CUDAException(this.res);
            }
            _isLocked = false;
            Monitor.Exit(this.sync);          
        }

        public CUcontext Context
        {
            get
            {
                return this.ctx;
            }
        }

        public CUResult LastError
        {
            get
            {
                return this.res;
            }
            protected internal set
            {
                this.res = value;
            }
        }

        private volatile bool _isLocked = false;

        public bool IsLocked 
        {
            get { return _isLocked;  }     
        }
    }
}

