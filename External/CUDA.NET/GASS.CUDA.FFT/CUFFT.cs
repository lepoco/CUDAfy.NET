namespace GASS.CUDA.FFT
{
    using GASS.CUDA;
    using GASS.CUDA.FFT.Types;
    using GASS.CUDA.Types;
    using System;
    [Obsolete]
    public class CUFFT
    {
        //private GASS.CUDA.CUDA cuda;
        private CUFFTResult lastError;
        private cufftHandle plan = new cufftHandle();
        private bool useRuntimeExceptions = true;
        private ICUFFTDriver _driver;
        public CUFFT(GASS.CUDA.CUDA cuda)
        {
            //this.cuda = cuda;
            if (IntPtr.Size == 8)
                _driver = new CUFFTDriver64();
            else
                _driver = new CUFFTDriver32();
        }

        public void Destroy()
        {
            this.Destroy(this.plan);
        }

        public void Destroy(cufftHandle plan)
        {
            this.LastError = _driver.cufftDestroy(plan);
        }

        //public void Execute1D(cuFloatComplex[] input, cuFloatComplex[] output, int nx, int batch)
        //{
        //    this.Execute1D(input, output, nx, batch, CUFFTDirection.Forward);
        //}

        //public void Execute1D(cuFloatComplex[] input, cuFloatReal[] output, int nx, int batch)
        //{
        //    cufftHandle plan = new cufftHandle();
        //    this.LastError = CUFFTDriver.cufftPlan1d(ref plan, nx, CUFFTType.C2R, batch);
        //    CUdeviceptr idata = this.cuda.CopyHostToDevice<cuFloatComplex>(input);
        //    CUdeviceptr odata = this.cuda.Allocate<cuFloatReal>(output);
        //    CUFFTDriver.cufftExecC2R(plan, idata, odata);
        //    this.cuda.CopyDeviceToHost<cuFloatReal>(odata, output);
        //    CUFFTDriver.cufftDestroy(plan);
        //    this.cuda.Free(idata);
        //    this.cuda.Free(odata);
        //}

        //public void Execute1D(cuFloatReal[] input, cuFloatComplex[] output, int nx, int batch)
        //{
        //    cufftHandle plan = new cufftHandle();
        //    this.LastError = CUFFTDriver.cufftPlan1d(ref plan, nx, CUFFTType.R2C, batch);
        //    CUdeviceptr idata = this.cuda.CopyHostToDevice<cuFloatReal>(input);
        //    CUdeviceptr odata = this.cuda.Allocate<cuFloatComplex>(output);
        //    CUFFTDriver.cufftExecR2C(plan, idata, odata);
        //    this.cuda.CopyDeviceToHost<cuFloatComplex>(odata, output);
        //    CUFFTDriver.cufftDestroy(plan);
        //    this.cuda.Free(idata);
        //    this.cuda.Free(odata);
        //}

        //public void Execute1D(cuFloatComplex[] input, cuFloatComplex[] output, int nx, int batch, CUFFTDirection direction)
        //{
        //    cufftHandle plan = new cufftHandle();
        //    this.LastError = CUFFTDriver.cufftPlan1d(ref plan, nx, CUFFTType.C2C, batch);
        //    CUdeviceptr idata = this.cuda.CopyHostToDevice<cuFloatComplex>(input);
        //    CUdeviceptr odata = this.cuda.Allocate<cuFloatComplex>(output);
        //    CUFFTDriver.cufftExecC2C(plan, idata, odata, direction);
        //    this.cuda.CopyDeviceToHost<cuFloatComplex>(odata, output);
        //    CUFFTDriver.cufftDestroy(plan);
        //    this.cuda.Free(idata);
        //    this.cuda.Free(odata);
        //}

        //public void Execute2D(cuFloatComplex[] input, cuFloatComplex[] output, int nx, int ny)
        //{
        //    this.Execute2D(input, output, nx, ny, CUFFTDirection.Forward);
        //}

        //public void Execute2D(cuFloatComplex[] input, cuFloatReal[] output, int nx, int ny)
        //{
        //    cufftHandle plan = new cufftHandle();
        //    this.LastError = CUFFTDriver.cufftPlan2d(ref plan, nx, ny, CUFFTType.C2R);
        //    CUdeviceptr idata = this.cuda.CopyHostToDevice<cuFloatComplex>(input);
        //    CUdeviceptr odata = this.cuda.Allocate<cuFloatReal>(output);
        //    CUFFTDriver.cufftExecC2R(plan, idata, odata);
        //    this.cuda.CopyDeviceToHost<cuFloatReal>(odata, output);
        //    CUFFTDriver.cufftDestroy(plan);
        //    this.cuda.Free(idata);
        //    this.cuda.Free(odata);
        //}

        //public void Execute2D(cuFloatReal[] input, cuFloatComplex[] output, int nx, int ny)
        //{
        //    cufftHandle plan = new cufftHandle();
        //    this.LastError = CUFFTDriver.cufftPlan2d(ref plan, nx, ny, CUFFTType.R2C);
        //    CUdeviceptr idata = this.cuda.CopyHostToDevice<cuFloatReal>(input);
        //    CUdeviceptr odata = this.cuda.Allocate<cuFloatComplex>(output);
        //    CUFFTDriver.cufftExecR2C(plan, idata, odata);
        //    this.cuda.CopyDeviceToHost<cuFloatComplex>(odata, output);
        //    CUFFTDriver.cufftDestroy(plan);
        //    this.cuda.Free(idata);
        //    this.cuda.Free(odata);
        //}

        //public void Execute2D(cuFloatComplex[] input, cuFloatComplex[] output, int nx, int ny, CUFFTDirection direction)
        //{
        //    cufftHandle plan = new cufftHandle();
        //    this.LastError = CUFFTDriver.cufftPlan2d(ref plan, nx, ny, CUFFTType.C2C);
        //    CUdeviceptr idata = this.cuda.CopyHostToDevice<cuFloatComplex>(input);
        //    CUdeviceptr odata = this.cuda.Allocate<cuFloatComplex>(output);
        //    CUFFTDriver.cufftExecC2C(plan, idata, odata, direction);
        //    this.cuda.CopyDeviceToHost<cuFloatComplex>(odata, output);
        //    CUFFTDriver.cufftDestroy(plan);
        //    this.cuda.Free(idata);
        //    this.cuda.Free(odata);
        //}

        //public void Execute3D(cuFloatComplex[] input, cuFloatComplex[] output, int nx, int ny, int nz)
        //{
        //    this.Execute3D(input, output, nx, ny, nz, CUFFTDirection.Forward);
        //}

        //public void Execute3D(cuFloatComplex[] input, cuFloatReal[] output, int nx, int ny, int nz)
        //{
        //    cufftHandle plan = new cufftHandle();
        //    this.LastError = CUFFTDriver.cufftPlan3d(ref plan, nx, ny, nz, CUFFTType.C2R);
        //    CUdeviceptr idata = this.cuda.CopyHostToDevice<cuFloatComplex>(input);
        //    CUdeviceptr odata = this.cuda.Allocate<cuFloatReal>(output);
        //    CUFFTDriver.cufftExecC2R(plan, idata, odata);
        //    this.cuda.CopyDeviceToHost<cuFloatReal>(odata, output);
        //    CUFFTDriver.cufftDestroy(plan);
        //    this.cuda.Free(idata);
        //    this.cuda.Free(odata);
        //}

        //public void Execute3D(cuFloatReal[] input, cuFloatComplex[] output, int nx, int ny, int nz)
        //{
        //    cufftHandle plan = new cufftHandle();
        //    this.LastError = CUFFTDriver.cufftPlan3d(ref plan, nx, ny, nz, CUFFTType.R2C);
        //    CUdeviceptr idata = this.cuda.CopyHostToDevice<cuFloatReal>(input);
        //    CUdeviceptr odata = this.cuda.Allocate<cuFloatComplex>(output);
        //    CUFFTDriver.cufftExecR2C(plan, idata, odata);
        //    this.cuda.CopyDeviceToHost<cuFloatComplex>(odata, output);
        //    CUFFTDriver.cufftDestroy(plan);
        //    this.cuda.Free(idata);
        //    this.cuda.Free(odata);
        //}

        //public void Execute3D(cuFloatComplex[] input, cuFloatComplex[] output, int nx, int ny, int nz, CUFFTDirection direction)
        //{
        //    cufftHandle plan = new cufftHandle();
        //    this.LastError = CUFFTDriver.cufftPlan3d(ref plan, nx, ny, nz, CUFFTType.C2C);
        //    CUdeviceptr idata = this.cuda.CopyHostToDevice<cuFloatComplex>(input);
        //    CUdeviceptr odata = this.cuda.Allocate<cuFloatComplex>(output);
        //    CUFFTDriver.cufftExecC2C(plan, idata, odata, direction);
        //    this.cuda.CopyDeviceToHost<cuFloatComplex>(odata, output);
        //    CUFFTDriver.cufftDestroy(plan);
        //    this.cuda.Free(idata);
        //    this.cuda.Free(odata);
        //}

        public void ExecuteComplexToComplex(CUdeviceptr input, CUdeviceptr output, CUFFTDirection direction)
        {
            this.ExecuteComplexToComplex(this.plan, input, output, direction);
        }

        public void ExecuteComplexToComplex(cufftHandle plan, CUdeviceptr input, CUdeviceptr output, CUFFTDirection direction)
        {
            this.LastError = _driver.cufftExecC2C(plan, input, output, direction);
        }

        public void ExecuteComplexToReal(CUdeviceptr input, CUdeviceptr output)
        {
            this.ExecuteComplexToReal(this.plan, input, output);
        }

        public void ExecuteComplexToReal(cufftHandle plan, CUdeviceptr input, CUdeviceptr output)
        {
            this.LastError = _driver.cufftExecC2R(plan, input, output);
        }

        public void ExecuteRealToComplex(CUdeviceptr input, CUdeviceptr output)
        {
            this.ExecuteRealToComplex(this.plan, input, output);
        }

        public void ExecuteRealToComplex(cufftHandle plan, CUdeviceptr input, CUdeviceptr output)
        {
            this.LastError = _driver.cufftExecR2C(plan, input, output);
        }

        public cufftHandle Plan1D(int nx, CUFFTType type, int batch)
        {
            this.plan = new cufftHandle();
            this.LastError = _driver.cufftPlan1d(ref this.plan, nx, type, batch);
            return this.plan;
        }

        public cufftHandle Plan2D(int nx, int ny, CUFFTType type)
        {
            this.plan = new cufftHandle();
            this.LastError = _driver.cufftPlan2d(ref this.plan, nx, ny, type);
            return this.plan;
        }

        public cufftHandle Plan3D(int nx, int ny, int nz, CUFFTType type)
        {
            this.plan = new cufftHandle();
            this.LastError = _driver.cufftPlan3d(ref this.plan, nx, ny, nz, type);
            return this.plan;
        }

        public CUFFTResult LastError
        {
            get
            {
                return this.lastError;
            }
            private set
            {
                this.lastError = value;
                if (this.useRuntimeExceptions && (this.lastError != CUFFTResult.Success))
                {
                    throw new CUFFTException(this.lastError);
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

        public static int GetVersion()
        {
            int version = -1;
            CUFFTResult res;
            try
            {
                
                ICUFFTDriver driver;
                if (IntPtr.Size == 8)
                    driver = new CUFFTDriver64();
                else
                    driver = new CUFFTDriver32();
                res = driver.cufftGetVersion(ref version);
                if(res != CUFFTResult.Success)
                    throw new CUFFTException(res);
            }
            catch (EntryPointNotFoundException ex)
            {
                System.Diagnostics.Debug.WriteLine("GetVersion(): " + ex.Message);
            }

            return version;
        }
    }
}

