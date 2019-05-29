namespace GASS.CUDA.FFT
{
    using System;

    public class CUFFTException : Exception
    {
        private CUFFTResult error;

        public CUFFTException(CUFFTResult error)
        {
            this.error = error;
        }

        public CUFFTException(CUFFTResult error, string message, Exception e) : base(message, e)
        {
            this.error = error;
        }

        public override string ToString()
        {
            return this.CUFFTError.ToString();
        }

        public CUFFTResult CUFFTError
        {
            get
            {
                return this.error;
            }
        }
    }
}

