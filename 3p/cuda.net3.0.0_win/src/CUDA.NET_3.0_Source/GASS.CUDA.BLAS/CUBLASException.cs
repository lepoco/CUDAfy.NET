namespace GASS.CUDA.BLAS
{
    using System;

    public class CUBLASException : Exception
    {
        private CUBLASStatus error;

        public CUBLASException(CUBLASStatus error)
        {
            this.error = error;
        }

        public CUBLASException(CUBLASStatus error, string message, Exception e) : base(message, e)
        {
            this.error = error;
        }

        public override string ToString()
        {
            return this.CUBLASError.ToString();
        }

        public CUBLASStatus CUBLASError
        {
            get
            {
                return this.error;
            }
        }
    }
}

