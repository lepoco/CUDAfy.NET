namespace GASS.CUDA
{
    using System;
    using System.Xml;

    [global::System.Serializable]
    public class CUDAException : Exception
    {
        private CUResult error;

        public CUDAException(CUResult error)
        {
            this.error = error;
        }

        public CUDAException(CUResult error, string message, Exception e) : base(message, e)
        {
            this.error = error;
        }

        public override string Message
        {
            get
            {
                return CUDAError.ToString();
            }
        }

        public override string ToString()
        {
            return this.CUDAError.ToString();
        }

        public CUResult CUDAError
        {
            get
            {
                return this.error;
            }
        }

        protected CUDAException(
          System.Runtime.Serialization.SerializationInfo info,
          System.Runtime.Serialization.StreamingContext context)
            : base(info, context) { }
    }
}

