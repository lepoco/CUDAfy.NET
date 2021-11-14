namespace GASS.CUDA
{
    using System;

    public class DeviceProperties
    {
        private int clockRate;
        private int[] maxGridSize;
        private int[] maxThreadsDim;
        private int maxThreadsPerBlock;
        private int memPitch;
        private int regsPerBlock;
        private int sharedMemPerBlock;
        private int simdWidth;
        private int textureAlign;
        private int totalConstantMemory;

        public int ClockRate
        {
            get
            {
                return this.clockRate;
            }
            internal set
            {
                this.clockRate = value;
            }
        }

        public int[] MaxGridSize
        {
            get
            {
                return this.maxGridSize;
            }
            internal set
            {
                this.maxGridSize = value;
            }
        }

        public int[] MaxThreadsDim
        {
            get
            {
                return this.maxThreadsDim;
            }
            internal set
            {
                this.maxThreadsDim = value;
            }
        }

        public int MaxThreadsPerBlock
        {
            get
            {
                return this.maxThreadsPerBlock;
            }
            internal set
            {
                this.maxThreadsPerBlock = value;
            }
        }

        public int MemoryPitch
        {
            get
            {
                return this.memPitch;
            }
            internal set
            {
                this.memPitch = value;
            }
        }

        public int RegistersPerBlock
        {
            get
            {
                return this.regsPerBlock;
            }
            internal set
            {
                this.regsPerBlock = value;
            }
        }

        public int SharedMemoryPerBlock
        {
            get
            {
                return this.sharedMemPerBlock;
            }
            internal set
            {
                this.sharedMemPerBlock = value;
            }
        }

        public int SIMDWidth
        {
            get
            {
                return this.simdWidth;
            }
            internal set
            {
                this.simdWidth = value;
            }
        }

        public int TextureAlign
        {
            get
            {
                return this.textureAlign;
            }
            set
            {
                this.textureAlign = value;
            }
        }

        public int TotalConstantMemory
        {
            get
            {
                return this.totalConstantMemory;
            }
            internal set
            {
                this.totalConstantMemory = value;
            }
        }
    }
}

