namespace GASS.CUDA
{
    using GASS.CUDA.Types;
    using System;

    public class Device
    {
        private Version computeCapability;
        private CUdevice handle;
        private string name;
        private int ordinal;
        private DeviceProperties properties;
        private ulong totalMemory;

        public Version ComputeCapability
        {
            get
            {
                return this.computeCapability;
            }
            internal set
            {
                this.computeCapability = value;
            }
        }

        public CUdevice Handle
        {
            get
            {
                return this.handle;
            }
            internal set
            {
                this.handle = value;
            }
        }

        public string Name
        {
            get
            {
                return this.name;
            }
            internal set
            {
                this.name = value;
            }
        }

        public int Ordinal
        {
            get
            {
                return this.ordinal;
            }
            internal set
            {
                this.ordinal = value;
            }
        }

        public DeviceProperties Properties
        {
            get
            {
                return this.properties;
            }
            internal set
            {
                this.properties = value;
            }
        }

        public ulong TotalMemory
        {
            get
            {
                return this.totalMemory;
            }
            internal set
            {
                this.totalMemory = value;
            }
        }
    }
}

