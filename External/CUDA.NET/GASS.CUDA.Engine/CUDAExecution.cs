namespace GASS.CUDA.Engine
{
    using GASS.CUDA;
    using GASS.CUDA.Types;
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Reflection;
    using System.Runtime.InteropServices;

    public sealed class CUDAExecution
    {
        private GASS.CUDA.CUDA cuda;
        private CUfunction cudaFunction;
        private CUmodule cudaModule;
        private string function;
        private float lastRunTime;
        private string module;
        private int parameterOffset;
        private List<Parameter> parameters;
        private CUevent start;
        private CUevent stop;
        private List<CUtexref> textures;

        public CUDAExecution(string module, string function) : this(0, module, function)
        {
        }

        public CUDAExecution(GASS.CUDA.CUDA cuda, string module, string function)
        {
            this.parameters = new List<Parameter>();
            this.textures = new List<CUtexref>();
            this.CUDAInstance = cuda;
            if (!module.EndsWith("cubin"))
            {
                module = module + ".cubin";
            }
            FileInfo info = new FileInfo(module);
            this.Module = info.FullName;
            this.Function = function;
            this.CUDAModule = cuda.LoadModule(this.module);
            this.CUDAFunction = cuda.GetModuleFunction(this.function);
            this.start = cuda.CreateEvent();
            this.stop = cuda.CreateEvent();
        }

        public CUDAExecution(int device, string module, string function) : this(new GASS.CUDA.CUDA(device, true), module, function)
        {
        }

        public int AddParameter(Parameter parameter)
        {
            int size = 0;
            switch (parameter.Type)
            {
                case ParameterType.Scalar:
                    if (!(parameter.Value is float))
                    {
                        if ((((parameter.Value is byte) || (parameter.Value is sbyte)) || ((parameter.Value is short) || (parameter.Value is ushort))) || ((parameter.Value is int) || (parameter.Value is uint)))
                        {
                            this.cuda.SetParameter(this.CUDAFunction, this.parameterOffset, (uint) parameter.Value);
                        }
                        break;
                    }
                    this.cuda.SetParameter(this.CUDAFunction, this.parameterOffset, (float) parameter.Value);
                    break;

                case ParameterType.Buffer:
                    this.cuda.SetParameter(this.CUDAFunction, this.parameterOffset, ((CUdeviceptr) parameter.Value).Pointer);
                    break;

                case ParameterType.Vector:
                    this.cuda.SetParameter<object>(this.CUDAFunction, this.parameterOffset, parameter.Value);
                    break;

                case ParameterType.Texture:
                    this.cuda.SetParameter(this.CUDAFunction, (CUtexref) parameter.Value);
                    break;
            }
            switch (parameter.Type)
            {
                case ParameterType.Scalar:
                case ParameterType.Vector:
                    size = Marshal.SizeOf(parameter.Value);
                    break;

                case ParameterType.Buffer:
                    size = IntPtr.Size;
                    break;
            }
            this.parameterOffset += size;
            this.parameters.Add(parameter);
            return (this.parameters.Count - 1);
        }

        public int AddParameter(string name, float data)
        {
            Parameter parameter = new Parameter(name, ParameterType.Scalar, ParameterDirection.In, data);
            return this.AddParameter(parameter);
        }

        public int AddParameter(string name, uint data)
        {
            Parameter parameter = new Parameter(name, ParameterType.Scalar, ParameterDirection.In, data);
            return this.AddParameter(parameter);
        }

        public int AddParameter<T>(string name, T data)
        {
            Parameter parameter = new Parameter(name, ParameterType.Vector, ParameterDirection.In, data);
            return this.AddParameter(parameter);
        }

        public int AddParameter<T>(string name, T[] data)
        {
            return this.AddParameter<T>(name, data, ParameterDirection.In);
        }

        public int AddParameter<T>(string name, T[] data, ParameterDirection direction)
        {
            CUdeviceptr udeviceptr = new CUdeviceptr();
            switch (direction)
            {
                case ParameterDirection.In:
                case ParameterDirection.InOut:
                    udeviceptr = this.cuda.CopyHostToDevice<T>(data);
                    break;

                case ParameterDirection.Out:
                    udeviceptr = this.cuda.Allocate<T>(data);
                    break;
            }
            Parameter parameter = new Parameter(name, ParameterType.Buffer, direction, udeviceptr);
            return this.AddParameter(parameter);
        }

        public void Clear()
        {
            foreach (CUtexref utexref in this.textures)
            {
                this.cuda.DestroyTexture(utexref);
            }
            this.textures.Clear();
            foreach (Parameter parameter in this.parameters)
            {
                if (parameter.Type == ParameterType.Buffer)
                {
                    this.cuda.Free((CUdeviceptr) parameter.Value);
                }
            }
            this.parameters.Clear();
            this.parameterOffset = 0;
        }

        public float Launch(Int3 blocks, Int3 threads)
        {
            return this.Launch(blocks.x, blocks.y, threads.x, threads.y, threads.z);
        }

        public float Launch(int blocksX, int blocksY, int threadsX, int threadsY, int threadsZ)
        {
            this.cuda.SetFunctionBlockShape(this.CUDAFunction, threadsX, threadsY, threadsZ);
            this.cuda.SetParameterSize(this.CUDAFunction, (uint) this.parameterOffset);
            this.cuda.RecordEvent(this.start);
            this.cuda.Launch(this.CUDAFunction, blocksX, blocksY);
            this.cuda.RecordEvent(this.stop);
            this.cuda.SynchronizeEvent(this.stop);
            this.LastRunTime = this.cuda.ElapsedTime(this.start, this.stop);
            return this.LastRunTime;
        }

        public void ReadData<T>(T[] output, Parameter parameter)
        {
            if (parameter.Type == ParameterType.Buffer)
            {
                this.cuda.CopyDeviceToHost<T>((CUdeviceptr) parameter.Value, output);
            }
        }

        public void ReadData<T>(T[] output, int paramIndex)
        {
            this.ReadData<T>(output, this[paramIndex]);
        }

        public void ReadData<T>(T[] output, string paramName)
        {
            this.ReadData<T>(output, this[paramName]);
        }

        public CUfunction CUDAFunction
        {
            get
            {
                return this.cudaFunction;
            }
            private set
            {
                this.cudaFunction = value;
            }
        }

        public GASS.CUDA.CUDA CUDAInstance
        {
            get
            {
                return this.cuda;
            }
            private set
            {
                this.cuda = value;
            }
        }

        public CUmodule CUDAModule
        {
            get
            {
                return this.cudaModule;
            }
            private set
            {
                this.cudaModule = value;
            }
        }

        public string Function
        {
            get
            {
                return this.function;
            }
            private set
            {
                this.function = value;
            }
        }

        public Parameter this[string name]
        {
            get
            {
                foreach (Parameter parameter in this.parameters)
                {
                    if (parameter.Name == name)
                    {
                        return parameter;
                    }
                }
                return null;
            }
            set
            {
                for (int i = 0; i < this.parameters.Count; i++)
                {
                    if (this.parameters[i].Name == name)
                    {
                        this.parameters[i] = value;
                    }
                }
            }
        }

        public Parameter this[int index]
        {
            get
            {
                return this.parameters[index];
            }
            set
            {
                this.parameters[index] = value;
            }
        }

        public float LastRunTime
        {
            get
            {
                return this.lastRunTime;
            }
            private set
            {
                this.lastRunTime = value;
            }
        }

        public string Module
        {
            get
            {
                return this.module;
            }
            private set
            {
                this.module = value;
            }
        }
    }
}

