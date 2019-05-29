namespace GASS.CUDA.Engine
{
    using System;

    public class Parameter
    {
        private ParameterDirection direction;
        private string name;
        private ParameterType type;
        private object value;

        public Parameter(string name)
        {
            this.Name = name;
        }

        public Parameter(string name, ParameterType type) : this(name)
        {
            this.Type = type;
        }

        public Parameter(string name, ParameterType type, ParameterDirection direction) : this(name, type)
        {
            this.Direction = direction;
        }

        public Parameter(string name, ParameterType type, ParameterDirection direction, object value) : this(name, type, direction)
        {
            this.Value = value;
        }

        public ParameterDirection Direction
        {
            get
            {
                return this.direction;
            }
            set
            {
                this.direction = value;
            }
        }

        public string Name
        {
            get
            {
                return this.name;
            }
            private set
            {
                this.name = value;
            }
        }

        public ParameterType Type
        {
            get
            {
                return this.type;
            }
            set
            {
                this.type = value;
            }
        }

        public object Value
        {
            get
            {
                return this.value;
            }
            set
            {
                this.value = value;
            }
        }
    }
}

