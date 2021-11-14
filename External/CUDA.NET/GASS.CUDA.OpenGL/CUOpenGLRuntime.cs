namespace GASS.CUDA.OpenGL
{
    using GASS.CUDA;
    using System;
    using System.Runtime.InteropServices;

    public class CUOpenGLRuntime
    {
        [Obsolete("Use new graphics API."), DllImport("cudart")]
        public static extern cudaError cudaGLMapBufferObject(ref IntPtr devPtr, uint bufObj);
        [Obsolete("Use new graphics API."), DllImport("cudart")]
        public static extern cudaError cudaGLMapBufferObjectAsync(ref IntPtr devPtr, uint bufObj, cudaStream stream);
        [Obsolete("Use new graphics API."), DllImport("cudart")]
        public static extern cudaError cudaGLRegisterBufferObject(uint bufObj);
        [Obsolete("Use new graphics API."), DllImport("cudart")]
        public static extern cudaError cudaGLSetBufferObjectMapFlags(uint bufObj, uint flags);
        [DllImport("cudart")]
        public static extern cudaError cudaGLSetGLDevice(int device);
        [Obsolete("Use new graphics API."), DllImport("cudart")]
        public static extern cudaError cudaGLUnmapBufferObject(uint bufObj);
        [Obsolete("Use new graphics API."), DllImport("cudart")]
        public static extern cudaError cudaGLUnmapBufferObjectAsync(uint bufObj, cudaStream stream);
        [Obsolete("Use new graphics API."), DllImport("cudart")]
        public static extern cudaError cudaGLUnregisterBufferObject(uint bufObj);
        [DllImport("cudart")]
        public static extern cudaError cudaGraphicsGLRegisterBuffer([In] cudaGraphicsResource[] resource, uint buffer, uint Flags);
        [DllImport("cudart")]
        public static extern cudaError cudaGraphicsGLRegisterImage([In] cudaGraphicsResource[] resource, uint image, uint target, uint Flags);
        [DllImport("cudart")]
        public static extern cudaError cudaWGLGetDevice(ref int device, IntPtr hGpu);
    }
}

