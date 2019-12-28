# CUDAfy.NET - CUDA 10.2 & Visual Studio 2019 & NET.Framework 4.8
CUDAfy.NET access to work with Visual Studio 2019 and the latest NVIDIA Toolkit CUDA 10.2 library

I was helped by what [Cr33zz](https://github.com/Cr33zz) did in the [library's processing at VS 2017](https://github.com/Cr33zz/CUDAfy.NET).

## How to start with CUDAfy?

#### Required components
- Visual Studio 2019
- MSVC v142 x64 / 86 build tools (v.14.24) or higher
- .NET Framework 4.8 SDK

#### Launching
1. Download the latest repository
2. Open the Cudafy.sln project
3. Choose the "Debug" solution
4. Choose the "x64" solution platforms
5. Choose the "CudafyByExample" startup project
6. Rebuild whole project
7. Start "CudafyByExample"

### What works?
- [x] The library starts correctly in the .NET Framework 4.8
- [x] The library works correctly (for my knowledge) with NVIDIA Toolkit CUDA 10.2
- [x] The library works correctly with Visual Studio 2019 Enterprise 16.4.2
- [x] Everything starts correctly in the 64-bit version.

### What's new?
- I added automatic support for versions 10.2 and 10.1.
- A new way to detect Visual Studio locations with the MSVC package

### Where can I find CUDA Toolkit?
You can download the latest CUDA version from the official NVIDIA website
[NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Example program
Here is an example program you can run using compiled libraries.
```c#
class Program
{
    public const int N = 10;
    public static void Main()
    {
        Console.WriteLine("CUDAfy Example\nCollecting necessary resources...");

        CudafyModes.Target = eGPUType.Cuda; // To use OpenCL, change this enum
        CudafyModes.DeviceId = 0;
        CudafyTranslator.Language = CudafyModes.Target == eGPUType.OpenCL ? eLanguage.OpenCL : eLanguage.Cuda;

        //Check for available devices
        if (CudafyHost.GetDeviceCount(CudafyModes.Target) == 0)
            throw new System.ArgumentException("No suitable devices found.", "original");

        //Init device
        GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
        Console.WriteLine("Running example using {0}", gpu.GetDeviceProperties(false).Name);

        //Load module for GPU
        CudafyModule km = CudafyTranslator.Cudafy();
        gpu.LoadModule(km);

        //Define local arrays
        int[] a = new int[N];
        int[] b = new int[N];
        int[] c = new int[N];

        // allocate the memory on the GPU
        int[] dev_c = gpu.Allocate<int>(c);

        // fill the arrays 'a' and 'b' on the CPU
        for (int i = 0; i < N; i++)
        {
            a[i] = i;
            b[i] = i * i;
        }

        // copy the arrays 'a' and 'b' to the GPU
        int[] dev_a = gpu.CopyToDevice(a);
        int[] dev_b = gpu.CopyToDevice(b);

        gpu.Launch(1, N).add(dev_a, dev_b, dev_c);

        // copy the array 'c' back from the GPU to the CPU
        gpu.CopyFromDevice(dev_c, c);

        // display the results
        for (int i = 0; i < N; i++)
            Console.WriteLine("{0} + {1} = {2}", a[i], b[i], c[i]);

        // free the memory allocated on the GPU
        gpu.FreeAll();

        Console.WriteLine("Done!");
        Console.ReadKey();
    }

    [Cudafy]
    public static void add(GThread thread, int[] a, int[] b, int[] c)
    {
        int tid = thread.threadIdx.x;
        if (tid < N)
            c[tid] = a[tid] + b[tid];
    }
}
```

## ATTENTION
Cudafy.NET is created by [HYBRIDDSP](http://hybriddsp.com/products/cudafynet/) under LGPL v2.1 License.
I only used sources on the Internet and searched the files to adapt them to the latest version of CUDA 10.2 and .NET Framework 4.8

I am not the creator of this library, but only a fan who wants to help in using CUDA in newer versions.

### Copyright
The LGPL v2.1 License applies to CUDAfy .NET. If you wish to modify the code then changes should be re-submitted to Hybrid DSP. If you wish to incorporate Cudafy.NET into your own application instead of redistributing the dll's then please consider a commerical license. Visit http://www.hybriddsp.com. This will also provide you with priority support and contribute to on-going development.

The following libraries are made use of:
The MIT license applies to ILSpy, NRefactory and ICSharpCode.Decompiler (Copyright (c) 2011 AlphaSierraPapa for the SharpDevelop team).
Mono.Cecil also uses the MIT license (Copyright JB Evain).
CUDA.NET is a free for use license (Copyright Company for Advanced Supercomputing Solutions Ltd)
