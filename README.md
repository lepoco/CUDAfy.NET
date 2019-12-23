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
4. Choose the "Any CPU" solution platforms
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

## ATTENTION
Cudafy.NET is created by [HYBRIDDSP](http://hybriddsp.com/products/cudafynet/) under LGPL v2.1 License.
I only used sources on the Internet and searched the files to adapt them to the latest version of CUDA 10.1

I am not the creator of this library, but only a fan who wants to help in using CUDA in newer versions.

### Copyright
The LGPL v2.1 License applies to CUDAfy .NET. If you wish to modify the code then changes should be re-submitted to Hybrid DSP. If you wish to incorporate Cudafy.NET into your own application instead of redistributing the dll's then please consider a commerical license. Visit http://www.hybriddsp.com. This will also provide you with priority support and contribute to on-going development.

The following libraries are made use of:
The MIT license applies to ILSpy, NRefactory and ICSharpCode.Decompiler (Copyright (c) 2011 AlphaSierraPapa for the SharpDevelop team).
Mono.Cecil also uses the MIT license (Copyright JB Evain).
CUDA.NET is a free for use license (Copyright Company for Advanced Supercomputing Solutions Ltd)
