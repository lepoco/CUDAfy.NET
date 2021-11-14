Readme file for CUDA.NET
------------------------
http://www.hoopoe-cloud.com/Solutions/cuda.net

CUDA.NET aims to provide access to CUDA functionality from .NET based applications
on windows and linux systems.

CUDA is a GPU computing library provided by NVIDIA Corp.
For more information you may take a look at: http://www.nvidia.com/object/cuda

Releases
--------
Release 3.0.0:
	* Supporting CUDA 3.0 API.
	* Added CUDAContextSynchronizer for multi-threaded applications.
	* Improved API effeciency for generic operators (memory copies etc.)
	* Extended CUDA class API to support all driver functions (memset).

Release 2.3.7:
	* Using SizeT in runtime types.

Release 2.3.6:
	* Added SizeT structure for better 32/64 bit support.
	* Extended Driver and Runtime API to work with this object.

Release 2.3.5:
	* Fixed support for 32 and 64 bit platforms with runtime API.

Release 2.3.4:
	* Removed all SysUInt marshaling types from native functions.

Release 2.3.3:
	* Minor additions of copy functions to CUDA class.

Release 2.3.2:
	* Changed DirectX API functions so that parameters are passed as ref and not out.
	* Changed OpenGL API functions so that parameters are passed as ref and not out.

Release 2.3.1:
	* Renamed CUFFT plans to reflect real CUDA names and support double precision.

Release 2.3:
	* Supporting CUDA 2.3 API.
	* Extended CUFFT function support (double precision).

Release 2.2.4:
	* API changes in CUDARuntime class, all "out" arguments are marked as "ref".

Release 2.2.3:
	* Fixed API for driver function cuModuleLoadDataEx to allow JIT compilation.
	* Added operators to CUdeviceptr structure to support offset manipulation.

Release 2.2.2:
	* Updated code for automatic array creation in CUDA class.

Release 2.2.1:
	* Fixed version information for the library.
	* Fixed old issue with device name retrieval to be in correct length.

Release 2.2:
	* Added support for CUDA 2.2 API.
	* CUDA class was extended.
	* Removed functions from CUFFT class.
	* Removed async copy functions from driver API.
	* Extended runtime API for copy operations.
	* Renamed flags enums to be consistent with CUDA API.
	* Added documentation.

Release 2.1:
	* Supporting new CUDA 2.1 API.
	* Added support for DirectX 10 API.
	* Added support for JIT compilation.
	* Updated data structures to reflect latest CUDA API.
	* Fix in CUDAExecution class.

Release 2.0.4:
	* Added suport for emulation modes of CUFFT and CUBLAS.
	* CUDA class now supports all function from CUDA driver (including texture and host memory functions).
	* Added new Engine namespace with CUDAExecution class to make robust execution of CUDA kernels.

Release 2.0.3:
	* Added support for CUDA runtime API through CUDARuntime class.
	* Added support for Direct3D and OpenGL interoperability routines for both driver and runtime APIs.
	* Fixed issues with copying vector arrays data from the device to the CPU.
	* Added support in CUDA class for: texture functions, 3D array creation and manipulation, 3D copy.
	* Fixed 2D unaligned copy function in CUDA class.

Release 2.0.2:
	* Fixed minor issues with CUFFT, CUBLAS classes and Mono.
	* Added support for ToString() in CUFFTException, CUBLASException.

Release 2.0.1:
	* Fixed minor issues with CUDA class and Mono.
	* Added support for ToString() in CUDAException.

Release 2.0:
	* Now supporting CUDA 2.0 interface.

Release 1.1:
	* Added object model for CUDA through CUDA, CUFFT and CUBLAS objects.

Release 1.1 beta 3:
	* CUDADriver, CUFFTDriver and CUBLASDriver API are generelized - now all .NET primitives and CUDA 
	  structures can be used with function calls.

Release 1.1 beta 2:
	* CUDADriver API is generelized - now all .NET primitives and CUDA 
	  structures can be used with function calls.
	* Added support for CUFFT and CUBLAS routines.
	* Added sample program to demonstrate loading of kernels and executing functions
	  on the device.

Release 1.1 beta:
	The attached library conforms to CUDA 1.1 API.
	Please note that prior to using the library that your system has a CUDA
	toolkit installed, version 1.1, and you have a CUDA compatible computing
	device (such as Tesla C870, GeForce 9600GT etc.)
	If using the library under Windows, verify that the driver nvcuda.dll is 
	located under system32 directory of your windows installation.
	If used under linux, verify that libcuda.so is located in one of $LD_LIBRARY_PATH
	directories (specifically, under the installation directory of CUDA toolkit.)

Release 1.1 alpha:
	The attached library conforms to CUDA 1.1 API.
	Please note that prior to using the library that your system has a CUDA
	toolkit installed, version 1.1, and you have a CUDA compatible computing
	device (such as Tesla C870, GeForce 9600GT etc.)
	
License Agreement, rights and privacy
-------------------------------------
The library is free for use. You may distribute it with your applications.
Please add the relevant details on a credit if possible.

Company details:
----------------
Company for Advanced Supercomputing Solutions Ltd
Bosmat 2a St., 60850, Shoham
Israel
support@hoopoe-cloud.com

All rights reserved (c) 2008