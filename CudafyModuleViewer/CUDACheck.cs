using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Cudafy;
using Cudafy.Compilers;
using Cudafy.Host;
using Cudafy.Maths.FFT;
using Cudafy.Maths.BLAS;
using Cudafy.Maths.RAND;
using Cudafy.Maths.SPARSE;
using Cudafy.Translator;
namespace CudafyModuleViewer
{
    public class CUDACheck
    {
        public static int DeviceCount { get; private set; }

        public static bool IsDriverInstalled { get; private set; }
        
        public static IEnumerable<string> EnumerateDevices(bool openCL = false)//out string info, out bool driverInstalled)
        {
            int count = 0;
            string message = string.Empty;
            string info = string.Empty;
            IsDriverInstalled = true;
            List<string> sb = new List<string>();
            List<GPGPUProperties> deviceProps = new List<GPGPUProperties>();
            bool failed = true;
            try
            {
                count = openCL ? OpenCLDevice.GetDeviceCount() : CudaGPU.GetDeviceCount();
                deviceProps = CudafyHost.GetDeviceProperties(openCL ? eGPUType.OpenCL : eGPUType.Cuda).ToList();
                failed = false;
            }
            catch (DllNotFoundException dnfe)
            {
                sb.Add("Suitable driver not installed. " + dnfe.Message);
                IsDriverInstalled = false;
            }
            catch (GASS.CUDA.CUDAException ex)
            {
                if (ex.Message == "ErrorNotInitialized")
                    sb.Add("Found 0 CUDA devices.");
                else
                    sb.Add("CUDAException: " + ex.Message);
            }
            catch (Exception ex)
            {
                sb.Add("Error: " + ex.Message);
            }

            if (failed)
            {
                foreach (var s in sb)
                {
                    yield return s;
                }
            }
            else
            {
                yield return string.Format("Found {0} devices.\r\n", count);
                foreach (var prop in deviceProps)
                {
                    yield return ("Name: " + prop.Name);
                    if(openCL)
                        yield return ("OpenCL Version: " + prop.Capability.ToString());
                    else
                        yield return ("Compute capability: " + prop.Capability.ToString());
                    if (!openCL && prop.Capability < new Version(1, 4))
                       yield return ("Note: This device will not support default calls to Cudafy(). Use overloads to give specific value.");
                    yield return (string.Empty);
                }
            }
        }

        public static IEnumerable<string> TestCUDASDK()
        {
            StringBuilder sb = new StringBuilder();

            NvccCompilerOptions nvcc = null;
            if (IntPtr.Size == 8)
                nvcc = NvccCompilerOptions.Createx64();
            else
                nvcc = NvccCompilerOptions.Createx86();
            yield return (string.Format("Platform={0}", nvcc.Platform));
            yield return ("Checking for CUDA SDK at " + nvcc.CompilerPath);
            if (!nvcc.TryTest())
                yield return ("Could not locate CUDA Include directory.");
            else
            {
                yield return (string.Format("CUDA SDK Version={0}", nvcc.Version));

                yield return ("Attempting to cudafy a kernel function.");
                var mod = CudafyTranslator.Cudafy(nvcc.Platform, eArchitecture.sm_11, nvcc.Version, false, typeof(CUDACheck));
                yield return ("Successfully translated to CUDA C.");

                yield return ("Attempting to compile CUDA C code.");
                string s = mod.Compile(eGPUCompiler.CudaNvcc, true);
                yield return ("Successfully compiled CUDA C into a module.");

                if (CudafyHost.GetDeviceCount(eGPUType.Cuda) > 0)
                {
                    yield return ("Attempting to instantiate CUDA device object (GPGPU).");
                    var gpu = CudafyHost.GetDevice(eGPUType.Cuda, 0);
                    yield return ("Successfully got CUDA device 0.");

                    yield return ("Attempting to load module.");
                    gpu.LoadModule(mod);
                    yield return ("Successfully loaded module.");

                    yield return ("Attempting to transfer data to GPU.");
                    int[] a = new int[1024];
                    int[] b = new int[1024];
                    int[] c = new int[1024];
                    Random rand = new Random();
                    for (int i = 0; i < 1024; i++)
                    {
                        a[i] = rand.Next(16384);
                        b[i] = rand.Next(16384);
                    }
                    int[] dev_a = gpu.CopyToDevice(a);
                    int[] dev_b = gpu.CopyToDevice(b);
                    int[] dev_c = gpu.Allocate(c);
                    yield return ("Successfully transferred data to GPU.");

                    yield return ("Attempting to launch function on GPU.");
                    gpu.Launch(1, 1024).TestKernelFunction(dev_a, dev_b, dev_c);
                    yield return ("Successfully launched function on GPU.");

                    yield return ("Attempting to transfer results back from GPU.");
                    gpu.CopyFromDevice(dev_c, c);
                    yield return ("Successfully transferred results from GPU.");

                    yield return ("Testing results.");
                    int errors = 0;
                    for (int i = 0; i < 1024; i++)
                    {
                        if (a[i] + b[i] != c[i])
                            errors++;
                    }
                    if (errors == 0)
                        yield return ("Successfully tested results.");
                    else
                        yield return ("Test failed - results not as expected.");

                    yield return ("Checking for math libraries (FFT, BLAS, SPARSE, RAND).");
                    var fft = GPGPUFFT.Create(gpu);
                    int version = fft.GetVersion();
                    if (version > 0)
                        yield return ("Successfully detected.");
                }
            }
        }

        public static IEnumerable<string> TestOpenCL()
        {
            yield return ("Attempting to cudafy a kernel function.");
            //CudafyTranslator.Language = eLanguage.OpenCL;
            var mod = CudafyTranslator.Cudafy(ePlatform.Auto, eArchitecture.OpenCL, null, false, typeof(CUDACheck));
            yield return ("Successfully translated to OpenCL C.");

            for (int id = 0; id < CudafyHost.GetDeviceCount(eGPUType.OpenCL); id++)
            {
                yield return ("Attempting to instantiate OpenCL device object (GPGPU).");
                var gpu = CudafyHost.GetDevice(eGPUType.OpenCL, id);
                yield return (string.Format("Successfully got OpenCL device {0}.", id));
                yield return ("Name: " + gpu.GetDeviceProperties(false).Name);
                yield return ("Attempting to load module.");
                gpu.LoadModule(mod);
                yield return ("Successfully loaded module.");

                yield return ("Attempting to transfer data to device.");
                int[] a = new int[1024];
                int[] b = new int[1024];
                int[] c = new int[1024];
                Random rand = new Random();
                for (int i = 0; i < 1024; i++)
                {
                    a[i] = rand.Next(16384);
                    b[i] = rand.Next(16384);
                }
                int[] dev_a = gpu.CopyToDevice(a);
                int[] dev_b = gpu.CopyToDevice(b);
                int[] dev_c = gpu.Allocate(c);
                yield return ("Successfully transferred data to device.");

                yield return ("Attempting to launch function on device.");
                gpu.Launch(4, 256).TestKernelFunction(dev_a, dev_b, dev_c);
                yield return ("Successfully launched function on device.");

                yield return ("Attempting to transfer results back from device.");
                gpu.CopyFromDevice(dev_c, c);
                yield return ("Successfully transferred results from device.");

                yield return ("Testing results.");
                int errors = 0;
                for (int i = 0; i < 1024; i++)
                {
                    if (a[i] + b[i] != c[i])
                        errors++;
                }
                if (errors == 0)
                    yield return ("Successfully tested results.\r\n\r\n");
                else
                    yield return ("Test failed - results not as expected.\r\n\r\n");
            }
        }

        [Cudafy]
        public static void TestKernelFunction(GThread thread, int[] a, int[] b, int[] c)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            c[i] = a[i] + b[i];
        }
    }
}
