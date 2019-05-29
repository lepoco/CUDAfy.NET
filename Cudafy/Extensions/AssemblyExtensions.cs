using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.Diagnostics;
namespace Cudafy
{
    /// <summary>
    /// Extensions to the Assembly class for handling related Cudafy Modules
    /// </summary>
    public static class AssemblyExtensions
    {
        /// <summary>
        /// Determines whether the assembly has a cudafy module embedded.
        /// </summary>
        /// <param name="assembly">The assembly.</param>
        /// <returns>
        ///   <c>true</c> if it has cudafy module; otherwise, <c>false</c>.
        /// </returns>
        public static bool HasCudafyModule(this Assembly assembly)
        {
            return CudafyModule.HasCudafyModule(assembly);
        }

        /// <summary>
        /// Gets the embedded cudafy module from the assembly.
        /// </summary>
        /// <param name="assembly">The assembly.</param>
        /// <returns>Cudafy module.</returns>
        public static CudafyModule GetCudafyModule(this Assembly assembly)
        {
            return CudafyModule.GetFromAssembly(assembly);
        }

        /// <summary>
        /// Cudafies the assembly producing a *.cdfy file with same name as assembly. Architecture is 2.0.
        /// </summary>
        /// <param name="assembly">The assembly.</param>
        /// <param name="arch">The architecture.</param>
        /// <returns>Output messages of the cudafycl.exe process.</returns>
        public static string Cudafy(this Assembly assembly, eArchitecture arch = eArchitecture.sm_20)
        {
            string messages;
            if(!TryCudafy(assembly, out messages, arch))
                throw new CudafyCompileException(CudafyCompileException.csCOMPILATION_ERROR_X, messages);
            return messages;
        }

        /// <summary>
        /// Tries cudafying the assembly producing a *.cdfy file with same name as assembly. Architecture is 2.0.
        /// </summary>
        /// <param name="assembly">The assembly.</param>
        /// <param name="arch">The architecture.</param>
        /// <returns>
        ///   <c>true</c> if successful; otherwise, <c>false</c>.
        /// </returns>
        public static bool TryCudafy(this Assembly assembly, eArchitecture arch = eArchitecture.sm_20)
        {
            string messages;
            return TryCudafy(assembly, out messages, arch);

        }
        /// <summary>
        /// Tries cudafying the assembly producing a *.cdfy file with same name as assembly. Architecture is 2.0.
        /// </summary>
        /// <param name="assembly">The assembly.</param>
        /// <param name="messages">Output messages of the cudafycl.exe process.</param>
        /// <param name="arch">The architecture.</param>
        /// <returns>
        ///   <c>true</c> if successful; otherwise, <c>false</c>.
        /// </returns>
        public static bool TryCudafy(this Assembly assembly, out string messages, eArchitecture arch = eArchitecture.sm_20)
        {
            var assemblyName = assembly.Location;
            Process process = new Process();
            process.StartInfo.UseShellExecute = false;
            process.StartInfo.RedirectStandardOutput = true;
            process.StartInfo.RedirectStandardError = true;
            process.StartInfo.FileName = "cudafycl.exe";
            StringBuilder sb = new StringBuilder();
            process.StartInfo.Arguments = string.Format("{0} -arch={1} -cdfy", assemblyName, arch);
            process.Start();
            while (!process.HasExited)
                System.Threading.Thread.Sleep(10);
            if (process.ExitCode != 0)
            {
                messages = process.StandardError.ReadToEnd() + "\r\n";
                messages += process.StandardOutput.ReadToEnd();
                return false;
            }
            else
            {
                messages = process.StandardOutput.ReadToEnd();
                return true;
            }
        }
    }
}
