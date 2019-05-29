/*
CUDAfy.NET - LGPL 2.1 License
Please consider purchasing a commerical license - it helps development, frees you from LGPL restrictions
and provides you with support.  Thank you!
Copyright (C) 2011 Hybrid DSP Systems
http://www.hybriddsp.com

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Diagnostics;
namespace Cudafy.Compilers
{
    /// <summary>
    /// Compiler options.
    /// </summary>
    public class NvccCompilerOptions : CompilerOptions
    {
        private const string csNVCC = "nvcc";
        
        /// <summary>
        /// Initializes a new instance of the <see cref="NvccCompilerOptions"/> class.
        /// </summary>
        /// <param name="name">The name.</param>
        public NvccCompilerOptions(string name)
            : base(name, csNVCC, string.Empty, null, IntPtr.Size == 8 ? ePlatform.x64 : ePlatform.x86)
        {

        } 

        /// <summary>
        /// Initializes a new instance of the <see cref="NvccCompilerOptions"/> class.
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="compiler">The compiler.</param>
        /// <param name="includeDirectory">The include directory.</param>
        /// <param name="compilerVersion">Compiler/toolkit version (e.g. CUDA V5.0).</param>
        public NvccCompilerOptions(string name, string compiler, string includeDirectory, Version compilerVersion, ePlatform platform)
            : base(name, compiler, includeDirectory, compilerVersion, platform)
        {

        }

        /// <summary>
        /// Gets the arguments.
        /// </summary>
        /// <returns></returns>
        public override string GetArguments()
        {
            string command = string.Empty;

            string includeDir = string.IsNullOrEmpty(Include) ? string.Empty : @" -I""" + Include + @"""";
            command += includeDir;
            foreach (string opt in Options)
                command += string.Format(" {0} ", opt);
//#if !Linux
            if (GenerateDebugInfo)
            {
                if(Version != null && Version.Major >= 5)
                    command += " -G ";
                else
                    command += " -G0 ";
            }
//#endif
            if (Sources.Count() == 0)
                throw new CudafyCompileException(CudafyCompileException.csNO_SOURCES);
            bool generateBinary = (CompileMode & eCudafyCompileMode.Binary) == eCudafyCompileMode.Binary;
            if (generateBinary)
                command += " -c ";
            foreach (string src in Sources)
                command += string.Format(@" ""{0}"" ", src);

            if (!generateBinary && Outputs.Count() == 1)
                command += string.Format(@" -o ""{0}"" ", Outputs.Take(1).FirstOrDefault());
            if (!generateBinary)
               command += " -ptx";
            return command;
        }


        private const string csGPUTOOLKIT = @"NVIDIA GPU Computing Toolkit\CUDA\";

        /// <summary>
        /// Creates a default x86 instance. Architecture is 2.0.
        /// </summary>
        /// <returns></returns>
        public static NvccCompilerOptions Create()
        {
            NvccCompilerOptions opt = Createx86(null, eArchitecture.sm_20);
            opt.CanEdit = true;
            return opt;
        }

        /// <summary>
        /// Creates a default x86 instance. Architecture is 2.0.
        /// </summary>
        /// <returns></returns>
        public static NvccCompilerOptions Createx86()
        {
            return Createx86(null, eArchitecture.sm_20);
        }

        private static void AddArchOptions(CompilerOptions co, eArchitecture arch)
        {
            //if (arch == eArchitecture.sm_11)
            //    co.AddOption("-arch=sm_11");
            //else if (arch == eArchitecture.sm_12)
            //    co.AddOption("-arch=sm_12");
            //else if (arch == eArchitecture.sm_13)
            //    co.AddOption("-arch=sm_13");
            //else if (arch == eArchitecture.sm_20)
            //    co.AddOption("-arch=sm_20");
            //else if (arch == eArchitecture.sm_21)
            //    co.AddOption("-arch=sm_21");
            //else if (arch == eArchitecture.sm_30)
            //    co.AddOption("-arch=sm_30");
            //else if (arch == eArchitecture.sm_35)
            //    co.AddOption("-arch=sm_35");
            //else
            //    throw new NotImplementedException(arch.ToString());
            co.AddOption("-arch=" + arch.ToString());
            co.Architecture = arch;
            
        }

        /// <summary>
        /// Creates a default x86 instance for specified architecture.
        /// </summary>
        /// <param name="arch">The architecture.</param>
        /// <returns></returns>
        public static NvccCompilerOptions Createx86(eArchitecture arch)
        {
            return Createx86(null, arch);
        }

        /// <summary>
        /// Creates a compiler instance for creating 32-bit apps.
        /// </summary>
        /// <param name="cudaVersion">The cuda version.</param>
        /// <param name="arch">Architecture.</param>
        /// <returns></returns>
        public static NvccCompilerOptions Createx86(Version cudaVersion, eArchitecture arch)
        {
            string progFiles = Utility.ProgramFiles();
            string toolkitbasedir = progFiles + Path.DirectorySeparatorChar + csGPUTOOLKIT;
            Version selVer;
            string cvStr = GetCudaVersion(cudaVersion, toolkitbasedir, out selVer);
            if (string.IsNullOrEmpty(cvStr))
            {
                progFiles = "C:\\Program Files";
                toolkitbasedir = progFiles + Path.DirectorySeparatorChar + csGPUTOOLKIT;
                cvStr = GetCudaVersion(cudaVersion, toolkitbasedir);
            }


            Debug.WriteLineIf(!string.IsNullOrEmpty(cvStr), "Compiler version: " + cvStr);
            string gpuToolKit = progFiles + Path.DirectorySeparatorChar + csGPUTOOLKIT + cvStr;
            string compiler = gpuToolKit + Path.DirectorySeparatorChar + @"bin" + Path.DirectorySeparatorChar + csNVCC;
            string includeDir = gpuToolKit + Path.DirectorySeparatorChar + @"include";
            NvccCompilerOptions opt = new NvccCompilerOptions("NVidia CC (x86)", compiler, includeDir, selVer, ePlatform.x86);
            if (!opt.TryTest())
            {
                opt = new NvccCompilerOptions("NVidia CC (x86)", csNVCC, string.Empty, selVer, ePlatform.x86);
//#if DEBUG
//                throw new CudafyCompileException("Test failed for NvccCompilerOptions for x86");
//#endif
            }
            opt.AddOption("-m32");
            opt.Platform = ePlatform.x86;
            AddArchOptions(opt, arch);
            return opt;
        }

        /// <summary>
        /// Creates a default x64 instance. Architecture is 2.0.
        /// </summary>
        /// <returns></returns>
        public static NvccCompilerOptions Createx64()
        {
            return Createx64(null, eArchitecture.sm_20);
        }

        /// <summary>
        /// Creates a default x64 instance for specified architecture.
        /// </summary>
        /// <param name="arch">The architecture.</param>
        /// <returns></returns>
        public static NvccCompilerOptions Createx64(eArchitecture arch)
        {
            return Createx64(null, arch);
        }

        /// <summary>
        /// Creates a compiler instance for creating 64-bit apps.
        /// </summary>
        /// <param name="cudaVersion">The cuda version or null for auto.</param>
        /// <param name="arch">Architecture.</param>
        /// <returns></returns>
        /// <exception cref="NotSupportedException">ProgramFilesx64 not found.</exception>
        public static NvccCompilerOptions Createx64(Version cudaVersion, eArchitecture arch)
        {
            string progFiles = Utility.ProgramFiles();
            string toolkitbasedir = progFiles + Path.DirectorySeparatorChar + csGPUTOOLKIT;
            Version selVer;
            string cvStr = GetCudaVersion(cudaVersion, toolkitbasedir, out selVer);
            Debug.WriteLineIf(!string.IsNullOrEmpty(cvStr), "Compiler version: " + cvStr);
            string gpuToolKit = progFiles + Path.DirectorySeparatorChar + csGPUTOOLKIT + cvStr;// cudaVersion;
            string compiler = gpuToolKit + Path.DirectorySeparatorChar + @"bin" + Path.DirectorySeparatorChar + csNVCC;
            string includeDir = gpuToolKit + Path.DirectorySeparatorChar + @"include";
            NvccCompilerOptions opt = new NvccCompilerOptions("NVidia CC (x64)", compiler, includeDir, selVer, ePlatform.x64);
            if (!opt.TryTest())
            {
                opt = new NvccCompilerOptions("NVidia CC (x64)", csNVCC, string.Empty, selVer, ePlatform.x64);
//#if DEBUG
//                throw new CudafyCompileException("Test failed for NvccCompilerOptions for x64");
//#endif
            }
            opt.AddOption("-m64");
            //opt.AddOption("-DCUDA_FORCE_API_VERSION=3010"); //For mixed bitness mode
            //if(Directory.Exists(@"C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\include"))
            //    opt.AddOption(@"-I""C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\include""");
            //else
            //    opt.AddOption(@"-I""C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\include""");
            opt.Platform = ePlatform.x64;
            AddArchOptions(opt, arch);
            return opt;
        }

        private static string GetCudaVersion(Version cudaVersion, string gpuToolKitDir)
        {
            Version v;
            return GetCudaVersion(cudaVersion, gpuToolKitDir, out v);
        }

        private static string GetCudaVersion(Version cudaVersion, string gpuToolKitDir, out Version selectedVersion)
        {
            string s = "v{0}.{1}";
            selectedVersion = cudaVersion;
            if (cudaVersion != null)
            {
                string version = string.Format(s, cudaVersion.Major, cudaVersion.Minor);
                string dir = gpuToolKitDir + version;
                if (System.IO.Directory.Exists(dir))
                    return version;
                else
                    return string.Empty;
            }
            for (int j = 9; j >= 4; j--)
                for (int i = 9; i >= 0; i--)
                {
                    string version = string.Format(s, j, i);
                    string dir = gpuToolKitDir + version;
                    if (System.IO.Directory.Exists(dir))
                    {
                        selectedVersion = new Version(string.Format("{0}.{1}", j, i));
                        return version;
                    }
                }
            return string.Empty;
        }
    }
}
