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
using System.Reflection;
using System.IO;
using System.Diagnostics;
using Cudafy;
using Cudafy.Compilers;
using Cudafy.Translator;
using Mono.Cecil;
namespace cudafycl
{
    class Program
    {
        static string cGUID = "63D6AC4F-CEC9-4E81-8DE7-7668EC9A3A0C";
        
        /// <summary>
        /// Usage: cudafycl.exe myassembly.dll [-arch=sm_11|sm_12|sm_13|sm_20|sm_21|sm_30|sm_35|sm_37|sm_50|sm_52]
        /// </summary>
        /// <param name="args"></param>
        static int Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("Usage: cudafycl.exe myassembly.dll [-arch=sm_11|sm_12|sm_13|sm_20|sm_21|sm_30|sm_35|sm_37|sm_50|sm_52] [-cdfy]");
                Console.WriteLine("\t-arch: CUDA architecture. Optional. Default is sm_20.");
                Console.WriteLine("\t-cdfy: cudafy the assembly and create the *.cdfy output file where * is assembly name. Optional.");
                return -1;
            }
            try
            {
                if (!args.Contains(cGUID))
                {
                    Process process = new Process();
                    process.StartInfo.UseShellExecute = false;
                    process.StartInfo.RedirectStandardOutput = true;
                    process.StartInfo.RedirectStandardError = true;
                    process.StartInfo.FileName = "cudafycl.exe";
                    StringBuilder sb = new StringBuilder();
                    foreach (var arg in args)
                        sb.AppendFormat("{0} ", arg);
                    sb.Append(cGUID);
                    process.StartInfo.Arguments = sb.ToString();
                    process.Start();
                    while (!process.HasExited)
                        System.Threading.Thread.Sleep(10);
                    if (process.ExitCode != 0)
                    {
                        string s = process.StandardError.ReadToEnd() + "\r\n";
                        s += process.StandardOutput.ReadToEnd();
                        throw new CudafyCompileException(CudafyCompileException.csCOMPILATION_ERROR_X, s);
                    }
                    else if(!args.Contains("-cdfy"))
                        EmbedInAssembly(args[0]);
                }
                else
                {
                    var arch = args.Where(a => a.StartsWith("-arch")).Select(a => 
                    {
                        string[] parts = a.Split('=');
                        eArchitecture ar = eArchitecture.sm_20;
                        if (parts.Length > 1)
                        {                            
                            bool pass = Enum.TryParse<eArchitecture>(parts[1], out ar);
                            return pass ? ar : eArchitecture.sm_20;
                        }
                        else
                            return ar;
                    }
                    ).FirstOrDefault();
                    
                    Console.WriteLine(string.Format(@"CreateCudafyModule(""{0}"", {1});", args[0], arch));
                    CreateCudafyModule(args[0], arch);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: {0}", ex.ToString());
                return -1;
            }
            return 0;
        }

        private static void CreateCudafyModule(string dllName, eArchitecture arch)
        {
            var assembly = Assembly.LoadFrom(dllName);//Assembly.LoadFrom(dllName);
            var types = assembly.GetTypes();
            var cm = CudafyTranslator.Cudafy(ePlatform.All, arch, types);
            var newFilename = Path.ChangeExtension(dllName, "cdfy");
            cm.Serialize(newFilename);
        }

        private static void EmbedInAssembly(string dllName)
        {
            var readerParameters = new ReaderParameters { ReadSymbols = true };
            var assemblyDefinition = AssemblyDefinition.ReadAssembly(dllName, readerParameters);
            var newFilename = Path.ChangeExtension(dllName, "cdfy");
            var resourceName = Path.GetFileName(newFilename);
            var filestream = new FileStream(newFilename, FileMode.Open);

            var existingResource = assemblyDefinition.MainModule.Resources.Where(res => res.Name == resourceName).FirstOrDefault();
            if (existingResource != null)
                assemblyDefinition.MainModule.Resources.Remove(existingResource);

            EmbeddedResource erTemp = new EmbeddedResource(resourceName, ManifestResourceAttributes.Public, filestream);
            assemblyDefinition.MainModule.Resources.Add(erTemp);
            WriterParameters wp = new WriterParameters()
            {
                WriteSymbols = true
            };
            assemblyDefinition.Write(dllName, wp);
        }
    }
}
