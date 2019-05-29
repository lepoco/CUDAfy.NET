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
using System.IO;
using System.Text;
using System.Reflection;
using System.Diagnostics;
using ICSharpCode.Decompiler;
using ICSharpCode.ILSpy;
using Cudafy;
using Cudafy.Compilers;
using Mono.Cecil;
namespace Cudafy.Translator
{


    /// <summary>
    /// Implements translation of .NET code to CUDA C.
    /// </summary>
    public class CudafyTranslator
    {
        static CudafyTranslator()
        {
            TimeOut = 60000;
            AllowClasses = false;
        }
        
        private static CUDALanguage _cl = new CUDALanguage(eLanguage.Cuda);

        internal static CUDAfyLanguageSpecifics LanguageSpecifics = new CUDAfyLanguageSpecifics();

        private static IEnumerable<Type> GetNestedTypes(Type type)
        {
            foreach (var nestedType in type.GetNestedTypes(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic))
            {
                if (nestedType.GetNestedTypes().Count() > 0)
                {
                    foreach (var nestedNestedType in GetNestedTypes(nestedType))
                        yield return nestedNestedType;
                }
                else
                {
                    //if(nestedType.IsClass)
                        yield return nestedType;
                }
            }
        }

        private static IEnumerable<Type> GetWithNestedTypes(Type[] types)
        {
            List<Type> typesList = new List<Type>();
            foreach (Type type in types.Distinct())
            {
                if (type == null)
                    continue;
                foreach (Type nestedType in GetNestedTypes(type))
                    typesList.Add(nestedType);
                typesList.Add(type);
            }
            return typesList.Distinct();
        }

        /// <summary>
        /// Gets or sets the language to generate.
        /// </summary>
        /// <value>
        /// The language.
        /// </value>
        public static eLanguage Language
        {
            get { return LanguageSpecifics.Language; }
            set 
            { 
                if (value != LanguageSpecifics.Language) 
                    _cl = new CUDALanguage(value); 
                LanguageSpecifics.Language = value; 
            }
        }

        /// <summary>
        /// Gets or sets the working directory for the compiler. The compiler must write temporary files to disk. This
        /// can be an issue if the application does not have write access of your application directly.
        /// </summary>
        public static string WorkingDirectory { get; set; }

        /// <summary>
        /// Gets or sets the time out for compilation.
        /// </summary>
        /// <value>
        /// The time out in milliseconds.
        /// </value>
        public static int TimeOut { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to compile for debug.
        /// </summary>
        /// <value>
        ///   <c>true</c> if compile for debug; otherwise, <c>false</c>.
        /// </value>
        public static bool GenerateDebug { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to allow classes to be Cudafyed. 
        /// Note that DeviceClassHelper utility can be used to move class objects from Host to Device.
        /// </summary>
        /// <value>
        ///   <c>true</c> if classes are permitted; otherwise, <c>false</c>.
        /// </value>
        public static bool AllowClasses { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to delete any temporary files.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if delete temporary files; otherwise, <c>false</c>.
        /// </value>
        public static bool DeleteTempFiles { get; set; }

        /// <summary>
        /// Tries to use a previous serialized CudafyModule else cudafies and compiles the type in which the calling method is located. 
        /// CUDA architecture is 2.0; platform is set to the current application's (x86 or x64); and the CUDA version is the 
        /// latest official release found on the current machine. 
        /// </summary>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule Cudafy()
        {
            StackTrace stackTrace = new StackTrace();
            Type type = stackTrace.GetFrame(1).GetMethod().ReflectedType;
            CudafyModule km = CudafyModule.TryDeserialize(type.Name);
            if (km == null || !km.TryVerifyChecksums())
            {
                km = Cudafy(ePlatform.Auto, eArchitecture.Unknown, type);
                km.Name = type.Name;
                km.TrySerialize();
            }
            return km;
        }

        /// <summary>
        /// Tries to use a previous serialized CudafyModule else cudafies and compiles the type in which the calling method is located. 
        /// CUDA architecture is as specified; platform is set to the current application's (x86 or x64); and the CUDA version is the 
        /// latest official release found on the current machine. 
        /// </summary>
        /// <param name="arch">The CUDA or OpenCL architecture.</param>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule Cudafy(eArchitecture arch)
        {
            StackTrace stackTrace = new StackTrace();
            Type type = stackTrace.GetFrame(1).GetMethod().ReflectedType;
            CudafyModule km = CudafyModule.TryDeserialize(type.Name);
            if (km == null || !km.TryVerifyChecksums(ePlatform.Auto, arch))
            {
                km = Cudafy(ePlatform.Auto, arch, type);
                km.Name = type.Name;
                km.TrySerialize();
            }
            return km;
        }

        /// <summary>
        /// Tries to use a previous serialized CudafyModule else cudafies and compiles the type in which the calling method is located. 
        /// CUDA architecture is 2.0; platform is as specified; and the CUDA version is the 
        /// latest official release found on the current machine. 
        /// </summary>
        /// <param name="platform">The platform.</param>
        /// <returns></returns>
        public static CudafyModule Cudafy(ePlatform platform)
        {
            StackTrace stackTrace = new StackTrace();
            Type type = stackTrace.GetFrame(1).GetMethod().ReflectedType;
            CudafyModule km = CudafyModule.TryDeserialize(type.Name);
            if (km == null || !km.TryVerifyChecksums() || !km.HasPTXForPlatform(platform))
            {
                km = Cudafy(platform, eArchitecture.Unknown, type);
                km.Name = type.Name;
                km.TrySerialize();
            }
            return km;
        }

        /// <summary>
        /// Cudafies for the specified platform.
        /// </summary>
        /// <param name="platform">The platform.</param>
        /// <param name="arch">The CUDA or OpenCL architecture.</param>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule Cudafy(ePlatform platform, eArchitecture arch)
        {
            StackTrace stackTrace = new StackTrace();
            Type type = stackTrace.GetFrame(1).GetMethod().ReflectedType;
            CudafyModule km = CudafyModule.TryDeserialize(type.Name);
            if (km == null || !km.TryVerifyChecksums())
            {
                km = Cudafy(platform, arch, type);
                km.Name = type.Name;
                km.TrySerialize();
            }
            return km;
        }

        /// <summary>
        /// Cudafies and compiles the type of the specified object with default settings. 
        /// CUDA architecture is 2.0; platform is set to the current application's (x86 or x64); and the CUDA version is the 
        /// latest official release found on the current machine. 
        /// </summary>
        /// <param name="o">An instance of the type to cudafy. Typically pass 'this'.</param>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule Cudafy(object o)
        {
            Type currentType = o.GetType();
            return Cudafy(currentType);
        }

        /// <summary>
        /// Cudafies and compiles the specified types with default settings. 
        /// CUDA architecture is 2.0; platform is set to the current application's (x86 or x64); and the CUDA version is the 
        /// latest official release found on the current machine. 
        /// </summary>
        /// <param name="types">The types.</param>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule Cudafy(params Type[] types)
        {
            return Cudafy(ePlatform.Auto, eArchitecture.Unknown, null, true, types);
        }

        /// <summary>
        /// Cudafies the specified types for the specified platform.
        /// </summary>
        /// <param name="platform">The platform.</param>
        /// <param name="arch">The CUDA or OpenCL architecture.</param>
        /// <param name="types">The types.</param>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule Cudafy(ePlatform platform, eArchitecture arch, params Type[] types)
        {
            return Cudafy(platform, arch, null, true, types);
        }

        /// <summary>
        /// Cudafies the specified types for the specified architecture on automatic platform.
        /// </summary>
        /// <param name="arch">The CUDA or OpenCL architecture.</param>
        /// <param name="types">The types.</param>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule Cudafy(eArchitecture arch, params Type[] types)
        {
            return Cudafy(ePlatform.Auto, arch, null, true, types);
        }


        /// <summary>
        /// Translates the specified types for the specified architecture without compiling. You can later call Compile method on the CudafyModule.
        /// </summary>
        /// <param name="arch">The CUDA or OpenCL architecture.</param>
        /// <param name="types">The types.</param>
        /// <returns></returns>
        public static CudafyModule Translate(eArchitecture arch, params Type[] types)
        {
            return Cudafy(ePlatform.Auto, arch, null, false, types);
        }

        /// <summary>
        /// Cudafies the specified types. Working directory will be as per CudafyTranslator.WorkingDirectory.
        /// </summary>
        /// <param name="platform">The platform.</param>
        /// <param name="arch">The CUDA or OpenCL architecture.</param>
        /// <param name="cudaVersion">The CUDA version. Specify null to automatically use the highest installed version.</param>
        /// <param name="compile">if set to <c>true</c> compile to PTX.</param>
        /// <param name="types">The types.</param>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule CudafyOld(ePlatform platform, eArchitecture arch, Version cudaVersion, bool compile, params Type[] types)
        {
            CudafyModule km = null;
            CUDALanguage.ComputeCapability = GetComputeCapability(arch);
            _architecture = arch;
            if (arch > eArchitecture.OpenCL)
                CudafyTranslator.Language = eLanguage.OpenCL;
            km = DoCudafy(null, types);
            if (km == null)
                throw new CudafyFatalException(CudafyFatalException.csUNEXPECTED_STATE_X, "CudafyModule km = null");
            km.WorkingDirectory = WorkingDirectory;
            if (compile && LanguageSpecifics.Language == eLanguage.Cuda)
            {
                if (platform == ePlatform.Auto)
                    platform = IntPtr.Size == 8 ? ePlatform.x64 : ePlatform.x86;
                if (platform != ePlatform.x86)
                    km.CompilerOptionsList.Add(NvccCompilerOptions.Createx64(cudaVersion, arch));
                if (platform != ePlatform.x64)
                    km.CompilerOptionsList.Add(NvccCompilerOptions.Createx86(cudaVersion, arch));
                km.GenerateDebug = GenerateDebug;
                km.TimeOut = TimeOut;
                km.Compile(eGPUCompiler.CudaNvcc, DeleteTempFiles);
            }
            Type lastType = types.Last(t => t != null);
            if(lastType != null)
                km.Name = lastType.Name;
            return km;
        }

        /// <summary>
        /// Cudafies the specified types. Working directory will be as per CudafyTranslator.WorkingDirectory.
        /// </summary>
        /// <param name="platform">The platform.</param>
        /// <param name="arch">The CUDA or OpenCL architecture.</param>
        /// <param name="cudaVersion">The CUDA version. Specify null to automatically use the highest installed version.</param>
        /// <param name="compile">if set to <c>true</c> compile to PTX.</param>
        /// <param name="types">The types.</param>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule Cudafy(ePlatform platform, eArchitecture arch, Version cudaVersion, bool compile, params Type[] types)
        {
            var cp = CompilerHelper.Create(ePlatform.Auto, arch, eCudafyCompileMode.Default, WorkingDirectory, GenerateDebug);
            if (!compile)
                cp.CompileMode = eCudafyCompileMode.TranslateOnly;
            return Cudafy(cp, types);
        }

        public static CudafyModule Cudafy(CompileProperties prop, params Type[] types)
        {
            var list = new List<CompileProperties>();
            list.Add(prop);
            return Cudafy(list, types);
        }

        /// <summary>
        /// Translates and compiles the given types against specified compilation properties.
        /// </summary>
        /// <param name="props">The settings.</param>
        /// <param name="types">Types to search and translate.</param>
        /// <returns></returns>
        public static CudafyModule Cudafy(IEnumerable<CompileProperties> props, params Type[] types)
        {
            CudafyModule km = null;
            //var uniqueLanguages = new List<eLanguage>();
            //if (props.Any(p => p.Language == eLanguage.Cuda))
            //    uniqueLanguages.Add(eLanguage.Cuda);
            //if (props.Any(p => p.Language == eLanguage.OpenCL))
            //    uniqueLanguages.Add(eLanguage.OpenCL);
            //foreach (var lang in uniqueLanguages)
            foreach(var p in props)
            {
                CudafyTranslator.Language = p.Language;
                _architecture = p.Architecture;
                CUDALanguage.ComputeCapability = GetComputeCapability(p.Architecture);
                
                km = DoCudafy(km, types);
                if (km == null)
                    throw new CudafyFatalException(CudafyFatalException.csUNEXPECTED_STATE_X, "CudafyModule km = null");
            }
            km.WorkingDirectory = WorkingDirectory;
            km.Compile(props.ToArray());
            
            Type lastType = types.Last(t => t != null);
            if (lastType != null)
                km.Name = lastType.Name;
            return km;
        }

        private static Version GetComputeCapability(eArchitecture arch, params Type[] types)
        {
            if (arch == eArchitecture.Emulator)
                return new Version(0, 1);
            else if (arch == eArchitecture.sm_10)
                return new Version(1, 0);
            else if (arch == eArchitecture.sm_11)
                return new Version(1, 1);
            else if (arch == eArchitecture.sm_12)
                return new Version(1, 2);
            else if (arch == eArchitecture.sm_13)
                return new Version(1, 3);
            else if (arch == eArchitecture.sm_20)
                return new Version(2, 0);
            else if (arch == eArchitecture.sm_21)
                return new Version(2, 1);
            else if (arch == eArchitecture.sm_30)
                return new Version(3, 0);
            else if (arch == eArchitecture.sm_35)
                return new Version(3, 5);
            else if (arch == eArchitecture.sm_37)
                return new Version(3, 7);
            else if (arch == eArchitecture.sm_50)
                return new Version(5, 0);
            else if (arch == eArchitecture.sm_52)
                return new Version(5, 2);
            else if (arch == eArchitecture.OpenCL)
                return new Version(1, 0);
            else if (arch == eArchitecture.OpenCL11)
                return new Version(1, 1);
            else if (arch == eArchitecture.OpenCL12)
                return new Version(1, 2);
            else if (arch == eArchitecture.Unknown && Language == eLanguage.OpenCL)
                return new Version(1, 0);
            else if (arch == eArchitecture.Unknown && Language == eLanguage.Cuda)
                return new Version(1, 3);
            throw new ArgumentException("Unknown architecture.");
        }

        private static eArchitecture _architecture; 
        
        private static CudafyModule DoCudafy(CudafyModule cm, params Type[] types)
        {
            MemoryStream output = new MemoryStream();
            var outputSw = new StreamWriter(output);
            
            MemoryStream structs = new MemoryStream();
            var structsSw = new StreamWriter(structs);
            var structsPto = new PlainTextOutput(structsSw);
            
            MemoryStream declarations = new MemoryStream();
            var declarationsSw = new StreamWriter(declarations);
            var declarationsPto = new PlainTextOutput(declarationsSw);

            MemoryStream code = new MemoryStream();
            var codeSw = new StreamWriter(code);
            var codePto = new PlainTextOutput(codeSw);

            bool isDummy = false;
            eCudafyDummyBehaviour behaviour = eCudafyDummyBehaviour.Default;

            Dictionary<string, ModuleDefinition> modules = new Dictionary<string,ModuleDefinition>();

            var compOpts = new DecompilationOptions { FullDecompilation = true };

            CUDALanguage.Reset();
            bool firstPass = true;
            if(cm == null)
                cm = new CudafyModule();// #######!!!
            else
                firstPass = false;
            
            // Test structs
            //foreach (var strct in types.Where(t => !t.IsClass))
            //    if (strct.GetCustomAttributes(typeof(CudafyAttribute), false).Length == 0)
            //        throw new CudafyLanguageException(CudafyLanguageException.csCUDAFY_ATTRIBUTE_IS_MISSING_ON_X, strct.Name);

            IEnumerable<Type> typeList = GetWithNestedTypes(types);
            foreach (var type in typeList)
            {
                if(!modules.ContainsKey(type.Assembly.Location))
                    modules.Add(type.Assembly.Location, ModuleDefinition.ReadModule(type.Assembly.Location));                
            }
            
            // Additional loop to compile in order
            foreach (var requestedType in typeList)
            {
                foreach (var kvp in modules)
                {
                    foreach (var td in kvp.Value.Types)
                    {
                        List<TypeDefinition> tdList = new List<TypeDefinition>();
                        tdList.Add(td);
                        tdList.AddRange(td.NestedTypes);

                        Type type = null;
                        foreach (var t in tdList)
                        {
                            //type = typeList.Where(tt => tt.FullName.Replace("+", "") == t.FullName.Replace("/", "")).FirstOrDefault();
                            // Only select type if this matches the requested type (to ensure order is maintained).
                            type = requestedType.FullName.Replace("+", "") == t.FullName.Replace("/", "") ? requestedType : null;

                            if (type == null)
                                continue;
                            Debug.WriteLine(t.FullName);
                            // Types                      
                            var attr = t.GetCudafyType(out isDummy, out behaviour);
                            if (attr != null)
                            {
                                _cl.DecompileType(t, structsPto, compOpts);
                                if (firstPass)
                                    cm.Types.Add(type.FullName.Replace("+", ""), new KernelTypeInfo(type, isDummy, behaviour));// #######!!!
                            }
                            else if (t.Name == td.Name)
                            {
                                // Fields
                                foreach (var fi in td.Fields)
                                {
                                    attr = fi.GetCudafyType(out isDummy, out behaviour);
                                    if (attr != null)
                                    {
                                        VerifyMemberName(fi.Name);
                                        System.Reflection.FieldInfo fieldInfo = type.GetField(fi.Name, BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
                                        if (fieldInfo == null)
                                            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Non-static fields");
                                        int[] dims = _cl.GetFieldInfoDimensions(fieldInfo);
                                        _cl.DecompileCUDAConstantField(fi, dims, codePto, compOpts);
                                        var kci = new KernelConstantInfo(fi.Name, fieldInfo, isDummy);
                                        if (firstPass)
                                            cm.Constants.Add(fi.Name, kci);// #######!!!
                                        CUDALanguage.AddConstant(kci);
                                    }
                                }
#warning TODO Only Global Methods can be called from host
#warning TODO For OpenCL may need to do Methods once all Constants have been handled
                                // Methods
                                foreach (var med in td.Methods)
                                {
                                    attr = med.GetCudafyType(out isDummy, out behaviour);
                                    if (attr != null)
                                    {
                                        if (!med.IsStatic)
                                            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Non-static methods");
                                        _cl.DecompileMethodDeclaration(med, declarationsPto, new DecompilationOptions { FullDecompilation = false });
                                        _cl.DecompileMethod(med, codePto, compOpts);
                                        MethodInfo mi = type.GetMethod(med.Name, BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
                                        if (mi == null)
                                            continue;
                                        VerifyMemberName(med.Name);
                                        eKernelMethodType kmt = eKernelMethodType.Device;
                                        kmt = GetKernelMethodType(attr, mi);
                                        if (firstPass)
                                            cm.Functions.Add(med.Name, new KernelMethodInfo(type, mi, kmt, isDummy, behaviour, cm));// #######!!!
                                    }
                                }
                            }
                        }
                    }
                }
            }

            codeSw.Flush();

            if (CudafyTranslator.Language == eLanguage.OpenCL)
            {
                outputSw.WriteLine("#if defined(cl_khr_fp64)");
                outputSw.WriteLine("#pragma OPENCL EXTENSION cl_khr_fp64: enable");
                outputSw.WriteLine("#elif defined(cl_amd_fp64)");
                outputSw.WriteLine("#pragma OPENCL EXTENSION cl_amd_fp64: enable");
                outputSw.WriteLine("#endif");
            }

            foreach (var oh in CUDALanguage.OptionalHeaders)
            {
                if (oh.Used && !oh.AsResource)
                    outputSw.WriteLine(oh.IncludeLine);
                else if (oh.Used)
                    outputSw.WriteLine(GetResourceString(oh.IncludeLine));
            }
            foreach (var oh in CUDALanguage.OptionalFunctions)
            {
                if (oh.Used)
                    outputSw.WriteLine(oh.Code);
            }

            declarationsSw.WriteLine();
            declarationsSw.Flush();

            structsSw.WriteLine();
            structsSw.Flush();

            foreach (var def in cm.GetDummyDefines())
                outputSw.WriteLine(def);
            foreach (var inc in cm.GetDummyStructIncludes())
                outputSw.WriteLine(inc);
            foreach (var inc in cm.GetDummyIncludes())
                outputSw.WriteLine(inc);
            outputSw.Flush();

            output.Write(structs.GetBuffer(), 0, (int)structs.Length);
            output.Write(declarations.GetBuffer(), 0, (int)declarations.Length);
            output.Write(code.GetBuffer(), 0, (int)code.Length);
            outputSw.Flush();
#if DEBUG
            using (FileStream fs = new FileStream("output.cu", FileMode.Create))
            {
                fs.Write(output.GetBuffer(), 0, (int)output.Length);
            }
#endif
            String s = Encoding.UTF8.GetString(output.GetBuffer(), 0, (int)output.Length);
            //cm.SourceCode = s;// #######!!!
            var scf = new SourceCodeFile(s, Language, _architecture);
            cm.AddSourceCodeFile(scf);
            return cm;
        }

        private static string GetResourceString(string name)
        {
            var assembly = Assembly.GetExecutingAssembly();
#if DEBUG
            foreach (var s in assembly.GetManifestResourceNames())
                Debug.WriteLine(s);
#endif
            using (var stream = assembly.GetManifestResourceStream("Cudafy.Translator.Resources." + name))
            {
                using (var sr = new StreamReader(stream))
                {
                    string s = sr.ReadToEnd();
                    return s;
                }
            }
        }

        private static string[] OpenCLReservedNames = new string[] { "kernel", "global" };

        private static void VerifyMemberName(string name)
        {
            if (LanguageSpecifics.Language == eLanguage.OpenCL && OpenCLReservedNames.Any(rn => rn == name))
                throw new CudafyLanguageException(CudafyLanguageException.csX_IS_A_RESERVED_KEYWORD, name);
        }

        private static eKernelMethodType GetKernelMethodType(eCudafyType? attr, MethodInfo mi)
        {
            eKernelMethodType kmt;
            if (attr == eCudafyType.Auto)
                kmt = mi.ReturnType.Name == "Void" ? eKernelMethodType.Global : eKernelMethodType.Device;
            else if (attr == eCudafyType.Device)
                kmt = eKernelMethodType.Device;
            else if (attr == eCudafyType.Global && mi.ReturnType.Name != "Void")
                throw new CudafyException(CudafyException.csX_NOT_SUPPORTED, "Return values on global methods");
            else if (attr == eCudafyType.Global)
                kmt = eKernelMethodType.Global;
            else if (attr == eCudafyType.Struct)
                throw new CudafyException(CudafyException.csX_NOT_SUPPORTED, "Cudafy struct attribute on methods");
            else
                throw new CudafyFatalException(attr.ToString());
            return kmt;
        }
    }

    public class CUDAfyLanguageSpecifics
    {
        public eLanguage Language {
            get { return CudafyModes.Language; }
            set { CudafyModes.Language = value; }
        }

        public string KernelFunctionModifiers
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return @"extern ""C"" __global__ ";
                else
                    return "__kernel ";
            }
        }

        public string DeviceFunctionModifiers
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "__device__ ";
                else
                    return " ";
            }
        }

        public string MemorySpaceSpecifier
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "";
                else
                    return "global";
            }
        }

        public string SharedModifier
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "__shared__";
                else
                    return "__local";
            }
        }

        public string ConstantModifier
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "__constant__";
                else
                    return "__constant";
            }
        }

        public string GetInlineModifier(eCudafyInlineMode mode)
        {
            if (Language != eLanguage.Cuda)
                return " ";
            if(mode == eCudafyInlineMode.Force)
                return "__forceinline__ ";
            else if(mode == eCudafyInlineMode.No)
                return "__noinline__ ";
            return " ";
        }


        public string GetAddressSpaceQualifier(eCudafyAddressSpace qualifier)
        {
            string addressSpaceQualifier = string.Empty;
            if (Language == eLanguage.OpenCL)
            {
                if ((qualifier & eCudafyAddressSpace.Global) == eCudafyAddressSpace.Global)
                {
                    return "global";
                }
                else if ((qualifier & eCudafyAddressSpace.Constant) == eCudafyAddressSpace.Constant)
                {
                    return "constant";
                }
                else if ((qualifier & eCudafyAddressSpace.Shared) == eCudafyAddressSpace.Shared)
                {
                    return "local";
                }
                else if ((qualifier & eCudafyAddressSpace.Private) == eCudafyAddressSpace.Private)
                {
                    return "private";
                }
            }
            return addressSpaceQualifier;
        }

        public string Int64Translation
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "long long";
                else
                    return "long";
            }
        }

        public string UInt64Translation
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "unsigned long long";
                else
                    return "ulong";
            }
        }

        public string PositiveInfinitySingle
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "0x7ff00000";
                else
                    return "INFINITY";
            }
        }

        public string NegativeInfinitySingle
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "0xfff00000";
                else
                    return "INFINITY";
            }
        }

        public string NaNSingle
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "logf(-1.0F)";
                else
                    return "NAN";
            }
        }

        public string PositiveInfinityDouble
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "0x7ff0000000000000";
                else
                    return "INFINITY";
            }
        }

        public string NegativeInfinityDouble
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "0xfff0000000000000";
                else
                    return "INFINITY";
            }
        }

        public string NaNDouble
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "log(-1.0)";
                else
                    return "NAN";
            }
        }
    }
}
