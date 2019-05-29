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
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel.Composition;
using System.Linq;
using System.Text;
using System.Globalization;
using System.Diagnostics;
using System.Reflection;
using System.IO;
using ICSharpCode.ILSpy;
using Mono.Cecil;
using ICSharpCode.NRefactory.CSharp;
using ICSharpCode.Decompiler;
using ICSharpCode.Decompiler.Ast.Transforms;
namespace Cudafy.Translator
{
#pragma warning disable 1591
    
    public class SpecialMemberFormatter : SpecialMember
    {
        public SpecialMemberFormatter(string declaringType, string original, Func<MemberReferenceExpression, object, string> func, bool callFunc = true, bool noSemiColon = false, System.Reflection.MethodInfo method = null)
            : base(declaringType, original, func, callFunc, noSemiColon)
        {
            Method = method;
        }
        public System.Reflection.MethodInfo Method { get; private set; }
    }
    
    public class SpecialMember
    {
        public SpecialMember(string declaringType, string original, Func<MemberReferenceExpression, object, string> func, bool callFunc = true, bool noSemiColon = false, string[] additionalLiterals = null) :
            this(new string[] { declaringType }, original, func, callFunc, noSemiColon, additionalLiterals)
        {

        }
        
        public SpecialMember(string[] declaringTypes, string original, Func<MemberReferenceExpression, object, string> func, bool callFunc = true, bool noSemiColon = false, string[] additionalLiterals = null)
        {
            OriginalName = original;
            AdditionalLiteralParams = additionalLiterals;
            DeclaringTypes = declaringTypes;
            Function = func;
            CallFunction = callFunc;
            NoSemicolon = noSemiColon;
        }

        public bool CallFunction { get; private set; }

        public string[] AdditionalLiteralParams { get; private set; }

        public string[] DeclaringTypes { get; private set; }
        
        public string OriginalName { get; private set; }

        public bool NoSemicolon { get; private set; }

        public Func<MemberReferenceExpression, object, string> Function { get; private set; }

        public string OptionalHeader { get; set; }

        public virtual string GetTranslation(MemberReferenceExpression mre, object data = null)
        {
            if (OptionalHeader != null)
                CUDALanguage.UseOptionalHeader(OptionalHeader);
            return Function(mre, data);
        }
    }

    public class OptionalHeader
    {
        public OptionalHeader(string name, string includeLine, bool asResource = false)
        {
            Name = name;
            IncludeLine = includeLine;
            AsResource = asResource;
        }
        
        public string Name { get; private set; }
        public string IncludeLine { get; private set; }
        public bool Used { get; set; }
        public bool AsResource { get; private set; }
    }

    public class OptionalFunction
    {
        public OptionalFunction(string name, string code)
        {
            Name = name;
            Code = code;
        }

        public string Name { get; private set; }
        public string Code { get; private set; }
        public bool Used { get; set; }
    }
    
    //[Export(typeof(Language))]
    public class CUDALanguage : Language
    {
        public CUDALanguage(eLanguage language)
        {
            _language = language;
            InitializeCommon();
            if (language == eLanguage.Cuda)
                InitializeCUDA();
            else if (language == eLanguage.OpenCL)
                InitializeOpenCL();
        }

                // key is "Class.Method"
        private static Dictionary<string, SpecialMemberFormatter> CachedFormatters;
        private const string FormatterSuffix = "_formatter";
        /// <summary>
        /// Initializes the <see cref="CUDALanguage"/> class.
        /// </summary>
        static CUDALanguage()
        {
            CachedFormatters = new Dictionary<string, SpecialMemberFormatter>();
            // TO-DO: skip system assemblies
            foreach (System.Reflection.Assembly assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                foreach (System.Type type in assembly.GetTypes())
                {
                    foreach (System.Reflection.MethodInfo method in type.GetMethods(System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic).Where(u => u.Name.EndsWith(FormatterSuffix)))
                    {
                        System.Reflection.ParameterInfo[] parameter = method.GetParameters();
                        if (parameter.Length == 2 && parameter[0].ParameterType.Equals(typeof(Cudafy.eLanguage)) && parameter[1].ParameterType.Equals(typeof(System.String[])) && method.ReturnType.Equals(typeof(System.String)))
                        {
                            SpecialMemberFormatter sm = new SpecialMemberFormatter(type.Name, method.Name, new Func<MemberReferenceExpression, object, string>(TranslateFormatterCode), false, false, method);
                            string key = string.Format("{0}.{1}", type.Name, method.Name.Substring(0, method.Name.Length - FormatterSuffix.Length));
                            if (CachedFormatters.ContainsKey(key))
                                throw new CudafyLanguageException(CudafyLanguageException.csMETHOD_X_ALREADY_ADDED_TO_THIS_MODULE, key);
                            CachedFormatters.Add(key, sm);
                        }
                    }
                }
            }
        }

        private eLanguage _language;
        
        private Predicate<IAstTransform> transformAbortCondition = null;

        public static bool DisableSmartArray { get; set; }
        
        public override string FileExtension
        {
            get { return ".cu"; }
        }

        public override string Name
        {
            get { return "CUDA"; }
        }

        public static Version ComputeCapability { get; set; }

        public int[] GetFieldInfoDimensions(System.Reflection.FieldInfo fieldInfo)
        {            
            Array array = fieldInfo.GetValue(null) as Array;
            if (array == null)
                return new int[0];

            List<int> dims = new List<int>();
            for (int i = 0; i < array.Rank; i++)
            {
                int len = array.GetLength(i);// GetUpperBound(i) + 1;
                dims.Add(len);
            }
            return dims.ToArray();
        }

        public override void DecompileField(FieldDefinition field, ITextOutput output, DecompilationOptions options)
        {
            WriteCommentLine(output, TypeToString(field.DeclaringType, includeNamespace: true));
            CUDAAstBuilder codeDomBuilder = CreateCUDAAstBuilder(options, currentType: field.DeclaringType, isSingleMember: true);
            codeDomBuilder.AddField(field);
            RunTransformsAndGenerateCode(codeDomBuilder, output, options);
        }

        public void DecompileCUDAConstantField(FieldDefinition field, int[] dims, ITextOutput output, DecompilationOptions options)
        {
            WriteCommentLine(output, TypeToString(field.DeclaringType, includeNamespace: true));
            CUDAAstBuilder codeDomBuilder = CreateCUDAAstBuilder(options, currentType: field.DeclaringType, isSingleMember: true);
            codeDomBuilder.AddField(field);
            RunTransformsAndGenerateCode(codeDomBuilder, output, options, dims);
        }

        public override void DecompileMethod(MethodDefinition method, ICSharpCode.Decompiler.ITextOutput output, DecompilationOptions options)
        {
            WriteCommentLine(output, TypeToString(method.DeclaringType, includeNamespace: true));
            CUDAAstBuilder codeDomBuilder = CreateCUDAAstBuilder(options, currentType: method.DeclaringType, isSingleMember: true);
            codeDomBuilder.AddMethod(method);
            RunTransformsAndGenerateCode(codeDomBuilder, output, options);
        }

        public void DecompileMethodDeclaration(MethodDefinition method, ICSharpCode.Decompiler.ITextOutput output, DecompilationOptions options)
        {
            WriteCommentLine(output, TypeToString(method.DeclaringType, includeNamespace: true));
            CUDAAstBuilder codeDomBuilder = CreateCUDAAstBuilder(options, currentType: method.DeclaringType, isSingleMember: true);
            codeDomBuilder.DecompileMethodBodies = false;
            codeDomBuilder.AddMethod(method);
            RunTransformsAndGenerateCode(codeDomBuilder, output, options);
        }

        public override void DecompileType(TypeDefinition type, ITextOutput output, DecompilationOptions options)
        {
            CUDAAstBuilder codeDomBuilder = CreateCUDAAstBuilder(options, currentType: type);
            codeDomBuilder.AddType(type);
            RunTransformsAndGenerateCode(codeDomBuilder, output, options);
        }

        void RunTransformsAndGenerateCode(CUDAAstBuilder astBuilder, ITextOutput output, DecompilationOptions options, int[] lastDims = null)
        {
            astBuilder.RunTransformations(transformAbortCondition);
            //if (options.DecompilerSettings.ShowXmlDocumentation)
            //    AddXmlDocTransform.Run(astBuilder.CompilationUnit);
            astBuilder.ConstantDims = lastDims;
            astBuilder.GenerateCode(output);
        }

        CUDAAstBuilder CreateCUDAAstBuilder(DecompilationOptions options, ModuleDefinition currentModule = null, TypeDefinition currentType = null, bool isSingleMember = false)
        {
            if (currentModule == null)
                currentModule = currentType.Module;
            DecompilerSettings settings = options.DecompilerSettings;
            if (isSingleMember)
            {
                settings = settings.Clone();
                settings.UsingDeclarations = false;
            }
            return new CUDAAstBuilder(
                new DecompilerContext(currentModule)
                {
                    CancellationToken = options.CancellationToken,
                    CurrentType = currentType,
                    Settings = settings
                });
        }

        ///// <summary>
        ///// Initializes the <see cref="CUDALanguage"/> class.
        ///// </summary>
        //static CUDALanguage()
        //{
        //    InitializeStatic();
        //}

        private static void InitializeCommon()
        {
            SpecialMethods.Clear();
            SpecialMethods.Add(new SpecialMember("GThread", "InsertCode", new Func<MemberReferenceExpression, object, string>(TranslateInsertCode), false, true));
            SpecialMethods.Add(new SpecialMember("Trace", null, new Func<MemberReferenceExpression, object, string>(CommentMeOut), false));
            SpecialMethods.Add(new SpecialMember("Debug", null, new Func<MemberReferenceExpression, object, string>(CommentMeOut), false));
            SpecialMethods.Add(new SpecialMember("Console", null, new Func<MemberReferenceExpression, object, string>(CommentMeOut), false));
            SpecialMethods.Add(new SpecialMember("Console", "Write", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            SpecialMethods.Add(new SpecialMember("Console", "WriteLine", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            SpecialMethods.Add(new SpecialMember("Debug", "Write", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            SpecialMethods.Add(new SpecialMember("Debug", "WriteIf", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            SpecialMethods.Add(new SpecialMember("Debug", "WriteLine", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            SpecialMethods.Add(new SpecialMember("Debug", "WriteLineIf", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            SpecialMethods.Add(new SpecialMember("ArrayType", "GetLength", new Func<MemberReferenceExpression, object, string>(TranslateArrayGetLength), false));
            SpecialMethods.Add(new SpecialMember("double", "IsNaN", new Func<MemberReferenceExpression, object, string>(TranslateFloatingPointMemberName)));
            SpecialMethods.Add(new SpecialMember("float", "IsNaN", new Func<MemberReferenceExpression, object, string>(TranslateFloatingPointMemberName)));
            SpecialMethods.Add(new SpecialMember("double", "IsInfinity", new Func<MemberReferenceExpression, object, string>(TranslateFloatingPointMemberName)));
            SpecialMethods.Add(new SpecialMember("float", "IsInfinity", new Func<MemberReferenceExpression, object, string>(TranslateFloatingPointMemberName)));

            //SIMD-in-a-word functions
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vabs2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vabsdiffs2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vabsdiffu2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vabsss2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vadd2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vaddss2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vaddus2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vavgs2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vavgu2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmpeq2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmpges2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmpgeu2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmpgts2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmpgtu2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmples2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmpleu2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmplts2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmpltu2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmpne2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vhaddu2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vmaxs2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vmaxu2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vmins2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vminu2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vneg2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vnegss2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsads2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsadu2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vseteq2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetges2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetgeu2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetgts2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetgtu2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetles2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetleu2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetlts2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetltu2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetne2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsub2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsubss2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsubus2", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vabs4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vabsdiffs4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vabsdiffu4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vabsss4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vadd4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vaddss4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vaddus4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vavgs4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vavgu4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmpeq4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmpges4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmpgeu4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmpgts4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmpgtu4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmples4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmpleu4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmplts4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmpltu4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vcmpne4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vhaddu4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vmaxs4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vmaxu4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vmins4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vminu4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vneg4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vnegss4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsads4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsadu4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vseteq4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetges4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetgeu4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetgts4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetgtu4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetles4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetleu4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetlts4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetltu4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsetne4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsub4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsubss4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "vsubus4", new Func<MemberReferenceExpression, object, string>(GetMemberName)) { OptionalHeader = csSIMDFUNCS });

            SpecialProperties.Clear();       
            SpecialProperties.Add(new SpecialMember("ArrayType", "Length", new Func<MemberReferenceExpression, object, string>(TranslateArrayLength)));
            SpecialProperties.Add(new SpecialMember("ArrayType", "LongLength", new Func<MemberReferenceExpression, object, string>(TranslateArrayLength)));
            SpecialProperties.Add(new SpecialMember("ArrayType", "IsFixedSize", new Func<MemberReferenceExpression, object, string>(TranslateToTrue)));
            SpecialProperties.Add(new SpecialMember("ArrayType", "IsReadOnly", new Func<MemberReferenceExpression, object, string>(TranslateToFalse)));
            SpecialProperties.Add(new SpecialMember("ArrayType", "IsSynchronized", new Func<MemberReferenceExpression, object, string>(TranslateToFalse)));
            SpecialProperties.Add(new SpecialMember("ArrayType", "Rank", new Func<MemberReferenceExpression, object, string>(TranslateArrayRank)));
            SpecialProperties.Add(new SpecialMember("System.String", "Length", new Func<MemberReferenceExpression, object, string>(TranslateStringLength)));

            SpecialTypes.Clear();
            SpecialTypes.Add("IntPtr", new SpecialTypeProps() { Name = "void", OptionalHeader = null });

            OptionalHeaders = new List<OptionalHeader>();

            OptionalFunctions = new List<OptionalFunction>();

            DisableSmartArray = false;
        }

        private static void InitializeCUDA()
        {
            ComputeCapability = new Version(1, 3);
            
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "get_global_size", new Func<MemberReferenceExpression, object, string>(GetOptionalFunctionMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "get_global_id", new Func<MemberReferenceExpression, object, string>(GetOptionalFunctionMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "get_local_id", new Func<MemberReferenceExpression, object, string>(GetOptionalFunctionMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "get_group_id", new Func<MemberReferenceExpression, object, string>(GetOptionalFunctionMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "get_local_size", new Func<MemberReferenceExpression, object, string>(GetOptionalFunctionMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "get_num_groups", new Func<MemberReferenceExpression, object, string>(GetOptionalFunctionMemberName)));
            
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "SyncThreads", new Func<MemberReferenceExpression, object, string>(TranslateSyncThreads)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "SyncThreadsCount", new Func<MemberReferenceExpression, object, string>(TranslateSyncThreadsCount)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "All", new Func<MemberReferenceExpression, object, string>(TranslateAll)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "Any", new Func<MemberReferenceExpression, object, string>(TranslateAny)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "Ballot", new Func<MemberReferenceExpression, object, string>(TranslateBallot)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "Shuffle", new Func<MemberReferenceExpression, object, string>(TranslateShuffle)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "ShuffleUp", new Func<MemberReferenceExpression, object, string>(TranslateShuffleUp)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "ShuffleDown", new Func<MemberReferenceExpression, object, string>(TranslateShuffleDown)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "ShuffleXor", new Func<MemberReferenceExpression, object, string>(TranslateShuffleXor)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicAdd", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicSub", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicExch", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicMin", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicMax", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicInc", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicDec", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicIncEx", new Func<MemberReferenceExpression, object, string>(TranslateCUDAAtomicIncDec), true, false, new string[] { "0xFFFFFFFF" }));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicDecEx", new Func<MemberReferenceExpression, object, string>(TranslateCUDAAtomicIncDec), true, false, new string[] { "0xFFFFFFFF" }));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicCAS", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicAnd", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicOr", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicXor", new Func<MemberReferenceExpression, object, string>(GetMemberName)));



            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_init", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_log_normal", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_log_normal_double", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_normal", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_normal_double", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_uniform", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_uniform_double", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "skipahead", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "skipahead_sequence", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));

            SpecialMethods.Add(new SpecialMember("GMath", null, new Func<MemberReferenceExpression, object, string>(TranslateGMath)));
            SpecialMethods.Add(new SpecialMember("Math", null, new Func<MemberReferenceExpression, object, string>(TranslateMath)));

            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "popcount", new Func<MemberReferenceExpression, object, string>(TranslateCUDAIntegerFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "popcountll", new Func<MemberReferenceExpression, object, string>(TranslateCUDAIntegerFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "clz", new Func<MemberReferenceExpression, object, string>(TranslateCUDAIntegerFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "clzll", new Func<MemberReferenceExpression, object, string>(TranslateCUDAIntegerFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "mul24", new Func<MemberReferenceExpression, object, string>(TranslateCUDAIntegerFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "mul64hi", new Func<MemberReferenceExpression, object, string>(TranslateCUDAIntegerFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "mulhi", new Func<MemberReferenceExpression, object, string>(TranslateCUDAIntegerFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "umul24", new Func<MemberReferenceExpression, object, string>(TranslateCUDAIntegerFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "umul64hi", new Func<MemberReferenceExpression, object, string>(TranslateCUDAIntegerFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "umulhi", new Func<MemberReferenceExpression, object, string>(TranslateCUDAIntegerFunc)));

            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "SynchronizeDevice", new Func<MemberReferenceExpression, object, string>(TranslateDynamicParallelismFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "GetDeviceCount", new Func<MemberReferenceExpression, object, string>(TranslateDynamicParallelismFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "GetDeviceID", new Func<MemberReferenceExpression, object, string>(TranslateDynamicParallelismFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "Launch", new Func<MemberReferenceExpression, object, string>(TranslateDynamicParallelismFunc), true, false));

            SpecialMethods.Add(new SpecialMember("ComplexD", null, new Func<MemberReferenceExpression, object, string>(TranslateComplexD)));
            SpecialMethods.Add(new SpecialMember("ComplexF", null, new Func<MemberReferenceExpression, object, string>(TranslateComplexF)));
           
            SpecialMethods.Add(new SpecialMember("ComplexD", "ctor", new Func<MemberReferenceExpression, object, string>(TranslateComplexDCtor)));
            SpecialMethods.Add(new SpecialMember("ComplexF", "ctor", new Func<MemberReferenceExpression, object, string>(TranslateComplexFCtor)));
            SpecialMethods.Add(new SpecialMember("Debug", "Assert", new Func<MemberReferenceExpression, object, string>(TranslateAssert), false));
            
 
            SpecialProperties.Add(new SpecialMember("Cudafy.GThread", "warpSize", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialProperties.Add(new SpecialMember("Math", "E", new Func<MemberReferenceExpression, object, string>(TranslateMathE)));
            SpecialProperties.Add(new SpecialMember("Math", "PI", new Func<MemberReferenceExpression, object, string>(TranslateMathPI)));
            SpecialProperties.Add(new SpecialMember("GMath", "E", new Func<MemberReferenceExpression, object, string>(TranslateGMathE)));
            SpecialProperties.Add(new SpecialMember("GMath", "PI", new Func<MemberReferenceExpression, object, string>(TranslateGMathPI)));
            
            SpecialTypes.Add("ComplexD", new SpecialTypeProps() { Name = "cuDoubleComplex", OptionalHeader = "cuComplex" });
            SpecialTypes.Add("ComplexF", new SpecialTypeProps() { Name = "cuFloatComplex", OptionalHeader = "cuComplex" });

            SpecialTypes.Add("RandStateXORWOW", new SpecialTypeProps() { Name = "curandStateXORWOW", OptionalHeader = csCURAND_KERNEL });
            SpecialTypes.Add("RandStateSobol32", new SpecialTypeProps() { Name = "curandStateSobol32", OptionalHeader = csCURAND_KERNEL });
            SpecialTypes.Add("RandStateScrambledSobol32", new SpecialTypeProps() { Name = "curandStateScrambledSobol32", OptionalHeader = csCURAND_KERNEL });
            SpecialTypes.Add("RandStateSobol64", new SpecialTypeProps() { Name = "curandStateSobol64", OptionalHeader = csCURAND_KERNEL });
            SpecialTypes.Add("RandStateScrambledSobol64", new SpecialTypeProps() { Name = "curandStateScrambledSobol64", OptionalHeader = csCURAND_KERNEL });

            
            OptionalHeaders.Add(new OptionalHeader("cuComplex", @"#include <cuComplex.h>"));
            OptionalHeaders.Add(new OptionalHeader(csCURAND_KERNEL, @"#include <curand_kernel.h>"));
            OptionalHeaders.Add(new OptionalHeader(csSTDIO, @"#include <stdio.h>"));
            OptionalHeaders.Add(new OptionalHeader(csASSERT, @"#include <assert.h>"));
            OptionalHeaders.Add(new OptionalHeader(csSIMDFUNCS, @"simd_functions.h", true));
            
            OptionalFunctions.Add(new OptionalFunction(csGET_GLOBAL_ID, OptionalStrings.get_global_id));
            OptionalFunctions.Add(new OptionalFunction(csGET_GLOBAL_SIZE, OptionalStrings.get_global_size));
            OptionalFunctions.Add(new OptionalFunction(csGET_GROUP_ID, OptionalStrings.get_group_id));
            OptionalFunctions.Add(new OptionalFunction(csGET_LOCAL_ID, OptionalStrings.get_local_id));
            OptionalFunctions.Add(new OptionalFunction(csGET_LOCAL_SIZE, OptionalStrings.get_local_size));
            OptionalFunctions.Add(new OptionalFunction(csGET_NUM_GROUPS, OptionalStrings.get_num_groups)); 
        }



        private static void InitializeOpenCL()
        {
            ComputeCapability = new Version(1, 3);

            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "get_global_size", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "get_global_id", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "get_local_id", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "get_group_id", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "get_local_size", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "get_num_groups", new Func<MemberReferenceExpression, object, string>(GetMemberName)));

            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "SyncThreads", new Func<MemberReferenceExpression, object, string>(TranslateSyncThreadsOpenCL), false));

            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "SyncThreadsCount", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "All", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "Any", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "Ballot", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "Shuffle", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "ShuffleUp", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "ShuffleDown", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "ShuffleXor", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicAdd", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLAtomic)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicSub", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLAtomic)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicExch", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLAtomic)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicInc", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLAtomic)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicDec", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLAtomic)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicIncEx", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLAtomic)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicDecEx", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLAtomic)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicCAS", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLAtomic)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicMin", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLAtomic)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicMax", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLAtomic)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicAnd", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLAtomic)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicOr", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLAtomic)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicXor", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLAtomic)));

            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_init", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_log_normal", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_log_normal_double", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_normal", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_normal_double", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_uniform", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_uniform_double", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "skipahead", new Func<MemberReferenceExpression, object, string>(NotSupported)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "skipahead_sequence", new Func<MemberReferenceExpression, object, string>(NotSupported)));

            SpecialMethods.Add(new SpecialMember("GMath", null, new Func<MemberReferenceExpression, object, string>(TranslateGMathOpenCL)));
            SpecialMethods.Add(new SpecialMember("Math", null, new Func<MemberReferenceExpression, object, string>(TranslateMathOpenCL)));

            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "popcount", new Func<MemberReferenceExpression, object, string>(GetOptionalFunctionMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "popcountll", new Func<MemberReferenceExpression, object, string>(GetOptionalFunctionMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "clz", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLIntegerFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "clzll", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLIntegerFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "mul24", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "mul64hi", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLIntegerFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "mulhi", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLIntegerFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "umul24", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLIntegerFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "umul64hi", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLIntegerFunc)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "umulhi", new Func<MemberReferenceExpression, object, string>(TranslateOpenCLIntegerFunc)));
            //SpecialMethods.Add(new SpecialMember("ComplexD", null, new Func<MemberReferenceExpression, object, string>(TranslateComplexD)));
            //SpecialMethods.Add(new SpecialMember("ComplexF", null, new Func<MemberReferenceExpression, object, string>(TranslateComplexF)));

            //SpecialMethods.Add(new SpecialMember("ArrayType", "GetLength", new Func<MemberReferenceExpression, object, string>(TranslateArrayGetLength), false));

            //SpecialMethods.Add(new SpecialMember("ComplexD", "ctor", new Func<MemberReferenceExpression, object, string>(TranslateComplexDCtor)));
            //SpecialMethods.Add(new SpecialMember("ComplexF", "ctor", new Func<MemberReferenceExpression, object, string>(TranslateComplexFCtor)));

            ////SpecialMethods.Add(new SpecialMember("Debug", null, new Func<MemberReferenceExpression, object, string>(CommentMeOut), false));
            ////SpecialMethods.Add(new SpecialMember("Console", null, new Func<MemberReferenceExpression, object, string>(CommentMeOut), false));

            //SpecialMethods.Add(new SpecialMember("Debug", "Write", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            //SpecialMethods.Add(new SpecialMember("Debug", "WriteIf", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            //SpecialMethods.Add(new SpecialMember("Debug", "WriteLine", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            //SpecialMethods.Add(new SpecialMember("Debug", "WriteLineIf", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            //SpecialMethods.Add(new SpecialMember("Debug", null, new Func<MemberReferenceExpression, object, string>(CommentMeOut), false));
            //SpecialMethods.Add(new SpecialMember("Console", "Write", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            //SpecialMethods.Add(new SpecialMember("Console", "WriteLine", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            //SpecialMethods.Add(new SpecialMember("Console", null, new Func<MemberReferenceExpression, object, string>(CommentMeOut), false));
            //SpecialMethods.Add(new SpecialMember("Debug", "Assert", new Func<MemberReferenceExpression, object, string>(TranslateAssert), false));
            //SpecialMethods.Add(new SpecialMember("Trace", null, new Func<MemberReferenceExpression, object, string>(CommentMeOut), false));

            SpecialProperties.Add(new SpecialMember("Cudafy.GThread", "warpSize", new Func<MemberReferenceExpression, object, string>(GetOpenCLDefaultWarpSize)));
            //SpecialProperties.Add(new SpecialMember("ArrayType", "Length", new Func<MemberReferenceExpression, object, string>(TranslateArrayLength)));
            //SpecialProperties.Add(new SpecialMember("ArrayType", "LongLength", new Func<MemberReferenceExpression, object, string>(TranslateArrayLength)));
            //SpecialProperties.Add(new SpecialMember("ArrayType", "IsFixedSize", new Func<MemberReferenceExpression, object, string>(TranslateToTrue)));
            //SpecialProperties.Add(new SpecialMember("ArrayType", "IsReadOnly", new Func<MemberReferenceExpression, object, string>(TranslateToFalse)));
            //SpecialProperties.Add(new SpecialMember("ArrayType", "IsSynchronized", new Func<MemberReferenceExpression, object, string>(TranslateToFalse)));
            //SpecialProperties.Add(new SpecialMember("ArrayType", "Rank", new Func<MemberReferenceExpression, object, string>(TranslateArrayRank)));
            //SpecialProperties.Add(new SpecialMember("Cudafy.GThread", "warpSize", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            ////
            //SpecialProperties.Add(new SpecialMember("System.String", "Length", new Func<MemberReferenceExpression, object, string>(TranslateStringLength)));


            //SpecialTypes.Add("ComplexD", new SpecialTypeProps() { Name = "cuDoubleComplex", OptionalHeader = "cuComplex" });
            //SpecialTypes.Add("ComplexF", new SpecialTypeProps() { Name = "cuFloatComplex", OptionalHeader = "cuComplex" });

            //SpecialTypes.Add("RandStateXORWOW", new SpecialTypeProps() { Name = "curandStateXORWOW", OptionalHeader = csCURAND_KERNEL });
            //SpecialTypes.Add("RandStateSobol32", new SpecialTypeProps() { Name = "curandStateSobol32", OptionalHeader = csCURAND_KERNEL });
            //SpecialTypes.Add("RandStateScrambledSobol32", new SpecialTypeProps() { Name = "curandStateScrambledSobol32", OptionalHeader = csCURAND_KERNEL });
            //SpecialTypes.Add("RandStateSobol64", new SpecialTypeProps() { Name = "curandStateSobol64", OptionalHeader = csCURAND_KERNEL });
            //SpecialTypes.Add("RandStateScrambledSobol64", new SpecialTypeProps() { Name = "curandStateScrambledSobol64", OptionalHeader = csCURAND_KERNEL });

            //OptionalHeaders = new List<OptionalHeader>();
            //OptionalHeaders.Add(new OptionalHeader("cuComplex", @"#include <cuComplex.h>"));
            //OptionalHeaders.Add(new OptionalHeader(csCURAND_KERNEL, @"#include <curand_kernel.h>"));
            //OptionalHeaders.Add(new OptionalHeader(csSTDIO, @"#include <stdio.h>"));
            //OptionalHeaders.Add(new OptionalHeader(csASSERT, @"#include <assert.h>"));
            OptionalHeaders.Add(new OptionalHeader(csSIMDFUNCS, @"simd_functions_opencl.h", true));

            OptionalFunctions.Add(new OptionalFunction(csPOPCOUNT, OptionalStrings.popCount));
            OptionalFunctions.Add(new OptionalFunction(csPOPCOUNTLL, OptionalStrings.popCountll));
        }

        private const string csCURAND_KERNEL = "curand_kernel";

        private const string csSTDIO = "stdio";

        private const string csASSERT = "assert";

        private const string csSIMDFUNCS = "simd_functions";

        private const string csGET_GLOBAL_ID = "get_global_id";
        private const string csGET_LOCAL_ID = "get_local_id";
        private const string csGET_GROUP_ID = "get_group_id";
        private const string csGET_LOCAL_SIZE = "get_local_size";
        private const string csGET_GLOBAL_SIZE = "get_global_size";
        private const string csGET_NUM_GROUPS = "get_num_groups";
        private const string csPOPCOUNT = "popcount";
        private const string csPOPCOUNTLL = "popcountll";
        public struct SpecialTypeProps
        {
            public string Name;
            public string OptionalHeader;
        }

        static string NormalizeDeclaringType(string declaringType)
        {
            if (declaringType.Contains("["))
                return "ArrayType";
            return declaringType;
        }

        public static bool IsSpecialProperty(string memberName, string declaringType)
        {
            return GetSpecialProperty(memberName, declaringType) != null;
        }

        public static SpecialMember GetSpecialProperty(string memberName, string declaringType)
        {
            declaringType = NormalizeDeclaringType(declaringType);           
            foreach (var item in SpecialProperties)
                if (item.DeclaringTypes.Contains(declaringType) && memberName == item.OriginalName)
                    return item;
            return null;
        }

        public static bool IsSpecialMethod(string memberName, string declaringType)
        {
            return GetSpecialMethod(memberName, declaringType) != null;
        }

        public static SpecialMember GetSpecialMethod(string memberName, string declaringType)
        {
            declaringType = NormalizeDeclaringType(declaringType);
            foreach (var item in SpecialMethods)
            {
                //if (memberName == "AllocateShared")
                //    Console.WriteLine("AllocateShared? "+item.OriginalName);
                if (item.DeclaringTypes.Contains(declaringType) && memberName == item.OriginalName)
                    return item;
            }
            string key = string.Format("{0}.{1}", declaringType, memberName);
            if (CachedFormatters.ContainsKey(key))
                return CachedFormatters[key];
            // We don't want to take a default method when there is a special property
            var prop = GetSpecialProperty(memberName, declaringType);
            if (prop == null)
            {
                foreach (var item in SpecialMethods)
                    if (item.DeclaringTypes.Contains(declaringType) && item.OriginalName == null)
                    {
                        return item;
                    }
            }
            return prop;
        }

        public static string TranslateSpecialType(string declaringType)
        {
            declaringType = NormalizeDeclaringType(declaringType);
            if (SpecialTypes.ContainsKey(declaringType))
            {
                var stp = SpecialTypes[declaringType];
                if (!string.IsNullOrEmpty(stp.OptionalHeader))
                    UseOptionalHeader(stp.OptionalHeader);
                return stp.Name;
            }
            else
                return declaringType.Replace(".", "");
        }

        public static void Reset()
        {
            foreach (var oh in OptionalHeaders)
                oh.Used = false;
            foreach (var oh in OptionalFunctions)
                oh.Used = false;
            _constants.Clear();
            DisableSmartArray = false;
        }
        
        private static List<KernelConstantInfo> _constants = new List<KernelConstantInfo>();

        public static IEnumerable<KernelConstantInfo> GetConstants()
        {
            return _constants;
        }

        public static void AddConstant(KernelConstantInfo kci)
        {
            _constants.Add(kci);
        }

        internal static void UseOptionalHeader(string name)
        {
            var oh = OptionalHeaders.Where(o => o.Name == name).FirstOrDefault();
            Debug.Assert(oh != null);
            oh.Used = true;
        }

        private static void UseOptionalFunction(string name)
        {
            var oh = OptionalFunctions.Where(o => o.Name == name).FirstOrDefault();
            Debug.Assert(oh != null);
            oh.Used = true;
        }


        public readonly static string csSyncThreads = "SyncThreads";
        public readonly static string csSyncThreadsCount = "SyncThreadsCount";
        
        public readonly static string csAll = "All";
        public readonly static string csAny = "Any";
        public readonly static string csBallot = "Ballot";
        public readonly static string csAllocateShared = "AllocateShared";

        public static List<SpecialMember> SpecialMethods = new List<SpecialMember>();
        public static List<SpecialMember> SpecialProperties = new List<SpecialMember>();
        public static Dictionary<string, SpecialTypeProps> SpecialTypes = new Dictionary<string, SpecialTypeProps>();
        public static List<OptionalHeader> OptionalHeaders;
        public static List<OptionalFunction> OptionalFunctions;

        static string TranslateStringLength(MemberReferenceExpression mre, object data)
        {
            string length = mre.Target.ToString() + "Len";
            return length;
        }

        static string TranslateArrayLength(MemberReferenceExpression mre, object data)
        {
            string rank, length;
            bool rc = mre.TranslateArrayLengthAndRank(out length, out rank);
            Debug.Assert(rc);
            return length;
        }

        static string TranslateArrayRank(MemberReferenceExpression mre, object data)
        {
            string rank, length;
            bool rc = mre.TranslateArrayLengthAndRank(out length, out rank);
            Debug.Assert(rc);
            return rank;
        }

        static string TranslateToTrue(MemberReferenceExpression mre, object data)
        {
            return "true";
        }

        static string TranslateToFalse(MemberReferenceExpression mre, object data)
        {
            return "false";
        }

        static string TranslateFormatterCode(MemberReferenceExpression mre, object data)
        {
            var anl = ((Expression)data).Children.ToList();
            string value = string.Empty;
            string key = string.Format("{0}.{1}", mre.Target.ToString(), mre.MemberName);
            SpecialMemberFormatter formatter = CachedFormatters[key];
            if (anl.Count > 1)
            {
                object[] args = anl.Skip(1).ToArray();

                // too restrictive.
                //foreach (object o in args)
                //    if (!(o is IdentifierExpression) && !(o is PrimitiveExpression))
                //        throw new CudafyLanguageException(CudafyLanguageException.csMETHOD_X_X_ONLY_SUPPORTS_X, mre.Target.ToString(), mre.MemberName + FormatterSuffix, "identifiers and primitives as list of arguments to text formatting");

                string[] str_args = args.Select(k => k.ToString()).ToArray();
                value = formatter.Method.Invoke(null, new object[] { CudafyTranslator.Language, str_args }).ToString();
            }
            else
                value = formatter.Method.Invoke(null, new object[] { CudafyTranslator.Language }).ToString();
            return value;
        }

        static string TranslateInsertCode(MemberReferenceExpression mre, object data)
        {
            var anl = ((Expression)data).Children.ToList();
            PrimitiveExpression pe = anl[1] as PrimitiveExpression;
            string value = string.Empty;
            if (pe == null)
                throw new CudafyLanguageException(CudafyLanguageException.csMETHOD_X_X_ONLY_SUPPORTS_X, "GThread", "InsertCode", "strings");

            value = pe.Value.ToString();
            if(anl.Count > 2)
            {
                string format = value;
                ArrayCreateExpression ace = anl[2] as ArrayCreateExpression;
                if(ace == null)
                    throw new CudafyLanguageException(CudafyLanguageException.csMETHOD_X_X_ONLY_SUPPORTS_X, "GThread", "InsertCode", "list of arguments to text formatting");
                var acecl = ace.Children.ToList();
                if (acecl.Count > 1)
                {
                    ArrayInitializerExpression aie = acecl[1] as ArrayInitializerExpression;
                    if (aie != null)
                    {
                        object[] args = aie.Children.ToArray();
                        foreach(object o in args)
                            if(!(o is IdentifierExpression) && !(o is PrimitiveExpression))
                                throw new CudafyLanguageException(CudafyLanguageException.csMETHOD_X_X_ONLY_SUPPORTS_X, "GThread", "InsertText", "identifiers and primitives as list of arguments to text formatting"); 
                        value = string.Format(format, args);
                    }
                }
            }
                            
            return value;
        }

        static string TranslateSyncThreads(MemberReferenceExpression mre, object data)
        {
            return "__syncthreads";
        }

        static string TranslateSyncThreadsOpenCL(MemberReferenceExpression mre, object data)
        {
            return "barrier(CLK_LOCAL_MEM_FENCE)";
        }

        static string TranslateSyncThreadsCount(MemberReferenceExpression mre, object data)
        {
            if (ComputeCapability < new Version(2, 0))
                throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED_FOR_COMPUTE_X, mre.MemberName, ComputeCapability);
            return "__syncthreads_count";
        }

        static string TranslateAll(MemberReferenceExpression mre, object data)
        {
            return "__all";
        }

        static string TranslateAny(MemberReferenceExpression mre, object data)
        {
            return "__any";
        }

        static string TranslateBallot(MemberReferenceExpression mre, object data)
        {
            if (ComputeCapability < new Version(2, 0))
                throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED_FOR_COMPUTE_X, mre.MemberName, ComputeCapability);
            return "__ballot";
        }

        static string TranslateShuffle(MemberReferenceExpression mre, object data)
        {
            if (ComputeCapability < new Version(3, 0))
                throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED_FOR_COMPUTE_X, mre.MemberName, ComputeCapability);
            return "__shfl";
        }

        static string TranslateShuffleUp(MemberReferenceExpression mre, object data)
        {
            if (ComputeCapability < new Version(3, 0))
                throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED_FOR_COMPUTE_X, mre.MemberName, ComputeCapability);
            return "__shfl_up";
        }

        static string TranslateShuffleDown(MemberReferenceExpression mre, object data)
        {
            if (ComputeCapability < new Version(3, 0))
                throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED_FOR_COMPUTE_X, mre.MemberName, ComputeCapability);
            return "__shfl_down";
        }

        static string TranslateShuffleXor(MemberReferenceExpression mre, object data)
        {
            if (ComputeCapability < new Version(3, 0))
                throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED_FOR_COMPUTE_X, mre.MemberName, ComputeCapability);
            return "__shfl_xor";
        }

        static string TranslateAtomicAddFloat(MemberReferenceExpression mre, object data)
        {
            if (ComputeCapability < new Version(2, 0))
                throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED_FOR_COMPUTE_X, mre.MemberName, ComputeCapability);
            return "atomicAdd";
        }

        static string TranslateToPrintF(MemberReferenceExpression mre, object data)
        {
            if (ComputeCapability < new Version(2, 0) && CudafyTranslator.Language == eLanguage.Cuda)
                return CommentMeOut(mre, data);
            if (CudafyTranslator.Language == eLanguage.Cuda)
                UseOptionalHeader(csSTDIO);
            string dbugwrite = string.Empty;
            dbugwrite = mre.TranslateToPrintF(data);
            return dbugwrite;
        }

        static string TranslateCUDAAtomicIncDec(MemberReferenceExpression mre, object data)
        {
            switch (mre.MemberName)
            {
                case "atomicIncEx":
                    return "atomicInc";
                case "atomicDecEx":
                    return "atomicDec";
                default:
                    break;
            }
            throw new NotSupportedException(mre.MemberName);
        }


#warning TODO Support atomicInc and atomicDec in OpenCL - CUDA implementation is different - http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/atomic_inc.html
        static string TranslateOpenCLAtomic(MemberReferenceExpression mre, object data)
        {
            switch (mre.MemberName)
            {
                case "atomicAdd":
                    return "atomic_add";
                case "atomicSub":
                    return "atomic_sub";
                case "atomicExch":
                    return "atomic_xchg";
                case "atomicIncEx":
                    return "atomic_inc";
                case "atomicDecEx":
                    return "atomic_dec";
                case "atomicCAS":
                    return "atomic_cmpxchg";
                case "atomicMin":
                    return "atomic_min";
                case "atomicMax":
                    return "atomic_max";
                case "atomicAnd":
                    return "atomic_and";
                case "atomicOr":
                    return "atomic_or";
                case "atomicXor":
                    return "atomic_xor";
                default:
                    break;
            }
            throw new NotSupportedException(mre.MemberName);
        }

        static string TranslateAssert(MemberReferenceExpression mre, object data)
        {
            if (ComputeCapability < new Version(2, 0))
                return CommentMeOut(mre, data);
            UseOptionalHeader(csASSERT);
            string assert = string.Empty;
            assert = mre.TranslateAssert(data);
            return assert;
        }

        static string GetMemberName(MemberReferenceExpression mre, object data)
        {
            return mre.MemberName;
        }

        static string NotSupported(MemberReferenceExpression mre, object data)
        {
            throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED_IN_X, mre.MemberName, CudafyTranslator.LanguageSpecifics.Language);
        }


        static string GetOpenCLDefaultWarpSize(MemberReferenceExpression mre, object data)
        {
            return "32";
        }

        static string GetOptionalFunctionMemberName(MemberReferenceExpression mre, object data)
        {
            UseOptionalFunction(mre.MemberName);
            return mre.MemberName;
        }

        static string GetCURANDMemberName(MemberReferenceExpression mre, object data)
        {
            UseOptionalHeader("curand_kernel");
            DisableSmartArray = true;
            return mre.MemberName;
        }



        static string TranslateComplexDCtor(MemberReferenceExpression mre, object data)
        {
            UseOptionalHeader("cuComplex");
            return "make_cuDoubleComplex";
        }

        static string TranslateComplexFCtor(MemberReferenceExpression mre, object data)
        {
            UseOptionalHeader("cuComplex");
            return "make_cuFloatComplex";
        }

        static string TranslateArrayGetLength(MemberReferenceExpression mre, object data)
        {
            string length = string.Empty;
            length = mre.TranslateArrayGetLength(data);
            return length;
        }

        static string CommentMeOut(MemberReferenceExpression mre, object data)
        {
            return string.Format("// {0}", mre.ToString());
        }

        static string TranslateFloatingPointMemberName(MemberReferenceExpression mre, object data)
        {
            switch (mre.MemberName)
            {
                case "IsNaN":
                    return "isnan";
                case "IsInfinity":
                    return "isinf";

                default:
                    break;
            }
            throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED, mre.MemberName);
        }

        private static string TranslateDynamicParallelismFunc(MemberReferenceExpression mre, object data)
        {
            switch (mre.MemberName)
            {
                case "SynchronizeDevice":
                    return "cudaDeviceSynchronize";
                case "GetDeviceCount":
                    return "cudaGetDeviceCount";
                case "GetDeviceID":
                    return "cudaGetDevice";
                case "Launch":
                    var list = (data as InvocationExpression).Arguments.Take(4).Cast<Expression>().ToList();
                    object gridSize = list[0].ToString();
                    object blockSize = list[1].ToString();
                    string name = list[2].ToString().Trim('"');
                    list[0] = new PrimitiveExpression("IGNOREMEE01B67F3" + gridSize.ToString());
                    list[1] = new PrimitiveExpression("IGNOREMEE01B67F3" + blockSize.ToString());
                    list[2] = new PrimitiveExpression("IGNOREMEE01B67F3" + name);
                    var args = (data as InvocationExpression).Arguments.ToList()[3];
                    
                    foreach(Expression elem in ((ICSharpCode.NRefactory.CSharp.ArrayCreateExpression)(args)).Initializer.Elements)//.ToList();
                        list.Add(elem.Clone());
                    //var explist = list.Cast<Expression>().ToList();
                    //list.AddRange(elems);
                    (data as InvocationExpression).Arguments.ReplaceWith(list);
                    return string.Format("{0}<<<{1},{2}>>>",name,gridSize,blockSize);
                    
                    
                default:
                    break;
            }
            throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED, mre.MemberName);
        }

        static string TranslateCUDAIntegerFunc(MemberReferenceExpression mre, object data)
        {
            switch (mre.MemberName)
            {
                case "popcount":
                    return "__popc";
                case "popcountll":
                    return "__popcll";
                case "clz":
                    return "__clz";
                case "clzll":
                    return "__clzll";
                case "mul24":
                    return "__mul24";
                case "mul64hi":
                    return "__mul64hi";
                case "mulhi":
                    return "__mulhi";
                case "umul24":
                    return "__umul24";
                case "umul64hi":
                    return "__umul64hi";
                case "umulhi":
                    return "__umulhi";
                default:
                    break;
            }
            throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED, mre.MemberName);
        }

        static string TranslateOpenCLIntegerFunc(MemberReferenceExpression mre, object data)
        {
            switch (mre.MemberName)
            {
                case "clz":
                    return "clz";
                case "clzll":
                    return "clz";
                case "popcountll":
                    return "popcountll";
                case "mul64hi":
                    return "mul_hi";
                case "mulhi":
                    return "mul_hi";
                case "umulhi":
                    return "mul_hi";
                case "umul64hi":
                    return "mul_hi";
                case "umul24":
                    return "mul24";
                default:
                    break;
            }
            throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED, mre.MemberName);
        }

        static string TranslateGMath(MemberReferenceExpression mre, object data)
        {
            switch (mre.MemberName)
            {
                case "Abs":
                    return "fabsf";
                case "Max":
                    return "fmaxf";
                case "Min":
                    return "fminf";
                default:
                    break;
            }
            return TranslateMath(mre, data) + "f";
        }

        static string TranslateGMathOpenCL(MemberReferenceExpression mre, object data)
        {
            if (mre.MemberName == "PI")
                return "M_PI_F";
            else if (mre.MemberName == "E")
                return "M_E_F";
            return TranslateMath(mre, data);
        }

        static string TranslateMathOpenCL(MemberReferenceExpression mre, object data)
        {
            if (mre.MemberName == "PI")
                return "M_PI";
            else if (mre.MemberName == "E")
                return "M_E";
            return TranslateMath(mre, data);
        }

        static string TranslateMath(MemberReferenceExpression mre, object data)
        {            
            switch (mre.MemberName)
            {
                case "Round":
                    return "rint";
                case "Truncate":
                    return "trunc";
                case "Ceiling":
                    return "ceil";
                    //Math.Sign
                case "DivRem":
                    throw new NotSupportedException(mre.MemberName);
                case "IEEERemainder":
                    throw new NotSupportedException(mre.MemberName);
                case "Sign"://http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
                    throw new NotSupportedException(mre.MemberName);
                case "BigMul":
                    throw new NotSupportedException(mre.MemberName);
                default:
                    break;
            }
            return mre.MemberName.ToLower();
        }

        static string TranslateMathE(MemberReferenceExpression mre, object data)
        {
            return Math.E.ToString(CultureInfo.InvariantCulture);
        }

        static string TranslateMathPI(MemberReferenceExpression mre, object data)
        {
            return Math.PI.ToString(CultureInfo.InvariantCulture);
        }

        static string TranslateGMathE(MemberReferenceExpression mre, object data)
        {
            return GMath.E.ToString(CultureInfo.InvariantCulture);
        }

        static string TranslateGMathPI(MemberReferenceExpression mre, object data)
        {
            return GMath.PI.ToString(CultureInfo.InvariantCulture);
        }

        static string TranslateComplexD(MemberReferenceExpression mre, object data)
        {
            switch (mre.MemberName)
            {
                case "Conj":
                    return "cuConj";
                case "Add":
                    return "cuCadd";
                case "Subtract":
                    return "cuCsub";
                case "Multiply":
                    return "cuCmul";
                case "Divide":
                    return "cuCdiv";
                case "Abs":
                    return "cuCabs";
                default:
                    throw new NotSupportedException(mre.MemberName);
            }
        }

        static string TranslateComplexF(MemberReferenceExpression mre, object data)
        {
            return TranslateComplexD(mre, data) + "f";
        }
    }
#pragma warning restore 1591
}
