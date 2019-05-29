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
using Mono.Cecil;
using ICSharpCode.NRefactory.CSharp;

using CL = Cudafy.Translator.CUDALanguage;
namespace Cudafy.Translator
{
    /// <summary>
    /// Internal use.
    /// </summary>
    public static class ExtensionMethods
    {
        private static string csCUDAFYATTRIBUTE = typeof(CudafyAttribute).Name;

        private static string csCUDAFYDUMMYATTRIBUTE = typeof(CudafyDummyAttribute).Name;

        private static string csCUDAFYIGNOREATTRIBUTE = typeof(CudafyIgnoreAttribute).Name;

        private static string csCUDAFYINLINEATTRIBUTE = typeof(CudafyInlineAttribute).Name;

#pragma warning disable 1591
        public static eCudafyType? GetCudafyType(this ICustomAttributeProvider med, out bool isDummy, out eCudafyDummyBehaviour behaviour)
        {
            bool ignore;
            eCudafyInlineMode inlineMode;
            return GetCudafyType(med, out isDummy, out ignore, out behaviour, out inlineMode);
        }

        public static eCudafyType? GetCudafyType(this ICustomAttributeProvider med, out bool isDummy, out bool ignore, out eCudafyDummyBehaviour behaviour, out eCudafyInlineMode inlineMode)
        {
            isDummy = false;
            behaviour = eCudafyDummyBehaviour.Default;
            inlineMode = eCudafyInlineMode.Auto;
            ignore = false;
            if (med is TypeDefinition)
                med = med as TypeDefinition;

            var customInline = med.CustomAttributes.Where(ca => ca.AttributeType.Name == csCUDAFYINLINEATTRIBUTE).FirstOrDefault();
            if (customInline != null)
            {
                if (customInline.ConstructorArguments.Count() > 0)
                    inlineMode = (eCudafyInlineMode)customInline.ConstructorArguments.First().Value;
            }

            var customAttr = med.CustomAttributes.Where(ca => ca.AttributeType.Name == csCUDAFYATTRIBUTE).FirstOrDefault();
            if (customAttr == null)
            {
                customAttr = med.CustomAttributes.Where(ca => ca.AttributeType.Name == csCUDAFYDUMMYATTRIBUTE).FirstOrDefault();
                isDummy = customAttr != null;
            }
            if (customAttr == null)
            {
                customAttr = med.CustomAttributes.Where(ca => ca.AttributeType.Name == csCUDAFYIGNOREATTRIBUTE).FirstOrDefault();
                ignore = true;
            }
            else
            {
                eCudafyType et = eCudafyType.Auto;
                if (customAttr.ConstructorArguments.Count() > 0)
                    et = (eCudafyType)customAttr.ConstructorArguments.First().Value;
                if (customAttr.ConstructorArguments.Count() > 1)
                    behaviour = (eCudafyDummyBehaviour)customAttr.ConstructorArguments.ElementAt(1).Value;
                return et;
            }
            return null;
        }

        //public static CudafyDummyAttribute GetCudafyDummyAttribute(this ICustomAttributeProvider med)
        //{
        //    customAttr = med.CustomAttributes.Where(ca => ca.AttributeType.Name == csCUDAFYDUMMYATTRIBUTE).FirstOrDefault();
        //}

        public static eCudafyType? GetCudafyType(this ICustomAttributeProvider med)
        {
            var customAttr = med.CustomAttributes.Where(ca => ca.AttributeType.Name == csCUDAFYATTRIBUTE).FirstOrDefault();
            if (customAttr != null)
            {
                eCudafyType et = eCudafyType.Auto;
                if (customAttr.ConstructorArguments.Count() > 0)
                    et = (eCudafyType)customAttr.ConstructorArguments.First().Value;
                return et;
            }
            else
                return null;
        }

        public static eCudafyType? GetCudafyDummyType(this ICustomAttributeProvider med)
        {
            var customAttr = med.CustomAttributes.Where(ca => ca.AttributeType.Name == csCUDAFYDUMMYATTRIBUTE).FirstOrDefault();
            if (customAttr != null)
            {
                eCudafyType et = eCudafyType.Auto;
                if (customAttr.ConstructorArguments.Count() > 0)
                    et = (eCudafyType)customAttr.ConstructorArguments.First().Value;
                return et;
            }
            else
                return null;
        }

        public static bool HasCudafyAttribute(this ICustomAttributeProvider med)
        {
            return med.HasCustomAttributes && med.CustomAttributes.Count(ca => ca.AttributeType.Name == csCUDAFYATTRIBUTE) > 0;
        }

        public static bool HasCudafyDummyAttribute(this ICustomAttributeProvider med)
        {
            return med.HasCustomAttributes && med.CustomAttributes.Count(ca => ca.AttributeType.Name == csCUDAFYDUMMYATTRIBUTE) > 0;
        }

        public static bool HasCudafyIgnoreAttribute(this ICustomAttributeProvider med)
        {
            return med.HasCustomAttributes && med.CustomAttributes.Count(ca => ca.AttributeType.Name == csCUDAFYIGNOREATTRIBUTE) > 0;
        }

        public static bool IsThreadIdVar(this MemberReferenceExpression mre)
        {
            if (mre.Target.Annotations.Count() > 0)
            {
                foreach (var ann in mre.Target.Annotations)
                {
                    PropertyDefinition pd = ann as PropertyDefinition;
                    if (pd != null && (pd.DeclaringType.ToString().Contains("Cudafy.GThread"))) 
                        return true;

                }
            }
            return false;
        }

        public static bool IsSpecialProperty(this MemberReferenceExpression mre)
        {
            IEnumerable<object> annotations = mre.Annotations.Any() ? mre.Annotations : mre.Target.Annotations;
            if (annotations.Count() > 0)
            {
                foreach (var ann in mre.Target.Annotations)
                {
                    var pd = ann as ICSharpCode.Decompiler.ILAst.ILVariable;
                    if (pd != null)
                    {
                        if (CUDALanguage.IsSpecialProperty(mre.MemberName, pd.Type.FullName))
                            return true;
                    }
                    else
                    {
                        var fd = ann as FieldDefinition;
                        if (fd != null)
                        {
                            if (CUDALanguage.IsSpecialProperty(mre.MemberName, fd.FieldType.GetType().Name))
                                return true;
                        }
                        else if (mre.NodeType == NodeType.Expression && !(mre.Target is ThisReferenceExpression))
                        {
                            if (mre.MemberName == "Length")
                                return true;
                        }
                    }
                }
            }
            return false;
        }

        public static bool IsSpecialMethod(this MemberReferenceExpression mre)
        {
            if (mre.Target.Annotations.Count() > 0)
            {
                foreach (var ann in mre.Target.Annotations)
                {
                    var pd = ann as ICSharpCode.Decompiler.ILAst.ILVariable;
                    if (pd != null)
                    {
                        if (CUDALanguage.IsSpecialMethod(mre.MemberName, pd.Type.FullName))//.GetType().Name))
                            return true;
                    }
                    else 
                    {
                        var fd = ann as FieldDefinition;
                        if (fd != null)
                        {
                            if (CUDALanguage.IsSpecialMethod(mre.MemberName, fd.FieldType.GetType().Name))
                                return true;
                        }
                    }
                }
            }
            else
                return CUDALanguage.IsSpecialMethod(mre.MemberName, mre.Target.ToString());
            return false;
        }

        public static SpecialMember GetSpecialMethod(this MemberReferenceExpression mre)
        {
            SpecialMember sm = null;
            if (mre.Target.Annotations.Count() > 0)
            {
                foreach (var ann in mre.Target.Annotations)
                {
                    var pd = ann as ICSharpCode.Decompiler.ILAst.ILVariable;
                    if (pd != null)
                    {
                        sm = CUDALanguage.GetSpecialMethod(mre.MemberName, pd.Type.FullName);//.GetType().Name))
                        if (sm != null)
                            return sm;
                    }
                    else
                    {
                        var fd = ann as FieldDefinition;
                        if (fd != null)
                        {
                            sm = CUDALanguage.GetSpecialMethod(mre.MemberName, fd.FieldType.GetType().Name);
                            if (sm != null)
                                return sm;
                        }
                    }
                }
            }
            else
                return CUDALanguage.GetSpecialMethod(mre.MemberName, mre.Target.ToString());
            return sm;
        }

        public static string TranslateSpecialProperty(this MemberReferenceExpression mre)
        {
            IEnumerable<object> annotations = mre.Target.Annotations;//mre.Annotations.Any() ? mre.Annotations : 
            if (annotations.Count() > 0)
            {
                foreach (var ann in mre.Target.Annotations)
                {
                    var pd = ann as ICSharpCode.Decompiler.ILAst.ILVariable;
                    if (pd != null)
                    {
                        SpecialMember sm = CUDALanguage.GetSpecialProperty(mre.MemberName, pd.Type.FullName);
                        if (sm != null)
                            return sm.GetTranslation(mre);
                        else
                            return mre.MemberName;
                    }
                    else
                    {
                        var fd = ann as FieldDefinition;
                        if (fd != null)
                        {
                            SpecialMember sm = CUDALanguage.GetSpecialProperty(mre.MemberName, fd.FieldType.GetType().Name);
                            return sm.GetTranslation(mre);
                        }
                        else if(mre.NodeType == NodeType.Expression)
                        {
                            if (mre.MemberName == "Length")
                                return mre.Target.ToString().Replace(".", "->") + "Len0"; // This should be made nicer
                        }
                    }
                }
            }
            throw new InvalidOperationException("SpecialProperty not found.");
        }

        public static string TranslateSpecialMethod(this MemberReferenceExpression mre, object data, out bool noSemicolon)
        {
            //callFunc = true;
            noSemicolon = false;
            if (mre.Target.Annotations.Count() > 0)
            {
                foreach (var ann in mre.Target.Annotations)
                {
                    var pd = ann as ICSharpCode.Decompiler.ILAst.ILVariable;
                    if (pd != null)
                    {
                        SpecialMember sm = CUDALanguage.GetSpecialMethod(mre.MemberName, pd.Type.FullName);
                        //callFunc = sm.CallFunction;
                        noSemicolon = sm.NoSemicolon;
                        return sm.GetTranslation(mre, data);
                    }
                    else 
                    {
                        var fd = ann as FieldDefinition;
                        if (fd != null)
                        {
                            SpecialMember sm = CUDALanguage.GetSpecialMethod(mre.MemberName, fd.FieldType.GetType().Name);
                            return sm.GetTranslation(mre, data);
                        }
                    }
                }
            }
            else
            {
                SpecialMember sm = CUDALanguage.GetSpecialMethod(mre.MemberName, mre.Target.ToString());// .GetType().Name);
                noSemicolon = sm.NoSemicolon;
                return sm.GetTranslation(mre, data);
            }
            throw new InvalidOperationException("SpecialMethod not found.");
        }

        public static bool TranslateArrayLengthAndRank(this MemberReferenceExpression mre, out string length, out string rank)
        {
            length = string.Empty;
            rank = string.Empty;
            if (mre.Target.Annotations.Count() > 0)
            {
                foreach (var ann in mre.Target.Annotations)
                {
                    var pd = ann as ICSharpCode.Decompiler.ILAst.ILVariable;
                    if (pd != null)
                    {
                        var at = pd.Type as Mono.Cecil.ArrayType;
                        if (at != null)
                        {
                            rank = at.Rank.ToString();
                            string s = string.Empty;
                            for (int i = 0; i < at.Rank; i++)
                            {
                                s += string.Format("{0}Len{1}", mre.Target, i);
                                if (i < at.Rank - 1)
                                    s += " * ";
                            }
                            length = s;
                            return true;
                        }
                    }
                    else
                    {
                        var fd = ann as FieldDefinition;
                        var at = fd.FieldType as Mono.Cecil.ArrayType;
                        if (at != null)
                        {
                            rank = at.Rank.ToString();
                            string s = string.Empty;
                            for (int i = 0; i < at.Rank; i++)
                            {
                                s += string.Format("{0}Len{1}", (mre.Target as MemberReferenceExpression).MemberName, i);
                                if (i < at.Rank - 1)
                                    s += " * ";
                            }
                            length = s;
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        public static string TranslateArrayGetLength(this MemberReferenceExpression mre, object data)
        {
            var ex = data as InvocationExpression;
            if (ex == null)
                throw new ArgumentNullException("data as InvocationExpression");
            PrimitiveExpression pe = ex.Arguments.First() as PrimitiveExpression;
            if (pe == null)
                throw new ArgumentNullException("PrimitiveExpression pe");
            foreach (var ann in mre.Target.Annotations)
            {
                var pd = ann as ICSharpCode.Decompiler.ILAst.ILVariable;
                if (pd != null)
                {
                    var at = pd.Type as Mono.Cecil.ArrayType;
                    if (at != null)
                    {
                        return string.Format("{0}Len{1}", mre.Target, pe.Value);
                    }
                }
                else 
                {
                    var fd = ann as FieldDefinition;
                    if (fd != null)
                    {
                        var at = fd.FieldType as Mono.Cecil.ArrayType;
                        if (at != null)
                        {
                            return string.Format("{0}Len{1}", mre.Target.ToString().Replace(".", "->"), pe.Value);
                        }
                    }   
                }
            }
            return string.Empty;
        }

        public static string TranslateToPrintF(this MemberReferenceExpression mre, object data)
        {
            TextWriter output = new StringWriter();
            CUDAOutputVisitor visitor = new CUDAOutputVisitor(new TextWriterOutputFormatter(output), new CSharpFormattingOptions());
            var ex = data as InvocationExpression;
            if (ex == null)
                throw new ArgumentNullException("data as InvocationExpression");

            bool isWriteLine = ((MemberReferenceExpression)ex.Target).MemberName.StartsWith("WriteLine");
            bool hasIfCondition = ((MemberReferenceExpression)ex.Target).MemberName.EndsWith("If");
            List<Expression> arguments = ex.Arguments.ToList();
            int i = 0;

            if (hasIfCondition)
            {
                i = -1;
                output.Write("if(");
                arguments[0].AcceptVisitor(visitor, data);
                output.Write(") ");
            }

            output.Write("printf(");
            foreach (var arg in arguments)
            {
                if (i == -1) { /* skip it, it was an if condition */ }
                else if (i == 0)
                {
                    if (!(arg is PrimitiveExpression))
                        throw new CudafyLanguageException("When using Debug.Write" + (isWriteLine ? "Line" : "") + (hasIfCondition ? "If" : "") + "() the first parameter must be a string literal");

                    string strFormat = arg.ToString();
                    if (!strFormat.StartsWith("\""))
                        throw new CudafyLanguageException("When using Debug.Write" + (isWriteLine ? "Line" : "") + "() the first parameter must be a string literal");

                    if (hasIfCondition)
                    {
                        // Since debug if condition messages don't support parameters, escape any % to avoid ErrorUnknown
                        strFormat = strFormat.Replace("%", "%%");
                    }

                    // NOTE: unless you know the type of each parameter passed to printf, 
                    // then I don't see a good way to convert String.Format() rules to printf() rules
                    //strFormat = CUDALanguage.TranslateStringFormat(strFormat);

                    if (isWriteLine)
                        strFormat = strFormat.Insert(strFormat.Length - 1, "\\n");

                    output.Write(strFormat);
                }
                else if (hasIfCondition)
                {
                    // Skip any other parameters if they are there, debug if conditions don't support formatting
                }
                else if (arg is ArrayCreateExpression)
                {
                    var arrayCreateExpression = arg as ArrayCreateExpression;
                    foreach (var arrayArg in arrayCreateExpression.Initializer.Elements)
                    {
                        output.Write(',');
                        arrayArg.AcceptVisitor(visitor, data);
                    }
                }
                else
                {
                    output.Write(',');
                    arg.AcceptVisitor(visitor, data);
                }
                i++;
            }
            output.Write(")");
            return output.ToString();
        }

        public static string TranslateAssert(this MemberReferenceExpression mre, object data)
        {
            TextWriter output = new StringWriter();
            CUDAOutputVisitor visitor = new CUDAOutputVisitor(new TextWriterOutputFormatter(output), new CSharpFormattingOptions());
            var ex = data as InvocationExpression;
            if (ex == null)
                throw new ArgumentNullException("data as InvocationExpression");

            bool isWriteLine = ((MemberReferenceExpression)ex.Target).MemberName == "WriteLine";
            int i = 0;
            List<Expression> arguments = ex.Arguments.ToList();
            if (arguments.Count > 1)
            {
                // An If statement and Debug.WriteLine should go along with this assert
                output.Write("if(!(");
                arguments[0].AcceptVisitor(visitor, data);
                output.Write(")) {");

                bool shortMessageIsNull = arguments[1] is NullReferenceExpression;
                // Argument #2 is a unformatted short message, print it out without any parameters
                if (!(arguments[1] is PrimitiveExpression || shortMessageIsNull))
                    throw new CudafyLanguageException("When using Debug.Assert() the second parameter must be a string literal");
                string shortMessage = arguments[1].ToString();
                if (!shortMessageIsNull && !shortMessage.StartsWith("\""))
                    throw new CudafyLanguageException("When using Debug.Assert() the second parameter must be a string literal");

                // Skip printing this first message if the value is NULL. 
                // Why? Because I assume the user wants to print the message with parameters without the hassle of two messages
                if (!shortMessageIsNull)
                {
                    // Insert a newline into the end of the message
                    shortMessage = shortMessage.Insert(shortMessage.Length - 1, "\\n");
                    // Since this message does not support parameters, escape any % to avoid ErrorUnknown
                    shortMessage = shortMessage.Replace("%", "%%");

                    output.Write("printf(" + shortMessage + ");");
                }

                if (arguments.Count > 2)
                {
                    // Argument #3 is an un/formated detailed message with or without parameters
                    if (!(arguments[2] is PrimitiveExpression))
                        throw new CudafyLanguageException("When using Debug.Assert() the third parameter must be a string literal");
                    string detailedMessageFormat = arguments[2].ToString();
                    if (!detailedMessageFormat.StartsWith("\""))
                        throw new CudafyLanguageException("When using Debug.Assert() the third parameter must be a string literal");

                    // Insert a newline into the end of the message
                    detailedMessageFormat = detailedMessageFormat.Insert(detailedMessageFormat.Length - 1, "\\n");
                    if (arguments.Count > 3)
                    {
                        output.Write("printf(" + detailedMessageFormat);
                        if (arguments[3] is ArrayCreateExpression)
                        {
                            var arrayCreateExpression = arguments[3] as ArrayCreateExpression;
                            foreach (var arrayArg in arrayCreateExpression.Initializer.Elements)
                            {
                                output.Write(',');
                                arrayArg.AcceptVisitor(visitor, data);
                            }
                        }
                        else
                        {
                            output.Write(',');
                            arguments[3].AcceptVisitor(visitor, data);
                        }
                        output.Write(");");
                    }
                    else if (arguments.Count == 3)
                    {
                        // Since this message does not support parameters, escape any % to avoid ErrorUnknown
                        detailedMessageFormat = detailedMessageFormat.Replace("%", "%%");
                        output.Write("printf(" + detailedMessageFormat + ");");
                    }
                }
                output.Write("assert(0);");
                output.Write("}");
            }
            else
            {
                output.Write("assert(");
                arguments[0].AcceptVisitor(visitor, data);
                output.Write(")");
            }
            return output.ToString();
        }

        public static bool IsSyncThreads(this MemberReferenceExpression mre)
        {
            return mre.MemberName == CL.csSyncThreads;
        }

        public static bool IsSyncThreadsCount(this MemberReferenceExpression mre)
        {
            return mre.MemberName == CL.csSyncThreadsCount;
        }

        public static bool IsAll(this MemberReferenceExpression mre)
        {
            return mre.MemberName == CL.csAll;
        }

        public static bool IsAny(this MemberReferenceExpression mre)
        {
            return mre.MemberName == CL.csAny;
        }

        public static bool IsBallot(this MemberReferenceExpression mre)
        {
            return mre.MemberName == CL.csBallot;
        }

        public static bool IsAllocateShared(this MemberReferenceExpression mre)
        {
            return mre.MemberName == CL.csAllocateShared;
        }
#pragma warning restore 1591
    }
}
