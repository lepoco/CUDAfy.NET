﻿/*
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
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

using ICSharpCode.NRefactory.PatternMatching;
using ICSharpCode.NRefactory.TypeSystem;
using ICSharpCode.NRefactory.CSharp;
using Cudafy;
using CL = Cudafy.Translator.CUDALanguage;

namespace Cudafy.Translator
{
#pragma warning disable 1591

    /// <summary>
	/// Outputs the AST.
	/// </summary>
	public class CUDAOutputVisitor : IAstVisitor<object, object>, IPatternAstVisitor<object, object>
	{
		readonly IOutputFormatter formatter;
		readonly CSharpFormattingOptions policy;
		
		readonly Stack<AstNode> containerStack = new Stack<AstNode>();
		readonly Stack<AstNode> positionStack = new Stack<AstNode>();

        private List<string> _methodDeclarations = new List<string>();
		
		/// <summary>
		/// Used to insert the minimal amount of spaces so that the lexer recognizes the tokens that were written.
		/// </summary>
		LastWritten lastWritten;
		
		enum LastWritten
		{
			Whitespace,
			Other,
			KeywordOrIdentifier,
			Plus,
			Minus,
			Ampersand,
			QuestionMark,
			Division
		}
		
		public CUDAOutputVisitor(TextWriter textWriter, CSharpFormattingOptions formattingPolicy)
		{
			if (textWriter == null)
				throw new ArgumentNullException("textWriter");
			if (formattingPolicy == null)
				throw new ArgumentNullException("formattingPolicy");
			this.formatter = new TextWriterOutputFormatter(textWriter);
			this.policy = formattingPolicy;
            DisableSmartArray = false;
		}

        public CUDAOutputVisitor(IOutputFormatter formatter, CSharpFormattingOptions formattingPolicy)
		{
			if (formatter == null)
				throw new ArgumentNullException("formatter");
			if (formattingPolicy == null)
				throw new ArgumentNullException("formattingPolicy");
			this.formatter = formatter;
			this.policy = formattingPolicy;
            DisableSmartArray = false;
		}
		
		#region StartNode/EndNode
		void StartNode(AstNode node)
		{
			// Ensure that nodes are visited in the proper nested order.
			// Jumps to different subtrees are allowed only for the child of a placeholder node.
			Debug.Assert(containerStack.Count == 0 || node.Parent == containerStack.Peek() || containerStack.Peek().NodeType == NodeType.Pattern);
			if (positionStack.Count > 0)
				WriteSpecialsUpToNode(node);
			containerStack.Push(node);
			positionStack.Push(node.FirstChild);
			formatter.StartNode(node);
		}
		
		object EndNode(AstNode node)
		{
			Debug.Assert(node == containerStack.Peek());
			AstNode pos = positionStack.Pop();
			Debug.Assert(pos == null || pos.Parent == node);
			WriteSpecials(pos, null);
			containerStack.Pop();
			formatter.EndNode(node);
			return null;
		}
		#endregion
		
		#region WriteSpecials
		/// <summary>
		/// Writes all specials from start to end (exclusive). Does not touch the positionStack.
		/// </summary>
		void WriteSpecials(AstNode start, AstNode end)
		{
			for (AstNode pos = start; pos != end; pos = pos.NextSibling) {
				if (pos.Role == AstNode.Roles.Comment) {
					pos.AcceptVisitor(this, null);
				}
			}
		}
		
		/// <summary>
		/// Writes all specials between the current position (in the positionStack) and the next
		/// node with the specified role. Advances the current position.
		/// </summary>
		void WriteSpecialsUpToRole(Role role)
		{
			for (AstNode pos = positionStack.Peek(); pos != null; pos = pos.NextSibling) {
				if (pos.Role == role) {
					WriteSpecials(positionStack.Pop(), pos);
					positionStack.Push(pos);
					break;
				}
			}
		}
		
		/// <summary>
		/// Writes all specials between the current position (in the positionStack) and the specified node.
		/// Advances the current position.
		/// </summary>
		void WriteSpecialsUpToNode(AstNode node)
		{
			for (AstNode pos = positionStack.Peek(); pos != null; pos = pos.NextSibling) {
				if (pos == node) {
					WriteSpecials(positionStack.Pop(), pos);
					positionStack.Push(pos);
					break;
				}
			}
		}
		
		void WriteSpecialsUpToRole(Role role, AstNode nextNode)
		{
			// Look for the role between the current position and the nextNode.
			for (AstNode pos = positionStack.Peek(); pos != null && pos != nextNode; pos = pos.NextSibling) {
				if (pos.Role == AstNode.Roles.Comma) {
					WriteSpecials(positionStack.Pop(), pos);
					positionStack.Push(pos);
					break;
				}
			}
		}
		#endregion
		
		#region Comma
		/// <summary>
		/// Writes a comma.
		/// </summary>
		/// <param name="nextNode">The next node after the comma.</param>
		/// <param name="noSpaceAfterComma">When set prevents printing a space after comma.</param>
		void Comma(AstNode nextNode, bool noSpaceAfterComma = false)
		{
			WriteSpecialsUpToRole(AstNode.Roles.Comma, nextNode);
			Space(policy.SpaceBeforeBracketComma); // TODO: Comma policy has changed.
			formatter.WriteToken(",");
			lastWritten = LastWritten.Other;
			Space(!noSpaceAfterComma && policy.SpaceAfterBracketComma); // TODO: Comma policy has changed.
		}
		
		public void WriteCommaSeparatedList(IEnumerable<AstNode> list)
		{
			bool isFirst = true;
            bool isIgnored = false;
			foreach (AstNode node in list) 
            {
                isIgnored = (node is ArrayCreateExpression) || IsIgnored(node);
                if (isIgnored)
                    continue;
				if (isFirst) 
                {
					isFirst = false;
				} 
                else 
                {
                    if (!isIgnored)
					    Comma(node);
				}
				    node.AcceptVisitor(this, null);
			}
		}

        void WriteCommaSeparatedList(IEnumerable<string> list, bool isFirst = true)
        {
            foreach (string s in list)
            {
                if (isFirst)
                {
                    isFirst = false;
                }
                else
                {
                    formatter.WriteToken(",");
                }
                formatter.WriteKeyword(s);
            }
        }

        bool IsIgnored(AstNode node)
        {
            var ie = node as IdentifierExpression;
            if (ie != null)
            {
                if (ie.Annotations.Count() > 0)
                {
                    var typeRef = ((object[])(node.Annotations))[0] as ICSharpCode.Decompiler.ILAst.ILVariable;//as Mono.Cecil.TypeReference;
                    if (typeRef != null && (typeRef.Type.Name == "GThread"))//+		[0]	{thread}	object {ICSharpCode.Decompiler.ILAst.ILVariable}
                        return true;

                }
            }
            var pe = node as PrimitiveExpression;
            if (pe != null)
            {
                if (pe.Value.ToString().StartsWith("IGNOREMEE01B67F3"))
                    return true;
            }
            return false;

        }

        void WriteCUDAParametersCommaSeparatedList(IEnumerable<AstNode> list)
        {
            bool isFirst = true;
            foreach (AstNode node in list)
            {
                ParameterDeclaration pd = node as ParameterDeclaration;
                if(pd != null && (pd.Type.ToString() == typeof(GThread).Name))
                    continue;

                if (isFirst)
                {
                    isFirst = false;
                }
                else
                {
                    Comma(node);
                }
                if (CudafyTranslator.LanguageSpecifics.Language == eLanguage.OpenCL)
                {
                    if (pd.Type is SimpleType)
                        WriteKeyword("struct");
                }
                node.AcceptVisitor(this, null);
            }
            if (CudafyTranslator.LanguageSpecifics.Language == eLanguage.OpenCL)
            {
                foreach (var kci in CUDALanguage.GetConstants())
                {
                    var elemType = kci.Information.FieldType.GetElementType();
                    string keyword = elemType.Name;
                    var primitiveType = CUDAAstBuilder.ConvertToPrimitiveType(keyword) as PrimitiveType;
                    string name = primitiveType.Keyword;
                    string structType = elemType.IsPrimitive ? "" : "struct";
                    WriteKeyword(string.Format(", {0} {1} {2}* {3}", CudafyTranslator.LanguageSpecifics.ConstantModifier, structType, name, kci.Name));
                }
            }
        }
		
		void WriteCommaSeparatedListInParenthesis(IEnumerable<AstNode> list, bool spaceWithin, string[] additionalArguments = null, bool writeConstants = false)
		{
			LPar();
			if (list.Any()) {
				Space(spaceWithin);
				WriteCommaSeparatedList(list);
                if (additionalArguments != null)
                    WriteCommaSeparatedList(additionalArguments, list.Count() == 0);
				Space(spaceWithin);
			}
            if (writeConstants && CudafyTranslator.LanguageSpecifics.Language == eLanguage.OpenCL)
            {
                foreach (var kci in CUDALanguage.GetConstants())
                {
                    //string keyword = kci.Information.FieldType.GetElementType().Name;
                    //var primitiveType = CUDAAstBuilder.ConvertToPrimitiveType(keyword) as PrimitiveType;
                    //string name = primitiveType.Keyword;
                    WriteKeyword(string.Format(", {0}", kci.Name));
                }
            }
			RPar();
		}

        void WriteCUDAParametersInParenthesis(IEnumerable<AstNode> list, bool spaceWithin)
        {
            LPar();
            if (list.Any())
            {
                Space(spaceWithin);
                WriteCUDAParametersCommaSeparatedList(list);
                Space(spaceWithin);
            }
            RPar();
        }
		
		#if DOTNET35
		void WriteCommaSeparatedList(IEnumerable<VariableInitializer> list)
		{
			WriteCommaSeparatedList(list.SafeCast<VariableInitializer, AstNode>());
		}
		
		void WriteCommaSeparatedList(IEnumerable<AstType> list)
		{
			WriteCommaSeparatedList(list.SafeCast<AstType, AstNode>());
		}
		
		void WriteCommaSeparatedListInParenthesis(IEnumerable<Expression> list, bool spaceWithin)
		{
			WriteCommaSeparatedListInParenthesis(list.SafeCast<Expression, AstNode>(), spaceWithin);
		}
		
		void WriteCommaSeparatedListInParenthesis(IEnumerable<ParameterDeclaration> list, bool spaceWithin)
		{
			WriteCommaSeparatedListInParenthesis(list.SafeCast<ParameterDeclaration, AstNode>(), spaceWithin);
		}

		#endif

		void WriteCommaSeparatedListInBrackets(IEnumerable<ParameterDeclaration> list, bool spaceWithin)
		{
			WriteToken("[", AstNode.Roles.LBracket);
			if (list.Any()) {
				Space(spaceWithin);
				WriteCommaSeparatedList(list.SafeCast<ParameterDeclaration, AstNode>());
				Space(spaceWithin);
			}
			WriteToken("]", AstNode.Roles.RBracket);
		}

		void WriteCommaSeparatedListInBrackets(IEnumerable<Expression> list)
		{
			WriteToken ("[", AstNode.Roles.LBracket);
			if (list.Any ()) {
				Space (policy.SpacesWithinBrackets);
				WriteCommaSeparatedList (list.SafeCast<Expression, AstNode> ());
				Space (policy.SpacesWithinBrackets);
			}
			WriteToken ("]", AstNode.Roles.RBracket);
		}

        void WriteCommaSeparatedIndexerListInBrackets(IEnumerable<Expression> list)
        {
            WriteToken("[", AstNode.Roles.LBracket);
            if (list.Any())
            {
                Space(policy.SpacesWithinBrackets);
                WriteCommaSeparatedIndexerList(list.SafeCast<Expression, AstNode>());
                Space(policy.SpacesWithinBrackets);
            }
            WriteToken("]", AstNode.Roles.RBracket);
        }

        void WriteCommaSeparatedIndexerList(IEnumerable<AstNode> list)
        {
            bool isFirst = true;
            int dims = list.Count();
            int i = 0;
            foreach (AstNode node in list)
            {
                if (isFirst)
                {
                    isFirst = false;
                }
                else
                {
                    //CommaEx(node);
                    int pos = i;
                    while (pos < dims)
                    {
                        var trgt = (node.Parent as IndexerExpression).Target;
                        string target = (node.Parent as IndexerExpression).Target.ToString();
                        if (trgt is MemberReferenceExpression)
                            target = (trgt as MemberReferenceExpression).MemberName;
                        string s = string.Format(" * {0}Len{1}", target, pos); 
                        formatter.WriteKeyword(s);
                        pos++;
                    }
                     
                    formatter.WriteKeyword(" + ");
                      
                }
                formatter.WriteToken("("); 
                node.AcceptVisitor(this, null);
                formatter.WriteToken(")");
                i++;
            }
        }

        void CommaEx(AstNode nextNode, bool noSpaceAfterComma = false)
        {
            WriteSpecialsUpToRole(AstNode.Roles.Comma, nextNode);
            Space(policy.SpaceBeforeBracketComma); // TODO: Comma policy has changed.
            formatter.WriteToken("");
            lastWritten = LastWritten.Other;
            Space(!noSpaceAfterComma && policy.SpaceAfterBracketComma); // TODO: Comma policy has changed.
        }
		#endregion
		
		#region Write tokens
		/// <summary>
		/// Writes a keyword, and all specials up to
		/// </summary>
		void WriteKeyword(string keyword, Role<CSharpTokenNode> tokenRole = null)
		{
			WriteSpecialsUpToRole(tokenRole ?? AstNode.Roles.Keyword);
			if (lastWritten == LastWritten.KeywordOrIdentifier)
				formatter.Space();
			formatter.WriteKeyword(keyword);
			lastWritten = LastWritten.KeywordOrIdentifier;
		}
		
		void WriteIdentifier(string identifier, Role<Identifier> identifierRole = null)
		{
			WriteSpecialsUpToRole(identifierRole ?? AstNode.Roles.Identifier);
			if (IsKeyword(identifier, containerStack.Peek())) {
				if (lastWritten == LastWritten.KeywordOrIdentifier)
					Space(); // this space is not strictly required, so we call Space()
				//formatter.WriteToken("@");
			} else if (lastWritten == LastWritten.KeywordOrIdentifier) {
				formatter.Space(); // this space is strictly required, so we directly call the formatter
			}
			formatter.WriteIdentifier(identifier);
			lastWritten = LastWritten.KeywordOrIdentifier;
		}
		
		void WriteToken(string token, Role<CSharpTokenNode> tokenRole)
		{
			WriteSpecialsUpToRole(tokenRole);
			// Avoid that two +, - or ? tokens are combined into a ++, -- or ?? token.
			// Note that we don't need to handle tokens like = because there's no valid
			// C# program that contains the single token twice in a row.
			// (for +, - and &, this can happen with unary operators;
			// for ?, this can happen in "a is int? ? b : c" or "a as int? ?? 0";
			// and for /, this can happen with "1/ *ptr" or "1/ //comment".)
			if (lastWritten == LastWritten.Plus && token[0] == '+'
			    || lastWritten == LastWritten.Minus && token[0] == '-'
			    || lastWritten == LastWritten.Ampersand && token[0] == '&'
			    || lastWritten == LastWritten.QuestionMark && token[0] == '?'
			    || lastWritten == LastWritten.Division && token[0] == '*')
			{
				formatter.Space();
			}
			formatter.WriteToken(token);
			if (token == "+")
				lastWritten = LastWritten.Plus;
			else if (token == "-")
				lastWritten = LastWritten.Minus;
			else if (token == "&")
				lastWritten = LastWritten.Ampersand;
			else if (token == "?")
				lastWritten = LastWritten.QuestionMark;
			else if (token == "/")
				lastWritten = LastWritten.Division;
			else
				lastWritten = LastWritten.Other;
		}
		
		void LPar()
		{
			WriteToken("(", AstNode.Roles.LPar);
		}
		
		void RPar()
		{
			WriteToken(")", AstNode.Roles.LPar);
		}
		
		/// <summary>
		/// Marks the end of a statement
		/// </summary>
		void Semicolon()
		{
			Role role = containerStack.Peek().Role; // get the role of the current node
			if (!(role == ForStatement.InitializerRole || role == ForStatement.IteratorRole || role == UsingStatement.ResourceAcquisitionRole)) 
            {
				if(!_noSemicolon)
                    WriteToken(";", AstNode.Roles.Semicolon);
                _noSemicolon = false;
				NewLine();
			}
		}
		
		/// <summary>
		/// Writes a space depending on policy.
		/// </summary>
		void Space(bool addSpace = true)
		{
			if (addSpace) {
				formatter.Space();
				lastWritten = LastWritten.Whitespace;
			}
		}
		
		void NewLine()
		{
			formatter.NewLine();
			lastWritten = LastWritten.Whitespace;
		}
		
		void OpenBrace(BraceStyle style)
		{
			WriteSpecialsUpToRole(AstNode.Roles.LBrace);
			formatter.OpenBrace(style);
			lastWritten = LastWritten.Other;
		}
		
		void CloseBrace(BraceStyle style)
		{
			WriteSpecialsUpToRole(AstNode.Roles.RBrace);
			formatter.CloseBrace(style);
			lastWritten = LastWritten.Other;
		}
		#endregion
		
		#region IsKeyword Test
		static readonly HashSet<string> unconditionalKeywords = new HashSet<string> {
			"abstract", "as", "base", "bool", "break", "byte", "case", "catch",
			"char", "checked", "class", "const", "continue", "decimal", "default", "delegate",
			"do", "double", "else", "enum", "event", "explicit", "extern", "false",
			"finally", "fixed", "float", "for", "foreach", "goto", "if", "implicit",
			"in", "int", "interface", "internal", "is", "lock", "long", "namespace",
			"new", "null", "object", "operator", "out", "override", "params", "private",
			"protected", "public", "readonly", "ref", "return", "sbyte", "sealed", "short",
			"sizeof", "stackalloc", "static", "string", "struct", "switch", "this", "throw",
			"true", "try", "typeof", "uint", "ulong", "unchecked", "unsafe", "ushort",
			"using", "virtual", "void", "volatile", "while"
		};
		
		static readonly HashSet<string> queryKeywords = new HashSet<string> {
			"from", "where", "join", "on", "equals", "into", "let", "orderby",
			"ascending", "descending", "select", "group", "by"
		};
		
		/// <summary>
		/// Determines whether the specified identifier is a keyword in the given context.
		/// </summary>
		public static bool IsKeyword(string identifier, AstNode context)
		{
			if (unconditionalKeywords.Contains(identifier))
				return true;
			if (context.Ancestors.Any(a => a is QueryExpression)) {
				if (queryKeywords.Contains(identifier))
					return true;
			}
			return false;
		}
		#endregion
		
		#region Write constructs
		void WriteTypeArguments(IEnumerable<AstType> typeArguments)
		{
			if (typeArguments.Any()) {
				WriteToken("<", AstNode.Roles.LChevron);
				WriteCommaSeparatedList(typeArguments);
				WriteToken(">", AstNode.Roles.RChevron);
			}
		}
		
		void WriteTypeParameters(IEnumerable<TypeParameterDeclaration> typeParameters)
		{
			if (typeParameters.Any()) {
				WriteToken("<", AstNode.Roles.LChevron);
				WriteCommaSeparatedList(typeParameters.SafeCast<TypeParameterDeclaration, AstNode>());
				WriteToken(">", AstNode.Roles.RChevron);
			}
		}
		
		void WriteModifiers(IEnumerable<CSharpModifierToken> modifierTokens)
		{
			foreach (CSharpModifierToken modifier in modifierTokens) {
				modifier.AcceptVisitor(this, null);
			}
		}

		
		void WriteQualifiedIdentifier(IEnumerable<Identifier> identifiers)
		{
			bool first = true;
			foreach (Identifier ident in identifiers) {
				if (first) {
					first = false;
					if (lastWritten == LastWritten.KeywordOrIdentifier)
						formatter.Space();
				} else {
					WriteSpecialsUpToRole(AstNode.Roles.Dot, ident);
					formatter.WriteToken(".");
					lastWritten = LastWritten.Other;
				}
				WriteSpecialsUpToNode(ident);
				formatter.WriteIdentifier(ident.Name);
				lastWritten = LastWritten.KeywordOrIdentifier;
			}
		}
		
		void WriteEmbeddedStatement(Statement embeddedStatement)
		{
			if (embeddedStatement.IsNull)
				return;
			BlockStatement block = embeddedStatement as BlockStatement;
			if (block != null)
				VisitBlockStatement(block, null);
			else
				embeddedStatement.AcceptVisitor(this, null);
		}
		
		void WriteMethodBody(BlockStatement body)
		{
			if (body.IsNull)
				Semicolon();
			else
				VisitBlockStatement(body, null);
		}
		
		void WriteAttributes(IEnumerable<AttributeSection> attributes)
		{
			foreach (AttributeSection attr in attributes) {
				attr.AcceptVisitor(this, null);
			}
		}
		
		void WritePrivateImplementationType(AstType privateImplementationType)
		{
			if (!privateImplementationType.IsNull) {
				privateImplementationType.AcceptVisitor(this, null);
				WriteToken(".", AstNode.Roles.Dot);
			}
		}
		#endregion
		
		#region Expressions
		public object VisitAnonymousMethodExpression(AnonymousMethodExpression anonymousMethodExpression, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Anonymous methods");
            //StartNode(anonymousMethodExpression);
            //WriteKeyword("delegate");
            //if (anonymousMethodExpression.HasParameterList) {
            //    Space(policy.SpaceBeforeMethodDeclarationParentheses);
            //    WriteCommaSeparatedListInParenthesis(anonymousMethodExpression.Parameters, policy.SpaceWithinMethodDeclarationParentheses);
            //}
            //anonymousMethodExpression.Body.AcceptVisitor(this, data);
            //return EndNode(anonymousMethodExpression);
		}
		
		public object VisitUndocumentedExpression(UndocumentedExpression undocumentedExpression, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Undocumented expressions");
            //StartNode(undocumentedExpression);
            //switch (undocumentedExpression.UndocumentedExpressionType) {
            //case UndocumentedExpressionType.ArgList:
            //case UndocumentedExpressionType.ArgListAccess:
            //WriteKeyword("__arglist");
            //    break;
            //case UndocumentedExpressionType.MakeRef:
            //    WriteKeyword("__makeref");
            //    break;
            //case UndocumentedExpressionType.RefType:
            //    WriteKeyword("__reftype");
            //    break;
            //case UndocumentedExpressionType.RefValue:
            //    WriteKeyword("__refvalue");
            //    break;
            //}
            //if (undocumentedExpression.Arguments.Count > 0) {
            //    Space(policy.SpaceBeforeMethodCallParentheses);
            //    WriteCommaSeparatedListInParenthesis(undocumentedExpression.Arguments, policy.SpaceWithinMethodCallParentheses);
            //}
            //return EndNode(undocumentedExpression);
		}
		
		public object VisitArrayCreateExpression(ArrayCreateExpression arrayCreateExpression, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Array create expressions");
            //StartNode(arrayCreateExpression);
            //WriteKeyword("new");
            //arrayCreateExpression.Type.AcceptVisitor(this, data);
            //WriteCommaSeparatedListInBrackets(arrayCreateExpression.Arguments);
            //foreach (var specifier in arrayCreateExpression.AdditionalArraySpecifiers)
            //    specifier.AcceptVisitor(this, data);
            //arrayCreateExpression.Initializer.AcceptVisitor(this, data);
            //return EndNode(arrayCreateExpression);
            //return "";
		}
		
		public object VisitArrayInitializerExpression(ArrayInitializerExpression arrayInitializerExpression, object data)
		{
			StartNode(arrayInitializerExpression);
			BraceStyle style;
			if (policy.PlaceArrayInitializersOnNewLine == ArrayInitializerPlacement.AlwaysNewLine)
				style = BraceStyle.NextLine;
			else
				style = BraceStyle.EndOfLine;
			OpenBrace(style);
			bool isFirst = true;
			foreach (AstNode node in arrayInitializerExpression.Elements) {
				if (isFirst) {
					isFirst = false;
				} else {
					Comma(node);
					NewLine();
				}
				node.AcceptVisitor(this, null);
			}
			NewLine();
			CloseBrace(style);
			return EndNode(arrayInitializerExpression);
		}
		
		public object VisitAsExpression(AsExpression asExpression, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "As expressions");
            //StartNode(asExpression);
            //asExpression.Expression.AcceptVisitor(this, data);
            //Space();
            //WriteKeyword("as");
            //Space();
            //asExpression.Type.AcceptVisitor(this, data);
            //return EndNode(asExpression);
		}
		
		public object VisitAssignmentExpression(AssignmentExpression assignmentExpression, object data)
		{
			StartNode(assignmentExpression);
			assignmentExpression.Left.AcceptVisitor(this, data);
			Space(policy.SpaceAroundAssignment);
			WriteToken(AssignmentExpression.GetOperatorSymbol(assignmentExpression.Operator), AssignmentExpression.OperatorRole);
			Space(policy.SpaceAroundAssignment);
			assignmentExpression.Right.AcceptVisitor(this, data);
			return EndNode(assignmentExpression);
		}
		
		public object VisitBaseReferenceExpression(BaseReferenceExpression baseReferenceExpression, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Base reference expressions");
            //StartNode(baseReferenceExpression);
            //WriteKeyword("base");
            //return EndNode(baseReferenceExpression);
		}
		
		public object VisitBinaryOperatorExpression(BinaryOperatorExpression binaryOperatorExpression, object data)
		{
			StartNode(binaryOperatorExpression);
			binaryOperatorExpression.Left.AcceptVisitor(this, data);
			bool spacePolicy;
			switch (binaryOperatorExpression.Operator) {
				case BinaryOperatorType.BitwiseAnd:
				case BinaryOperatorType.BitwiseOr:
				case BinaryOperatorType.ExclusiveOr:
					spacePolicy = policy.SpaceAroundBitwiseOperator;
					break;
				case BinaryOperatorType.ConditionalAnd:
				case BinaryOperatorType.ConditionalOr:
					spacePolicy = policy.SpaceAroundLogicalOperator;
					break;
				case BinaryOperatorType.GreaterThan:
				case BinaryOperatorType.GreaterThanOrEqual:
				case BinaryOperatorType.LessThanOrEqual:
				case BinaryOperatorType.LessThan:
					spacePolicy = policy.SpaceAroundRelationalOperator;
					break;
				case BinaryOperatorType.Equality:
				case BinaryOperatorType.InEquality:
					spacePolicy = policy.SpaceAroundEqualityOperator;
					break;
				case BinaryOperatorType.Add:
				case BinaryOperatorType.Subtract:
					spacePolicy = policy.SpaceAroundAdditiveOperator;
					break;
				case BinaryOperatorType.Multiply:
				case BinaryOperatorType.Divide:
				case BinaryOperatorType.Modulus:
					spacePolicy = policy.SpaceAroundMultiplicativeOperator;
					break;
				case BinaryOperatorType.ShiftLeft:
				case BinaryOperatorType.ShiftRight:
					spacePolicy = policy.SpaceAroundShiftOperator;
					break;
				case BinaryOperatorType.NullCoalescing:
					spacePolicy = true;
					break;
				default:
					throw new NotSupportedException("Invalid value for BinaryOperatorType");
			}
			Space(spacePolicy);
			WriteToken(BinaryOperatorExpression.GetOperatorSymbol(binaryOperatorExpression.Operator), BinaryOperatorExpression.OperatorRole);
			Space(spacePolicy);
			binaryOperatorExpression.Right.AcceptVisitor(this, data);
			return EndNode(binaryOperatorExpression);
		}
		
		public object VisitCastExpression(CastExpression castExpression, object data)
		{
			
            StartNode(castExpression);
            string castExpressionStr = castExpression.Type.ToString(); 
            if (!castExpressionStr.Contains("IntPtr") && !castExpressionStr.Contains("object"))
            {
                LPar();
                Space(policy.SpacesWithinCastParentheses);
                castExpression.Type.AcceptVisitor(this, data);
                Space(policy.SpacesWithinCastParentheses);
                RPar();
                Space(policy.SpaceAfterTypecast);
            }
            castExpression.Expression.AcceptVisitor(this, data);
            
			return EndNode(castExpression);
		}
		
		public object VisitCheckedExpression(CheckedExpression checkedExpression, object data)
		{
            //throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Checked expressions");
            StartNode(checkedExpression);
            //WriteKeyword("checked");
            //LPar();
            //Space(policy.SpacesWithinCheckedExpressionParantheses);
            checkedExpression.Expression.AcceptVisitor(this, data);
            //Space(policy.SpacesWithinCheckedExpressionParantheses);
            //RPar();
            return EndNode(checkedExpression);
		}
		
		public object VisitConditionalExpression(ConditionalExpression conditionalExpression, object data)
		{
			StartNode(conditionalExpression);
			conditionalExpression.Condition.AcceptVisitor(this, data);
			
			Space(policy.SpaceBeforeConditionalOperatorCondition);
			WriteToken("?", ConditionalExpression.QuestionMarkRole);
			Space(policy.SpaceAfterConditionalOperatorCondition);
			
			conditionalExpression.TrueExpression.AcceptVisitor(this, data);
			
			Space(policy.SpaceBeforeConditionalOperatorSeparator);
			WriteToken(":", ConditionalExpression.ColonRole);
			Space(policy.SpaceAfterConditionalOperatorSeparator);
			
			conditionalExpression.FalseExpression.AcceptVisitor(this, data);
			
			return EndNode(conditionalExpression);
		}
		
		public object VisitDefaultValueExpression(DefaultValueExpression defaultValueExpression, object data)
		{
			StartNode(defaultValueExpression);
			
			//WriteKeyword("default");
			//LPar();
			//Space(policy.SpacesWithinTypeOfParentheses);
			defaultValueExpression.Type.AcceptVisitor(this, data);
			//Space(policy.SpacesWithinTypeOfParentheses);
			//RPar();
            LPar();
            RPar();
			return EndNode(defaultValueExpression);
		}
		
		public object VisitDirectionExpression(DirectionExpression directionExpression, object data)
		{
			StartNode(directionExpression);
			
			switch (directionExpression.FieldDirection) {
				case FieldDirection.Out:
					//WriteKeyword("out");
                    formatter.WriteToken("&");
					break;
				case FieldDirection.Ref:
					//WriteKeyword("ref");
                    formatter.WriteToken("&");
					break;
				default:
					throw new NotSupportedException("Invalid value for FieldDirection");
			}
			//Space();
			directionExpression.Expression.AcceptVisitor(this, data);
			
			return EndNode(directionExpression);
		}

        public bool DisableSmartArray { get; set; }
		
		public object VisitIdentifierExpression(IdentifierExpression identifierExpression, object data)
		{
			StartNode(identifierExpression);
            bool dereference = false;
            if ((((ICSharpCode.NRefactory.CSharp.AstNode)(identifierExpression)).Annotations.Count() > 0))
            {
                //if(identifierExpression.Annotations.First() is ByReferenceType)
                var type = ((ICSharpCode.Decompiler.ILAst.ILVariable)(((object[])(identifierExpression.Annotations))[0])).Type;
                if (type.IsByReference)// || (!type.IsValueType && !type.IsArray))
                    dereference = true;
            }
			if (dereference)
                WriteIdentifier("(*" + identifierExpression.Identifier + ")");
            else
                WriteIdentifier(identifierExpression.Identifier);
            if (identifierExpression.Parent is InvocationExpression)
            {
                foreach (var x in identifierExpression.Annotations)
                {
                    var xVar = x as ICSharpCode.Decompiler.ILAst.ILVariable;
                    if (xVar != null)
                    {
                        var array = xVar.Type as Mono.Cecil.ArrayType;
                        if (array != null && !DisableSmartArray)
                        {
                            for (int r = 0; r < array.Rank; r++)
                                formatter.WriteKeyword(string.Format(", {0}Len{1}", identifierExpression.Identifier, r));
                        }
                        else
                        {
                            var typeRef = xVar.Type as Mono.Cecil.TypeReference;
                            if (typeRef.FullName == "System.String")
                            {
                                formatter.WriteKeyword(string.Format(", {0}Len", identifierExpression.Identifier));
                            }
                        }
                    }
                }
            }
			WriteTypeArguments(identifierExpression.TypeArguments);
			return EndNode(identifierExpression);
		}

		public object VisitIndexerExpression(IndexerExpression indexerExpression, object data)
		{
			StartNode(indexerExpression);
			indexerExpression.Target.AcceptVisitor(this, data);
			Space(policy.SpaceBeforeMethodCallParentheses);
			//WriteCommaSeparatedListInBrackets(indexerExpression.Arguments);
            WriteCommaSeparatedIndexerListInBrackets(indexerExpression.Arguments);
			return EndNode(indexerExpression);
		}
		
		public object VisitInvocationExpression(InvocationExpression invocationExpression, object data)
		{
			StartNode(invocationExpression);
            var mre = invocationExpression.Target as MemberReferenceExpression;
            SpecialMember sm = null;
            if (mre != null && mre.IsSpecialMethod())
                sm = mre.GetSpecialMethod();
            invocationExpression.Target.AcceptVisitor(this, invocationExpression);//data);
            this.DisableSmartArray = CUDALanguage.DisableSmartArray;
			Space(policy.SpaceBeforeMethodCallParentheses);
            if(sm == null || sm.CallFunction)
			    WriteCommaSeparatedListInParenthesis(invocationExpression.Arguments, policy.SpaceWithinMethodCallParentheses, (sm != null ? sm.AdditionalLiteralParams : null), sm == null);
            CUDALanguage.DisableSmartArray = false;
			return EndNode(invocationExpression);
		}
		
		public object VisitIsExpression(IsExpression isExpression, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Is expressions");
            //StartNode(isExpression);
            //isExpression.Expression.AcceptVisitor(this, data);
            //Space();
            //WriteKeyword("is");
            //isExpression.Type.AcceptVisitor(this, data);
            //return EndNode(isExpression);
		}
		
		public object VisitLambdaExpression(LambdaExpression lambdaExpression, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Lambda expressions");
            //StartNode(lambdaExpression);
            //if (LambdaNeedsParenthesis(lambdaExpression)) {
            //    WriteCommaSeparatedListInParenthesis(lambdaExpression.Parameters, policy.SpaceWithinMethodDeclarationParentheses);
            //} else {
            //    lambdaExpression.Parameters.Single().AcceptVisitor(this, data);
            //}
            //Space();
            //WriteToken("=>", LambdaExpression.ArrowRole);
            //Space();
            //lambdaExpression.Body.AcceptVisitor(this, data);
            //return EndNode(lambdaExpression);
		}
		
		bool LambdaNeedsParenthesis(LambdaExpression lambdaExpression)
		{
			if (lambdaExpression.Parameters.Count != 1)
				return true;
			var p = lambdaExpression.Parameters.Single();
			return !(p.Type.IsNull && p.ParameterModifier == ParameterModifier.None);
		}

        private bool _noSemicolon = false;

		public object VisitMemberReferenceExpression(MemberReferenceExpression mre, object data)
		{
			StartNode(mre);
            if (mre.Target.ToString() == typeof(eCudafyAddressSpace).Name)
            {
                eCudafyAddressSpace cudafyQualifier = (eCudafyAddressSpace)Enum.Parse(typeof(eCudafyAddressSpace), mre.MemberName);
                string parameterQualifier = CudafyTranslator.LanguageSpecifics.GetAddressSpaceQualifier(cudafyQualifier);
                formatter.WriteIdentifier(parameterQualifier);
                _lastAddressSpace = cudafyQualifier;
                return EndNode(mre);
            }
            bool isGThread = mre.IsThreadIdVar();
            bool isFixedElementField = mre.MemberName == "FixedElementField";
            bool isSpecialProp = !isGThread && mre.IsSpecialProperty();
            bool isSpecialMethod = !isGThread && !isSpecialProp && mre.IsSpecialMethod(); // JLM
            //Console.WriteLine(mre.ToString());
            //Console.WriteLine(mre.MemberName + string.Format(" is {0}special method", isSpecialMethod ? "" : "not "));
            Debug.Assert(!mre.IsAllocateShared());
            bool callFunc = !isSpecialProp;
            string builtinTranslation = null;
            if (isSpecialMethod)
            {
                bool noSemicolon = false;
                string s = mre.TranslateSpecialMethod(data, out noSemicolon);
                _noSemicolon = noSemicolon;
                formatter.WriteIdentifier(s);
            }
            else if (isSpecialProp)
            {
                string s = mre.TranslateSpecialProperty();
                formatter.WriteIdentifier(s);
            }
            else if (!(mre.Target is IdentifierExpression && (bool)IsGThread))
            {
                bool curIsGThread = IsGThread;
                IsGThread = isGThread;
                if (!(mre.Target is ThisReferenceExpression) && !(mre.Target is TypeReferenceExpression))
                {
                    
                    if (CudafyTranslator.LanguageSpecifics.Language == eLanguage.OpenCL)
                    {
                        string target = mre.Target.ToString();
                        string dim = mre.MemberName;
                        builtinTranslation = ConvertToOpenCLBuiltinFunction(target, dim);                      
                    }
                    if (builtinTranslation == null)
                    {
                        mre.Target.AcceptVisitor(this, data);
                        if (!isFixedElementField)
						{
                        	bool indexed = (mre.Target is IndexerExpression);
                        	var target = indexed ? (mre.Target as IndexerExpression).Target : mre.Target;
                        	var variable = target.Annotations.FirstOrDefault() as ICSharpCode.Decompiler.ILAst.ILVariable;
                            var field = target.Annotations.FirstOrDefault() as Mono.Cecil.FieldDefinition;
                            Mono.Cecil.TypeReference type = null;
                            if (variable != null)
                                type = indexed ? variable.Type.GetElementType() : variable.Type;
                        	else if (field != null)
                                type = indexed ? field.FieldType.GetElementType() : field.FieldType;
                        	if (type != null && !type.IsArray && !type.IsValueType)
                            	WriteToken("->", MemberReferenceExpression.Roles.Dot);
                        	else
                            	WriteToken(".", MemberReferenceExpression.Roles.Dot);
						}
                    }
                    else
                    {
                        WriteKeyword(builtinTranslation);
                    }
                }
                IsGThread = curIsGThread;
            }

            if (!isSpecialMethod && !isSpecialProp && !isFixedElementField && builtinTranslation == null)
            {
                WriteIdentifier(mre.MemberName);
                WriteTypeArguments(mre.TypeArguments);
            }

            // Feed in array lengths
            if (mre.Parent is InvocationExpression)
            {
                foreach (var x in mre.Annotations)
                {
                    var xVar = x as Mono.Cecil.FieldReference;
                    if (xVar != null)
                    {
                        var array = xVar.FieldType as Mono.Cecil.ArrayType;
                        if (array != null && !DisableSmartArray)
                        {
                            for (int r = 0; r < array.Rank; r++)
                                formatter.WriteKeyword(string.Format(", {0}Len{1}", mre.MemberName, r));
                        }
                        else
                        {
                            var typeRef = xVar.FieldType as Mono.Cecil.TypeReference;
                            if (typeRef.FullName == "System.String")
                            {
                                formatter.WriteKeyword(string.Format(", {0}Len", mre.MemberName));
                            }
                        }
                    }
                }
            }

			return EndNode(mre);
		}

        private string ConvertToOpenCLBuiltinFunction(string target, string dim)
        {
            string function = null;
            int dimension = 0;
            if (dim == "y")
                dimension = 1;
            else if (dim == "z")
                dimension = 2;
            if (target.Contains(".threadIdx"))
                function = "get_local_id";
            else if (target.Contains(".blockDim"))
                function = "get_local_size";
            else if (target.Contains(".blockIdx"))
                function = "get_group_id";
            else if (target.Contains(".gridDim"))
                function = "get_num_groups";
            else
                return null;
            string call = string.Format("{0}({1})", function, dimension);
            return call;
        }

        private bool IsGThread { get; set; }
		
		public object VisitNamedArgumentExpression(NamedArgumentExpression namedArgumentExpression, object data)
		{
			StartNode(namedArgumentExpression);
			WriteIdentifier(namedArgumentExpression.Identifier);
			if (namedArgumentExpression.Parent is ArrayInitializerExpression) {
				Space();
				WriteToken("=", NamedArgumentExpression.Roles.Assign);
			} else {
				WriteToken(":", NamedArgumentExpression.Roles.Colon);
			}
			Space();
			namedArgumentExpression.Expression.AcceptVisitor(this, data);
			return EndNode(namedArgumentExpression);
		}
		
		public object VisitNullReferenceExpression(NullReferenceExpression nullReferenceExpression, object data)
		{
			StartNode(nullReferenceExpression);
			WriteKeyword("NULL");
			return EndNode(nullReferenceExpression);
		}
		
		public object VisitObjectCreateExpression(ObjectCreateExpression objectCreateExpression, object data)
		{
			StartNode(objectCreateExpression);
			//WriteKeyword("new");
            var sm = CUDALanguage.GetSpecialMethod("ctor", objectCreateExpression.Type.ToString());
            if (sm != null)
            {
                string name = sm.Function(null, null);
                formatter.WriteIdentifier(name);
            }
            else
			    objectCreateExpression.Type.AcceptVisitor(this, data);
			bool useParenthesis = objectCreateExpression.Arguments.Any() || objectCreateExpression.Initializer.IsNull;
			// also use parenthesis if there is an '(' token and this isn't an anonymous type
			if (!objectCreateExpression.LParToken.IsNull && !objectCreateExpression.Type.IsNull)
				useParenthesis = true;
			if (useParenthesis) {
				Space(policy.SpaceBeforeMethodCallParentheses);
				WriteCommaSeparatedListInParenthesis(objectCreateExpression.Arguments, policy.SpaceWithinMethodCallParentheses);
			}
			objectCreateExpression.Initializer.AcceptVisitor(this, data);
			return EndNode(objectCreateExpression);
		}
		
		public object VisitAnonymousTypeCreateExpression(AnonymousTypeCreateExpression anonymousTypeCreateExpression, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Anonymous type create expressions");
            //StartNode(anonymousTypeCreateExpression);
            //WriteKeyword("new");
            //Space();
            //LPar();
            //RPar();
            //Space();
            //OpenBrace(policy.AnonymousMethodBraceStyle);
            //foreach (AstNode node in anonymousTypeCreateExpression.Initializer) {
            //    node.AcceptVisitor(this, null);
            //    if (node.NextSibling != null)
            //        Comma(node);
            //    NewLine ();
            //}
            //CloseBrace(policy.AnonymousMethodBraceStyle);
            //return EndNode(anonymousTypeCreateExpression);
		}

		public object VisitParenthesizedExpression(ParenthesizedExpression parenthesizedExpression, object data)
		{
			StartNode(parenthesizedExpression);
			LPar();
			Space(policy.SpacesWithinParentheses);
			parenthesizedExpression.Expression.AcceptVisitor(this, data);
			Space(policy.SpacesWithinParentheses);
			RPar();
			return EndNode(parenthesizedExpression);
		}
		
		public object VisitPointerReferenceExpression (PointerReferenceExpression pointerReferenceExpression, object data)
		{
			StartNode (pointerReferenceExpression);
			pointerReferenceExpression.Target.AcceptVisitor (this, data);
			WriteToken ("->", PointerReferenceExpression.ArrowRole);
			WriteIdentifier (pointerReferenceExpression.MemberName);
			WriteTypeArguments (pointerReferenceExpression.TypeArguments);
			return EndNode (pointerReferenceExpression);
		}
		
		public object VisitEmptyExpression (EmptyExpression emptyExpression, object data)
		{
			return EndNode (emptyExpression);
		}
		#region VisitPrimitiveExpression
		public object VisitPrimitiveExpression(PrimitiveExpression primitiveExpression, object data)
		{
			StartNode(primitiveExpression);
            bool isIndexer = primitiveExpression.Parent is IndexerExpression;
            bool isInvocationExp = primitiveExpression.Parent is InvocationExpression;
			WritePrimitiveValue(primitiveExpression.Value, isIndexer, isInvocationExp);
			return EndNode(primitiveExpression);
		}

        void WritePrimitiveValue(object val, bool isIndexer = false, bool isInvocationExp = false)
		{
			if (val == null) {
				// usually NullReferenceExpression should be used for this, but we'll handle it anyways
				WriteKeyword("null");
				return;
			}
			
			if (val is bool) {
				if ((bool)val) {
					WriteKeyword("true");
				} else {
					WriteKeyword("false");
				}
				return;
			}
			
			if (val is string) {//(unsigned short*)
                //formatter.WriteToken(string.Format("(__wchar_t{0})L\"", isIndexer ? "" : "*") + ConvertString(val.ToString()) + "\"");
                formatter.WriteToken(string.Format("(unsigned short{0})L\"", isIndexer ? "" : "*") + ConvertString(val.ToString()) + "\"");
                if (isInvocationExp)
                    formatter.WriteToken(string.Format(", {0}", val.ToString().Length));
				lastWritten = LastWritten.Other;
			} else if (val is char) {
                byte[] ba = Encoding.Unicode.GetBytes(new char[] { (char)val });
                ushort shrt = BitConverter.ToUInt16(ba, 0);
                formatter.WriteToken(shrt.ToString());
				//formatter.WriteToken("'" + ConvertCharLiteral((char)val) + "'");
				lastWritten = LastWritten.Other;
			} else if (val is decimal) {
				formatter.WriteToken(((decimal)val).ToString(NumberFormatInfo.InvariantInfo) + "m");
				lastWritten = LastWritten.Other;
			} else if (val is float) {
				float f = (float)val;
                if (float.IsInfinity(f) || float.IsNaN(f)) {
                //    // Strictly speaking, these aren't PrimitiveExpressions;
                //    // but we still support writing these to make life easier for code generators.
                //    WriteKeyword("float");
                //    WriteToken(".", AstNode.Roles.Dot);
                if (float.IsPositiveInfinity(f))
                    WriteIdentifier(CudafyTranslator.LanguageSpecifics.PositiveInfinitySingle); // INFINITY OpenCL  //"PositiveInfinity");
                else if (float.IsNegativeInfinity(f))
                    WriteIdentifier(CudafyTranslator.LanguageSpecifics.NegativeInfinitySingle);// INFINITY OpenCL//NegativeInfinity");
                else
                    WriteIdentifier(CudafyTranslator.LanguageSpecifics.NaNSingle);// NAN OpenCL//"NaN");
                    return;
                }
				//formatter.WriteToken(f.ToString("R", NumberFormatInfo.InvariantInfo) +"f");// + "f");
                //if(!f.ToString().Contains("."))
                //    formatter.WriteToken(f.ToString("F1", NumberFormatInfo.InvariantInfo) + "f");// + "f");
                //else
                //    formatter.WriteToken(f.ToString() + "f");

                string number = f.ToString(NumberFormatInfo.InvariantInfo);//"R", NumberFormatInfo.InvariantInfo);
                if (number.IndexOf('.') < 0 && number.IndexOf('E') < 0)
                    number += ".0";
                formatter.WriteToken(number+"f");
				lastWritten = LastWritten.Other;
			} else if (val is double) {
				double f = (double)val;
                if (double.IsInfinity(f) || double.IsNaN(f))
                {
                    // Strictly speaking, these aren't PrimitiveExpressions;
                    // but we still support writing these to make life easier for code generators.
                    //WriteKeyword("double");
                    //WriteToken(".", AstNode.Roles.Dot);
                    if (double.IsPositiveInfinity(f))
                        WriteIdentifier(CudafyTranslator.LanguageSpecifics.PositiveInfinityDouble);//"PositiveInfinity");
                    else if (double.IsNegativeInfinity(f))
                        WriteIdentifier(CudafyTranslator.LanguageSpecifics.NegativeInfinityDouble);//NegativeInfinity");
                    else
                        WriteIdentifier(CudafyTranslator.LanguageSpecifics.NaNDouble);//"NaN");
                    return;
                }
                string number = f.ToString(NumberFormatInfo.InvariantInfo);//"R", NumberFormatInfo.InvariantInfo);
				if (number.IndexOf('.') < 0 && number.IndexOf('E') < 0)
					number += ".0";
				formatter.WriteToken(number);
				// needs space if identifier follows number; this avoids mistaking the following identifier as type suffix
				lastWritten = LastWritten.KeywordOrIdentifier;
			} else if (val is IFormattable) {
				StringBuilder b = new StringBuilder();
//				if (primitiveExpression.LiteralFormat == LiteralFormat.HexadecimalNumber) {
//					b.Append("0x");
//					b.Append(((IFormattable)val).ToString("x", NumberFormatInfo.InvariantInfo));
//				} else {
				b.Append(((IFormattable)val).ToString(null, NumberFormatInfo.InvariantInfo));
//				}
				if (val is uint || val is ulong) {
					b.Append("u");
				}
				if (val is long || val is ulong) {
					b.Append("L");
				}
				formatter.WriteToken(b.ToString());
				// needs space if identifier follows number; this avoids mistaking the following identifier as type suffix
				lastWritten = LastWritten.KeywordOrIdentifier;
			} else {
				formatter.WriteToken(val.ToString());
				lastWritten = LastWritten.Other;
			}
		}
		
		static string ConvertCharLiteral(char ch)
		{
			if (ch == '\'') return "\\'";
			return ConvertChar(ch);
		}
		
		static string ConvertChar(char ch)
		{
			switch (ch) {
				case '\\':
					return "\\\\";
				case '\0':
					return "\\0";
				case '\a':
					return "\\a";
				case '\b':
					return "\\b";
				case '\f':
					return "\\f";
				case '\n':
					return "\\n";
				case '\r':
					return "\\r";
				case '\t':
					return "\\t";
				case '\v':
					return "\\v";
				default:
					//if (char.IsControl(ch) || char.IsSurrogate(ch)) {
						return "\\x"+ ((int)ch).ToString("x4");// 
					//} else {
                    //    byte[] ba = Encoding.Unicode.GetBytes(new char[] { (char)ch });
                    //    ushort shrt = BitConverter.ToUInt16(ba, 0);
                    //    return "\\u"+shrt.ToString("X");
						//return ch.ToString();
					//}
			}
		}
		
		static string ConvertString(string str)
		{
			StringBuilder sb = new StringBuilder();
			foreach (char ch in str) {
				if (ch == '"')
					sb.Append("\\\"");
				else
					sb.Append(ConvertChar(ch));
			}
			return sb.ToString();
		}
		#endregion
		
		public object VisitSizeOfExpression(SizeOfExpression sizeOfExpression, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "SizeOf expressions");
            //StartNode(sizeOfExpression);
			
            //WriteKeyword("sizeof");
            //LPar();
            //Space(policy.SpacesWithinSizeOfParentheses);
            //sizeOfExpression.Type.AcceptVisitor(this, data);
            //Space(policy.SpacesWithinSizeOfParentheses);
            //RPar();
			
            //return EndNode(sizeOfExpression);
		}
		
		public object VisitStackAllocExpression(StackAllocExpression stackAllocExpression, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "StackAlloc expressions");
            //StartNode(stackAllocExpression);
            //WriteKeyword("stackalloc");
            //stackAllocExpression.Type.AcceptVisitor(this, data);
            //WriteCommaSeparatedListInBrackets(new[] { stackAllocExpression.CountExpression });
            //return EndNode(stackAllocExpression);
		}
		
		public object VisitThisReferenceExpression(ThisReferenceExpression thisReferenceExpression, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "This Reference expressions");
            //StartNode(thisReferenceExpression);
            //WriteKeyword("this");
            //return EndNode(thisReferenceExpression);
		}
		
		public object VisitTypeOfExpression(TypeOfExpression typeOfExpression, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "TypeOf expressions");
            //StartNode(typeOfExpression);
			
            //WriteKeyword("typeof");
            //LPar();
            //Space(policy.SpacesWithinTypeOfParentheses);
            //typeOfExpression.Type.AcceptVisitor(this, data);
            //Space(policy.SpacesWithinTypeOfParentheses);
            //RPar();
			
            //return EndNode(typeOfExpression);
		}
		
		public object VisitTypeReferenceExpression(TypeReferenceExpression typeReferenceExpression, object data)
		{
			StartNode(typeReferenceExpression);
			typeReferenceExpression.Type.AcceptVisitor(this, data);
			return EndNode(typeReferenceExpression);
		}
		
		public object VisitUnaryOperatorExpression(UnaryOperatorExpression unaryOperatorExpression, object data)
		{
			StartNode(unaryOperatorExpression);
			UnaryOperatorType opType = unaryOperatorExpression.Operator;
			string opSymbol = UnaryOperatorExpression.GetOperatorSymbol(opType);
			if (!(opType == UnaryOperatorType.PostIncrement || opType == UnaryOperatorType.PostDecrement) && !unaryOperatorExpression.ToString().Contains("FixedElementField"))
				WriteToken(opSymbol, UnaryOperatorExpression.OperatorRole);
			unaryOperatorExpression.Expression.AcceptVisitor(this, data);
			if (opType == UnaryOperatorType.PostIncrement || opType == UnaryOperatorType.PostDecrement)
				WriteToken(opSymbol, UnaryOperatorExpression.OperatorRole);
			return EndNode(unaryOperatorExpression);
		}
		
		public object VisitUncheckedExpression(UncheckedExpression uncheckedExpression, object data)
		{
            //throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Unchecked expressions");
            StartNode(uncheckedExpression);
            //WriteKeyword("unchecked");
            //LPar();
            //Space(policy.SpacesWithinCheckedExpressionParantheses);
            uncheckedExpression.Expression.AcceptVisitor(this, data);
            //Space(policy.SpacesWithinCheckedExpressionParantheses);
            //RPar();
            return EndNode(uncheckedExpression);
		}
		#endregion
		
		#region Query Expressions
		public object VisitQueryExpression(QueryExpression queryExpression, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Query expressions");
            //StartNode(queryExpression);
            //bool indent = !(queryExpression.Parent is QueryContinuationClause);
            //if (indent) {
            //    formatter.Indent();
            //    NewLine();
            //}
            //bool first = true;
            //foreach (var clause in queryExpression.Clauses) {
            //    if (first)
            //        first = false;
            //    else
            //        if (!(clause is QueryContinuationClause))
            //        NewLine();
            //    clause.AcceptVisitor(this, data);
            //}
            //if (indent)
            //    formatter.Unindent();
            //return EndNode(queryExpression);
		}
		
		public object VisitQueryContinuationClause(QueryContinuationClause queryContinuationClause, object data)
		{
			StartNode(queryContinuationClause);
			queryContinuationClause.PrecedingQuery.AcceptVisitor(this, data);
			Space();
			WriteKeyword("into", QueryContinuationClause.IntoKeywordRole);
			Space();
			WriteIdentifier(queryContinuationClause.Identifier);
			return EndNode(queryContinuationClause);
		}
		
		public object VisitQueryFromClause(QueryFromClause queryFromClause, object data)
		{
			StartNode(queryFromClause);
			WriteKeyword("from", QueryFromClause.FromKeywordRole);
			queryFromClause.Type.AcceptVisitor(this, data);
			Space();
			WriteIdentifier(queryFromClause.Identifier);
			WriteKeyword("in", QueryFromClause.InKeywordRole);
			queryFromClause.Expression.AcceptVisitor(this, data);
			return EndNode(queryFromClause);
		}
		
		public object VisitQueryLetClause(QueryLetClause queryLetClause, object data)
		{
			StartNode(queryLetClause);
			WriteKeyword("let");
			Space();
			WriteIdentifier(queryLetClause.Identifier);
			Space(policy.SpaceAroundAssignment);
			WriteToken("=", QueryLetClause.Roles.Assign);
			Space(policy.SpaceAroundAssignment);
			queryLetClause.Expression.AcceptVisitor(this, data);
			return EndNode(queryLetClause);
		}
		
		public object VisitQueryWhereClause(QueryWhereClause queryWhereClause, object data)
		{
			StartNode(queryWhereClause);
			WriteKeyword("where");
			Space();
			queryWhereClause.Condition.AcceptVisitor(this, data);
			return EndNode(queryWhereClause);
		}
		
		public object VisitQueryJoinClause(QueryJoinClause queryJoinClause, object data)
		{
			StartNode(queryJoinClause);
			WriteKeyword("join", QueryJoinClause.JoinKeywordRole);
			queryJoinClause.Type.AcceptVisitor(this, data);
			Space();
			WriteIdentifier(queryJoinClause.JoinIdentifier, QueryJoinClause.JoinIdentifierRole);
			Space();
			WriteKeyword("in", QueryJoinClause.InKeywordRole);
			Space();
			queryJoinClause.InExpression.AcceptVisitor(this, data);
			Space();
			WriteKeyword("on", QueryJoinClause.OnKeywordRole);
			Space();
			queryJoinClause.OnExpression.AcceptVisitor(this, data);
			Space();
			WriteKeyword("equals", QueryJoinClause.EqualsKeywordRole);
			Space();
			queryJoinClause.EqualsExpression.AcceptVisitor(this, data);
			if (queryJoinClause.IsGroupJoin) {
				Space();
				WriteKeyword("into", QueryJoinClause.IntoKeywordRole);
				WriteIdentifier(queryJoinClause.IntoIdentifier, QueryJoinClause.IntoIdentifierRole);
			}
			return EndNode(queryJoinClause);
		}
		
		public object VisitQueryOrderClause(QueryOrderClause queryOrderClause, object data)
		{
			StartNode(queryOrderClause);
			WriteKeyword("orderby");
			Space();
			WriteCommaSeparatedList(queryOrderClause.Orderings.SafeCast<QueryOrdering, AstNode>());
			return EndNode(queryOrderClause);
		}
		
		public object VisitQueryOrdering(QueryOrdering queryOrdering, object data)
		{
			StartNode(queryOrdering);
			queryOrdering.Expression.AcceptVisitor(this, data);
			switch (queryOrdering.Direction) {
				case QueryOrderingDirection.Ascending:
					Space();
					WriteKeyword("ascending");
					break;
				case QueryOrderingDirection.Descending:
					Space();
					WriteKeyword("descending");
					break;
			}
			return EndNode(queryOrdering);
		}
		
		public object VisitQuerySelectClause(QuerySelectClause querySelectClause, object data)
		{
			StartNode(querySelectClause);
			WriteKeyword("select");
			Space();
			querySelectClause.Expression.AcceptVisitor(this, data);
			return EndNode(querySelectClause);
		}
		
		public object VisitQueryGroupClause(QueryGroupClause queryGroupClause, object data)
		{
			StartNode(queryGroupClause);
			WriteKeyword("group", QueryGroupClause.GroupKeywordRole);
			Space();
			queryGroupClause.Projection.AcceptVisitor(this, data);
			Space();
			WriteKeyword("by", QueryGroupClause.ByKeywordRole);
			Space();
			queryGroupClause.Key.AcceptVisitor(this, data);
			return EndNode(queryGroupClause);
		}
		#endregion
		
		#region GeneralScope
		public object VisitAttribute(ICSharpCode.NRefactory.CSharp.Attribute attribute, object data)
		{
			StartNode(attribute);
			attribute.Type.AcceptVisitor(this, data);
			Space(policy.SpaceBeforeMethodCallParentheses);
			if (attribute.Arguments.Count != 0 || !attribute.GetChildByRole(AstNode.Roles.LPar).IsNull)
                WriteCommaSeparatedList(attribute.Arguments);//, policy.SpaceWithinMethodCallParentheses);
				//WriteCommaSeparatedListInParenthesis(attribute.Arguments, policy.SpaceWithinMethodCallParentheses);
			return EndNode(attribute);
		}
		
		public object VisitAttributeSection(AttributeSection attributeSection, object data)
		{
			StartNode(attributeSection);
			//WriteToken("[", AstNode.Roles.LBracket);
            //if (!string.IsNullOrEmpty (attributeSection.AttributeTarget)) {
            //    WriteToken(attributeSection.AttributeTarget, AttributeSection.TargetRole);
            //    WriteToken(":", AttributeSection.Roles.Colon);
            //    Space();
            //}
			WriteCommaSeparatedList(attributeSection.Attributes.SafeCast<ICSharpCode.NRefactory.CSharp.Attribute, AstNode>());
			//WriteToken("]", AstNode.Roles.RBracket);
			if (attributeSection.Parent is ParameterDeclaration || attributeSection.Parent is TypeParameterDeclaration)
				Space();
			else
				NewLine();
			return EndNode(attributeSection);
		}
		
		public object VisitDelegateDeclaration(DelegateDeclaration delegateDeclaration, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Delegates");
            //StartNode(delegateDeclaration);
            //WriteAttributes(delegateDeclaration.Attributes);
            //WriteModifiers(delegateDeclaration.ModifierTokens);
            //WriteKeyword("delegate");
            //delegateDeclaration.ReturnType.AcceptVisitor(this, data);
            //Space();
            //WriteIdentifier(delegateDeclaration.Name);
            //WriteTypeParameters(delegateDeclaration.TypeParameters);
            //Space(policy.SpaceBeforeDelegateDeclarationParentheses);
            //WriteCommaSeparatedListInParenthesis(delegateDeclaration.Parameters, policy.SpaceWithinMethodDeclarationParentheses);
            //foreach (Constraint constraint in delegateDeclaration.Constraints) {
            //    constraint.AcceptVisitor(this, data);
            //}
            //Semicolon();
            //return EndNode(delegateDeclaration);
		}
		
		public object VisitNamespaceDeclaration(NamespaceDeclaration namespaceDeclaration, object data)
		{
			StartNode(namespaceDeclaration);
            //WriteKeyword("namespace");
            //WriteQualifiedIdentifier(namespaceDeclaration.Identifiers);
            //OpenBrace(policy.NamespaceBraceStyle);
            foreach (var member in namespaceDeclaration.Members)
                member.AcceptVisitor(this, data);
            //CloseBrace(policy.NamespaceBraceStyle);
            //NewLine();
			return EndNode(namespaceDeclaration);
		}
		
		public object VisitTypeDeclaration(TypeDeclaration typeDeclaration, object data)
		{
			StartNode(typeDeclaration);
            //typeDeclaration.Attributes.ToList().ForEach(a => (a.Children.ToList().ForEach(c => Console.WriteLine(c.ToString())));
            //if (typeDeclaration.Attributes.ToList().Count(a => a.ToString().Contains("Flag")) > 0)
            //    throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED, "Flags");
			//WriteAttributes(typeDeclaration.Attributes);
			//WriteModifiers(typeDeclaration.ModifierTokens);
			BraceStyle braceStyle = policy.StructBraceStyle;
            if (typeDeclaration.ClassType != ClassType.Struct && typeDeclaration.ClassType != ClassType.Enum)
            {
                if (typeDeclaration.ClassType != ClassType.Class || !CudafyTranslator.AllowClasses) throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED, typeDeclaration.ClassType.ToString());
            }

            var typeDeclarationEx = typeDeclaration as TypeDeclarationEx;
            if ((typeDeclarationEx != null && typeDeclarationEx.IsDummy))
            {
                formatter.WriteComment(CommentType.SingleLine, string.Format("Type {0} is a dummy.", typeDeclaration.Name));
            }
            else
            {
                 #region Commented Out
                switch (typeDeclaration.ClassType)
                {
                    case ClassType.Enum:
                        WriteKeyword("enum");
                        braceStyle = policy.EnumBraceStyle;
                        break;
                    //case ClassType.Interface:
                    //    WriteKeyword("interface");
                    //    braceStyle = policy.InterfaceBraceStyle;
                    //    break;
                    case ClassType.Class:
                        WriteKeyword("struct");
                        braceStyle = policy.StructBraceStyle;
                        break;
                    case ClassType.Struct:
                        WriteKeyword("struct");
                        braceStyle = policy.StructBraceStyle;
                        break;
                    //default:
                    //    WriteKeyword("class");
                    //    braceStyle = policy.ClassBraceStyle;
                    //    break;
                }
            #endregion
                string typeName;
              
               
                if (typeDeclarationEx == null)
                    typeName = typeDeclaration.Name;
                else
                    //typeName = typeDeclarationEx.Name;
                    typeName = typeDeclarationEx.FullName;
                typeName = typeName.Replace('<', '_');
                typeName = typeName.Replace('>', '_');
                WriteIdentifier(typeName);
                //WriteTypeParameters(typeDeclaration.TypeParameters);
			    if (typeDeclaration.BaseTypes.Any()) {
                    var type = typeDeclaration.BaseTypes.First().Annotations.FirstOrDefault() as Mono.Cecil.ICustomAttributeProvider;
                    var attr = type.GetCudafyType();
                    if (attr != null)
                    {
                        Space();
                        WriteToken(":", TypeDeclaration.ColonRole);
                        Space();
                        WriteCommaSeparatedList(typeDeclaration.BaseTypes);
                    }
			    }
                // Ignore constraints
			    //foreach (Constraint constraint in typeDeclaration.Constraints) 
                //{
                //constraint.AcceptVisitor(this, data);
			    //}
			    OpenBrace(braceStyle);
			    if (typeDeclaration.ClassType == ClassType.Enum) {
				    bool first = true;
				    foreach (var member in typeDeclaration.Members) {
					    if (first) {
						    first = false;
					    } else {
						    Comma(member, noSpaceAfterComma: true);
						    NewLine();
					    }
					    member.AcceptVisitor(this, data);
				    }
				    NewLine();
			    } else {
                    IsTypeTranslationDepth++;
                    try
                    {
                        // Make an default constructor for CUDA
                        if (CudafyTranslator.LanguageSpecifics.Language == eLanguage.Cuda)
                        {
                            WriteIdentifier(string.Format("{0} {1}()", CudafyTranslator.LanguageSpecifics.DeviceFunctionModifiers, typeName));
                            OpenBrace(braceStyle);
                            CloseBrace(braceStyle);
                            NewLine();
                        }
                        Dictionary<string, int> classSizes = new Dictionary<string, int>();
                        foreach (var member in typeDeclaration.Members)
                        {
                            var td = member as TypeDeclarationEx;
                            if (td != null && td.Name.Contains(">e__FixedBuffer"))
                            {
                                int classSize1 = ((Mono.Cecil.TypeDefinition)(((object[])(((ICSharpCode.NRefactory.CSharp.AstNode)(member)).Annotations))[0])).ClassSize;
                                classSizes.Add(td.Name, classSize1);
                            }
                            var fd = member as FieldDeclarationEx;
                            int classSize = 0;
                            string returnType = "unknown";
                            bool ignoreElemSize = false;
                            if (fd != null)
                            {
                                if (fd.ReturnType != null && fd.ReturnType is MemberType)
                                {
                                    var mt = fd.ReturnType as MemberType;
                                    if (classSizes.ContainsKey(mt.MemberName))
                                    {
                                        classSize = classSizes[mt.MemberName];
                                        returnType = _FixedElementFields[mt.MemberName];
                                        _FixedElementFields.Remove(mt.MemberName);
                                    }
                                }
                                else
                                {
                                    var vi = fd.Variables.Where(v => v is VariableInitializer).FirstOrDefault();
                                    if (vi != null && vi.Name == "FixedElementField")
                                        _FixedElementFields.Add((fd.Parent as TypeDeclarationEx).Name, fd.ReturnType.ToString());
                                    else if(vi != null)
                                    {
                                        var attr = fd as AstNode;// fd.Attributes.Take(1);
                                        if (attr != null && attr.Annotations.Count() > 0)
                                        {
                                            var anno = attr.Annotations.Take(1);
                                            var fieldDef = anno.FirstOrDefault() as Mono.Cecil.FieldDefinition;
                                            if (fieldDef != null && fieldDef.HasMarshalInfo)
                                            {
                                                var mi = fieldDef.MarshalInfo as Mono.Cecil.FixedSysStringMarshalInfo;
                                                if (mi != null)
                                                {
                                                    returnType = fd.ReturnType.ToString();
                                                    classSize = mi.Size;
                                                    ignoreElemSize = true;
                                                }
                                                else
                                                {
                                                    var fami = fieldDef.MarshalInfo as Mono.Cecil.FixedArrayMarshalInfo;
                                                    if (fami != null)
                                                    {
                                                        returnType = fd.ReturnType.ToString();
                                                        classSize = fami.Size;
                                                        ignoreElemSize = true;
                                                    }
                                                }
                                            }
                                        }
                                        //var mi = ((Mono.Cecil.FieldDefinition)(((object[])(((ICSharpCode.NRefactory.CSharp.AstNode)(((ICSharpCode.NRefactory.CSharp.FieldDeclaration)(fd)).Variables.node)).Annotations))[0])).MarshalInfo;
                                        //var fssmi = mi as Mono.Cecil.FixedSysStringMarshalInfo;
                                        //int size = fssmi == null ? 0 : fssmi.Size;
                                    }
                                }
                            }
                            member.AcceptVisitor(this, classSize == 0 ? data : new Tuple<string,int,bool>(returnType, classSize, ignoreElemSize));
                        }
                    }
                    catch(Exception)
                    {
                        throw;
                    }
                    finally
                    {
                        IsTypeTranslationDepth--;
                    }
			    }
			    CloseBrace(braceStyle);
                Semicolon();
			    NewLine();
            }
			return EndNode(typeDeclaration);
		}

        private Dictionary<string,string> _FixedElementFields = new Dictionary<string,string>();

        public bool IsTypeTranslation 
        {
            get { return IsTypeTranslationDepth > 0; }
        }

        private int _typeTranslationDepth = 0;

        private int IsTypeTranslationDepth 
        {
            get { return _typeTranslationDepth; }
            set { _typeTranslationDepth = value; Debug.Assert(value >= 0); } 
        }
		
		public object VisitUsingAliasDeclaration(UsingAliasDeclaration usingAliasDeclaration, object data)
		{
			StartNode(usingAliasDeclaration);
            //WriteKeyword("using");
            //WriteIdentifier(usingAliasDeclaration.Alias, UsingAliasDeclaration.AliasRole);
            //Space(policy.SpaceAroundEqualityOperator);
            //WriteToken("=", AstNode.Roles.Assign);
            //Space(policy.SpaceAroundEqualityOperator);
            //usingAliasDeclaration.Import.AcceptVisitor(this, data);
            //Semicolon();
			return EndNode(usingAliasDeclaration);
		}
		
		public object VisitUsingDeclaration(UsingDeclaration usingDeclaration, object data)
		{
			StartNode(usingDeclaration);
            //WriteKeyword("using");
            //usingDeclaration.Import.AcceptVisitor(this, data);
            //Semicolon();
			return EndNode(usingDeclaration);
		}
		
		public object VisitExternAliasDeclaration(ExternAliasDeclaration externAliasDeclaration, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Extern Alias expressions");
            //StartNode(externAliasDeclaration);
            //WriteKeyword("extern");
            //Space ();
            //WriteKeyword("alias");
            //Space ();
            //externAliasDeclaration.NameToken.AcceptVisitor(this, data);
            //Semicolon();
            //return EndNode(externAliasDeclaration);
		}


		#endregion
		
		#region Statements
		public object VisitBlockStatement(BlockStatement blockStatement, object data)
		{
			StartNode(blockStatement);
			BraceStyle style;
			if (blockStatement.Parent is AnonymousMethodExpression || blockStatement.Parent is LambdaExpression) {
				style = policy.AnonymousMethodBraceStyle;
			} else if (blockStatement.Parent is ConstructorDeclaration) {
				style = policy.ConstructorBraceStyle;
			} else if (blockStatement.Parent is DestructorDeclaration) {
				style = policy.DestructorBraceStyle;
			} else if (blockStatement.Parent is MethodDeclaration) {
				style = policy.MethodBraceStyle;
			} else if (blockStatement.Parent is Accessor) {
				if (blockStatement.Parent.Role == PropertyDeclaration.GetterRole)
					style = policy.PropertyGetBraceStyle;
				else if (blockStatement.Parent.Role == PropertyDeclaration.SetterRole)
					style = policy.PropertySetBraceStyle;
				else if (blockStatement.Parent.Role == CustomEventDeclaration.AddAccessorRole)
					style = policy.EventAddBraceStyle;
				else if (blockStatement.Parent.Role == CustomEventDeclaration.RemoveAccessorRole)
					style = policy.EventRemoveBraceStyle;
				else
					throw new NotSupportedException("Unknown type of accessor");
			} else {
				style = policy.StatementBraceStyle;
			}
			OpenBrace(style);
			foreach (var node in blockStatement.Statements) {
                //Console.WriteLine("{0} is {1}", node.ToString(), node.GetType());
				node.AcceptVisitor(this, data);
			}
			CloseBrace(style);
			NewLine();
			return EndNode(blockStatement);
		}
		
		public object VisitBreakStatement(BreakStatement breakStatement, object data)
		{
			StartNode(breakStatement);
			WriteKeyword("break");
			Semicolon();
			return EndNode(breakStatement);
		}
		
		public object VisitCheckedStatement(CheckedStatement checkedStatement, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Checked statements");
            //StartNode(checkedStatement);
            //WriteKeyword("checked");
            //checkedStatement.Body.AcceptVisitor(this, data);
            //return EndNode(checkedStatement);
		}
		
		public object VisitContinueStatement(ContinueStatement continueStatement, object data)
		{
			StartNode(continueStatement);
			WriteKeyword("continue");
			Semicolon();
			return EndNode(continueStatement);
		}
		
		public object VisitDoWhileStatement(DoWhileStatement doWhileStatement, object data)
		{
			StartNode(doWhileStatement);
			WriteKeyword("do", DoWhileStatement.DoKeywordRole);
			WriteEmbeddedStatement(doWhileStatement.EmbeddedStatement);
			WriteKeyword("while", DoWhileStatement.WhileKeywordRole);
			Space(policy.SpaceBeforeWhileParentheses);
			LPar();
			Space(policy.SpacesWithinWhileParentheses);
			doWhileStatement.Condition.AcceptVisitor(this, data);
			Space(policy.SpacesWithinWhileParentheses);
			RPar();
			Semicolon();
			return EndNode(doWhileStatement);
		}
		
		public object VisitEmptyStatement(EmptyStatement emptyStatement, object data)
		{
			StartNode(emptyStatement);
			Semicolon();
			return EndNode(emptyStatement);
		}
		
		public object VisitExpressionStatement(ExpressionStatement expressionStatement, object data)
		{
			StartNode(expressionStatement);
			expressionStatement.Expression.AcceptVisitor(this, data);
			Semicolon();
			return EndNode(expressionStatement);
		}
		
		public object VisitFixedStatement(FixedStatement fixedStatement, object data)
		{
            //throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "fixed statements");
            StartNode(fixedStatement);
            //WriteKeyword("fixed");
            //Space(policy.SpaceBeforeUsingParentheses);
            //LPar();
            //Space(policy.SpacesWithinUsingParentheses);
            fixedStatement.Type.AcceptVisitor(this, data);
            Space();
            WriteCommaSeparatedList(fixedStatement.Variables);
            Semicolon();
            //Space(policy.SpacesWithinUsingParentheses);
            //RPar();
            WriteEmbeddedStatement(fixedStatement.EmbeddedStatement);
            return EndNode(fixedStatement);
		}
		
		public object VisitForeachStatement(ForeachStatement foreachStatement, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Foreach statements");



            //StartNode(foreachStatement);

            //string format = "for(int {0} = 0; {0} < {1}; {0}++) {";
            //string guid = ShortGuid.NewGuid().ToString();

            //WriteKeyword("for");
            //Space(policy.SpaceBeforeForeachParentheses);
            //LPar();
            //Space(policy.SpacesWithinForeachParentheses);
            //foreachStatement.VariableType.AcceptVisitor(this, data);
            //Space();
            //string guid = Guid.NewGuid().ToString();
            //WriteIdentifier(guid);
            //WriteIdentifier(" = 0; ");
            //WriteIdentifier(guid);
            //WriteIdentifier(" < ");
            //foreachStatement.InExpression.AcceptVisitor(this, data);
            //formatter.WriteIdentifier("Len0;");
            //WriteIdentifier(guid);
            //WriteIdentifier("++");
            ////WriteIdentifier(foreachStatement.VariableName);
            ////WriteKeyword("in", ForeachStatement.Roles.InKeyword);
            ////Space();
            ////foreachStatement.InExpression.AcceptVisitor(this, data);
            ////Space(policy.SpacesWithinForeachParentheses);
            //RPar();
            //NewLine();
            //WriteIdentifier(foreachStatement.VariableName);
            //WriteEmbeddedStatement(foreachStatement.EmbeddedStatement);
            //return EndNode(foreachStatement);

            //StartNode(foreachStatement);
            //WriteKeyword("foreach");
            //Space(policy.SpaceBeforeForeachParentheses);
            //LPar();
            //Space(policy.SpacesWithinForeachParentheses);
            //foreachStatement.VariableType.AcceptVisitor(this, data);
            //Space();
            //WriteIdentifier(foreachStatement.VariableName);
            //WriteKeyword("in", ForeachStatement.Roles.InKeyword);
            //Space();
            //foreachStatement.InExpression.AcceptVisitor(this, data);
            //Space(policy.SpacesWithinForeachParentheses);
            //RPar();
            //WriteEmbeddedStatement(foreachStatement.EmbeddedStatement);
            return EndNode(foreachStatement);
		}
		
		public object VisitForStatement(ForStatement forStatement, object data)
		{
			StartNode(forStatement);
			WriteKeyword("for");
			Space(policy.SpaceBeforeForParentheses);
			LPar();
			Space(policy.SpacesWithinForParentheses);
			
			WriteCommaSeparatedList(forStatement.Initializers.SafeCast<Statement, AstNode>());
			Space (policy.SpaceBeforeForSemicolon);
			WriteToken(";", AstNode.Roles.Semicolon);
			Space (policy.SpaceAfterForSemicolon);
			
			forStatement.Condition.AcceptVisitor(this, data);
			Space (policy.SpaceBeforeForSemicolon);
			WriteToken(";", AstNode.Roles.Semicolon);
			Space(policy.SpaceAfterForSemicolon);
			
			WriteCommaSeparatedList(forStatement.Iterators.SafeCast<Statement, AstNode>());
			
			Space(policy.SpacesWithinForParentheses);
			RPar();
			WriteEmbeddedStatement(forStatement.EmbeddedStatement);
			return EndNode(forStatement);
		}
		
		public object VisitGotoCaseStatement(GotoCaseStatement gotoCaseStatement, object data)
		{
			StartNode(gotoCaseStatement);
			WriteKeyword("goto");
			WriteKeyword("case", GotoCaseStatement.CaseKeywordRole);
			Space();
			gotoCaseStatement.LabelExpression.AcceptVisitor(this, data);
			Semicolon();
			return EndNode(gotoCaseStatement);
		}
		
		public object VisitGotoDefaultStatement(GotoDefaultStatement gotoDefaultStatement, object data)
		{
			StartNode(gotoDefaultStatement);
			WriteKeyword("goto");
			WriteKeyword("default", GotoDefaultStatement.DefaultKeywordRole);
			Semicolon();
			return EndNode(gotoDefaultStatement);
		}
		
		public object VisitGotoStatement(GotoStatement gotoStatement, object data)
		{
			StartNode(gotoStatement);
			WriteKeyword("goto");
			WriteIdentifier(gotoStatement.Label);
			Semicolon();
			return EndNode(gotoStatement);
		}
		
		public object VisitIfElseStatement(IfElseStatement ifElseStatement, object data)
		{
			StartNode(ifElseStatement);
			WriteKeyword("if", IfElseStatement.IfKeywordRole);
			Space(policy.SpaceBeforeIfParentheses);
			LPar();
			Space(policy.SpacesWithinIfParentheses);
			ifElseStatement.Condition.AcceptVisitor(this, data);
			Space(policy.SpacesWithinIfParentheses);
			RPar();
			WriteEmbeddedStatement(ifElseStatement.TrueStatement);
			if (!ifElseStatement.FalseStatement.IsNull) {
				WriteKeyword("else", IfElseStatement.ElseKeywordRole);
				WriteEmbeddedStatement(ifElseStatement.FalseStatement);
			}
			return EndNode(ifElseStatement);
		}
		
		public object VisitLabelStatement(LabelStatement labelStatement, object data)
		{
			StartNode(labelStatement);
			WriteIdentifier(labelStatement.Label);
			WriteToken(":", LabelStatement.Roles.Colon);
			bool foundLabelledStatement = false;
			for (AstNode tmp = labelStatement.NextSibling; tmp != null; tmp = tmp.NextSibling) {
				if (tmp.Role == labelStatement.Role) {
					foundLabelledStatement = true;
				}
			}
			if (!foundLabelledStatement) {
				// introduce an EmptyStatement so that the output becomes syntactically valid
				WriteToken(";", LabelStatement.Roles.Semicolon);
			}
			NewLine();
			return EndNode(labelStatement);
		}
		
		public object VisitLockStatement(LockStatement lockStatement, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Lock statements");
            //StartNode(lockStatement);
            //WriteKeyword("lock");
            //Space(policy.SpaceBeforeLockParentheses);
            //LPar();
            //Space(policy.SpacesWithinLockParentheses);
            //lockStatement.Expression.AcceptVisitor(this, data);
            //Space(policy.SpacesWithinLockParentheses);
            //RPar();
            //WriteEmbeddedStatement(lockStatement.EmbeddedStatement);
            //return EndNode(lockStatement);
		}
		
		public object VisitReturnStatement(ReturnStatement returnStatement, object data)
		{
			StartNode(returnStatement);
			WriteKeyword("return");
			if (!returnStatement.Expression.IsNull) {
				Space();
				returnStatement.Expression.AcceptVisitor(this, data);
			}
			Semicolon();
			return EndNode(returnStatement);
		}
		
		public object VisitSwitchStatement(SwitchStatement switchStatement, object data)
		{
			StartNode(switchStatement);
			WriteKeyword("switch");
			Space(policy.SpaceBeforeSwitchParentheses);
			LPar();
			Space(policy.SpacesWithinSwitchParentheses);
			switchStatement.Expression.AcceptVisitor(this, data);
			Space(policy.SpacesWithinSwitchParentheses);
			RPar();
			OpenBrace(policy.StatementBraceStyle);
			foreach (var section in switchStatement.SwitchSections)
				section.AcceptVisitor(this, data);
			CloseBrace(policy.StatementBraceStyle);
			NewLine();
			return EndNode(switchStatement);
		}
		
		public object VisitSwitchSection(SwitchSection switchSection, object data)
		{
			StartNode(switchSection);
			bool first = true;
			foreach (var label in switchSection.CaseLabels) {
				if (!first)
					NewLine();
				label.AcceptVisitor(this, data);
				first = false;
			}
			foreach (var statement in switchSection.Statements)
				statement.AcceptVisitor(this, data);
			return EndNode(switchSection);
		}
		
		public object VisitCaseLabel(CaseLabel caseLabel, object data)
		{
			StartNode(caseLabel);
			if (caseLabel.Expression.IsNull) {
				WriteKeyword("default");
			} else {
				WriteKeyword("case");
				Space();
				caseLabel.Expression.AcceptVisitor(this, data);
			}
			WriteToken(":", CaseLabel.Roles.Colon);
			return EndNode(caseLabel);
		}
		
		public object VisitThrowStatement(ThrowStatement throwStatement, object data)
		{
            if (CudafyTranslator.Language != eLanguage.Cuda)
                throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Throw statements");


           StartNode(throwStatement);
            //WriteKeyword("throw");
            //if (!throwStatement.Expression.IsNull) {
            //    Space();
            //    throwStatement.Expression.AcceptVisitor(this, data);
            //}
            //Semicolon();
           //if (CUDALanguage.ComputeCapability.Major < 2)
               WriteKeyword(@"asm(""trap;"");");
           //else
           //    WriteKeyword(@"assert(0);");
           
            return EndNode(throwStatement);
		}
		
		public object VisitTryCatchStatement(TryCatchStatement tryCatchStatement, object data)
		{
			StartNode(tryCatchStatement);
            formatter.WriteComment(CommentType.SingleLine, "try");
			//WriteKeyword("try", TryCatchStatement.TryKeywordRole);
			tryCatchStatement.TryBlock.AcceptVisitor(this, data);
            if(tryCatchStatement.CatchClauses.Count > 0)
                formatter.WriteComment(CommentType.SingleLine, "catch");
			//foreach (var catchClause in tryCatchStatement.CatchClauses)
			//	catchClause.AcceptVisitor(this, data);
			if (!tryCatchStatement.FinallyBlock.IsNull) {
				//WriteKeyword("finally", TryCatchStatement.FinallyKeywordRole);
                formatter.WriteComment(CommentType.SingleLine, "finally");
				tryCatchStatement.FinallyBlock.AcceptVisitor(this, data);
			}
			return EndNode(tryCatchStatement);
		}
		
		public object VisitCatchClause(CatchClause catchClause, object data)
		{
			StartNode(catchClause);
			WriteKeyword("catch");
			if (!catchClause.Type.IsNull) {
				Space(policy.SpaceBeforeCatchParentheses);
				LPar();
				Space(policy.SpacesWithinCatchParentheses);
				catchClause.Type.AcceptVisitor(this, data);
				if (!string.IsNullOrEmpty(catchClause.VariableName)) {
					Space();
					WriteIdentifier(catchClause.VariableName);
				}
				Space(policy.SpacesWithinCatchParentheses);
				RPar();
			}
			catchClause.Body.AcceptVisitor(this, data);
			return EndNode(catchClause);
		}
		
		public object VisitUncheckedStatement(UncheckedStatement uncheckedStatement, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Unchecked statements");
            //StartNode(uncheckedStatement);
            //WriteKeyword("unchecked");
            //uncheckedStatement.Body.AcceptVisitor(this, data);
            //return EndNode(uncheckedStatement);
		}
		
		public object VisitUnsafeStatement(UnsafeStatement unsafeStatement, object data)
		{
            StartNode(unsafeStatement);
            //WriteKeyword("unsafe");
            unsafeStatement.Body.AcceptVisitor(this, data);
            return EndNode(unsafeStatement);
		}
		
		public object VisitUsingStatement(UsingStatement usingStatement, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Using statements");
            //StartNode(usingStatement);
            //WriteKeyword("using");
            //Space(policy.SpaceBeforeUsingParentheses);
            //LPar();
            //Space(policy.SpacesWithinUsingParentheses);
			
            //usingStatement.ResourceAcquisition.AcceptVisitor(this, data);
			
            //Space(policy.SpacesWithinUsingParentheses);
            //RPar();
			
            //WriteEmbeddedStatement(usingStatement.EmbeddedStatement);
			
            //return EndNode(usingStatement);
		}
		
		public object VisitVariableDeclarationStatement(VariableDeclarationStatement variableDeclarationStatement, object data)
		{
			StartNode(variableDeclarationStatement);

            bool shared = CheckForAllocateShared(variableDeclarationStatement);
            if (!shared)
            {
                data = (object)true;
                variableDeclarationStatement.Type.AcceptVisitor(this, data);

                // if Type is a reference type, we need an extra '*'
                var typeReference = variableDeclarationStatement.Type.Annotations.FirstOrDefault();
                if (typeReference != null && typeReference is Mono.Cecil.TypeReference)
                    if (!(typeReference as Mono.Cecil.TypeReference).IsValueType) WriteIdentifier("*");

                Space();
                WriteCommaSeparatedList(variableDeclarationStatement.Variables);
                Semicolon();
                // here we set textLen
                var pt = variableDeclarationStatement.FirstChild as PrimitiveType;
                if (pt != null && pt.Keyword == "string")
                {
                    var vi = variableDeclarationStatement.LastChild as VariableInitializer;
                    if (vi != null)
                    {
                        string str = null;
                        var pe = vi.Initializer as PrimitiveExpression;
                        if (pe != null)
                        {
                            str = pe.Value as string;
                            WriteIdentifier(string.Format("int {0}Len = {1};\r\n", vi.Name, str.Length));
                        }
                        else
                        {
                            var ie = vi.Initializer as IdentifierExpression;
                            if (ie != null)
                            {
                                str = ie.Identifier;
                                WriteIdentifier(string.Format("int {0}Len = {1}Len;\r\n", vi.Name, str));
                            }
                        }
                    }
                }
            }
      
                
			
			return EndNode(variableDeclarationStatement);
		}

        bool CheckForAllocateShared(VariableDeclarationStatement variableDeclarationStatement)
        {
            //Console.WriteLine("variableDeclarationStatement=" + variableDeclarationStatement.ToString());
            bool isAlloc = variableDeclarationStatement.ToString().Contains(CL.csAllocateShared);
            if (isAlloc)
            {
                WriteKeyword(CudafyTranslator.LanguageSpecifics.SharedModifier);
                AstType astType = (variableDeclarationStatement.Type as ComposedType).BaseType;
                string keyword = astType.ToString();
                if (CudafyTranslator.LanguageSpecifics.Language == eLanguage.OpenCL && !(astType is ICSharpCode.NRefactory.CSharp.PrimitiveType))
                   WriteKeyword("struct");
                keyword = ConvertPrimitiveType(keyword);
                keyword = CUDALanguage.TranslateSpecialType(keyword);
                WriteKeyword(keyword);
                
                foreach (var v in variableDeclarationStatement.Variables.ToList())
                {
                    List<object> dims = new List<object>();
                    WriteIdentifier(v.Name);
                    VariableInitializer vi = v as VariableInitializer;
                    if (vi != null)
                    {
                        InvocationExpression ie = vi.Initializer as InvocationExpression;
                        if (ie != null)
                        {
                            int ctr = 0;
                            formatter.WriteToken("[");
                            int argLen = ie.Arguments.Count;
                            foreach (var arg in ie.Arguments)
                            {
                                if (ctr > 0)
                                {
                                    if (!(arg is PrimitiveExpression))
                                        throw new CudafyLanguageException(CudafyLanguageException.csSHARED_MEMORY_MUST_BE_CONSTANT);
                                    object o = (arg as PrimitiveExpression).Value;

                                    //object o;
                                    //if (arg is PrimitiveExpression)
                                    //{
                                    //    o = (arg as PrimitiveExpression).Value;
                                    //}
                                    //else if ((arg is MemberReferenceExpression))
                                    //{
                                    //    var method = GetGetMethod(((MemberReferenceExpression)arg));
                                    //    if (method == null)   // TO-DO Revise the message string: csSHARED_MEMORY_MUST_BE_CONSTANT
                                    //        throw new CudafyLanguageException(CudafyLanguageException.csSHARED_MEMORY_MUST_BE_CONSTANT);
                                    //    o = new PrimitiveExpression(method.Invoke(method.DeclaringType, null));
                                    //}
                                    //else
                                    //{
                                    //    throw new CudafyLanguageException(CudafyLanguageException.csSHARED_MEMORY_MUST_BE_CONSTANT);
                                    //}

                                    formatter.WriteIdentifier(o.ToString());
                                    if (ctr < argLen - 1)
                                        formatter.WriteKeyword("*");
                                    dims.Add(o);
                                }
                                ctr++;
                            }
                            formatter.WriteToken("]");
                        }
                    }
                    Semicolon();
                    NewLine();
                    for (int d = 0; d < dims.Count; d++)
                    {
                        formatter.WriteIdentifier(string.Format("int {0}Len{1} = {2};", v.Name, d, dims[d]));
                        NewLine();
                    }
                }


            }
            return isAlloc;
        }

        private System.Reflection.MethodInfo GetGetMethod(MemberReferenceExpression member)
        {
            var method = (from a in member.Annotations
                          where a is Mono.Cecil.PropertyDefinition
                          select ((Mono.Cecil.PropertyDefinition)a).GetMethod
                        ).FirstOrDefault();
            if (method == null) return null;
            var type = (from ass in AppDomain.CurrentDomain.GetAssemblies()
                        where ass.GetType(method.DeclaringType.FullName) != null
                        select ass.GetType(method.DeclaringType.FullName)
                        ).LastOrDefault();
            if (type == null) return null;
            return (from p in type.GetProperties(System.Reflection.BindingFlags.Static
                                               | System.Reflection.BindingFlags.Public)
                    where p.Name == member.MemberName
                    select p.GetGetMethod()
                        ).FirstOrDefault();
        }
		
		public object VisitWhileStatement(WhileStatement whileStatement, object data)
		{
			StartNode(whileStatement);
			WriteKeyword("while", WhileStatement.WhileKeywordRole);
			Space(policy.SpaceBeforeWhileParentheses);
			LPar();
			Space(policy.SpacesWithinWhileParentheses);
			whileStatement.Condition.AcceptVisitor(this, data);
			Space(policy.SpacesWithinWhileParentheses);
			RPar();
			WriteEmbeddedStatement(whileStatement.EmbeddedStatement);
			return EndNode(whileStatement);
		}
		
		public object VisitYieldBreakStatement(YieldBreakStatement yieldBreakStatement, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Yield statements");
            //StartNode(yieldBreakStatement);
            //WriteKeyword("yield", YieldBreakStatement.YieldKeywordRole);
            //WriteKeyword("break", YieldBreakStatement.BreakKeywordRole);
            //Semicolon();
            //return EndNode(yieldBreakStatement);
		}
		
		public object VisitYieldStatement(YieldStatement yieldStatement, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Yield statements");
            //StartNode(yieldStatement);
            //WriteKeyword("yield", YieldStatement.YieldKeywordRole);
            //WriteKeyword("return", YieldStatement.ReturnKeywordRole);
            //Space();
            //yieldStatement.Expression.AcceptVisitor(this, data);
            //Semicolon();
            //return EndNode(yieldStatement);
		}
		#endregion
		
		#region TypeMembers
		public object VisitAccessor(Accessor accessor, object data)
		{
            if (accessor.Role == PropertyDeclaration.GetterRole || accessor.Role == PropertyDeclaration.SetterRole)
                throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Getter and setter accessors");
            else
                throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Add and remove accessors");
            //StartNode(accessor);
            //WriteAttributes(accessor.Attributes);
            //WriteModifiers(accessor.ModifierTokens);
            //if (accessor.Role == PropertyDeclaration.GetterRole) {
            //    WriteKeyword("get");
            //} else if (accessor.Role == PropertyDeclaration.SetterRole) {
            //    WriteKeyword("set");
            //} else if (accessor.Role == CustomEventDeclaration.AddAccessorRole) {
            //    WriteKeyword("add");
            //} else if (accessor.Role == CustomEventDeclaration.RemoveAccessorRole) {
            //    WriteKeyword("remove");
            //}
            //WriteMethodBody(accessor.Body);
            //return EndNode(accessor);
		}

		public object VisitConstructorDeclaration(ConstructorDeclaration constructorDeclaration, object data)
		{
			StartNode(constructorDeclaration);
			//WriteAttributes(constructorDeclaration.Attributes);
			//WriteModifiers(constructorDeclaration.ModifierTokens);
            if (CudafyTranslator.Language == eLanguage.OpenCL)
                throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED_IN_X, "Constructor", "OpenCL");
            WriteKeyword(CudafyTranslator.LanguageSpecifics.DeviceFunctionModifiers);
            TypeDeclaration type = constructorDeclaration.Parent as TypeDeclaration;
			TypeDeclarationEx typeEx = constructorDeclaration.Parent as TypeDeclarationEx;
            if (typeEx != null)
                WriteIdentifier(typeEx.FullName);
            else
                WriteIdentifier(type.Name);
			Space(policy.SpaceBeforeConstructorDeclarationParentheses);
			WriteCommaSeparatedListInParenthesis(constructorDeclaration.Parameters, policy.SpaceWithinMethodDeclarationParentheses);
			if (!constructorDeclaration.Initializer.IsNull) {
				Space();
				constructorDeclaration.Initializer.AcceptVisitor(this, data);
			}
			WriteMethodBody(constructorDeclaration.Body);
			return EndNode(constructorDeclaration);
		}
		
		public object VisitConstructorInitializer(ConstructorInitializer constructorInitializer, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Constructor initializers");
            //StartNode(constructorInitializer);
            //WriteToken(":", ConstructorInitializer.Roles.Colon);
            //Space();
            //if (constructorInitializer.ConstructorInitializerType == ConstructorInitializerType.This) {
            //    WriteKeyword("this");
            //} else {
            //    WriteKeyword("base");
            //}
            //Space(policy.SpaceBeforeMethodCallParentheses);
            //WriteCommaSeparatedListInParenthesis(constructorInitializer.Arguments, policy.SpaceWithinMethodCallParentheses);
            //return EndNode(constructorInitializer);
		}
		
		public object VisitDestructorDeclaration(DestructorDeclaration destructorDeclaration, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Destructors");
            //StartNode(destructorDeclaration);
            //WriteAttributes(destructorDeclaration.Attributes);
            //WriteModifiers(destructorDeclaration.ModifierTokens);
            //WriteToken("~", DestructorDeclaration.TildeRole);
            //TypeDeclaration type = destructorDeclaration.Parent as TypeDeclaration;
            //WriteIdentifier(type != null ? type.Name : destructorDeclaration.Name);
            //Space(policy.SpaceBeforeConstructorDeclarationParentheses);
            //LPar();
            //RPar();
            //WriteMethodBody(destructorDeclaration.Body);
            //return EndNode(destructorDeclaration);
		}
		
		public object VisitEnumMemberDeclaration(EnumMemberDeclaration enumMemberDeclaration, object data)
		{
            //throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Enumerators");
            StartNode(enumMemberDeclaration);
            WriteAttributes(enumMemberDeclaration.Attributes);
            WriteModifiers(enumMemberDeclaration.ModifierTokens);
            WriteIdentifier(enumMemberDeclaration.Name);
            if (!enumMemberDeclaration.Initializer.IsNull)
            {
                Space(policy.SpaceAroundAssignment);
                WriteToken("=", EnumMemberDeclaration.Roles.Assign);
                Space(policy.SpaceAroundAssignment);
                enumMemberDeclaration.Initializer.AcceptVisitor(this, data);
            }
            return EndNode(enumMemberDeclaration);
		}
		
		public object VisitEventDeclaration(EventDeclaration eventDeclaration, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Events");
            //StartNode(eventDeclaration);
            //WriteAttributes(eventDeclaration.Attributes);
            //WriteModifiers(eventDeclaration.ModifierTokens);
            //WriteKeyword("event");
            //eventDeclaration.ReturnType.AcceptVisitor(this, data);
            //Space();
            //WriteCommaSeparatedList(eventDeclaration.Variables);
            //Semicolon();
            //return EndNode(eventDeclaration);
		}
		
		public object VisitCustomEventDeclaration(CustomEventDeclaration customEventDeclaration, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Events");
            //StartNode(customEventDeclaration);
            //WriteAttributes(customEventDeclaration.Attributes);
            //WriteModifiers(customEventDeclaration.ModifierTokens);
            //WriteKeyword("event");
            //customEventDeclaration.ReturnType.AcceptVisitor(this, data);
            //Space();
            //WritePrivateImplementationType(customEventDeclaration.PrivateImplementationType);
            //WriteIdentifier(customEventDeclaration.Name);
            //OpenBrace(policy.EventBraceStyle);
            //// output add/remove in their original order
            //foreach (AstNode node in customEventDeclaration.Children) {
            //    if (node.Role == CustomEventDeclaration.AddAccessorRole || node.Role == CustomEventDeclaration.RemoveAccessorRole) {
            //        node.AcceptVisitor(this, data);
            //    }
            //}
            //CloseBrace(policy.EventBraceStyle);
            //NewLine();
            //return EndNode(customEventDeclaration);
		}


        /// <summary>
        /// Gets or sets the dims of the latest CUDA Constant.
        /// </summary>
        /// <value>
        /// The constant dims.
        /// </value>
        public int[] ConstantDims { get; set; }

        /// <summary>
        /// Visits the field declaration.
        /// </summary>
        /// <param name="fieldDeclaration">The field declaration.</param>
        /// <param name="data">The data.</param>
        /// <returns></returns>
		public object VisitFieldDeclaration (FieldDeclaration fieldDeclaration, object data)
		{
            eCudafyType? ct = null;
            var fieldDeclarationEx = fieldDeclaration as FieldDeclarationEx;
            if (fieldDeclarationEx != null)
                ct = fieldDeclarationEx.CudafyType;

            StartNode(fieldDeclaration);

            if (fieldDeclarationEx.IsDummy)
            {
                string varName = fieldDeclaration.Variables.First().Name;
                string msg = string.Format("Field {0} is a dummy.", varName);
                formatter.WriteComment(CommentType.SingleLine, string.Format("Field {0} is a dummy.", varName));
            }
            else if (ct != null)
            {
                if (fieldDeclaration.Variables.Count() > 1)
                    throw new CudafyLanguageException(CudafyLanguageException.csMULTIPLE_VARIABLE_DECLARATIONS_ARE_NOT_SUPPORTED);
                if (!(fieldDeclaration.ReturnType is ComposedType) && !(fieldDeclaration.ReturnType is PrimitiveType))
                    throw new CudafyLanguageException(CudafyLanguageException.csCONSTANTS_MUST_BE_INITIALIZED);
                int ctr = 0;
                string varName = fieldDeclaration.Variables.First().Name;
                if (CudafyTranslator.LanguageSpecifics.Language == eLanguage.Cuda)
                {
                    WriteKeyword(CudafyTranslator.LanguageSpecifics.ConstantModifier);
                    if (fieldDeclaration.ReturnType is ComposedType)
                        WriteKeyword((fieldDeclaration.ReturnType as ComposedType).BaseType.ToString().Replace(".", ""));
                    else
                        WriteKeyword((fieldDeclaration.ReturnType as PrimitiveType).Keyword);
                    
                    WriteIdentifier(varName);
                    
                    if (ConstantDims.Any())
                    {
                        formatter.WriteKeyword("[");
                        foreach (int dim in ConstantDims)
                        {
                            formatter.WriteKeyword(string.Format("{0}", dim));
                            if (ctr < ConstantDims.Length - 1)
                                formatter.WriteKeyword(" * ");
                            ctr++;
                        }
                        formatter.WriteKeyword("]");
                    }
                    Semicolon();
                }
                //formatter.NewLine();
                ctr = 0;
                foreach (int dim in ConstantDims)
                {
                    WriteKeyword(string.Format("#define {0}Len{1} {2}", varName, ctr++, dim));
                    formatter.NewLine();
                }
            }
            else
            {
                //WriteAttributes (fieldDeclaration.Attributes);
                //WriteModifiers(fieldDeclaration.ModifierTokens);

                if (data != null)
                {
                    var tuple = (Tuple<string, int, bool>)data;
                    int classSize = (int)tuple.Item2;
                    
                    if (!tuple.Item3)
                    {
                        int elementSize = GetSize(tuple.Item1);
                        classSize /= elementSize;
                    }
                    string returnType = tuple.Item1;
                    returnType = returnType.TrimEnd('[', ']');
                    WriteKeyword(returnType);
                    Space();
                    WriteCommaSeparatedList(fieldDeclaration.Variables);
                    WriteIdentifier(string.Format("[{0}]", classSize));
                }
                else
                {
                    fieldDeclaration.ReturnType.AcceptVisitor(this, false);//data
                    var type = (fieldDeclaration.ReturnType is ComposedType) ? (fieldDeclaration.ReturnType as ComposedType).BaseType.Annotations.FirstOrDefault() as Mono.Cecil.TypeReference
                        : fieldDeclaration.ReturnType.Annotations.FirstOrDefault() as Mono.Cecil.TypeReference;
                    if (type != null && !type.IsValueType) formatter.WriteKeyword("*");
                    Space();
                    WriteCommaSeparatedList(fieldDeclaration.Variables);
                }
                var fieldDefinition = fieldDeclaration.Annotations.FirstOrDefault() as Mono.Cecil.FieldDefinition;
                if (fieldDefinition != null)
                {
                    var arrayType = fieldDefinition.FieldType as Mono.Cecil.ArrayType;
                    if (arrayType != null && !DisableSmartArray)
                    {
                        for (int r = 0; r < arrayType.Rank; r++)
                            formatter.WriteKeyword(string.Format("; int {0}Len{1}", fieldDefinition.Name, r));
                    }
                }
                Semicolon();
            }			
			return EndNode (fieldDeclaration);
		}

        private int GetSize(string cudatype)
        {
            switch (cudatype)
            {
                case "char":
                    return 1;
                case "unsigned char":
                    return 1;
                case "unsigned short":
                    return 2;
                case "short":
                    return 2;
                case "unsigned int":
                    return 4;
                case "int":
                    return 4;
                case "unsigned long long":
                case "ulong":
                    return 8;
                case "long long":
                case "long":
                    return 8;
                case "float":
                    return 4;
                case "double":
                    return 8;
                case "bool":
                    return 1;
                default:
                    throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED_IN_X, cudatype, "structs");
            }
        }
		
		public object VisitFixedFieldDeclaration (FixedFieldDeclaration fixedFieldDeclaration, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Fixed fields");
            //StartNode(fixedFieldDeclaration);
            //WriteAttributes(fixedFieldDeclaration.Attributes);
            //WriteModifiers(fixedFieldDeclaration.ModifierTokens);
            //WriteKeyword("fixed");
            //Space();
            //fixedFieldDeclaration.ReturnType.AcceptVisitor (this, data);
            //Space();
            //WriteCommaSeparatedList(fixedFieldDeclaration.Variables);
            //Semicolon();
            //return EndNode(fixedFieldDeclaration);
		}
		
		public object VisitFixedVariableInitializer (FixedVariableInitializer fixedVariableInitializer, object data)
		{
			StartNode(fixedVariableInitializer);
			WriteIdentifier(fixedVariableInitializer.Name);
			if (!fixedVariableInitializer.CountExpression.IsNull) {
				WriteToken("[", AstNode.Roles.LBracket);
				Space(policy.SpacesWithinBrackets);
				fixedVariableInitializer.CountExpression.AcceptVisitor(this, data);
				Space(policy.SpacesWithinBrackets);
				WriteToken("]", AstNode.Roles.RBracket);
			}
			return EndNode(fixedVariableInitializer);
		}
		
		public object VisitIndexerDeclaration(IndexerDeclaration indexerDeclaration, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Indexer declarations");
            //StartNode(indexerDeclaration);
            //WriteAttributes(indexerDeclaration.Attributes);
            //WriteModifiers(indexerDeclaration.ModifierTokens);
            //indexerDeclaration.ReturnType.AcceptVisitor(this, data);
            //WritePrivateImplementationType(indexerDeclaration.PrivateImplementationType);
            //WriteKeyword ("this");
            //Space(policy.SpaceBeforeMethodDeclarationParentheses);
            //WriteCommaSeparatedListInBrackets(indexerDeclaration.Parameters, policy.SpaceWithinMethodDeclarationParentheses);
            //OpenBrace(policy.PropertyBraceStyle);
            //// output get/set in their original order
            //foreach (AstNode node in indexerDeclaration.Children) {
            //    if (node.Role == IndexerDeclaration.GetterRole || node.Role == IndexerDeclaration.SetterRole) {
            //        node.AcceptVisitor(this, data);
            //    }
            //}
            //CloseBrace(policy.PropertyBraceStyle);
            //NewLine();
            //return EndNode(indexerDeclaration);
		}

		public object VisitMethodDeclaration(MethodDeclaration methodDeclaration, object data)
		{
            eCudafyType? ct = eCudafyType.Auto;
            eCudafyInlineMode inlineMode = eCudafyInlineMode.Auto;
            bool isDummy = false;
            var methodDeclarationEx = methodDeclaration as MethodDeclarationEx;
            if (methodDeclarationEx != null)
            {
                ct = methodDeclarationEx.CudafyType;
                isDummy = methodDeclarationEx.IsDummy;
                inlineMode = methodDeclarationEx.InlineMode;
            }

            StartNode(methodDeclaration);
            if (isDummy)
            {
                string msg = string.Format("Method {0} is a dummy.", methodDeclaration.Name);
                formatter.WriteComment(CommentType.SingleLine, string.Format("Method {0} is a dummy.", methodDeclaration.Name));
            }
            else
            {
                WriteAttributes(methodDeclaration.Attributes);
                WriteCUDAModifiers(methodDeclaration.ModifierTokens, methodDeclaration.ReturnType, ct.Value);
                WriteInlineModifiers(inlineMode);
                methodDeclaration.ReturnType.AcceptVisitor(this, data);
                Space();
                WritePrivateImplementationType(methodDeclaration.PrivateImplementationType);
                WriteIdentifier(methodDeclaration.Name);
                WriteTypeParameters(methodDeclaration.TypeParameters);
                Space(policy.SpaceBeforeMethodDeclarationParentheses);
                WriteCUDAParametersInParenthesis(methodDeclaration.Parameters, policy.SpaceWithinMethodDeclarationParentheses);
                foreach (Constraint constraint in methodDeclaration.Constraints)
                {
                    constraint.AcceptVisitor(this, data);
                }
                WriteMethodBody(methodDeclaration.Body);
            }
			return EndNode(methodDeclaration);
		}


        void WriteCUDAModifiers(IEnumerable<CSharpModifierToken> modifierTokens, AstType returnType, eCudafyType ct)
        {
            if(returnType.ToString() == "void" && ct != eCudafyType.Device && IsTypeTranslationDepth == 0)
                formatter.WriteKeyword(CudafyTranslator.LanguageSpecifics.KernelFunctionModifiers);//@"extern ""C"" __global__ ");
            else
                formatter.WriteKeyword(CudafyTranslator.LanguageSpecifics.DeviceFunctionModifiers);//@"__device__ ");
        }

        void WriteInlineModifiers(eCudafyInlineMode mode)
        {
            formatter.WriteKeyword(CudafyTranslator.LanguageSpecifics.GetInlineModifier(mode));
        }
		
		public object VisitOperatorDeclaration(OperatorDeclaration operatorDeclaration, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Operator declarations");
            //StartNode(operatorDeclaration);
            //WriteAttributes(operatorDeclaration.Attributes);
            //WriteModifiers(operatorDeclaration.ModifierTokens);
            //if (operatorDeclaration.OperatorType == OperatorType.Explicit) {
            //    WriteKeyword("explicit", OperatorDeclaration.OperatorTypeRole);
            //} else if (operatorDeclaration.OperatorType == OperatorType.Implicit) {
            //    WriteKeyword("implicit", OperatorDeclaration.OperatorTypeRole);
            //} else {
            //    operatorDeclaration.ReturnType.AcceptVisitor(this, data);
            //}
            //WriteKeyword("operator", OperatorDeclaration.OperatorKeywordRole);
            //Space();
            //if (operatorDeclaration.OperatorType == OperatorType.Explicit
            //    || operatorDeclaration.OperatorType == OperatorType.Implicit)
            //{
            //    operatorDeclaration.ReturnType.AcceptVisitor(this, data);
            //} else {
            //    WriteToken(OperatorDeclaration.GetToken(operatorDeclaration.OperatorType), OperatorDeclaration.OperatorTypeRole);
            //}
            //Space(policy.SpaceBeforeMethodDeclarationParentheses);
            //WriteCommaSeparatedListInParenthesis(operatorDeclaration.Parameters, policy.SpaceWithinMethodDeclarationParentheses);
            //WriteMethodBody(operatorDeclaration.Body);
            //return EndNode(operatorDeclaration);
		}

        private eCudafyAddressSpace? _lastAddressSpace = null;

        /// <summary>
        /// Visits the parameter declaration.
        /// </summary>
        /// <param name="parameterDeclaration">The parameter declaration.</param>
        /// <param name="data">The data.</param>
        /// <returns></returns>
		public object VisitParameterDeclaration(ParameterDeclaration parameterDeclaration, object data)
		{
			StartNode(parameterDeclaration);
            if (parameterDeclaration.Attributes.Count > 0)
            {
                //WriteAttributes(parameterDeclaration.Attributes);
            }
            //foreach(var attr in parameterDeclaration.Attributes)
            //    attr.
            //switch (parameterDeclaration.ParameterModifier) {
            //    case ParameterModifier.Ref:
            //        WriteKeyword("ref", ParameterDeclaration.ModifierRole);
            //        break;
            //    case ParameterModifier.Out:
            //        WriteKeyword("out", ParameterDeclaration.ModifierRole);
            //        break;
            //    case ParameterModifier.Params:
            //        WriteKeyword("params", ParameterDeclaration.ModifierRole);
            //        break;
            //    case ParameterModifier.This:
            //        WriteKeyword("this", ParameterDeclaration.ModifierRole);
            //        break;
            //}
			parameterDeclaration.Type.AcceptVisitor(this, data);
			if (!parameterDeclaration.Type.IsNull && !string.IsNullOrEmpty(parameterDeclaration.Name))
				Space();

            if (parameterDeclaration.ParameterModifier == ParameterModifier.Out || parameterDeclaration.ParameterModifier == ParameterModifier.Ref)
            {
                if (parameterDeclaration.Type.ToString().Contains("["))
                    throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED, "Passing array by reference");
                formatter.WriteKeyword("*");
            }
            else if (parameterDeclaration.ParameterModifier != ParameterModifier.None)
                throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED, parameterDeclaration.ParameterModifier);
            var type = (parameterDeclaration.Type is ComposedType) ? (parameterDeclaration.Type as ComposedType).BaseType.Annotations.FirstOrDefault() as Mono.Cecil.TypeReference
                : parameterDeclaration.Type.Annotations.FirstOrDefault() as Mono.Cecil.TypeReference;
            if (type != null && !type.IsValueType) 
                formatter.WriteKeyword("*");

			if (!string.IsNullOrEmpty(parameterDeclaration.Name))
				WriteIdentifier(parameterDeclaration.Name);

            ComposedType composedType = parameterDeclaration.Type as ComposedType;
            if (composedType != null)
            {
                int ctr = 0;
                if (composedType.ToString().Contains("["))
                {
                    int rank = 1 + composedType.ToString().Count(c => c == ',');
                    for (int i = 0; i < rank; i++)
                        formatter.WriteKeyword(string.Format(", int {0}Len{1}", parameterDeclaration.Name, ctr++));
                }
            }
            else
            {
                var primType = parameterDeclaration.Type as PrimitiveType;
                if (primType != null && primType.OriginalType == "String")
                {
                    formatter.WriteKeyword(string.Format(", int {0}Len", parameterDeclaration.Name));
                }
            }



			if (!parameterDeclaration.DefaultExpression.IsNull) {
                //Space(policy.SpaceAroundAssignment);
                //WriteToken("=", ParameterDeclaration.Roles.Assign);
                //Space(policy.SpaceAroundAssignment);
                //parameterDeclaration.DefaultExpression.AcceptVisitor(this, data);
			}
			return EndNode(parameterDeclaration);
		}
		
		public object VisitPropertyDeclaration(PropertyDeclaration propertyDeclaration, object data)
		{
			StartNode(propertyDeclaration);
			WriteAttributes(propertyDeclaration.Attributes);
			WriteModifiers(propertyDeclaration.ModifierTokens);
			propertyDeclaration.ReturnType.AcceptVisitor(this, data);
			Space();
			WritePrivateImplementationType(propertyDeclaration.PrivateImplementationType);
			WriteIdentifier(propertyDeclaration.Name);
			OpenBrace(policy.PropertyBraceStyle);
			// output get/set in their original order
			foreach (AstNode node in propertyDeclaration.Children) {
				if (node.Role == IndexerDeclaration.GetterRole || node.Role == IndexerDeclaration.SetterRole) {
					node.AcceptVisitor(this, data);
				}
			}
			CloseBrace(policy.PropertyBraceStyle);
			NewLine();
			return EndNode(propertyDeclaration);
		}
		#endregion
		
		#region Other nodes
		public object VisitVariableInitializer(VariableInitializer variableInitializer, object data)
		{
			StartNode(variableInitializer);
			WriteIdentifier(variableInitializer.Name);
			if (!variableInitializer.Initializer.IsNull) {
				Space(policy.SpaceAroundAssignment);
				WriteToken("=", VariableInitializer.Roles.Assign);
				Space(policy.SpaceAroundAssignment);
				variableInitializer.Initializer.AcceptVisitor(this, data);
			}
			return EndNode(variableInitializer);
		}
		
		public object VisitCompilationUnit(CompilationUnit compilationUnit, object data)
		{
			// don't do node tracking as we visit all children directly
			foreach (AstNode node in compilationUnit.Children)
				node.AcceptVisitor(this, data);
			return null;
		}
		
		public object VisitSimpleType(SimpleType simpleType, object data)
		{
			StartNode(simpleType);
            var sti = CUDALanguage.TranslateSpecialType(simpleType.Identifier);

            var typeReference = simpleType.Annotations.FirstOrDefault() as Mono.Cecil.TypeReference;
            if (typeReference != null && typeReference.IsValueType)
                WriteIdentifier(sti);
            else
                WriteIdentifier(sti);
			// no type arguments
            //WriteTypeArguments(simpleType.TypeArguments);
			return EndNode(simpleType);
		}
		
		public object VisitMemberType(MemberType memberType, object data)
		{
			StartNode(memberType);
			memberType.Target.AcceptVisitor(this, data);
			if (memberType.IsDoubleColon)
				WriteToken("::", MemberType.Roles.Dot);
			//else
			//	//WriteToken(".", MemberType.Roles.Dot);
            WriteToken("", MemberType.Roles.Dot);
            string memberName = memberType.MemberName;
            memberName = memberName.Replace('<', '_');
            memberName = memberName.Replace('>', '_');
            WriteIdentifier(memberName);
			WriteTypeArguments(memberType.TypeArguments);
			return EndNode(memberType);
		}
		
		public object VisitComposedType(ComposedType composedType, object data)
		{
			StartNode(composedType);
            if (composedType.ArraySpecifiers.Count > 0 && _lastAddressSpace == null)
            {
                ParameterDeclaration pd = composedType.Parent as ParameterDeclaration;
                bool doWriteMemorySpaceSpecifier = true;
                if (pd != null && CudafyTranslator.Language == eLanguage.OpenCL)
                {
                    foreach (var a in pd.Attributes)
                    {
                        var v = a.Attributes.FirstOrDefault();
                        if (v == null)
                            continue;
                        var arg = v.Arguments.FirstOrDefault();
                        if (arg != null)
                        {
                            if (arg.ToString().StartsWith("eCudafyAddressSpace"))
                            {
                                eCudafyAddressSpace cas = (eCudafyAddressSpace)Enum.Parse(typeof(eCudafyAddressSpace), arg.ToString().Remove(0, "eCudafyAddressSpace.".Length));
                                string asq = CudafyTranslator.LanguageSpecifics.GetAddressSpaceQualifier(cas);
                                doWriteMemorySpaceSpecifier = false;
                                WriteKeyword(asq);
                                break;
                            }
                        }
                    }
                }
                if(doWriteMemorySpaceSpecifier)
                    WriteKeyword(CudafyTranslator.LanguageSpecifics.MemorySpaceSpecifier);
            }
            _lastAddressSpace = null;
            if (CudafyTranslator.LanguageSpecifics.Language == eLanguage.OpenCL && !(composedType.BaseType is PrimitiveType))
                WriteKeyword("struct");
			composedType.BaseType.AcceptVisitor(this, data);
			if (composedType.HasNullableSpecifier)
				WriteToken("?", ComposedType.NullableRole);
			for (int i = 0; i < composedType.PointerRank; i++)
				WriteToken("*", ComposedType.PointerRole);
#warning NK100511
            if(composedType.ArraySpecifiers.Count > 0)
                WriteToken("*", ComposedType.PointerRole);
			//foreach (var node in composedType.ArraySpecifiers)
			//	node.AcceptVisitor(this, data);
			return EndNode(composedType);
		}
		
		public object VisitArraySpecifier(ArraySpecifier arraySpecifier, object data)
		{
			StartNode(arraySpecifier);
			WriteToken("[", ArraySpecifier.Roles.LBracket);
			foreach (var comma in arraySpecifier.GetChildrenByRole(ArraySpecifier.Roles.Comma)) {
				WriteSpecialsUpToNode(comma);
				formatter.WriteToken(",");
				lastWritten = LastWritten.Other;
			}
			WriteToken("]", ArraySpecifier.Roles.RBracket);
			return EndNode(arraySpecifier);
		}
		
		public object VisitPrimitiveType(PrimitiveType primitiveType, object data)
		{
			StartNode(primitiveType);
            string keyword = primitiveType.Keyword;
           // if (primitiveType.OriginalType != null)
           //     Debug.WriteLine("original type=" + primitiveType.OriginalType);
            if(data == null || (bool)data == true)
                if(primitiveType.OriginalType != "SByte")
                    keyword = ConvertPrimitiveType(primitiveType.Keyword); 
                                  
            WriteKeyword(keyword);
            if (keyword == "new")
            {
				// new() constraint
				LPar();
				RPar();
			}
			return EndNode(primitiveType);
		}

        private string ConvertPrimitiveType(string keyword)
        {
            switch (keyword)
            {
                case "sbyte":
                    return "char";
                case "byte":
                    return "unsigned char";
                case "ushort":
                    return "unsigned short";
                case "uint":
                    return "unsigned int";
                case "ulong":
                    return CudafyTranslator.LanguageSpecifics.UInt64Translation;//"unsigned long long";
                case "long":
                    return CudafyTranslator.LanguageSpecifics.Int64Translation;//"long long";
                case "decimal":
                    return "double";
                case "char":
                    return "unsigned short";
                case "bool":
                    return "bool";
                case "string":
                    return "unsigned short*";//"__wchar_t*";
                default:
                    return keyword;
            }
        }
		
		public object VisitComment(Comment comment, object data)
		{
			if (lastWritten == LastWritten.Division) {
				// When there's a comment starting after a division operator
				// "1.0 / /*comment*/a", then we need to insert a space in front of the comment.
				formatter.Space();
			}
			formatter.WriteComment(comment.CommentType, comment.Content);
			lastWritten = LastWritten.Whitespace;
			return null;
		}
		
		public object VisitTypeParameterDeclaration(TypeParameterDeclaration typeParameterDeclaration, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Type parameter declarations");
            //StartNode(typeParameterDeclaration);
            //WriteAttributes(typeParameterDeclaration.Attributes);
            //switch (typeParameterDeclaration.Variance) {
            //    case VarianceModifier.Invariant:
            //        break;
            //    case VarianceModifier.Covariant:
            //        WriteKeyword("out");
            //        break;
            //    case VarianceModifier.Contravariant:
            //        WriteKeyword("in");
            //        break;
            //    default:
            //        throw new NotSupportedException("Invalid value for VarianceModifier");
            //}
            //WriteIdentifier(typeParameterDeclaration.Name);
            //return EndNode(typeParameterDeclaration);
		}
		
		public object VisitConstraint(Constraint constraint, object data)
		{
            throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Constraints");
            //StartNode(constraint);
            //Space();
            //WriteKeyword("where");
            //WriteIdentifier(constraint.TypeParameter);
            //Space();
            //WriteToken(":", Constraint.ColonRole);
            //Space();
            //WriteCommaSeparatedList(constraint.BaseTypes);
            //return EndNode(constraint);
		}
		
		public object VisitCSharpTokenNode(CSharpTokenNode cSharpTokenNode, object data)
		{
			CSharpModifierToken mod = cSharpTokenNode as CSharpModifierToken;
			if (mod != null) {
				StartNode(mod);
				WriteKeyword(CSharpModifierToken.GetModifierName(mod.Modifier));
				return EndNode(mod);
			} else {
				throw new NotSupportedException("Should never visit individual tokens");
			}
		}
		
		public object VisitIdentifier(Identifier identifier, object data)
		{
			StartNode(identifier);
			WriteIdentifier(identifier.Name);
			return EndNode(identifier);
		}
		#endregion
		
		#region Pattern Nodes
		public object VisitPatternPlaceholder(AstNode placeholder, ICSharpCode.NRefactory.PatternMatching.Pattern pattern, object data)
		{
			StartNode(placeholder);
			pattern.AcceptVisitor(this, data);
			return EndNode(placeholder);
		}
		
		object IPatternAstVisitor<object, object>.VisitAnyNode(AnyNode anyNode, object data)
		{
			if (!string.IsNullOrEmpty(anyNode.GroupName)) {
				WriteIdentifier(anyNode.GroupName);
				WriteToken(":", AstNode.Roles.Colon);
			}
			WriteKeyword("anyNode");
			return null;
		}
		
		object IPatternAstVisitor<object, object>.VisitBackreference(Backreference backreference, object data)
		{
			WriteKeyword("backreference");
			LPar();
			WriteIdentifier(backreference.ReferencedGroupName);
			RPar();
			return null;
		}
		
		object IPatternAstVisitor<object, object>.VisitIdentifierExpressionBackreference(IdentifierExpressionBackreference identifierExpressionBackreference, object data)
		{
			WriteKeyword("identifierBackreference");
			LPar();
			WriteIdentifier(identifierExpressionBackreference.ReferencedGroupName);
			RPar();
			return null;
		}
		
		object IPatternAstVisitor<object, object>.VisitChoice(Choice choice, object data)
		{
			WriteKeyword("choice");
			Space();
			LPar();
			NewLine();
			formatter.Indent();
			foreach (INode alternative in choice) {
				VisitNodeInPattern(alternative, data);
				if (alternative != choice.Last())
					WriteToken(",", AstNode.Roles.Comma);
				NewLine();
			}
			formatter.Unindent();
			RPar();
			return null;
		}
		
		object IPatternAstVisitor<object, object>.VisitNamedNode(NamedNode namedNode, object data)
		{
			if (!string.IsNullOrEmpty(namedNode.GroupName)) {
				WriteIdentifier(namedNode.GroupName);
				WriteToken(":", AstNode.Roles.Colon);
			}
			VisitNodeInPattern(namedNode.ChildNode, data);
			return null;
		}
		
		object IPatternAstVisitor<object, object>.VisitRepeat(Repeat repeat, object data)
		{
			WriteKeyword("repeat");
			LPar();
			if (repeat.MinCount != 0 || repeat.MaxCount != int.MaxValue) {
				WriteIdentifier(repeat.MinCount.ToString());
				WriteToken(",", AstNode.Roles.Comma);
				WriteIdentifier(repeat.MaxCount.ToString());
				WriteToken(",", AstNode.Roles.Comma);
			}
			VisitNodeInPattern(repeat.ChildNode, data);
			RPar();
			return null;
		}
		
		object IPatternAstVisitor<object, object>.VisitOptionalNode(OptionalNode optionalNode, object data)
		{
			WriteKeyword("optional");
			LPar();
			VisitNodeInPattern(optionalNode.ChildNode, data);
			RPar();
			return null;
		}
		
		void VisitNodeInPattern(INode childNode, object data)
		{
			AstNode astNode = childNode as AstNode;
			if (astNode != null) {
				astNode.AcceptVisitor(this, data);
			} else {
				Pattern pattern = childNode as Pattern;
				if (pattern != null) {
					pattern.AcceptVisitor(this, data);
				} else {
					throw new InvalidOperationException("Unknown node type in pattern");
				}
			}
		}
		#endregion
    }
#pragma warning restore 1591
}

