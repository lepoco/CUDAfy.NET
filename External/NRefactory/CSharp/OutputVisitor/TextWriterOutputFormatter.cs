﻿// Copyright (c) AlphaSierraPapa for the SharpDevelop Team (for details please see \doc\copyright.txt)
// This code is distributed under MIT X11 license (for details please see \doc\license.txt)

using System;
using System.IO;

namespace ICSharpCode.NRefactory.CSharp
{
	/// <summary>
	/// Writes C# code into a TextWriter.
	/// </summary>
	public class TextWriterOutputFormatter : IOutputFormatter
	{
		readonly TextWriter textWriter;
		int indentation;
		bool needsIndent = true;
		
		public TextWriterOutputFormatter(TextWriter textWriter)
		{
			if (textWriter == null)
				throw new ArgumentNullException("textWriter");
			this.textWriter = textWriter;
		}
		
		public void WriteIdentifier(string ident)
		{
			WriteIndentation();
			textWriter.Write(ident);
		}
		
		public void WriteKeyword(string keyword)
		{
			WriteIndentation();
			textWriter.Write(keyword);
		}
		
		public void WriteToken(string token)
		{
			WriteIndentation();
			textWriter.Write(token);
		}
		
		public void Space()
		{
			WriteIndentation();
			textWriter.Write(' ');
		}
		
		public void OpenBrace(BraceStyle style)
		{
			WriteIndentation();
			textWriter.Write(' ');
			textWriter.Write('{');
			Indent();
			NewLine();
		}
		
		public void CloseBrace(BraceStyle style)
		{
			Unindent();
			WriteIndentation();
			textWriter.Write('}');
		}
		
		void WriteIndentation()
		{
			if (needsIndent) {
				needsIndent = false;
				for (int i = 0; i < indentation; i++) {
					textWriter.Write('\t');
				}
			}
		}
		
		public void NewLine()
		{
			textWriter.WriteLine();
			needsIndent = true;
		}
		
		public void Indent()
		{
			indentation++;
		}
		
		public void Unindent()
		{
			indentation--;
		}
		
		public void WriteComment(CommentType commentType, string content)
		{
			WriteIndentation();
			switch (commentType) {
				case CommentType.SingleLine:
					textWriter.Write("//");
					textWriter.WriteLine(content);
					break;
				case CommentType.MultiLine:
					textWriter.Write("/*");
					textWriter.Write(content);
					textWriter.Write("*/");
					break;
				case CommentType.Documentation:
					textWriter.Write("///");
					textWriter.WriteLine(content);
					break;
			}
		}
		
		public virtual void StartNode(AstNode node)
		{
		}
		
		public virtual void EndNode(AstNode node)
		{
		}
	}
}
