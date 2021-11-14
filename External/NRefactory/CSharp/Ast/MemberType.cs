﻿// 
// FullTypeName.cs
//
// Author:
//       Mike Krüger <mkrueger@novell.com>
// 
// Copyright (c) 2010 Novell, Inc (http://www.novell.com)
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ICSharpCode.NRefactory.CSharp
{
	public class MemberType : AstType
	{
		public static readonly Role<AstType> TargetRole = new Role<AstType>("Target", AstType.Null);
		
		public bool IsDoubleColon { get; set; }
		
		public AstType Target {
			get { return GetChildByRole(TargetRole); }
			set { SetChildByRole(TargetRole, value); }
		}
		
		public string MemberName {
			get {
				return GetChildByRole (Roles.Identifier).Name;
			}
			set {
				SetChildByRole (Roles.Identifier, new Identifier(value, AstLocation.Empty));
			}
		}
		
		public AstNodeCollection<AstType> TypeArguments {
			get { return GetChildrenByRole (Roles.TypeArgument); }
		}
		
		public override S AcceptVisitor<T, S> (IAstVisitor<T, S> visitor, T data)
		{
			return visitor.VisitMemberType (this, data);
		}
		
		protected internal override bool DoMatch(AstNode other, PatternMatching.Match match)
		{
			MemberType o = other as MemberType;
			return o != null && this.IsDoubleColon == o.IsDoubleColon && MatchString(this.MemberName, o.MemberName) && this.Target.DoMatch(o.Target, match);
		}
		
		public override string ToString()
		{
			StringBuilder b = new StringBuilder();
			b.Append(this.Target);
			if (IsDoubleColon)
				b.Append("::");
			else
				b.Append('.');
			b.Append(this.MemberName);
			if (this.TypeArguments.Any()) {
				b.Append('<');
				b.Append(DotNet35Compat.StringJoin(", ", this.TypeArguments));
				b.Append('>');
			}
			return b.ToString();
		}
	}
}

