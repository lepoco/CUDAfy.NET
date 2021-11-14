﻿// Copyright (c) 2010 AlphaSierraPapa for the SharpDevelop Team (for details please see \doc\copyright.txt)
// This code is distributed under MIT X11 license (for details please see \doc\license.txt)

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;

namespace ICSharpCode.NRefactory.TypeSystem.Implementation
{
	/// <summary>
	/// Default implementation for IParameter.
	/// </summary>
	public sealed class DefaultParameter : AbstractFreezable, IParameter, ISupportsInterning
	{
		string name = string.Empty;
		ITypeReference type = SharedTypes.UnknownType;
		IList<IAttribute> attributes;
		IConstantValue defaultValue;
		DomRegion region;
		byte flags;
		
		public DefaultParameter(ITypeReference type, string name)
		{
			if (type == null)
				throw new ArgumentNullException("type");
			if (name == null)
				throw new ArgumentNullException("name");
			this.type = type;
			this.name = name;
		}
		
		/// <summary>
		/// Copy constructor
		/// </summary>
		public DefaultParameter(IParameter p)
		{
			this.name = p.Name;
			this.type = p.Type;
			this.attributes = CopyList(p.Attributes);
			this.defaultValue = p.DefaultValue;
			this.region = p.Region;
			this.IsRef = p.IsRef;
			this.IsOut = p.IsOut;
			this.IsParams = p.IsParams;
		}
		
		protected override void FreezeInternal()
		{
			attributes = FreezeList(attributes);
			if (defaultValue != null)
				defaultValue.Freeze();
			base.FreezeInternal();
		}
		
		public string Name {
			get { return name; }
			set {
				if (value == null)
					throw new ArgumentNullException();
				CheckBeforeMutation();
				name = value;
			}
		}
		
		public ITypeReference Type {
			get { return type; }
			set {
				if (value == null)
					throw new ArgumentNullException();
				CheckBeforeMutation();
				type = value;
			}
		}
		
		public IList<IAttribute> Attributes {
			get {
				if (attributes == null)
					attributes = new List<IAttribute>();
				return attributes;
			}
		}
		
		public IConstantValue DefaultValue {
			get { return defaultValue; }
			set {
				CheckBeforeMutation();
				defaultValue = value;
			}
		}
		
		public object GetDefaultValue(ITypeResolveContext context)
		{
			if (defaultValue == null)
				throw new InvalidOperationException();
			else
				return defaultValue.GetValue(context);
		}
		
		public DomRegion Region {
			get { return region; }
			set {
				CheckBeforeMutation();
				region = value;
			}
		}
		
		bool HasFlag(byte flag)
		{
			return (this.flags & flag) != 0;
		}
		void SetFlag(byte flag, bool value)
		{
			CheckBeforeMutation();
			if (value)
				this.flags |= flag;
			else
				this.flags &= unchecked((byte)~flag);
		}
		
		public bool IsRef {
			get { return HasFlag(1); }
			set { SetFlag(1, value); }
		}
		
		public bool IsOut {
			get { return HasFlag(2); }
			set { SetFlag(2, value); }
		}
		
		public bool IsParams {
			get { return HasFlag(4); }
			set { SetFlag(4, value); }
		}
		
		public bool IsOptional {
			get { return this.DefaultValue != null; }
		}
		
		void ISupportsInterning.PrepareForInterning(IInterningProvider provider)
		{
			name = provider.Intern(name);
			type = provider.Intern(type);
			attributes = provider.InternList(attributes);
			defaultValue = provider.Intern(defaultValue);
		}
		
		int ISupportsInterning.GetHashCodeForInterning()
		{
			return type.GetHashCode() ^ (attributes != null ? attributes.GetHashCode() : 0) ^ (defaultValue != null ? defaultValue.GetHashCode() : 0);
		}
		
		bool ISupportsInterning.EqualsForInterning(ISupportsInterning other)
		{
			DefaultParameter p = other as DefaultParameter;
			return p != null && type == p.type && attributes == p.attributes
				&& defaultValue == p.defaultValue && region == p.region && flags == p.flags;
		}
		
		public override string ToString()
		{
			StringBuilder b = new StringBuilder();
			if (IsRef)
				b.Append("ref ");
			if (IsOut)
				b.Append("out ");
			if (IsParams)
				b.Append("params ");
			b.Append(name);
			b.Append(':');
			b.Append(type.ToString());
			if (defaultValue != null) {
				b.Append(" = ");
				b.Append(defaultValue.ToString());
			}
			return b.ToString();
		}
		
		bool IVariable.IsConst {
			get { return false; }
		}
		
		IConstantValue IVariable.ConstantValue {
			get { return null; }
		}
	}
}
