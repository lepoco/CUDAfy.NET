﻿// Copyright (c) AlphaSierraPapa for the SharpDevelop Team (for details please see \doc\copyright.txt)
// This code is distributed under MIT X11 license (for details please see \doc\license.txt)

using System;
using System.Collections.Generic;
using System.Threading;
using ICSharpCode.Decompiler.Ast;
using ICSharpCode.NRefactory.TypeSystem;
using ICSharpCode.NRefactory.TypeSystem.Implementation;
using Mono.Cecil;

namespace ICSharpCode.Decompiler
{
	public class DecompilerContext
	{
		public ModuleDefinition CurrentModule;
		public CancellationToken CancellationToken;
		public TypeDefinition CurrentType;
		public MethodDefinition CurrentMethod;
		public DecompilerSettings Settings = new DecompilerSettings();
		
//		public ITypeResolveContext TypeResolveContext;
//		public IProjectContent ProjectContent;
		
		public DecompilerContext(ModuleDefinition currentModule)
		{
			if (currentModule == null)
				throw new ArgumentNullException("currentModule");
			this.CurrentModule = currentModule;
			
//			this.ProjectContent = new CecilTypeResolveContext(currentModule);
//			List<ITypeResolveContext> resolveContexts = new List<ITypeResolveContext>();
//			resolveContexts.Add(this.ProjectContent);
//			foreach (AssemblyNameReference r in currentModule.AssemblyReferences) {
//				AssemblyDefinition d = currentModule.AssemblyResolver.Resolve(r);
//				if (d != null) {
//					resolveContexts.Add(new CecilTypeResolveContext(d.MainModule));
//				}
//			}
//			this.TypeResolveContext = new CompositeTypeResolveContext(resolveContexts);
		}
		
		/// <summary>
		/// Used to pass variable names from a method to its anonymous methods.
		/// </summary>
		internal List<string> ReservedVariableNames = new List<string>();
		
		public DecompilerContext Clone()
		{
			DecompilerContext ctx = (DecompilerContext)MemberwiseClone();
			ctx.ReservedVariableNames = new List<string>(ctx.ReservedVariableNames);
			return ctx;
		}
	}
}
