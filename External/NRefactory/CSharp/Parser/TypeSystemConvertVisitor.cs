﻿// Copyright (c) AlphaSierraPapa for the SharpDevelop Team (for details please see \doc\copyright.txt)
// This code is distributed under MIT X11 license (for details please see \doc\license.txt)

using ICSharpCode.NRefactory.CSharp.Resolver;
using ICSharpCode.NRefactory.TypeSystem;
using ICSharpCode.NRefactory.TypeSystem.Implementation;
using ICSharpCode.NRefactory.Utils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace ICSharpCode.NRefactory.CSharp
{
    /// <summary>
    /// Produces type and member definitions from the DOM.
    /// </summary>
    public class TypeSystemConvertVisitor : DepthFirstAstVisitor<object, IEntity>
    {
        readonly ParsedFile parsedFile;
        UsingScope usingScope;
        DefaultTypeDefinition currentTypeDefinition;
        DefaultMethod currentMethod;

        /// <summary>
        /// Creates a new TypeSystemConvertVisitor.
        /// </summary>
        /// <param name="pc">The parent project content (used as owner for the types being created).</param>
        /// <param name="fileName">The file name (used for DomRegions).</param>
        public TypeSystemConvertVisitor(IProjectContent pc, string fileName)
        {
            if (pc == null)
                throw new ArgumentNullException("pc");
            if (fileName == null)
                throw new ArgumentNullException("fileName");
            this.parsedFile = new ParsedFile(fileName, new UsingScope(pc));
            this.usingScope = parsedFile.RootUsingScope;
        }

        /// <summary>
        /// Creates a new TypeSystemConvertVisitor and initializes it with a given context.
        /// </summary>
        /// <param name="parsedFile">The parsed file to which members should be added.</param>
        /// <param name="currentUsingScope">The current using scope.</param>
        /// <param name="currentTypeDefinition">The current type definition.</param>
        public TypeSystemConvertVisitor(ParsedFile parsedFile, UsingScope currentUsingScope = null, DefaultTypeDefinition currentTypeDefinition = null)
        {
            if (parsedFile == null)
                throw new ArgumentNullException("parsedFile");
            this.parsedFile = parsedFile;
            this.usingScope = currentUsingScope ?? parsedFile.RootUsingScope;
            this.currentTypeDefinition = currentTypeDefinition;
        }

        public ParsedFile ParsedFile
        {
            get { return parsedFile; }
        }

        DomRegion MakeRegion(AstLocation start, AstLocation end)
        {
            return new DomRegion(parsedFile.FileName, start.Line, start.Column, end.Line, end.Column);
        }

        DomRegion MakeRegion(AstNode node)
        {
            if (node == null || node.IsNull)
                return DomRegion.Empty;
            else
                return MakeRegion(node.StartLocation, node.EndLocation);
        }

        DomRegion MakeBraceRegion(AstNode node)
        {
            if (node == null || node.IsNull)
                return DomRegion.Empty;
            else
                return MakeRegion(node.GetChildByRole(AstNode.Roles.LBrace).StartLocation,
                                  node.GetChildByRole(AstNode.Roles.RBrace).EndLocation);
        }

        #region Using Declarations
        // TODO: extern aliases

        public override IEntity VisitUsingDeclaration(UsingDeclaration usingDeclaration, object data)
        {
            ITypeOrNamespaceReference u = ConvertType(usingDeclaration.Import, true) as ITypeOrNamespaceReference;
            if (u != null)
                usingScope.Usings.Add(u);
            return null;
        }

        public override IEntity VisitUsingAliasDeclaration(UsingAliasDeclaration usingDeclaration, object data)
        {
            ITypeOrNamespaceReference u = ConvertType(usingDeclaration.Import, true) as ITypeOrNamespaceReference;
            if (u != null)
                usingScope.UsingAliases.Add(new KeyValuePair<string, ITypeOrNamespaceReference>(usingDeclaration.Alias, u));
            return null;
        }
        #endregion

        #region Namespace Declaration
        public override IEntity VisitNamespaceDeclaration(NamespaceDeclaration namespaceDeclaration, object data)
        {
            DomRegion region = MakeRegion(namespaceDeclaration);
            UsingScope previousUsingScope = usingScope;
            foreach (Identifier ident in namespaceDeclaration.Identifiers)
            {
                usingScope = new UsingScope(usingScope, NamespaceDeclaration.BuildQualifiedName(usingScope.NamespaceName, ident.Name));
                usingScope.Region = region;
            }
            base.VisitNamespaceDeclaration(namespaceDeclaration, data);
            parsedFile.UsingScopes.Add(usingScope); // add after visiting children so that nested scopes come first
            usingScope = previousUsingScope;
            return null;
        }
        #endregion

        // TODO: assembly attributes

        #region Type Definitions
        DefaultTypeDefinition CreateTypeDefinition(string name)
        {
            DefaultTypeDefinition newType;
            if (currentTypeDefinition != null)
            {
                newType = new DefaultTypeDefinition(currentTypeDefinition, name);
                currentTypeDefinition.InnerClasses.Add(newType);
            }
            else
            {
                newType = new DefaultTypeDefinition(usingScope.ProjectContent, usingScope.NamespaceName, name);
                parsedFile.TopLevelTypeDefinitions.Add(newType);
            }
            return newType;
        }

        public override IEntity VisitTypeDeclaration(TypeDeclaration typeDeclaration, object data)
        {
            var td = currentTypeDefinition = CreateTypeDefinition(typeDeclaration.Name);
            td.ClassType = typeDeclaration.ClassType;
            td.Region = MakeRegion(typeDeclaration);
            td.BodyRegion = MakeBraceRegion(typeDeclaration);
            td.AddDefaultConstructorIfRequired = true;

            ConvertAttributes(td.Attributes, typeDeclaration.Attributes);
            ApplyModifiers(td, typeDeclaration.Modifiers);
            if (td.ClassType == ClassType.Interface)
                td.IsAbstract = true; // interfaces are implicitly abstract
            else if (td.ClassType == ClassType.Enum || td.ClassType == ClassType.Struct)
                td.IsSealed = true; // enums/structs are implicitly sealed

            ConvertTypeParameters(td.TypeParameters, typeDeclaration.TypeParameters, typeDeclaration.Constraints);

            foreach (AstType baseType in typeDeclaration.BaseTypes)
            {
                td.BaseTypes.Add(ConvertType(baseType));
            }

            foreach (AttributedNode member in typeDeclaration.Members)
            {
                member.AcceptVisitor(this, data);
            }

            currentTypeDefinition = (DefaultTypeDefinition)currentTypeDefinition.DeclaringTypeDefinition;
            return td;
        }

        public override IEntity VisitDelegateDeclaration(DelegateDeclaration delegateDeclaration, object data)
        {
            var td = currentTypeDefinition = CreateTypeDefinition(delegateDeclaration.Name);
            td.ClassType = ClassType.Delegate;
            td.Region = MakeRegion(delegateDeclaration);
            td.BaseTypes.Add(multicastDelegateReference);

            ConvertAttributes(td.Attributes, delegateDeclaration.Attributes);
            ApplyModifiers(td, delegateDeclaration.Modifiers);
            td.IsSealed = true; // delegates are implicitly sealed

            ConvertTypeParameters(td.TypeParameters, delegateDeclaration.TypeParameters, delegateDeclaration.Constraints);

            ITypeReference returnType = ConvertType(delegateDeclaration.ReturnType);
            List<IParameter> parameters = new List<IParameter>();
            ConvertParameters(parameters, delegateDeclaration.Parameters);
            AddDefaultMethodsToDelegate(td, returnType, parameters);

            currentTypeDefinition = (DefaultTypeDefinition)currentTypeDefinition.DeclaringTypeDefinition;
            return td;
        }

        static readonly ITypeReference multicastDelegateReference = typeof(MulticastDelegate).ToTypeReference();
        static readonly IParameter delegateObjectParameter = MakeParameter(KnownTypeReference.Object, "object");
        static readonly IParameter delegateIntPtrMethodParameter = MakeParameter(typeof(IntPtr).ToTypeReference(), "method");
        static readonly IParameter delegateAsyncCallbackParameter = MakeParameter(typeof(AsyncCallback).ToTypeReference(), "callback");
        static readonly IParameter delegateResultParameter = MakeParameter(typeof(IAsyncResult).ToTypeReference(), "result");

        static IParameter MakeParameter(ITypeReference type, string name)
        {
            DefaultParameter p = new DefaultParameter(type, name);
            p.Freeze();
            return p;
        }

        /// <summary>
        /// Adds the 'Invoke', 'BeginInvoke', 'EndInvoke' methods, and a constructor, to the <paramref name="delegateType"/>.
        /// </summary>
        public static void AddDefaultMethodsToDelegate(DefaultTypeDefinition delegateType, ITypeReference returnType, IEnumerable<IParameter> parameters)
        {
            if (delegateType == null)
                throw new ArgumentNullException("delegateType");
            if (returnType == null)
                throw new ArgumentNullException("returnType");
            if (parameters == null)
                throw new ArgumentNullException("parameters");

            DomRegion region = new DomRegion(delegateType.Region.FileName, delegateType.Region.BeginLine, delegateType.Region.BeginColumn);

            DefaultMethod invoke = new DefaultMethod(delegateType, "Invoke");
            invoke.Accessibility = ICSharpCode.NRefactory.TypeSystem.Accessibility.Public;
            invoke.IsSynthetic = true;
            invoke.Parameters.AddRange(parameters);
            invoke.ReturnType = returnType;
            invoke.Region = region;
            delegateType.Methods.Add(invoke);

            DefaultMethod beginInvoke = new DefaultMethod(delegateType, "BeginInvoke");
            beginInvoke.Accessibility = ICSharpCode.NRefactory.TypeSystem.Accessibility.Public;
            beginInvoke.IsSynthetic = true;
            beginInvoke.Parameters.AddRange(invoke.Parameters);
            beginInvoke.Parameters.Add(delegateAsyncCallbackParameter);
            beginInvoke.Parameters.Add(delegateObjectParameter);
            beginInvoke.ReturnType = delegateResultParameter.Type;
            beginInvoke.Region = region;
            delegateType.Methods.Add(beginInvoke);

            DefaultMethod endInvoke = new DefaultMethod(delegateType, "EndInvoke");
            endInvoke.Accessibility = ICSharpCode.NRefactory.TypeSystem.Accessibility.Public;
            endInvoke.IsSynthetic = true;
            endInvoke.Parameters.Add(delegateResultParameter);
            endInvoke.ReturnType = invoke.ReturnType;
            endInvoke.Region = region;
            delegateType.Methods.Add(endInvoke);

            DefaultMethod ctor = new DefaultMethod(delegateType, ".ctor");
            ctor.EntityType = EntityType.Constructor;
            ctor.Accessibility = ICSharpCode.NRefactory.TypeSystem.Accessibility.Public;
            ctor.IsSynthetic = true;
            ctor.Parameters.Add(delegateObjectParameter);
            ctor.Parameters.Add(delegateIntPtrMethodParameter);
            ctor.ReturnType = delegateType;
            ctor.Region = region;
            delegateType.Methods.Add(ctor);
        }
        #endregion

        #region Fields
        public override IEntity VisitFieldDeclaration(FieldDeclaration fieldDeclaration, object data)
        {
            bool isSingleField = fieldDeclaration.Variables.Count() == 1;
            Modifiers modifiers = fieldDeclaration.Modifiers;
            DefaultField field = null;
            foreach (VariableInitializer vi in fieldDeclaration.Variables)
            {
                field = new DefaultField(currentTypeDefinition, vi.Name);

                field.Region = isSingleField ? MakeRegion(fieldDeclaration) : MakeRegion(vi);
                field.BodyRegion = MakeRegion(vi);
                ConvertAttributes(field.Attributes, fieldDeclaration.Attributes);

                ApplyModifiers(field, modifiers);
                field.IsVolatile = (modifiers & Modifiers.Volatile) != 0;
                field.IsReadOnly = (modifiers & Modifiers.Readonly) != 0;

                field.ReturnType = ConvertType(fieldDeclaration.ReturnType);
                if ((modifiers & Modifiers.Fixed) != 0)
                {
                    field.ReturnType = PointerTypeReference.Create(field.ReturnType);
                }

                if ((modifiers & Modifiers.Const) != 0)
                {
                    field.ConstantValue = ConvertConstantValue(field.ReturnType, vi.Initializer);
                }

                currentTypeDefinition.Fields.Add(field);
            }
            return isSingleField ? field : null;
        }

        public override IEntity VisitEnumMemberDeclaration(EnumMemberDeclaration enumMemberDeclaration, object data)
        {
            DefaultField field = new DefaultField(currentTypeDefinition, enumMemberDeclaration.Name);
            field.Region = field.BodyRegion = MakeRegion(enumMemberDeclaration);
            ConvertAttributes(field.Attributes, enumMemberDeclaration.Attributes);

            field.ReturnType = currentTypeDefinition;
            field.Accessibility = ICSharpCode.NRefactory.TypeSystem.Accessibility.Public;
            field.IsStatic = true;
            if (!enumMemberDeclaration.Initializer.IsNull)
            {
                field.ConstantValue = ConvertConstantValue(currentTypeDefinition, enumMemberDeclaration.Initializer);
            }
            else
            {
                throw new NotImplementedException();
            }

            currentTypeDefinition.Fields.Add(field);
            return field;
        }
        #endregion

        #region Methods
        public override IEntity VisitMethodDeclaration(MethodDeclaration methodDeclaration, object data)
        {
            DefaultMethod m = new DefaultMethod(currentTypeDefinition, methodDeclaration.Name);
            currentMethod = m; // required for resolving type parameters
            m.Region = MakeRegion(methodDeclaration);
            m.BodyRegion = MakeRegion(methodDeclaration.Body);


            ConvertTypeParameters(m.TypeParameters, methodDeclaration.TypeParameters, methodDeclaration.Constraints);
            m.ReturnType = ConvertType(methodDeclaration.ReturnType);
            ConvertAttributes(m.Attributes, methodDeclaration.Attributes.Where(s => s.AttributeTarget != "return"));
            ConvertAttributes(m.ReturnTypeAttributes, methodDeclaration.Attributes.Where(s => s.AttributeTarget == "return"));

            ApplyModifiers(m, methodDeclaration.Modifiers);
            m.IsExtensionMethod = methodDeclaration.IsExtensionMethod;

            ConvertParameters(m.Parameters, methodDeclaration.Parameters);
            if (!methodDeclaration.PrivateImplementationType.IsNull)
            {
                m.Accessibility = ICSharpCode.NRefactory.TypeSystem.Accessibility.None;
                m.InterfaceImplementations.Add(ConvertInterfaceImplementation(methodDeclaration.PrivateImplementationType, m.Name));
            }

            currentTypeDefinition.Methods.Add(m);
            currentMethod = null;
            return m;
        }

        void ConvertTypeParameters(IList<ITypeParameter> output, IEnumerable<TypeParameterDeclaration> typeParameters, IEnumerable<Constraint> constraints)
        {
            if (typeParameters.Any())
                throw new NotImplementedException();
        }

        DefaultExplicitInterfaceImplementation ConvertInterfaceImplementation(AstType interfaceType, string memberName)
        {
            return new DefaultExplicitInterfaceImplementation(ConvertType(interfaceType), memberName);
        }
        #endregion

        #region Operators
        public override IEntity VisitOperatorDeclaration(OperatorDeclaration operatorDeclaration, object data)
        {
            DefaultMethod m = new DefaultMethod(currentTypeDefinition, operatorDeclaration.Name);
            m.EntityType = EntityType.Operator;
            m.Region = MakeRegion(operatorDeclaration);
            m.BodyRegion = MakeRegion(operatorDeclaration.Body);

            m.ReturnType = ConvertType(operatorDeclaration.ReturnType);
            ConvertAttributes(m.Attributes, operatorDeclaration.Attributes.Where(s => s.AttributeTarget != "return"));
            ConvertAttributes(m.ReturnTypeAttributes, operatorDeclaration.Attributes.Where(s => s.AttributeTarget == "return"));

            ApplyModifiers(m, operatorDeclaration.Modifiers);

            ConvertParameters(m.Parameters, operatorDeclaration.Parameters);

            currentTypeDefinition.Methods.Add(m);
            return m;
        }
        #endregion

        #region Constructors
        public override IEntity VisitConstructorDeclaration(ConstructorDeclaration constructorDeclaration, object data)
        {
            Modifiers modifiers = constructorDeclaration.Modifiers;
            bool isStatic = (modifiers & Modifiers.Static) != 0;
            DefaultMethod ctor = new DefaultMethod(currentTypeDefinition, isStatic ? ".cctor" : ".ctor");
            ctor.EntityType = EntityType.Constructor;
            ctor.Region = MakeRegion(constructorDeclaration);
            if (!constructorDeclaration.Initializer.IsNull)
            {
                ctor.BodyRegion = MakeRegion(constructorDeclaration.Initializer.StartLocation, constructorDeclaration.EndLocation);
            }
            else
            {
                ctor.BodyRegion = MakeRegion(constructorDeclaration.Body);
            }
            ctor.ReturnType = currentTypeDefinition;

            ConvertAttributes(ctor.Attributes, constructorDeclaration.Attributes);
            ConvertParameters(ctor.Parameters, constructorDeclaration.Parameters);

            if (isStatic)
                ctor.IsStatic = true;
            else
                ApplyModifiers(ctor, modifiers);

            currentTypeDefinition.Methods.Add(ctor);
            return ctor;
        }
        #endregion

        #region Destructors
        public override IEntity VisitDestructorDeclaration(DestructorDeclaration destructorDeclaration, object data)
        {
            DefaultMethod dtor = new DefaultMethod(currentTypeDefinition, "Finalize");
            dtor.EntityType = EntityType.Destructor;
            dtor.Region = MakeRegion(destructorDeclaration);
            dtor.BodyRegion = MakeRegion(destructorDeclaration.Body);
            dtor.Accessibility = ICSharpCode.NRefactory.TypeSystem.Accessibility.Protected;
            dtor.IsOverride = true;
            dtor.ReturnType = KnownTypeReference.Void;

            ConvertAttributes(dtor.Attributes, destructorDeclaration.Attributes);

            currentTypeDefinition.Methods.Add(dtor);
            return dtor;
        }
        #endregion

        #region Properties / Indexers
        public override IEntity VisitPropertyDeclaration(PropertyDeclaration propertyDeclaration, object data)
        {
            DefaultProperty p = new DefaultProperty(currentTypeDefinition, propertyDeclaration.Name);
            p.Region = MakeRegion(propertyDeclaration);
            p.BodyRegion = MakeBraceRegion(propertyDeclaration);
            ApplyModifiers(p, propertyDeclaration.Modifiers);
            p.ReturnType = ConvertType(propertyDeclaration.ReturnType);
            ConvertAttributes(p.Attributes, propertyDeclaration.Attributes);
            if (!propertyDeclaration.PrivateImplementationType.IsNull)
            {
                p.Accessibility = ICSharpCode.NRefactory.TypeSystem.Accessibility.None;
                p.InterfaceImplementations.Add(ConvertInterfaceImplementation(propertyDeclaration.PrivateImplementationType, p.Name));
            }
            p.Getter = ConvertAccessor(propertyDeclaration.Getter, p.Accessibility);
            p.Setter = ConvertAccessor(propertyDeclaration.Setter, p.Accessibility);
            currentTypeDefinition.Properties.Add(p);
            return p;
        }

        public override IEntity VisitIndexerDeclaration(IndexerDeclaration indexerDeclaration, object data)
        {
            DefaultProperty p = new DefaultProperty(currentTypeDefinition, "Items");
            p.EntityType = EntityType.Indexer;
            p.Region = MakeRegion(indexerDeclaration);
            p.BodyRegion = MakeBraceRegion(indexerDeclaration);
            ApplyModifiers(p, indexerDeclaration.Modifiers);
            p.ReturnType = ConvertType(indexerDeclaration.ReturnType);
            ConvertAttributes(p.Attributes, indexerDeclaration.Attributes);
            if (!indexerDeclaration.PrivateImplementationType.IsNull)
            {
                p.Accessibility = ICSharpCode.NRefactory.TypeSystem.Accessibility.None;
                p.InterfaceImplementations.Add(ConvertInterfaceImplementation(indexerDeclaration.PrivateImplementationType, p.Name));
            }
            p.Getter = ConvertAccessor(indexerDeclaration.Getter, p.Accessibility);
            p.Setter = ConvertAccessor(indexerDeclaration.Setter, p.Accessibility);
            ConvertParameters(p.Parameters, indexerDeclaration.Parameters);
            currentTypeDefinition.Properties.Add(p);
            return p;
        }

        IAccessor ConvertAccessor(Accessor accessor, ICSharpCode.NRefactory.TypeSystem.Accessibility defaultAccessibility)
        {
            DefaultAccessor a = new DefaultAccessor();
            a.Accessibility = GetAccessibility(accessor.Modifiers) ?? defaultAccessibility;
            a.Region = MakeRegion(accessor);
            ConvertAttributes(a.Attributes, accessor.Attributes);
            return a;
        }
        #endregion

        #region Events
        public override IEntity VisitEventDeclaration(EventDeclaration eventDeclaration, object data)
        {
            bool isSingleEvent = eventDeclaration.Variables.Count() == 1;
            Modifiers modifiers = eventDeclaration.Modifiers;
            DefaultEvent ev = null;
            foreach (VariableInitializer vi in eventDeclaration.Variables)
            {
                ev = new DefaultEvent(currentTypeDefinition, vi.Name);

                ev.Region = isSingleEvent ? MakeRegion(eventDeclaration) : MakeRegion(vi);
                ev.BodyRegion = MakeRegion(vi);
                ConvertAttributes(ev.Attributes, eventDeclaration.Attributes);

                ApplyModifiers(ev, modifiers);

                ev.ReturnType = ConvertType(eventDeclaration.ReturnType);

                currentTypeDefinition.Events.Add(ev);
            }
            return isSingleEvent ? ev : null;
        }

        public override IEntity VisitCustomEventDeclaration(CustomEventDeclaration eventDeclaration, object data)
        {
            DefaultEvent e = new DefaultEvent(currentTypeDefinition, eventDeclaration.Name);
            e.Region = MakeRegion(eventDeclaration);
            e.BodyRegion = MakeBraceRegion(eventDeclaration);
            ApplyModifiers(e, eventDeclaration.Modifiers);
            e.ReturnType = ConvertType(eventDeclaration.ReturnType);
            ConvertAttributes(e.Attributes, eventDeclaration.Attributes);

            if (!eventDeclaration.PrivateImplementationType.IsNull)
            {
                e.Accessibility = ICSharpCode.NRefactory.TypeSystem.Accessibility.None;
                e.InterfaceImplementations.Add(ConvertInterfaceImplementation(eventDeclaration.PrivateImplementationType, e.Name));
            }

            e.AddAccessor = ConvertAccessor(eventDeclaration.AddAccessor, e.Accessibility);
            e.RemoveAccessor = ConvertAccessor(eventDeclaration.RemoveAccessor, e.Accessibility);

            currentTypeDefinition.Events.Add(e);
            return e;
        }
        #endregion

        #region Modifiers
        static void ApplyModifiers(DefaultTypeDefinition td, Modifiers modifiers)
        {
            td.Accessibility = GetAccessibility(modifiers) ?? (td.DeclaringTypeDefinition != null ? ICSharpCode.NRefactory.TypeSystem.Accessibility.Private : ICSharpCode.NRefactory.TypeSystem.Accessibility.Internal);
            td.IsAbstract = (modifiers & (Modifiers.Abstract | Modifiers.Static)) != 0;
            td.IsSealed = (modifiers & (Modifiers.Sealed | Modifiers.Static)) != 0;
            td.IsShadowing = (modifiers & Modifiers.New) != 0;
        }

        static void ApplyModifiers(TypeSystem.Implementation.AbstractMember m, Modifiers modifiers)
        {
            m.Accessibility = GetAccessibility(modifiers) ?? ICSharpCode.NRefactory.TypeSystem.Accessibility.Private;
            m.IsAbstract = (modifiers & Modifiers.Abstract) != 0;
            m.IsOverride = (modifiers & Modifiers.Override) != 0;
            m.IsSealed = (modifiers & Modifiers.Sealed) != 0;
            m.IsShadowing = (modifiers & Modifiers.New) != 0;
            m.IsStatic = (modifiers & Modifiers.Static) != 0;
            m.IsVirtual = (modifiers & Modifiers.Virtual) != 0;
        }

        static ICSharpCode.NRefactory.TypeSystem.Accessibility? GetAccessibility(Modifiers modifiers)
        {
            switch (modifiers & Modifiers.VisibilityMask)
            {
                case Modifiers.Private:
                    return ICSharpCode.NRefactory.TypeSystem.Accessibility.Private;
                case Modifiers.Internal:
                    return ICSharpCode.NRefactory.TypeSystem.Accessibility.Internal;
                case Modifiers.Protected | Modifiers.Internal:
                    return ICSharpCode.NRefactory.TypeSystem.Accessibility.ProtectedOrInternal;
                case Modifiers.Protected:
                    return ICSharpCode.NRefactory.TypeSystem.Accessibility.Protected;
                case Modifiers.Public:
                    return ICSharpCode.NRefactory.TypeSystem.Accessibility.Public;
                default:
                    return null;
            }
        }
        #endregion

        #region Attributes
        void ConvertAttributes(IList<IAttribute> outputList, IEnumerable<AttributeSection> attributes)
        {
            foreach (AttributeSection section in attributes)
            {
                throw new NotImplementedException();
            }
        }
        #endregion

        #region Types
        ITypeReference ConvertType(AstType type, bool isInUsingDeclaration = false)
        {
            return ConvertType(type, currentTypeDefinition, currentMethod, usingScope, isInUsingDeclaration);
        }

        internal static ITypeReference ConvertType(AstType type, ITypeDefinition parentTypeDefinition, IMethod parentMethodDefinition, UsingScope parentUsingScope, bool isInUsingDeclaration)
        {
            SimpleType s = type as SimpleType;
            if (s != null)
            {
                List<ITypeReference> typeArguments = new List<ITypeReference>();
                foreach (var ta in s.TypeArguments)
                {
                    typeArguments.Add(ConvertType(ta, parentTypeDefinition, parentMethodDefinition, parentUsingScope, isInUsingDeclaration));
                }
                if (typeArguments.Count == 0 && parentMethodDefinition != null)
                {
                    // SimpleTypeOrNamespaceReference doesn't support method type parameters,
                    // so we directly handle them here.
                    foreach (ITypeParameter tp in parentMethodDefinition.TypeParameters)
                    {
                        if (tp.Name == s.Identifier)
                            return tp;
                    }
                }
                return new SimpleTypeOrNamespaceReference(s.Identifier, typeArguments, parentTypeDefinition, parentUsingScope, isInUsingDeclaration);
            }

            PrimitiveType p = type as PrimitiveType;
            if (p != null)
            {
                switch (p.Keyword)
                {
                    case "string":
                        return KnownTypeReference.String;
                    case "int":
                        return KnownTypeReference.Int32;
                    case "uint":
                        return KnownTypeReference.UInt32;
                    case "object":
                        return KnownTypeReference.Object;
                    case "bool":
                        return KnownTypeReference.Boolean;
                    case "sbyte":
                        return KnownTypeReference.SByte;
                    case "byte":
                        return KnownTypeReference.Byte;
                    case "short":
                        return KnownTypeReference.Int16;
                    case "ushort":
                        return KnownTypeReference.UInt16;
                    case "long":
                        return KnownTypeReference.Int64;
                    case "ulong":
                        return KnownTypeReference.UInt64;
                    case "float":
                        return KnownTypeReference.Single;
                    case "double":
                        return KnownTypeReference.Double;
                    case "decimal":
                        return ReflectionHelper.ToTypeReference(TypeCode.Decimal);
                    case "char":
                        return KnownTypeReference.Char;
                    case "void":
                        return KnownTypeReference.Void;
                    default:
                        return SharedTypes.UnknownType;
                }
            }
            MemberType m = type as MemberType;
            if (m != null)
            {
                ITypeOrNamespaceReference t;
                if (m.IsDoubleColon)
                {
                    SimpleType st = m.Target as SimpleType;
                    if (st != null)
                    {
                        t = new AliasNamespaceReference(st.Identifier, parentUsingScope);
                    }
                    else
                    {
                        t = null;
                    }
                }
                else
                {
                    t = ConvertType(m.Target, parentTypeDefinition, parentMethodDefinition, parentUsingScope, isInUsingDeclaration) as ITypeOrNamespaceReference;
                }
                if (t == null)
                    return SharedTypes.UnknownType;
                List<ITypeReference> typeArguments = new List<ITypeReference>();
                foreach (var ta in m.TypeArguments)
                {
                    typeArguments.Add(ConvertType(ta, parentTypeDefinition, parentMethodDefinition, parentUsingScope, isInUsingDeclaration));
                }
                return new MemberTypeOrNamespaceReference(t, m.MemberName, typeArguments, parentTypeDefinition, parentUsingScope);
            }
            ComposedType c = type as ComposedType;
            if (c != null)
            {
                ITypeReference t = ConvertType(c.BaseType, parentTypeDefinition, parentMethodDefinition, parentUsingScope, isInUsingDeclaration);
                if (c.HasNullableSpecifier)
                {
                    t = NullableType.Create(t);
                }
                for (int i = 0; i < c.PointerRank; i++)
                {
                    t = PointerTypeReference.Create(t);
                }
                foreach (var a in c.ArraySpecifiers.Reverse())
                {
                    t = ArrayTypeReference.Create(t, a.Dimensions);
                }
                return t;
            }
            Debug.WriteLine("Unknown node used as type: " + type);
            return SharedTypes.UnknownType;
        }
        #endregion

        #region Constant Values
        IConstantValue ConvertConstantValue(ITypeReference targetType, AstNode expression)
        {
            // TODO: implement ConvertConstantValue
            return new SimpleConstantValue(targetType, null);
        }
        #endregion

        #region Parameters
        void ConvertParameters(IList<IParameter> outputList, IEnumerable<ParameterDeclaration> parameters)
        {
            foreach (ParameterDeclaration pd in parameters)
            {
                DefaultParameter p = new DefaultParameter(ConvertType(pd.Type), pd.Name);
                p.Region = MakeRegion(pd);
                ConvertAttributes(p.Attributes, pd.Attributes);
                switch (pd.ParameterModifier)
                {
                    case ParameterModifier.Ref:
                        p.IsRef = true;
                        p.Type = ByReferenceTypeReference.Create(p.Type);
                        break;
                    case ParameterModifier.Out:
                        p.IsOut = true;
                        p.Type = ByReferenceTypeReference.Create(p.Type);
                        break;
                    case ParameterModifier.Params:
                        p.IsParams = true;
                        break;
                }
                if (!pd.DefaultExpression.IsNull)
                    p.DefaultValue = ConvertConstantValue(p.Type, pd.DefaultExpression);
                outputList.Add(p);
            }
        }
        #endregion
    }
}
