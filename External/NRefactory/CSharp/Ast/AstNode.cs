﻿// 
// AstNode.cs
//
// Author:
//       Mike Krüger <mkrueger@novell.com>
// 
// Copyright (c) 2009 Novell, Inc (http://www.novell.com)
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
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;

namespace ICSharpCode.NRefactory.CSharp
{
	public abstract class AstNode : PatternMatching.INode
	{
		#region Null
		public static readonly AstNode Null = new NullAstNode ();
		
		sealed class NullAstNode : AstNode
		{
			public override NodeType NodeType {
				get {
					return NodeType.Unknown;
				}
			}
			
			public override bool IsNull {
				get {
					return true;
				}
			}
			
			public override S AcceptVisitor<T, S> (IAstVisitor<T, S> visitor, T data)
			{
				return default (S);
			}
			
			protected internal override bool DoMatch(AstNode other, PatternMatching.Match match)
			{
				return other == null || other.IsNull;
			}
		}
		#endregion
		
		#region PatternPlaceholder
		public static implicit operator AstNode(PatternMatching.Pattern pattern)
		{
			return pattern != null ? new PatternPlaceholder(pattern) : null;
		}
		
		sealed class PatternPlaceholder : AstNode, PatternMatching.INode
		{
			readonly PatternMatching.Pattern child;
			
			public PatternPlaceholder(PatternMatching.Pattern child)
			{
				this.child = child;
			}
			
			public override NodeType NodeType {
				get { return NodeType.Pattern; }
			}
			
			public override S AcceptVisitor<T, S>(IAstVisitor<T, S> visitor, T data)
			{
				return visitor.VisitPatternPlaceholder(this, child, data);
			}
			
			protected internal override bool DoMatch(AstNode other, PatternMatching.Match match)
			{
				return child.DoMatch(other, match);
			}
			
			bool PatternMatching.INode.DoMatchCollection(Role role, PatternMatching.INode pos, PatternMatching.Match match, PatternMatching.BacktrackingInfo backtrackingInfo)
			{
				return child.DoMatchCollection(role, pos, match, backtrackingInfo);
			}
		}
		#endregion
		
		AstNode parent;
		AstNode prevSibling;
		AstNode nextSibling;
		AstNode firstChild;
		AstNode lastChild;
		Role role = RootRole;
		
		public abstract NodeType NodeType {
			get;
		}
		
		public virtual bool IsNull {
			get {
				return false;
			}
		}
		
		public virtual AstLocation StartLocation {
			get {
				var child = firstChild;
				if (child == null)
					return AstLocation.Empty;
				return child.StartLocation;
			}
		}
		
		public virtual AstLocation EndLocation {
			get {
				var child = lastChild;
				if (child == null)
					return AstLocation.Empty;
				return child.EndLocation;
			}
		}
		
		public AstNode Parent {
			get { return parent; }
		}
		
		public Role Role {
			get { return role; }
		}
		
		public AstNode NextSibling {
			get { return nextSibling; }
		}
		
		public AstNode PrevSibling {
			get { return prevSibling; }
		}
		
		public AstNode FirstChild {
			get { return firstChild; }
		}
		
		public AstNode LastChild {
			get { return lastChild; }
		}
		
		public IEnumerable<AstNode> Children {
			get {
				AstNode next;
				for (AstNode cur = firstChild; cur != null; cur = next) {
					Debug.Assert(cur.parent == this);
					// Remember next before yielding cur.
					// This allows removing/replacing nodes while iterating through the list.
					next = cur.nextSibling;
					yield return cur;
				}
			}
		}
		
		/// <summary>
		/// Gets the ancestors of this node (excluding this node itself)
		/// </summary>
		public IEnumerable<AstNode> Ancestors {
			get {
				for (AstNode cur = parent; cur != null; cur = cur.parent) {
					yield return cur;
				}
			}
		}
		
		/// <summary>
		/// Gets all descendants of this node (excluding this node itself).
		/// </summary>
		public IEnumerable<AstNode> Descendants {
			get {
				return Utils.TreeTraversal.PreOrder(this.Children, n => n.Children);
			}
		}
		
		/// <summary>
		/// Gets all descendants of this node (including this node itself).
		/// </summary>
		public IEnumerable<AstNode> DescendantsAndSelf {
			get {
				return Utils.TreeTraversal.PreOrder(this, n => n.Children);
			}
		}
		
		/// <summary>
		/// Gets the first child with the specified role.
		/// Returns the role's null object if the child is not found.
		/// </summary>
		public T GetChildByRole<T>(Role<T> role) where T : AstNode
		{
			if (role == null)
				throw new ArgumentNullException("role");
			for (var cur = firstChild; cur != null; cur = cur.nextSibling) {
				if (cur.role == role)
					return (T)cur;
			}
			return role.NullObject;
		}
		
		public AstNodeCollection<T> GetChildrenByRole<T>(Role<T> role) where T : AstNode
		{
			return new AstNodeCollection<T>(this, role);
		}
		
		protected void SetChildByRole<T>(Role<T> role, T newChild) where T : AstNode
		{
			AstNode oldChild = GetChildByRole(role);
			if (oldChild.IsNull)
				AddChild(newChild, role);
			else
				oldChild.ReplaceWith(newChild);
		}
		
		public void AddChild<T>(T child, Role<T> role) where T : AstNode
		{
			if (role == null)
				throw new ArgumentNullException("role");
			if (child == null || child.IsNull)
				return;
			if (this.IsNull)
				throw new InvalidOperationException("Cannot add children to null nodes");
			if (child.parent != null)
				throw new ArgumentException ("Node is already used in another tree.", "child");
			AddChildUnsafe(child, role);
		}
		
		/// <summary>
		/// Adds a child without performing any safety checks.
		/// </summary>
		void AddChildUnsafe(AstNode child, Role role)
		{
			child.parent = this;
			child.role = role;
			if (firstChild == null) {
				lastChild = firstChild = child;
			} else {
				lastChild.nextSibling = child;
				child.prevSibling = lastChild;
				lastChild = child;
			}
		}
		
		public void InsertChildBefore<T>(AstNode nextSibling, T child, Role<T> role) where T : AstNode
		{
			if (role == null)
				throw new ArgumentNullException("role");
			if (nextSibling == null || nextSibling.IsNull) {
				AddChild(child, role);
				return;
			}
			
			if (child == null || child.IsNull)
				return;
			if (child.parent != null)
				throw new ArgumentException ("Node is already used in another tree.", "child");
			if (nextSibling.parent != this)
				throw new ArgumentException ("NextSibling is not a child of this node.", "nextSibling");
			// No need to test for "Cannot add children to null nodes",
			// as there isn't any valid nextSibling in null nodes.
			InsertChildBeforeUnsafe(nextSibling, child, role);
		}
		
		
		void InsertChildBeforeUnsafe(AstNode nextSibling, AstNode child, Role role)
		{
			child.parent = this;
			child.role = role;
			child.nextSibling = nextSibling;
			child.prevSibling = nextSibling.prevSibling;
			
			if (nextSibling.prevSibling != null) {
				Debug.Assert(nextSibling.prevSibling.nextSibling == nextSibling);
				nextSibling.prevSibling.nextSibling = child;
			} else {
				Debug.Assert(firstChild == nextSibling);
				firstChild = child;
			}
			nextSibling.prevSibling = child;
		}
		
		public void InsertChildAfter<T>(AstNode prevSibling, T child, Role<T> role) where T : AstNode
		{
			InsertChildBefore((prevSibling == null || prevSibling.IsNull) ? firstChild : prevSibling.nextSibling, child, role);
		}
		
		/// <summary>
		/// Removes this node from its parent.
		/// </summary>
		public void Remove()
		{
			if (parent != null) {
				if (prevSibling != null) {
					Debug.Assert(prevSibling.nextSibling == this);
					prevSibling.nextSibling = nextSibling;
				} else {
					Debug.Assert(parent.firstChild == this);
					parent.firstChild = nextSibling;
				}
				if (nextSibling != null) {
					Debug.Assert(nextSibling.prevSibling == this);
					nextSibling.prevSibling = prevSibling;
				} else {
					Debug.Assert(parent.lastChild == this);
					parent.lastChild = prevSibling;
				}
				parent = null;
				role = Roles.Root;
				prevSibling = null;
				nextSibling = null;
			}
		}
		
		/// <summary>
		/// Replaces this node with the new node.
		/// </summary>
		public void ReplaceWith(AstNode newNode)
		{
			if (newNode == null || newNode.IsNull) {
				Remove();
				return;
			}
			if (newNode == this)
				return; // nothing to do...
			if (parent == null) {
				throw new InvalidOperationException(this.IsNull ? "Cannot replace the null nodes" : "Cannot replace the root node");
			}
			// Because this method doesn't statically check the new node's type with the role,
			// we perform a runtime test:
			if (!role.IsValid(newNode)) {
				throw new ArgumentException (string.Format("The new node '{0}' is not valid in the role {1}", newNode.GetType().Name, role.ToString()), "newNode");
			}
			if (newNode.parent != null) {
				// newNode is used within this tree?
				if (newNode.Ancestors.Contains(this)) {
					// e.g. "parenthesizedExpr.ReplaceWith(parenthesizedExpr.Expression);"
					// enable automatic removal
					newNode.Remove();
				} else {
					throw new ArgumentException ("Node is already used in another tree.", "newNode");
				}
			}
			
			newNode.parent = parent;
			newNode.role = role;
			newNode.prevSibling = prevSibling;
			newNode.nextSibling = nextSibling;
			if (parent != null) {
				if (prevSibling != null) {
					Debug.Assert(prevSibling.nextSibling == this);
					prevSibling.nextSibling = newNode;
				} else {
					Debug.Assert(parent.firstChild == this);
					parent.firstChild = newNode;
				}
				if (nextSibling != null) {
					Debug.Assert(nextSibling.prevSibling == this);
					nextSibling.prevSibling = newNode;
				} else {
					Debug.Assert(parent.lastChild == this);
					parent.lastChild = newNode;
				}
				parent = null;
				prevSibling = null;
				nextSibling = null;
				role = Roles.Root;
			}
		}
		
		public AstNode ReplaceWith(Func<AstNode, AstNode> replaceFunction)
		{
			if (replaceFunction == null)
				throw new ArgumentNullException("replaceFunction");
			if (parent == null) {
				throw new InvalidOperationException(this.IsNull ? "Cannot replace the null nodes" : "Cannot replace the root node");
			}
			AstNode oldParent = parent;
			AstNode oldSuccessor = nextSibling;
			Role oldRole = role;
			Remove();
			AstNode replacement = replaceFunction(this);
			if (oldSuccessor != null && oldSuccessor.parent != oldParent)
				throw new InvalidOperationException("replace function changed nextSibling of node being replaced?");
			if (!(replacement == null || replacement.IsNull)) {
				if (replacement.parent != null)
					throw new InvalidOperationException("replace function must return the root of a tree");
				if (!oldRole.IsValid(replacement)) {
					throw new InvalidOperationException (string.Format("The new node '{0}' is not valid in the role {1}", replacement.GetType().Name, oldRole.ToString()));
				}
				
				if (oldSuccessor != null)
					oldParent.InsertChildBeforeUnsafe(oldSuccessor, replacement, oldRole);
				else
					oldParent.AddChildUnsafe(replacement, oldRole);
			}
			return replacement;
		}
		
		/// <summary>
		/// Clones the whole subtree starting at this AST node.
		/// </summary>
		/// <remarks>Annotations are copied over to the new nodes; and any annotations implementing ICloneable will be cloned.</remarks>
		public AstNode Clone()
		{
			AstNode copy = (AstNode)MemberwiseClone();
			// First, reset the shallow pointer copies
			copy.parent = null;
			copy.role = Roles.Root;
			copy.firstChild = null;
			copy.lastChild = null;
			copy.prevSibling = null;
			copy.nextSibling = null;
			
			// Then perform a deep copy:
			for (AstNode cur = firstChild; cur != null; cur = cur.nextSibling) {
				copy.AddChildUnsafe(cur.Clone(), cur.role);
			}
			
			// Finally, clone the annotation, if necessary
			ICloneable copiedAnnotations = copy.annotations as ICloneable; // read from copy (for thread-safety)
			if (copiedAnnotations != null)
				copy.annotations = copiedAnnotations.Clone();
			
			return copy;
		}
		
		#region Annotation support
		// Annotations: points either null (no annotations), to the single annotation,
		// or to an AnnotationList.
		// Once it is pointed at an AnnotationList, it will never change (this allows thread-safety support by locking the list)
		object annotations;
		
		sealed class AnnotationList : List<object>, ICloneable
		{
			// There are two uses for this custom list type:
			// 1) it's private, and thus (unlike List<object>) cannot be confused with real annotations
			// 2) It allows us to simplify the cloning logic by making the list behave the same as a clonable annotation.
			public AnnotationList(int initialCapacity) : base(initialCapacity)
			{
			}
			
			public object Clone()
			{
				lock (this) {
					AnnotationList copy = new AnnotationList(this.Count);
					for (int i = 0; i < this.Count; i++) {
						object obj = this[i];
						ICloneable c = obj as ICloneable;
						copy.Add(c != null ? c.Clone() : obj);
					}
					return copy;
				}
			}
		}
		
		public void AddAnnotation(object annotation)
		{
			if (annotation == null)
				throw new ArgumentNullException("annotation");
			if (this.IsNull)
				throw new InvalidOperationException("Cannot add annotations to the null node");
		retry: // Retry until successful
			object oldAnnotation = Interlocked.CompareExchange(ref this.annotations, annotation, null);
			if (oldAnnotation == null) {
				return; // we successfully added a single annotation
			}
			AnnotationList list = oldAnnotation as AnnotationList;
			if (list == null) {
				// we need to transform the old annotation into a list
				list = new AnnotationList(4);
				list.Add(oldAnnotation);
				list.Add(annotation);
				if (Interlocked.CompareExchange(ref this.annotations, list, oldAnnotation) != oldAnnotation) {
					// the transformation failed (some other thread wrote to this.annotations first)
					goto retry;
				}
			} else {
				// once there's a list, use simple locking
				lock (list) {
					list.Add(annotation);
				}
			}
		}
		
		public void RemoveAnnotations<T>() where T : class
		{
		retry: // Retry until successful
			object oldAnnotations = this.annotations;
			AnnotationList list = oldAnnotations as AnnotationList;
			if (list != null) {
				lock (list)
					list.RemoveAll(obj => obj is T);
			} else if (oldAnnotations is T) {
				if (Interlocked.CompareExchange(ref this.annotations, null, oldAnnotations) != oldAnnotations) {
					// Operation failed (some other thread wrote to this.annotations first)
					goto retry;
				}
			}
		}
		
		public void RemoveAnnotations(Type type)
		{
			if (type == null)
				throw new ArgumentNullException("type");
		retry: // Retry until successful
			object oldAnnotations = this.annotations;
			AnnotationList list = oldAnnotations as AnnotationList;
			if (list != null) {
				lock (list)
					list.RemoveAll(obj => type.IsInstanceOfType(obj));
			} else if (type.IsInstanceOfType(oldAnnotations)) {
				if (Interlocked.CompareExchange(ref this.annotations, null, oldAnnotations) != oldAnnotations) {
					// Operation failed (some other thread wrote to this.annotations first)
					goto retry;
				}
			}
		}
		
		public T Annotation<T>() where T: class
		{
			object annotations = this.annotations;
			AnnotationList list = annotations as AnnotationList;
			if (list != null) {
				lock (list) {
					foreach (object obj in list) {
						T t = obj as T;
						if (t != null)
							return t;
					}
					return null;
				}
			} else {
				return annotations as T;
			}
		}
		
		public object Annotation(Type type)
		{
			if (type == null)
				throw new ArgumentNullException("type");
			object annotations = this.annotations;
			AnnotationList list = annotations as AnnotationList;
			if (list != null) {
				lock (list) {
					foreach (object obj in list) {
						if (type.IsInstanceOfType(obj))
							return obj;
					}
				}
			} else {
				if (type.IsInstanceOfType(annotations))
					return annotations;
			}
			return null;
		}
		
		/// <summary>
		/// Gets all annotations stored on this AstNode.
		/// </summary>
		public IEnumerable<object> Annotations {
			get {
				object annotations = this.annotations;
				AnnotationList list = annotations as AnnotationList;
				if (list != null) {
					lock (list) {
						return list.ToArray();
					}
				} else {
					if (annotations != null)
						return new object[] { annotations };
					else
						return Enumerable.Empty<object>();
				}
			}
		}
		#endregion
		
		public abstract S AcceptVisitor<T, S> (IAstVisitor<T, S> visitor, T data);
		
		#region Pattern Matching
		protected static bool MatchString(string name1, string name2)
		{
			return string.IsNullOrEmpty(name1) || name1 == name2;
		}
		
		protected internal abstract bool DoMatch(AstNode other, PatternMatching.Match match);
		
		bool PatternMatching.INode.DoMatch(PatternMatching.INode other, PatternMatching.Match match)
		{
			AstNode o = other as AstNode;
			// try matching if other is null, or if other is an AstNode
			return (other == null || o != null) && DoMatch(o, match);
		}
		
		bool PatternMatching.INode.DoMatchCollection(Role role, PatternMatching.INode pos, PatternMatching.Match match, PatternMatching.BacktrackingInfo backtrackingInfo)
		{
			AstNode o = pos as AstNode;
			return (pos == null || o != null) && DoMatch (o, match);
		}
		
		PatternMatching.INode PatternMatching.INode.NextSibling {
			get { return nextSibling; }
		}
		
		PatternMatching.INode PatternMatching.INode.FirstChild {
			get { return firstChild; }
		}
		#endregion
		
		public AstNode GetNextNode ()
		{
			if (NextSibling != null)
				return NextSibling;
			if (Parent != null)
				return Parent.GetNextNode ();
			return null;
		}

		public AstNode GetPrevNode ()
		{
			if (PrevSibling != null)
				return PrevSibling;
			if (Parent != null)
				return Parent.GetPrevNode ();
			return null;
		}
		
		// filters all non c# nodes (comments, white spaces or pre processor directives)
		public AstNode GetCSharpNodeBefore (AstNode node)
		{
			var n = node.PrevSibling;
			while (n != null) {
				if (n.Role != Roles.Comment)
					return n;
				n = n.GetPrevNode ();
			}
			return null;
		}
		
		// the Root role must be available when creating the null nodes, so we can't put it in the Roles class
		static readonly Role<AstNode> RootRole = new Role<AstNode>("Root");
		
		public static class Roles
		{
			/// <summary>
			/// Root of an abstract syntax tree.
			/// </summary>
			public static readonly Role<AstNode> Root = RootRole;
			
			// some pre defined constants for common roles
			public static readonly Role<Identifier> Identifier = new Role<Identifier>("Identifier", CSharp.Identifier.Null);
			
			public static readonly Role<BlockStatement> Body = new Role<BlockStatement>("Body", CSharp.BlockStatement.Null);
			public static readonly Role<ParameterDeclaration> Parameter = new Role<ParameterDeclaration>("Parameter");
			public static readonly Role<Expression> Argument = new Role<Expression>("Argument", CSharp.Expression.Null);
			public static readonly Role<AstType> Type = new Role<AstType>("Type", CSharp.AstType.Null);
			public static readonly Role<Expression> Expression = new Role<Expression>("Expression", CSharp.Expression.Null);
			public static readonly Role<Expression> TargetExpression = new Role<Expression>("Target", CSharp.Expression.Null);
			public readonly static Role<Expression> Condition = new Role<Expression>("Condition", CSharp.Expression.Null);
			
			public static readonly Role<TypeParameterDeclaration> TypeParameter = new Role<TypeParameterDeclaration>("TypeParameter");
			public static readonly Role<AstType> TypeArgument = new Role<AstType>("TypeArgument", CSharp.AstType.Null);
			public readonly static Role<Constraint> Constraint = new Role<Constraint>("Constraint");
			public static readonly Role<VariableInitializer> Variable = new Role<VariableInitializer>("Variable");
			public static readonly Role<Statement> EmbeddedStatement = new Role<Statement>("EmbeddedStatement", CSharp.Statement.Null);
			
			public static readonly Role<CSharpTokenNode> Keyword = new Role<CSharpTokenNode>("Keyword", CSharpTokenNode.Null);
			public static readonly Role<CSharpTokenNode> InKeyword = new Role<CSharpTokenNode>("InKeyword", CSharpTokenNode.Null);
			
			// some pre defined constants for most used punctuation
			public static readonly Role<CSharpTokenNode> LPar = new Role<CSharpTokenNode>("LPar", CSharpTokenNode.Null);
			public static readonly Role<CSharpTokenNode> RPar = new Role<CSharpTokenNode>("RPar", CSharpTokenNode.Null);
			public static readonly Role<CSharpTokenNode> LBracket = new Role<CSharpTokenNode>("LBracket", CSharpTokenNode.Null);
			public static readonly Role<CSharpTokenNode> RBracket = new Role<CSharpTokenNode>("RBracket", CSharpTokenNode.Null);
			public static readonly Role<CSharpTokenNode> LBrace = new Role<CSharpTokenNode>("LBrace", CSharpTokenNode.Null);
			public static readonly Role<CSharpTokenNode> RBrace = new Role<CSharpTokenNode>("RBrace", CSharpTokenNode.Null);
			public static readonly Role<CSharpTokenNode> LChevron = new Role<CSharpTokenNode>("LChevron", CSharpTokenNode.Null);
			public static readonly Role<CSharpTokenNode> RChevron = new Role<CSharpTokenNode>("RChevron", CSharpTokenNode.Null);
			public static readonly Role<CSharpTokenNode> Comma = new Role<CSharpTokenNode>("Comma", CSharpTokenNode.Null);
			public static readonly Role<CSharpTokenNode> Dot = new Role<CSharpTokenNode>("Dot", CSharpTokenNode.Null);
			public static readonly Role<CSharpTokenNode> Semicolon = new Role<CSharpTokenNode>("Semicolon", CSharpTokenNode.Null);
			public static readonly Role<CSharpTokenNode> Assign = new Role<CSharpTokenNode>("Assign", CSharpTokenNode.Null);
			public static readonly Role<CSharpTokenNode> Colon = new Role<CSharpTokenNode>("Colon", CSharpTokenNode.Null);
			
			public static readonly Role<Comment> Comment = new Role<Comment>("Comment");
			
		}
	}
}
