﻿// Copyright (c) 2010 AlphaSierraPapa for the SharpDevelop Team (for details please see \doc\copyright.txt)
// This code is distributed under MIT X11 license (for details please see \doc\license.txt)

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

using ICSharpCode.NRefactory.Utils;

namespace ICSharpCode.NRefactory.TypeSystem.Implementation
{
	/// <summary>
	/// Stores a set of types and allows resolving them.
	/// </summary>
	/// <remarks>
	/// Concurrent read accesses are thread-safe, but a write access concurrent to any other access is not safe.
	/// </remarks>
	public sealed class TypeStorage : ITypeResolveContext
	{
		#region FullNameAndTypeParameterCount
		struct FullNameAndTypeParameterCount
		{
			public readonly string Namespace;
			public readonly string Name;
			public readonly int TypeParameterCount;
			
			public FullNameAndTypeParameterCount(string nameSpace, string name, int typeParameterCount)
			{
				this.Namespace = nameSpace;
				this.Name = name;
				this.TypeParameterCount = typeParameterCount;
			}
		}
		
		sealed class FullNameAndTypeParameterCountComparer : IEqualityComparer<FullNameAndTypeParameterCount>
		{
			public static readonly FullNameAndTypeParameterCountComparer Ordinal = new FullNameAndTypeParameterCountComparer(StringComparer.Ordinal);
			
			public readonly StringComparer NameComparer;
			
			public FullNameAndTypeParameterCountComparer(StringComparer nameComparer)
			{
				this.NameComparer = nameComparer;
			}
			
			public bool Equals(FullNameAndTypeParameterCount x, FullNameAndTypeParameterCount y)
			{
				return x.TypeParameterCount == y.TypeParameterCount 
					&& NameComparer.Equals(x.Name, y.Name)
					&& NameComparer.Equals(x.Namespace, y.Namespace);
			}
			
			public int GetHashCode(FullNameAndTypeParameterCount obj)
			{
				return NameComparer.GetHashCode(obj.Name) ^ NameComparer.GetHashCode(obj.Namespace) ^ obj.TypeParameterCount;
			}
		}
		#endregion
		
		#region Type Dictionary Storage
		volatile Dictionary<FullNameAndTypeParameterCount, ITypeDefinition>[] _typeDicts = {
			new Dictionary<FullNameAndTypeParameterCount, ITypeDefinition>(FullNameAndTypeParameterCountComparer.Ordinal)
		};
		readonly object dictsLock = new object();
		
		Dictionary<FullNameAndTypeParameterCount, ITypeDefinition> GetTypeDictionary(StringComparer nameComparer)
		{
			// Gets the dictionary for the specified comparer, creating it if necessary.
			// New dictionaries might be added during read accesses, so this method needs to be thread-safe,
			// as we allow concurrent read-accesses.
			var typeDicts = this._typeDicts;
			foreach (var dict in typeDicts) {
				FullNameAndTypeParameterCountComparer comparer = (FullNameAndTypeParameterCountComparer)dict.Comparer;
				if (comparer.NameComparer == nameComparer)
					return dict;
			}
			
			// ensure that no other thread can try to lazy-create this (or another) dict
			lock (dictsLock) {
				typeDicts = this._typeDicts; // fetch fresh value after locking
				// try looking for it again, maybe it was added while we were waiting for a lock
				// (double-checked locking pattern)
				foreach (var dict in typeDicts) {
					FullNameAndTypeParameterCountComparer comparer = (FullNameAndTypeParameterCountComparer)dict.Comparer;
					if (comparer.NameComparer == nameComparer)
						return dict;
				}
				
				// now create new dict
				var oldDict = typeDicts[0]; // Ordinal dict
				var newDict = new Dictionary<FullNameAndTypeParameterCount, ITypeDefinition>(
					oldDict.Count,
					new FullNameAndTypeParameterCountComparer(nameComparer));
				foreach (var pair in oldDict) {
					// don't use Add() as there might be conflicts in the target language
					newDict[pair.Key] = pair.Value;
				}
				
				// add the new dict to the array of dicts
				var newTypeDicts = new Dictionary<FullNameAndTypeParameterCount, ITypeDefinition>[typeDicts.Length + 1];
				Array.Copy(typeDicts, 0, newTypeDicts, 0, typeDicts.Length);
				newTypeDicts[typeDicts.Length] = newDict;
				this._typeDicts = newTypeDicts;
				return newDict;
			}
		}
		#endregion
		
		#region Namespace Storage
		class NamespaceEntry
		{
			public readonly string Name;
			public int ClassCount;
			
			public NamespaceEntry(string name)
			{
				this.Name = name;
			}
		}
		
		volatile Dictionary<string, NamespaceEntry>[] _namespaceDicts = {
			new Dictionary<string, NamespaceEntry>(StringComparer.Ordinal)
		};
		
		Dictionary<string, NamespaceEntry> GetNamespaceDictionary(StringComparer nameComparer)
		{
			// Gets the dictionary for the specified comparer, creating it if necessary.
			// New dictionaries might be added during read accesses, so this method needs to be thread-safe,
			// as we allow concurrent read-accesses.
			var namespaceDicts = this._namespaceDicts;
			foreach (var dict in namespaceDicts) {
				if (dict.Comparer == nameComparer)
					return dict;
			}
			
			// ensure that no other thread can try to lazy-create this (or another) dict
			lock (dictsLock) {
				namespaceDicts = this._namespaceDicts; // fetch fresh value after locking
				// try looking for it again, maybe it was added while we were waiting for a lock
				// (double-checked locking pattern)
				foreach (var dict in namespaceDicts) {
					if (dict.Comparer == nameComparer)
						return dict;
				}
				
				// now create new dict
				var newDict = _typeDicts[0].Values.GroupBy(c => c.Namespace, nameComparer)
					.Select(g => new NamespaceEntry(g.Key) { ClassCount = g.Count() })
					.ToDictionary(d => d.Name);
				
				// add the new dict to the array of dicts
				var newNamespaceDicts = new Dictionary<string, NamespaceEntry>[namespaceDicts.Length + 1];
				Array.Copy(namespaceDicts, 0, newNamespaceDicts, 0, namespaceDicts.Length);
				newNamespaceDicts[namespaceDicts.Length] = newDict;
				this._namespaceDicts = newNamespaceDicts;
				return newDict;
			}
		}
		#endregion
		
		#region ITypeResolveContext implementation
		/// <inheritdoc/>
		public ITypeDefinition GetClass(string nameSpace, string name, int typeParameterCount, StringComparer nameComparer)
		{
			if (nameSpace == null)
				throw new ArgumentNullException("nameSpace");
			if (name == null)
				throw new ArgumentNullException("name");
			if (nameComparer == null)
				throw new ArgumentNullException("nameComparer");
			
			var key = new FullNameAndTypeParameterCount(nameSpace, name, typeParameterCount);
			ITypeDefinition result;
			if (GetTypeDictionary(nameComparer).TryGetValue(key, out result))
				return result;
			else
				return null;
		}
		
		/// <inheritdoc/>
		public IEnumerable<ITypeDefinition> GetClasses()
		{
			return _typeDicts[0].Values;
		}
		
		/// <inheritdoc/>
		public IEnumerable<ITypeDefinition> GetClasses(string nameSpace, StringComparer nameComparer)
		{
			if (nameSpace == null)
				throw new ArgumentNullException("nameSpace");
			if (nameComparer == null)
				throw new ArgumentNullException("nameComparer");
			return GetClasses().Where(c => nameComparer.Equals(nameSpace, c.Namespace));
		}
		
		/// <inheritdoc/>
		public IEnumerable<string> GetNamespaces()
		{
			return _namespaceDicts[0].Keys;
		}
		
		/// <inheritdoc/>
		public string GetNamespace(string nameSpace, StringComparer nameComparer)
		{
			if (nameSpace == null)
				throw new ArgumentNullException("nameSpace");
			if (nameComparer == null)
				throw new ArgumentNullException("nameComparer");
			NamespaceEntry result;
			if (GetNamespaceDictionary(nameComparer).TryGetValue(nameSpace, out result))
				return result.Name;
			else
				return null;
		}
		#endregion
		
		#region Synchronize
		/// <summary>
		/// TypeStorage is mutable and does not provide any means for synchronization, so this method
		/// always throws a <see cref="NotSupportedException"/>.
		/// </summary>
		public ISynchronizedTypeResolveContext Synchronize()
		{
			throw new NotSupportedException();
		}
		
		/// <inheritdoc/>
		public CacheManager CacheManager {
			// TypeStorage is mutable, so caching is a bad idea.
			// We could provide a CacheToken if we update it on every modication, but
			// that's not worth the effort as TypeStorage is rarely directly used in resolve operations.
			get { return null; }
		}
		#endregion
		
		#region RemoveType
		/// <summary>
		/// Removes a type definition from this project content.
		/// </summary>
		public void RemoveType(ITypeDefinition typeDefinition)
		{
			if (typeDefinition == null)
				throw new ArgumentNullException("typeDefinition");
			var key = new FullNameAndTypeParameterCount(typeDefinition.Namespace, typeDefinition.Name, typeDefinition.TypeParameterCount);
			bool wasRemoved = false;
			foreach (var dict in _typeDicts) {
				ITypeDefinition defInDict;
				if (dict.TryGetValue(key, out defInDict)) {
					if (defInDict == typeDefinition) {
						wasRemoved = true;
						dict.Remove(key);
					}
				}
			}
			if (wasRemoved) {
				foreach (var dict in _namespaceDicts) {
					NamespaceEntry ns;
					if (dict.TryGetValue(typeDefinition.Namespace, out ns)) {
						if (--ns.ClassCount == 0)
							dict.Remove(typeDefinition.Namespace);
					}
				}
			}
		}
		#endregion
		
		#region UpdateType
		/// <summary>
		/// Adds the type definition to this project content.
		/// Replaces existing type definitions with the same name.
		/// </summary>
		public void UpdateType(ITypeDefinition typeDefinition)
		{
			if (typeDefinition == null)
				throw new ArgumentNullException("typeDefinition");
			var key = new FullNameAndTypeParameterCount(typeDefinition.Namespace, typeDefinition.Name, typeDefinition.TypeParameterCount);
			bool isNew = !_typeDicts[0].ContainsKey(key);
			foreach (var dict in _typeDicts) {
				dict[key] = typeDefinition;
			}
			if (isNew) {
				foreach (var dict in _namespaceDicts) {
					NamespaceEntry ns;
					if (dict.TryGetValue(typeDefinition.Namespace, out ns)) {
						++ns.ClassCount;
					} else {
						dict.Add(typeDefinition.Namespace, new NamespaceEntry(typeDefinition.Namespace) { ClassCount = 1 });
					}
				}
			}
		}
		#endregion
	}
}
