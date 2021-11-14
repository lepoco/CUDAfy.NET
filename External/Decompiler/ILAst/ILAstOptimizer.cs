// Copyright (c) 2011 AlphaSierraPapa for the SharpDevelop Team
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
// to whom the Software is furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using ICSharpCode.Decompiler.FlowAnalysis;
using ICSharpCode.NRefactory.Utils;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Mono.CSharp;

namespace ICSharpCode.Decompiler.ILAst
{
	public enum ILAstOptimizationStep
	{
		RemoveRedundantCode,
		ReduceBranchInstructionSet,
		InlineVariables,
		CopyPropagation,
		YieldReturn,
		PropertyAccessInstructions,
		SplitToMovableBlocks,
		TypeInference,
		SimplifyShortCircuit,
		SimplifyTernaryOperator,
		SimplifyNullCoalescing,
		JoinBasicBlocks,
		TransformDecimalCtorToConstant,
		SimplifyLdObjAndStObj,
		TransformArrayInitializers,
		TransformObjectInitializers,
		MakeAssignmentExpression,
		IntroducePostIncrement,
		InlineVariables2,
		FindLoops,
		FindConditions,
		FlattenNestedMovableBlocks,
		RemoveRedundantCode2,
		GotoRemoval,
		DuplicateReturns,
		ReduceIfNesting,
		InlineVariables3,
		CachedDelegateInitialization,
		IntroduceFixedStatements,
		RecombineVariables,
		TypeInference2,
		RemoveRedundantCode3,
		None
	}
	
	public partial class ILAstOptimizer
	{
		int nextLabelIndex = 0;
		
		DecompilerContext context;
		TypeSystem typeSystem;
		ILBlock method;
		
		public void Optimize(DecompilerContext context, ILBlock method, ILAstOptimizationStep abortBeforeStep = ILAstOptimizationStep.None)
		{
			this.context = context;
			this.typeSystem = context.CurrentMethod.Module.TypeSystem;
			this.method = method;
			
			if (abortBeforeStep == ILAstOptimizationStep.RemoveRedundantCode) return;
			RemoveRedundantCode(method);
			
			if (abortBeforeStep == ILAstOptimizationStep.ReduceBranchInstructionSet) return;
			foreach(ILBlock block in method.GetSelfAndChildrenRecursive<ILBlock>()) {
				ReduceBranchInstructionSet(block);
			}
			// ReduceBranchInstructionSet runs before inlining because the non-aggressive inlining heuristic
			// looks at which type of instruction consumes the inlined variable.
			
			if (abortBeforeStep == ILAstOptimizationStep.InlineVariables) return;
			// Works better after simple goto removal because of the following debug pattern: stloc X; br Next; Next:; ldloc X
			ILInlining inlining1 = new ILInlining(method);
			inlining1.InlineAllVariables();
			
			if (abortBeforeStep == ILAstOptimizationStep.CopyPropagation) return;
			inlining1.CopyPropagation();
			
			if (abortBeforeStep == ILAstOptimizationStep.YieldReturn) return;
			YieldReturnDecompiler.Run(context, method);
			
			if (abortBeforeStep == ILAstOptimizationStep.PropertyAccessInstructions) return;
			IntroducePropertyAccessInstructions(method);
			
			if (abortBeforeStep == ILAstOptimizationStep.SplitToMovableBlocks) return;
			foreach(ILBlock block in method.GetSelfAndChildrenRecursive<ILBlock>()) {
				SplitToBasicBlocks(block);
			}
			
			if (abortBeforeStep == ILAstOptimizationStep.TypeInference) return;
			// Types are needed for the ternary operator optimization
			TypeAnalysis.Run(context, method);
			
			foreach(ILBlock block in method.GetSelfAndChildrenRecursive<ILBlock>()) {
				bool modified;
				do {
					modified = false;
					
					if (abortBeforeStep == ILAstOptimizationStep.SimplifyShortCircuit) return;
					modified |= block.RunOptimization(new SimpleControlFlow(context, method).SimplifyShortCircuit);
					
					if (abortBeforeStep == ILAstOptimizationStep.SimplifyTernaryOperator) return;
					modified |= block.RunOptimization(new SimpleControlFlow(context, method).SimplifyTernaryOperator);
					
					if (abortBeforeStep == ILAstOptimizationStep.SimplifyNullCoalescing) return;
					modified |= block.RunOptimization(new SimpleControlFlow(context, method).SimplifyNullCoalescing);
					
					if (abortBeforeStep == ILAstOptimizationStep.JoinBasicBlocks) return;
					modified |= block.RunOptimization(new SimpleControlFlow(context, method).JoinBasicBlocks);
					
					if (abortBeforeStep == ILAstOptimizationStep.TransformDecimalCtorToConstant) return;
					modified |= block.RunOptimization(TransformDecimalCtorToConstant);
					modified |= block.RunOptimization(SimplifyLdcI4ConvI8);
					
					if (abortBeforeStep == ILAstOptimizationStep.SimplifyLdObjAndStObj) return;
					modified |= block.RunOptimization(SimplifyLdObjAndStObj);
					
					if (abortBeforeStep == ILAstOptimizationStep.TransformArrayInitializers) return;
					modified |= block.RunOptimization(TransformArrayInitializers);
					
					if (abortBeforeStep == ILAstOptimizationStep.TransformObjectInitializers) return;
					modified |= block.RunOptimization(TransformObjectInitializers);
					
					if (abortBeforeStep == ILAstOptimizationStep.MakeAssignmentExpression) return;
					modified |= block.RunOptimization(MakeAssignmentExpression);
					modified |= block.RunOptimization(MakeCompoundAssignments);
					
					if (abortBeforeStep == ILAstOptimizationStep.IntroducePostIncrement) return;
					modified |= block.RunOptimization(IntroducePostIncrement);
					
					if (abortBeforeStep == ILAstOptimizationStep.InlineVariables2) return;
					modified |= new ILInlining(method).InlineAllInBlock(block);
					new ILInlining(method).CopyPropagation();
					
				} while(modified);
			}
			
			if (abortBeforeStep == ILAstOptimizationStep.FindLoops) return;
			foreach(ILBlock block in method.GetSelfAndChildrenRecursive<ILBlock>()) {
				new LoopsAndConditions(context).FindLoops(block);
			}
			
			if (abortBeforeStep == ILAstOptimizationStep.FindConditions) return;
			foreach(ILBlock block in method.GetSelfAndChildrenRecursive<ILBlock>()) {
				new LoopsAndConditions(context).FindConditions(block);
			}
			
			if (abortBeforeStep == ILAstOptimizationStep.FlattenNestedMovableBlocks) return;
			FlattenBasicBlocks(method);
			
			if (abortBeforeStep == ILAstOptimizationStep.RemoveRedundantCode2) return;
			RemoveRedundantCode(method);
			
			if (abortBeforeStep == ILAstOptimizationStep.GotoRemoval) return;
			new GotoRemoval().RemoveGotos(method);
			
			if (abortBeforeStep == ILAstOptimizationStep.DuplicateReturns) return;
			DuplicateReturnStatements(method);
			
			if (abortBeforeStep == ILAstOptimizationStep.ReduceIfNesting) return;
			ReduceIfNesting(method);
			
			if (abortBeforeStep == ILAstOptimizationStep.InlineVariables3) return;
			// The 2nd inlining pass is necessary because DuplicateReturns and the introduction of ternary operators
			// open up additional inlining possibilities.
			new ILInlining(method).InlineAllVariables();
			
			if (abortBeforeStep == ILAstOptimizationStep.CachedDelegateInitialization) return;
			foreach(ILBlock block in method.GetSelfAndChildrenRecursive<ILBlock>()) {
				for (int i = 0; i < block.Body.Count; i++) {
					// TODO: Move before loops
					CachedDelegateInitializationWithField(block, ref i);
					CachedDelegateInitializationWithLocal(block, ref i);
				}
			}
			
			if (abortBeforeStep == ILAstOptimizationStep.IntroduceFixedStatements) return;
			// we need post-order traversal, not pre-order, for "fixed" to work correctly
			foreach (ILBlock block in TreeTraversal.PostOrder<ILNode>(method, n => n.GetChildren()).OfType<ILBlock>()) {
				for (int i = block.Body.Count - 1; i >= 0; i--) {
					// TODO: Move before loops
					if (i < block.Body.Count)
						IntroduceFixedStatements(block.Body, i);
				}
			}
			
			if (abortBeforeStep == ILAstOptimizationStep.RecombineVariables) return;
			RecombineVariables(method);
			
			if (abortBeforeStep == ILAstOptimizationStep.TypeInference2) return;
			TypeAnalysis.Reset(method);
			TypeAnalysis.Run(context, method);
			
			if (abortBeforeStep == ILAstOptimizationStep.RemoveRedundantCode3) return;
			GotoRemoval.RemoveRedundantCode(method);
			
			// ReportUnassignedILRanges(method);
		}
		
		/// <summary>
		/// Removes redundatant Br, Nop, Dup, Pop
		/// </summary>
		/// <param name="method"></param>
		void RemoveRedundantCode(ILBlock method)
		{
			Dictionary<ILLabel, int> labelRefCount = new Dictionary<ILLabel, int>();
			foreach (ILLabel target in method.GetSelfAndChildrenRecursive<ILExpression>(e => e.IsBranch()).SelectMany(e => e.GetBranchTargets())) {
				labelRefCount[target] = labelRefCount.GetOrDefault(target) + 1;
			}
			
			foreach(ILBlock block in method.GetSelfAndChildrenRecursive<ILBlock>()) {
				List<ILNode> body = block.Body;
				List<ILNode> newBody = new List<ILNode>(body.Count);
				for (int i = 0; i < body.Count; i++) {
					ILLabel target;
					ILExpression popExpr;
					if (body[i].Match(ILCode.Br, out target) && i+1 < body.Count && body[i+1] == target) {
						// Ignore the branch
						if (labelRefCount[target] == 1)
							i++;  // Ignore the label as well
					} else if (body[i].Match(ILCode.Nop)){
						// Ignore nop
					} else if (body[i].Match(ILCode.Pop, out popExpr)) {
						ILVariable v;
						if (!popExpr.Match(ILCode.Ldloc, out v))
							throw new Exception("Pop should have just ldloc at this stage");
						// Best effort to move the ILRange to previous statement
						ILVariable prevVar;
						ILExpression prevExpr;
						if (i - 1 >= 0 && body[i - 1].Match(ILCode.Stloc, out prevVar, out prevExpr) && prevVar == v)
							prevExpr.ILRanges.AddRange(((ILExpression)body[i]).ILRanges);
						// Ignore pop
					} else {
						newBody.Add(body[i]);
					}
				}
				block.Body = newBody;
			}
			
			// 'dup' removal
			foreach (ILExpression expr in method.GetSelfAndChildrenRecursive<ILExpression>()) {
				for (int i = 0; i < expr.Arguments.Count; i++) {
					ILExpression child;
					if (expr.Arguments[i].Match(ILCode.Dup, out child)) {
						child.ILRanges.AddRange(expr.Arguments[i].ILRanges);
						expr.Arguments[i] = child;
					}
				}
			}
		}
		
		/// <summary>
		/// Reduces the branch codes to just br and brtrue.
		/// Moves ILRanges to the branch argument
		/// </summary>
		void ReduceBranchInstructionSet(ILBlock block)
		{
			for (int i = 0; i < block.Body.Count; i++) {
				ILExpression expr = block.Body[i] as ILExpression;
				if (expr != null && expr.Prefixes == null) {
					switch(expr.Code) {
						case ILCode.Switch:
						case ILCode.Brtrue:
							expr.Arguments.Single().ILRanges.AddRange(expr.ILRanges);
							expr.ILRanges.Clear();
							continue;
							case ILCode.__Brfalse:  block.Body[i] = new ILExpression(ILCode.Brtrue, expr.Operand, new ILExpression(ILCode.LogicNot, null, expr.Arguments.Single())); break;
							case ILCode.__Beq:      block.Body[i] = new ILExpression(ILCode.Brtrue, expr.Operand, new ILExpression(ILCode.Ceq, null, expr.Arguments)); break;
							case ILCode.__Bne_Un:   block.Body[i] = new ILExpression(ILCode.Brtrue, expr.Operand, new ILExpression(ILCode.LogicNot, null, new ILExpression(ILCode.Ceq, null, expr.Arguments))); break;
							case ILCode.__Bgt:      block.Body[i] = new ILExpression(ILCode.Brtrue, expr.Operand, new ILExpression(ILCode.Cgt, null, expr.Arguments)); break;
							case ILCode.__Bgt_Un:   block.Body[i] = new ILExpression(ILCode.Brtrue, expr.Operand, new ILExpression(ILCode.Cgt_Un, null, expr.Arguments)); break;
							case ILCode.__Ble:      block.Body[i] = new ILExpression(ILCode.Brtrue, expr.Operand, new ILExpression(ILCode.LogicNot, null, new ILExpression(ILCode.Cgt, null, expr.Arguments))); break;
							case ILCode.__Ble_Un:   block.Body[i] = new ILExpression(ILCode.Brtrue, expr.Operand, new ILExpression(ILCode.LogicNot, null, new ILExpression(ILCode.Cgt_Un, null, expr.Arguments))); break;
							case ILCode.__Blt:      block.Body[i] = new ILExpression(ILCode.Brtrue, expr.Operand, new ILExpression(ILCode.Clt, null, expr.Arguments)); break;
							case ILCode.__Blt_Un:   block.Body[i] = new ILExpression(ILCode.Brtrue, expr.Operand, new ILExpression(ILCode.Clt_Un, null, expr.Arguments)); break;
							case ILCode.__Bge:	    block.Body[i] = new ILExpression(ILCode.Brtrue, expr.Operand, new ILExpression(ILCode.LogicNot, null, new ILExpression(ILCode.Clt, null, expr.Arguments))); break;
							case ILCode.__Bge_Un:   block.Body[i] = new ILExpression(ILCode.Brtrue, expr.Operand, new ILExpression(ILCode.LogicNot, null, new ILExpression(ILCode.Clt_Un, null, expr.Arguments))); break;
						default:
							continue;
					}
					((ILExpression)block.Body[i]).Arguments.Single().ILRanges.AddRange(expr.ILRanges);
				}
			}
		}
		
		/// <summary>
		/// Converts call and callvirt instructions that read/write properties into CallGetter/CallSetter instructions.
		/// 
		/// CallGetter/CallSetter is used to allow the ILAst to represent "while ((SomeProperty = value) != null)".
		/// </summary>
		void IntroducePropertyAccessInstructions(ILNode node)
		{
			ILExpression parentExpr = node as ILExpression;
			if (parentExpr != null) {
				for (int i = 0; i < parentExpr.Arguments.Count; i++) {
					ILExpression expr = parentExpr.Arguments[i];
					IntroducePropertyAccessInstructions(expr);
					IntroducePropertyAccessInstructions(expr, parentExpr, i);
				}
			} else {
				foreach (ILNode child in node.GetChildren()) {
					IntroducePropertyAccessInstructions(child);
					ILExpression expr = child as ILExpression;
					if (expr != null) {
						IntroducePropertyAccessInstructions(expr, null, -1);
					}
				}
			}
		}
		
		void IntroducePropertyAccessInstructions(ILExpression expr, ILExpression parentExpr, int posInParent)
		{
			if (expr.Code == ILCode.Call || expr.Code == ILCode.Callvirt) {
				MethodReference cecilMethod = (MethodReference)expr.Operand;
				if (cecilMethod.DeclaringType is ArrayType) {
					switch (cecilMethod.Name) {
						case "Get":
							expr.Code = ILCode.CallGetter;
							break;
						case "Set":
							expr.Code = ILCode.CallSetter;
							break;
						case "Address":
							ByReferenceType brt = cecilMethod.ReturnType as ByReferenceType;
							if (brt != null) {
								MethodReference getMethod = new MethodReference("Get", brt.ElementType, cecilMethod.DeclaringType);
								foreach (var p in cecilMethod.Parameters)
									getMethod.Parameters.Add(p);
								getMethod.HasThis = cecilMethod.HasThis;
								expr.Operand = getMethod;
							}
							expr.Code = ILCode.CallGetter;
							if (parentExpr != null) {
								parentExpr.Arguments[posInParent] = new ILExpression(ILCode.AddressOf, null, expr);
							}
							break;
					}
				} else {
					MethodDefinition cecilMethodDef = cecilMethod.Resolve();
					if (cecilMethodDef != null) {
						if (cecilMethodDef.IsGetter)
							expr.Code = (expr.Code == ILCode.Call) ? ILCode.CallGetter : ILCode.CallvirtGetter;
						else if (cecilMethodDef.IsSetter)
							expr.Code = (expr.Code == ILCode.Call) ? ILCode.CallSetter : ILCode.CallvirtSetter;
					}
				}
			}
		}
		
		/// <summary>
		/// Group input into a set of blocks that can be later arbitraliby schufled.
		/// The method adds necessary branches to make control flow between blocks
		/// explicit and thus order independent.
		/// </summary>
		void SplitToBasicBlocks(ILBlock block)
		{
			List<ILNode> basicBlocks = new List<ILNode>();
			
			ILLabel entryLabel = block.Body.FirstOrDefault() as ILLabel ?? new ILLabel() { Name = "Block_" + (nextLabelIndex++) };
			ILBasicBlock basicBlock = new ILBasicBlock();
			basicBlocks.Add(basicBlock);
			basicBlock.Body.Add(entryLabel);
			block.EntryGoto = new ILExpression(ILCode.Br, entryLabel);
			
			if (block.Body.Count > 0) {
				if (block.Body[0] != entryLabel)
					basicBlock.Body.Add(block.Body[0]);
				
				for (int i = 1; i < block.Body.Count; i++) {
					ILNode lastNode = block.Body[i - 1];
					ILNode currNode = block.Body[i];
					
					// Start a new basic block if necessary
					if (currNode is ILLabel ||
					    currNode is ILTryCatchBlock || // Counts as label
					    lastNode.IsConditionalControlFlow() ||
					    lastNode.IsUnconditionalControlFlow())
					{
						// Try to reuse the label
						ILLabel label = currNode is ILLabel ? ((ILLabel)currNode) : new ILLabel() { Name = "Block_" + (nextLabelIndex++) };
						
						// Terminate the last block
						if (!lastNode.IsUnconditionalControlFlow()) {
							// Explicit branch from one block to other
							basicBlock.Body.Add(new ILExpression(ILCode.Br, label));
						}
						
						// Start the new block
						basicBlock = new ILBasicBlock();
						basicBlocks.Add(basicBlock);
						basicBlock.Body.Add(label);
						
						// Add the node to the basic block
						if (currNode != label)
							basicBlock.Body.Add(currNode);
					} else {
						basicBlock.Body.Add(currNode);
					}
				}
			}
			
			block.Body = basicBlocks;
			return;
		}
		
		void DuplicateReturnStatements(ILBlock method)
		{
			Dictionary<ILLabel, ILNode> nextSibling = new Dictionary<ILLabel, ILNode>();
			
			// Build navigation data
			foreach(ILBlock block in method.GetSelfAndChildrenRecursive<ILBlock>()) {
				for (int i = 0; i < block.Body.Count - 1; i++) {
					ILLabel curr = block.Body[i] as ILLabel;
					if (curr != null) {
						nextSibling[curr] = block.Body[i + 1];
					}
				}
			}
			
			// Duplicate returns
			foreach(ILBlock block in method.GetSelfAndChildrenRecursive<ILBlock>()) {
				for (int i = 0; i < block.Body.Count; i++) {
					ILLabel targetLabel;
					if (block.Body[i].Match(ILCode.Br, out targetLabel) || block.Body[i].Match(ILCode.Leave, out targetLabel)) {
						// Skip extra labels
						while(nextSibling.ContainsKey(targetLabel) && nextSibling[targetLabel] is ILLabel) {
							targetLabel = (ILLabel)nextSibling[targetLabel];
						}
						
						// Inline return statement
						ILNode target;
						List<ILExpression> retArgs;
						if (nextSibling.TryGetValue(targetLabel, out target)) {
							if (target.Match(ILCode.Ret, out retArgs)) {
								ILVariable locVar;
								object constValue;
								if (retArgs.Count == 0) {
									block.Body[i] = new ILExpression(ILCode.Ret, null);
								} else if (retArgs.Single().Match(ILCode.Ldloc, out locVar)) {
									block.Body[i] = new ILExpression(ILCode.Ret, null, new ILExpression(ILCode.Ldloc, locVar));
								} else if (retArgs.Single().Match(ILCode.Ldc_I4, out constValue)) {
									block.Body[i] = new ILExpression(ILCode.Ret, null, new ILExpression(ILCode.Ldc_I4, constValue));
								}
							}
						} else {
							if (method.Body.Count > 0 && method.Body.Last() == targetLabel) {
								// It exits the main method - so it is same as return;
								block.Body[i] = new ILExpression(ILCode.Ret, null);
							}
						}
					}
				}
			}
		}
		
		/// <summary>
		/// Flattens all nested basic blocks, except the the top level 'node' argument
		/// </summary>
		void FlattenBasicBlocks(ILNode node)
		{
			ILBlock block = node as ILBlock;
			if (block != null) {
				List<ILNode> flatBody = new List<ILNode>();
				foreach (ILNode child in block.GetChildren()) {
					FlattenBasicBlocks(child);
					ILBasicBlock childAsBB = child as ILBasicBlock;
					if (childAsBB != null) {
						if (!(childAsBB.Body.FirstOrDefault() is ILLabel))
							throw new Exception("Basic block has to start with a label. \n" + childAsBB.ToString());
						if (childAsBB.Body.LastOrDefault() is ILExpression && !childAsBB.Body.LastOrDefault().IsUnconditionalControlFlow())
							throw new Exception("Basci block has to end with unconditional control flow. \n" + childAsBB.ToString());
						flatBody.AddRange(childAsBB.GetChildren());
					} else {
						flatBody.Add(child);
					}
				}
				block.EntryGoto = null;
				block.Body = flatBody;
			} else if (node is ILExpression) {
				// Optimization - no need to check expressions
			} else if (node != null) {
				// Recursively find all ILBlocks
				foreach(ILNode child in node.GetChildren()) {
					FlattenBasicBlocks(child);
				}
			}
		}
		
		/// <summary>
		/// Reduce the nesting of conditions.
		/// It should be done on flat data that already had most gotos removed
		/// </summary>
		void ReduceIfNesting(ILNode node)
		{
			ILBlock block = node as ILBlock;
			if (block != null) {
				for (int i = 0; i < block.Body.Count; i++) {
					ILCondition cond = block.Body[i] as ILCondition;
					if (cond != null) {
						bool trueExits = cond.TrueBlock.Body.LastOrDefault().IsUnconditionalControlFlow();
						bool falseExits = cond.FalseBlock.Body.LastOrDefault().IsUnconditionalControlFlow();
						
						if (trueExits) {
							// Move the false block after the condition
							block.Body.InsertRange(i + 1, cond.FalseBlock.GetChildren());
							cond.FalseBlock = new ILBlock();
						} else if (falseExits) {
							// Move the true block after the condition
							block.Body.InsertRange(i + 1, cond.TrueBlock.GetChildren());
							cond.TrueBlock = new ILBlock();
						}
						
						// Eliminate empty true block
						if (!cond.TrueBlock.GetChildren().Any() && cond.FalseBlock.GetChildren().Any()) {
							// Swap bodies
							ILBlock tmp = cond.TrueBlock;
							cond.TrueBlock = cond.FalseBlock;
							cond.FalseBlock = tmp;
							cond.Condition = new ILExpression(ILCode.LogicNot, null, cond.Condition);
						}
					}
				}
			}
			
			// We are changing the number of blocks so we use plain old recursion to get all blocks
			foreach(ILNode child in node.GetChildren()) {
				if (child != null && !(child is ILExpression))
					ReduceIfNesting(child);
			}
		}
		
		void RecombineVariables(ILBlock method)
		{
			// Recombine variables that were split when the ILAst was created
			// This ensures that a single IL variable is a single C# variable (gets assigned only one name)
			// The DeclareVariables transformation might then split up the C# variable again if it is used indendently in two separate scopes.
			Dictionary<VariableDefinition, ILVariable> dict = new Dictionary<VariableDefinition, ILVariable>();
			foreach (ILExpression expr in method.GetSelfAndChildrenRecursive<ILExpression>()) {
				ILVariable v = expr.Operand as ILVariable;
				if (v != null && v.OriginalVariable != null) {
					ILVariable combinedVariable;
					if (!dict.TryGetValue(v.OriginalVariable, out combinedVariable)) {
						dict.Add(v.OriginalVariable, v);
						combinedVariable = v;
					}
					expr.Operand = combinedVariable;
				}
			}
		}
		
		void ReportUnassignedILRanges(ILBlock method)
		{
			var unassigned = ILRange.Invert(method.GetSelfAndChildrenRecursive<ILExpression>().SelectMany(e => e.ILRanges), context.CurrentMethod.Body.CodeSize).ToList();
			if (unassigned.Count > 0)
				Debug.WriteLine(string.Format("Unassigned ILRanges for {0}.{1}: {2}", this.context.CurrentMethod.DeclaringType.Name, this.context.CurrentMethod.Name, string.Join(", ", unassigned.Select(r => r.ToString()))));
		}
	}
	
	public static class ILAstOptimizerExtensionMethods
	{
		/// <summary>
		/// Perform one pass of a given optimization on this block.
		/// This block must consist of only basicblocks.
		/// </summary>
		public static bool RunOptimization(this ILBlock block, Func<List<ILNode>, ILBasicBlock, int, bool> optimization)
		{
			bool modified = false;
			List<ILNode> body = block.Body;
			for (int i = body.Count - 1; i >= 0; i--) {
				if (i < body.Count && optimization(body, (ILBasicBlock)body[i], i)) {
					modified = true;
				}
			}
			return modified;
		}
		
		public static bool RunOptimization(this ILBlock block, Func<List<ILNode>, ILExpression, int, bool> optimization)
		{
			bool modified = false;
			foreach (ILBasicBlock bb in block.Body) {
				for (int i = bb.Body.Count - 1; i >= 0; i--) {
					ILExpression expr = bb.Body.ElementAtOrDefault(i) as ILExpression;
					if (expr != null && optimization(bb.Body, expr, i)) {
						modified = true;
					}
				}
			}
			return modified;
		}
		
		public static bool IsConditionalControlFlow(this ILNode node)
		{
			ILExpression expr = node as ILExpression;
			return expr != null && expr.Code.IsConditionalControlFlow();
		}
		
		public static bool IsUnconditionalControlFlow(this ILNode node)
		{
			ILExpression expr = node as ILExpression;
			return expr != null && expr.Code.IsUnconditionalControlFlow();
		}
		
		/// <summary>
		/// The expression has no effect on the program and can be removed
		/// if its return value is not needed.
		/// </summary>
		public static bool HasNoSideEffects(this ILExpression expr)
		{
			// Remember that if expression can throw an exception, it is a side effect
			
			switch(expr.Code) {
				case ILCode.Ldloc:
				case ILCode.Ldloca:
				case ILCode.Ldstr:
				case ILCode.Ldnull:
				case ILCode.Ldc_I4:
				case ILCode.Ldc_I8:
				case ILCode.Ldc_R4:
				case ILCode.Ldc_R8:
					return true;
				default:
					return false;
			}
		}
		
		public static bool IsStoreToArray(this ILCode code)
		{
			switch (code) {
				case ILCode.Stelem_Any:
				case ILCode.Stelem_I:
				case ILCode.Stelem_I1:
				case ILCode.Stelem_I2:
				case ILCode.Stelem_I4:
				case ILCode.Stelem_I8:
				case ILCode.Stelem_R4:
				case ILCode.Stelem_R8:
				case ILCode.Stelem_Ref:
					return true;
				default:
					return false;
			}
		}
		
		public static bool IsLoadFromArray(this ILCode code)
		{
			switch (code) {
				case ILCode.Ldelem_Any:
				case ILCode.Ldelem_I:
				case ILCode.Ldelem_I1:
				case ILCode.Ldelem_I2:
				case ILCode.Ldelem_I4:
				case ILCode.Ldelem_I8:
				case ILCode.Ldelem_U1:
				case ILCode.Ldelem_U2:
				case ILCode.Ldelem_U4:
				case ILCode.Ldelem_R4:
				case ILCode.Ldelem_R8:
				case ILCode.Ldelem_Ref:
					return true;
				default:
					return false;
			}
		}
		
		/// <summary>
		/// Can the expression be used as a statement in C#?
		/// </summary>
		public static bool CanBeExpressionStatement(this ILExpression expr)
		{
			switch(expr.Code) {
				case ILCode.Call:
				case ILCode.Callvirt:
					// property getters can't be expression statements, but all other method calls can be
					MethodReference mr = (MethodReference)expr.Operand;
					return !mr.Name.StartsWith("get_", StringComparison.Ordinal);
				case ILCode.Newobj:
				case ILCode.Newarr:
				case ILCode.Stloc:
					return true;
				default:
					return false;
			}
		}
		
		public static void RemoveTail(this List<ILNode> body, params ILCode[] codes)
		{
			for (int i = 0; i < codes.Length; i++) {
				if (((ILExpression)body[body.Count - codes.Length + i]).Code != codes[i])
					throw new Exception("Tailing code does not match expected.");
			}
			body.RemoveRange(body.Count - codes.Length, codes.Length);
		}
		
		public static V GetOrDefault<K,V>(this Dictionary<K, V> dict, K key)
		{
			V ret;
			dict.TryGetValue(key, out ret);
			return ret;
		}
		
		public static void RemoveOrThrow<T>(this ICollection<T> collection, T item)
		{
			if (!collection.Remove(item))
				throw new Exception("The item was not found in the collection");
		}
		
		public static void RemoveOrThrow<K,V>(this Dictionary<K,V> collection, K key)
		{
			if (!collection.Remove(key))
				throw new Exception("The key was not found in the dictionary");
		}
	}
}
