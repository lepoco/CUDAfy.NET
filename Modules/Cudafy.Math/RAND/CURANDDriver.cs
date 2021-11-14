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
using System.Runtime.InteropServices;
using GASS.Types;
using GASS.CUDA.Types;
namespace Cudafy.Maths.RAND
{
    internal interface ICURANDDriver
    {
        curandStatus CreateGenerator(ref RandGenerator generator, curandRngType rng_type);
        curandStatus CreateGeneratorHost(ref RandGenerator generator, curandRngType rng_type);
        curandStatus DestroyGenerator(RandGenerator generator);
        curandStatus SetPseudoRandomGeneratorSeed(RandGenerator generator, ulong seed);
        /// <summary>
        /// GenerateUniform
        /// </summary>
        /// <param name="generator">Handle</param>
        /// <param name="outputPtr">Single array</param>
        /// <param name="n">Count</param>
        /// <returns>Status</returns>
        curandStatus GenerateUniform(RandGenerator generator, IntPtr outputPtr, SizeT n);
        /// <summary>
        /// GenerateUniform
        /// </summary>
        /// <param name="generator">Handle</param>
        /// <param name="outputPtr">Double array</param>
        /// <param name="n">Count</param>
        /// <returns>Status</returns>
        curandStatus GenerateUniformDouble(RandGenerator generator, IntPtr outputPtr, SizeT n);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="generator"></param>
        /// <param name="outputPtr">Int32 array</param>
        /// <param name="num"></param>
        /// <returns></returns>
        curandStatus Generate(RandGenerator generator, IntPtr outputPtr, SizeT num);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="generator"></param>
        /// <param name="outputPtr">Float array</param>
        /// <param name="n"></param>
        /// <param name="mean"></param>
        /// <param name="stddev"></param>
        /// <returns></returns>
        curandStatus GenerateLogNormal(RandGenerator generator, IntPtr outputPtr, SizeT n, float mean, float stddev);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="generator"></param>
        /// <param name="outputPtr">Double array</param>
        /// <param name="n"></param>
        /// <param name="mean"></param>
        /// <param name="stddev"></param>
        /// <returns></returns>
        curandStatus GenerateLogNormalDouble(RandGenerator generator, IntPtr outputPtr, SizeT n, double mean, double stddev);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="generator"></param>
        /// <param name="outputPtr">UInt64 array</param>
        /// <param name="num"></param>
        /// <returns></returns>
        curandStatus GenerateLongLong(RandGenerator generator, IntPtr outputPtr, SizeT num);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="generator"></param>
        /// <param name="outputPtr">Single Array</param>
        /// <param name="n"></param>
        /// <param name="mean"></param>
        /// <param name="stddev"></param>
        /// <returns></returns>
        curandStatus GenerateNormal(RandGenerator generator, IntPtr outputPtr, SizeT n, float mean, float stddev);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="generator"></param>
        /// <param name="outputPtr">Double array</param>
        /// <param name="n"></param>
        /// <param name="mean"></param>
        /// <param name="stddev"></param>
        /// <returns></returns>
        curandStatus GenerateNormalDouble(RandGenerator generator, IntPtr outputPtr, SizeT n, double mean, double stddev);

        curandStatus GenerateSeeds(RandGenerator generator);

        curandStatus GetDirectionVectors32(ref RandDirectionVectors32 vectors, curandDirectionVectorSet set);

        curandStatus GetDirectionVectors64(ref RandDirectionVectors64 vectors, curandDirectionVectorSet set);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="constants">UInt32 array</param>
        /// <returns></returns>
        curandStatus GetScrambleConstants32(ref IntPtr constants);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="constants">UInt64 array</param>
        /// <returns></returns>
        curandStatus GetScrambleConstants64(ref IntPtr constants);

        curandStatus GetVersion(ref int version);

        curandStatus SetGeneratorOffset(RandGenerator generator, ulong offset);

        curandStatus SetGeneratorOrdering(RandGenerator generator, curandOrdering order);

        curandStatus SetQuasiRandomGeneratorDimensions(RandGenerator generator, uint num_dimensions);

        curandStatus SetStream(RandGenerator generator, CUstream stream);
    }

    internal class CURANDDriver64 : ICURANDDriver
    {
#if LINUX
        internal const string DLL_NAME = "libcurand";
#else
        internal const string DLL_NAME = "curand64_70";
#endif
        //curandStatus_t curandCreateGenerator (curandGenerator_t generator, curandRngType_t rng_type)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandCreateGenerator(ref RandGenerator generator, curandRngType rng_type);

        //curandStatus_t curandCreateGenerator (curandGenerator_t generator, curandRngType_t rng_type)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandCreateGeneratorHost(ref RandGenerator generator, curandRngType rng_type);

        //curandStatus_t curandDestroyGenerator (curandGenerator_t generator)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandDestroyGenerator(RandGenerator generator);

        //curandStatus_t curandGenerate (curandGenerator_t generator, unsigned int  outputPtr, size_t num)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerate(RandGenerator generator, IntPtr outputPtr, SizeT num);

        //curandStatus_t curandGenerateLogNormal (curandGenerator_t generator, float  outputPtr, size_t n, oat mean, oat stddev)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerateLogNormal(RandGenerator generator, IntPtr outputPtr, SizeT n, float mean, float stddev);

        //curandStatus_t curandGenerateLogNormalDouble (curandGenerator_t generator, double  outputPtr, size_t n, double mean, double stddev)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerateLogNormalDouble(RandGenerator generator, IntPtr outputPtr, SizeT n, double mean, double stddev);

        //curandStatus_t curandGenerateLongLong (curandGenerator_t generator, unsigned long long  outputPtr, size_t num)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerateLongLong(RandGenerator generator, IntPtr outputPtr, SizeT num);

        //curandStatus_t curandGenerateNormal (curandGenerator_t generator, float  outputPtr, size_t n, oat mean, oat stddev)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerateNormal(RandGenerator generator, IntPtr outputPtr, SizeT n, float mean, float stddev); 

        //curandStatus_t curandGenerateNormalDouble (curandGenerator_t generator, double  outputPtr, size_t n, double mean, double stddev)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerateNormalDouble(RandGenerator generator, IntPtr outputPtr, SizeT n, double mean, double stddev);

        //curandStatus_t curandGenerateSeeds (curandGenerator_t generator)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerateSeeds(RandGenerator generator);

        //curandStatus_t curandGenerateUniform (curandGenerator_t generator, oat  outputPtr, size_t num)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerateUniform(RandGenerator generator, IntPtr outputPtr, SizeT num);

        //curandStatus_t curandGenerateUniformDouble (curandGenerator_t generator, double  outputPtr, size_t num)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerateUniformDouble(RandGenerator generator, IntPtr outputPtr, SizeT num);

        //curandStatus_t curandGetDirectionVectors32 (curandDirectionVectors32_t  vectors[ ], curandDirectionVectorSet_t set)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGetDirectionVectors32(ref RandDirectionVectors32 vectors, curandDirectionVectorSet set);
 
        //curandStatus_t curandGetDirectionVectors64 (curandDirectionVectors64_t  vectors[ ], curandDirectionVectorSet_t set)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGetDirectionVectors64(ref RandDirectionVectors64 vectors, curandDirectionVectorSet set);

        //curandStatus_t curandGetScrambleConstants32 (unsigned int  constants)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGetScrambleConstants32(ref IntPtr constants);

        //curandStatus_t curandGetScrambleConstants64 (unsigned long long constants)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGetScrambleConstants64(ref IntPtr constants);

        //curandStatus_t curandGetVersion (int  version)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGetVersion(ref int version);

        //curandStatus_t curandSetGeneratorOffset (curandGenerator_t generator, unsigned long long offset)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandSetGeneratorOffset(RandGenerator generator, ulong offset);

        //curandStatus_t curandSetGeneratorOrdering (curandGenerator_t generator, curandOrdering_t order)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandSetGeneratorOrdering(RandGenerator generator, curandOrdering order);

        //curandStatus_t curandSetPseudoRandomGeneratorSeed (curandGenerator_t generator, unsigned long long seed)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandSetPseudoRandomGeneratorSeed(RandGenerator generator, ulong seed);

        //curandStatus_t curandSetQuasiRandomGeneratorDimensions (curandGenerator_t generator, unsigned int num_dimensions)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandSetQuasiRandomGeneratorDimensions(RandGenerator generator, uint num_dimensions);

        //curandStatus_t curandSetStream (curandGenerator_t generator, cudaStream_t stream)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandSetStream(RandGenerator generator, CUstream stream);


        public curandStatus CreateGenerator(ref RandGenerator generator, curandRngType rng_type)
        {
            return curandCreateGenerator(ref generator, rng_type);
        }

        public curandStatus CreateGeneratorHost(ref RandGenerator generator, curandRngType rng_type)
        {
            return curandCreateGeneratorHost(ref generator, rng_type);
        }

        public curandStatus DestroyGenerator(RandGenerator generator)
        {
            return curandDestroyGenerator(generator);
        }

        public curandStatus SetPseudoRandomGeneratorSeed(RandGenerator generator, ulong seed)
        {
            return curandSetPseudoRandomGeneratorSeed(generator, seed);
        }

        public curandStatus GenerateUniform(RandGenerator generator, IntPtr outputPtr, SizeT n)
        {
            return curandGenerateUniform(generator, outputPtr, n);
        }

        public curandStatus GenerateUniformDouble(RandGenerator generator, IntPtr outputPtr, SizeT n)
        {
            return curandGenerateUniformDouble(generator, outputPtr, n);
        }

        public curandStatus Generate(RandGenerator generator, IntPtr outputPtr, SizeT num)
        {
            return curandGenerate(generator, outputPtr, num);
        }

        public curandStatus GenerateLogNormal(RandGenerator generator, IntPtr outputPtr, SizeT n, float mean, float stddev)
        {
            return curandGenerateLogNormal(generator, outputPtr, n, mean, stddev);
        }

        public curandStatus GenerateLogNormalDouble(RandGenerator generator, IntPtr outputPtr, SizeT n, double mean, double stddev)
        {
            return curandGenerateLogNormalDouble(generator, outputPtr, n, mean, stddev);
        }

        public curandStatus GenerateLongLong(RandGenerator generator, IntPtr outputPtr, SizeT num)
        {
            return curandGenerateLongLong(generator, outputPtr, num);
        }

        public curandStatus GenerateNormal(RandGenerator generator, IntPtr outputPtr, SizeT n, float mean, float stddev)
        {
            return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
        }

        public curandStatus GenerateNormalDouble(RandGenerator generator, IntPtr outputPtr, SizeT n, double mean, double stddev)
        {
            return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
        }

        public curandStatus GenerateSeeds(RandGenerator generator)
        {
            return curandGenerateSeeds(generator);
        }

        public curandStatus GetDirectionVectors32(ref RandDirectionVectors32 vectors, curandDirectionVectorSet set)
        {
            return curandGetDirectionVectors32(ref vectors, set);
        }

        public curandStatus GetDirectionVectors64(ref RandDirectionVectors64 vectors, curandDirectionVectorSet set)
        {
            return curandGetDirectionVectors64(ref vectors, set);
        }

        public curandStatus GetScrambleConstants32(ref IntPtr constants)
        {
            return curandGetScrambleConstants32(ref constants);
        }

        public curandStatus GetScrambleConstants64(ref IntPtr constants)
        {
            return curandGetScrambleConstants64(ref constants);
        }

        public curandStatus GetVersion(ref int version)
        {
            return curandGetVersion(ref version);
        }

        public curandStatus SetGeneratorOffset(RandGenerator generator, ulong offset)
        {
            return curandSetGeneratorOffset(generator, offset);
        }

        public curandStatus SetGeneratorOrdering(RandGenerator generator, curandOrdering order)
        {
            return curandSetGeneratorOrdering(generator, order);
        }

        public curandStatus SetQuasiRandomGeneratorDimensions(RandGenerator generator, uint num_dimensions)
        {
            return curandSetQuasiRandomGeneratorDimensions(generator, num_dimensions);
        }

        public curandStatus SetStream(RandGenerator generator, CUstream stream)
        {
            return curandSetStream(generator, stream);
        }
    }

    internal class CURANDDriver32 : ICURANDDriver
    {
#if LINUX
        internal const string DLL_NAME = "libcurand";
#else
        internal const string DLL_NAME = "curand32_70";
#endif

        //curandStatus_t curandCreateGenerator (curandGenerator_t generator, curandRngType_t rng_type)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandCreateGenerator(ref RandGenerator generator, curandRngType rng_type);

        //curandStatus_t curandCreateGenerator (curandGenerator_t generator, curandRngType_t rng_type)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandCreateGeneratorHost(ref RandGenerator generator, curandRngType rng_type);

        //curandStatus_t curandDestroyGenerator (curandGenerator_t generator)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandDestroyGenerator(RandGenerator generator);

        //curandStatus_t curandGenerate (curandGenerator_t generator, unsigned int  outputPtr, size_t num)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerate(RandGenerator generator, IntPtr outputPtr, SizeT num);

        //curandStatus_t curandGenerateLogNormal (curandGenerator_t generator, float  outputPtr, size_t n, oat mean, oat stddev)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerateLogNormal(RandGenerator generator, IntPtr outputPtr, SizeT n, float mean, float stddev);

        //curandStatus_t curandGenerateLogNormalDouble (curandGenerator_t generator, double  outputPtr, size_t n, double mean, double stddev)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerateLogNormalDouble(RandGenerator generator, IntPtr outputPtr, SizeT n, double mean, double stddev);

        //curandStatus_t curandGenerateLongLong (curandGenerator_t generator, unsigned long long  outputPtr, size_t num)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerateLongLong(RandGenerator generator, IntPtr outputPtr, SizeT num);

        //curandStatus_t curandGenerateNormal (curandGenerator_t generator, float  outputPtr, size_t n, oat mean, oat stddev)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerateNormal(RandGenerator generator, IntPtr outputPtr, SizeT n, float mean, float stddev);

        //curandStatus_t curandGenerateNormalDouble (curandGenerator_t generator, double  outputPtr, size_t n, double mean, double stddev)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerateNormalDouble(RandGenerator generator, IntPtr outputPtr, SizeT n, double mean, double stddev);

        //curandStatus_t curandGenerateSeeds (curandGenerator_t generator)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerateSeeds(RandGenerator generator);

        //curandStatus_t curandGenerateUniform (curandGenerator_t generator, oat  outputPtr, size_t num)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerateUniform(RandGenerator generator, IntPtr outputPtr, SizeT num);

        //curandStatus_t curandGenerateUniformDouble (curandGenerator_t generator, double  outputPtr, size_t num)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGenerateUniformDouble(RandGenerator generator, IntPtr outputPtr, SizeT num);

        //curandStatus_t curandGetDirectionVectors32 (curandDirectionVectors32_t  vectors[ ], curandDirectionVectorSet_t set)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGetDirectionVectors32(ref RandDirectionVectors32 vectors, curandDirectionVectorSet set);

        //curandStatus_t curandGetDirectionVectors64 (curandDirectionVectors64_t  vectors[ ], curandDirectionVectorSet_t set)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGetDirectionVectors64(ref RandDirectionVectors64 vectors, curandDirectionVectorSet set);

        //curandStatus_t curandGetScrambleConstants32 (unsigned int  constants)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGetScrambleConstants32(ref IntPtr constants);

        //curandStatus_t curandGetScrambleConstants64 (unsigned long long constants)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGetScrambleConstants64(ref IntPtr constants);

        //curandStatus_t curandGetVersion (int  version)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandGetVersion(ref int version);

        //curandStatus_t curandSetGeneratorOffset (curandGenerator_t generator, unsigned long long offset)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandSetGeneratorOffset(RandGenerator generator, ulong offset);

        //curandStatus_t curandSetGeneratorOrdering (curandGenerator_t generator, curandOrdering_t order)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandSetGeneratorOrdering(RandGenerator generator, curandOrdering order);

        //curandStatus_t curandSetPseudoRandomGeneratorSeed (curandGenerator_t generator, unsigned long long seed)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandSetPseudoRandomGeneratorSeed(RandGenerator generator, ulong seed);

        //curandStatus_t curandSetQuasiRandomGeneratorDimensions (curandGenerator_t generator, unsigned int num_dimensions)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandSetQuasiRandomGeneratorDimensions(RandGenerator generator, uint num_dimensions);

        //curandStatus_t curandSetStream (curandGenerator_t generator, cudaStream_t stream)
        [DllImport(DLL_NAME)]
        private static extern curandStatus curandSetStream(RandGenerator generator, CUstream stream);


        public curandStatus CreateGenerator(ref RandGenerator generator, curandRngType rng_type)
        {
            return curandCreateGenerator(ref generator, rng_type);
        }

        public curandStatus CreateGeneratorHost(ref RandGenerator generator, curandRngType rng_type)
        {
            return curandCreateGeneratorHost(ref generator, rng_type);
        }

        public curandStatus DestroyGenerator(RandGenerator generator)
        {
            return curandDestroyGenerator(generator);
        }

        public curandStatus SetPseudoRandomGeneratorSeed(RandGenerator generator, ulong seed)
        {
            return curandSetPseudoRandomGeneratorSeed(generator, seed);
        }

        public curandStatus GenerateUniform(RandGenerator generator, IntPtr outputPtr, SizeT n)
        {
            return curandGenerateUniform(generator, outputPtr, n);
        }

        public curandStatus GenerateUniformDouble(RandGenerator generator, IntPtr outputPtr, SizeT n)
        {
            return curandGenerateUniformDouble(generator, outputPtr, n);
        }

        public curandStatus Generate(RandGenerator generator, IntPtr outputPtr, SizeT num)
        {
            return curandGenerate(generator, outputPtr, num);
        }

        public curandStatus GenerateLogNormal(RandGenerator generator, IntPtr outputPtr, SizeT n, float mean, float stddev)
        {
            return curandGenerateLogNormal(generator, outputPtr, n, mean, stddev);
        }

        public curandStatus GenerateLogNormalDouble(RandGenerator generator, IntPtr outputPtr, SizeT n, double mean, double stddev)
        {
            return curandGenerateLogNormalDouble(generator, outputPtr, n, mean, stddev);
        }

        public curandStatus GenerateLongLong(RandGenerator generator, IntPtr outputPtr, SizeT num)
        {
            return curandGenerateLongLong(generator, outputPtr, num);
        }

        public curandStatus GenerateNormal(RandGenerator generator, IntPtr outputPtr, SizeT n, float mean, float stddev)
        {
            return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
        }

        public curandStatus GenerateNormalDouble(RandGenerator generator, IntPtr outputPtr, SizeT n, double mean, double stddev)
        {
            return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
        }

        public curandStatus GenerateSeeds(RandGenerator generator)
        {
            return curandGenerateSeeds(generator);
        }

        public curandStatus GetDirectionVectors32(ref RandDirectionVectors32 vectors, curandDirectionVectorSet set)
        {
            return curandGetDirectionVectors32(ref vectors, set);
        }

        public curandStatus GetDirectionVectors64(ref RandDirectionVectors64 vectors, curandDirectionVectorSet set)
        {
            return curandGetDirectionVectors64(ref vectors, set);
        }

        public curandStatus GetScrambleConstants32(ref IntPtr constants)
        {
            return curandGetScrambleConstants32(ref constants);
        }

        public curandStatus GetScrambleConstants64(ref IntPtr constants)
        {
            return curandGetScrambleConstants64(ref constants);
        }

        public curandStatus GetVersion(ref int version)
        {
            return curandGetVersion(ref version);
        }

        public curandStatus SetGeneratorOffset(RandGenerator generator, ulong offset)
        {
            return curandSetGeneratorOffset(generator, offset);
        }

        public curandStatus SetGeneratorOrdering(RandGenerator generator, curandOrdering order)
        {
            return curandSetGeneratorOrdering(generator, order);
        }

        public curandStatus SetQuasiRandomGeneratorDimensions(RandGenerator generator, uint num_dimensions)
        {
            return curandSetQuasiRandomGeneratorDimensions(generator, num_dimensions);
        }

        public curandStatus SetStream(RandGenerator generator, CUstream stream)
        {
            return curandSetStream(generator, stream);
        }
    } 
}
