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
using Cudafy.Host;
namespace Cudafy.Maths.RAND
{
    internal sealed class HostRAND : GPGPURAND
    {
        internal HostRAND(GPGPU gpu, curandRngType rng_type)
        {
            throw new CudafyMathException(CudafyMathException.csX_NOT_CURRENTLY_SUPPORTED, "HostRand");
        }

        protected override void Shutdown()
        {
            throw new NotImplementedException();
        }

        public override void SetPseudoRandomGeneratorSeed(ulong seed)
        {
            throw new NotImplementedException();
        }

        public override void GenerateUniform(float[] array, int n = 0)
        {
            throw new NotImplementedException();
        }

        public override void GenerateUniform(double[] array, int n = 0)
        {
            throw new NotImplementedException();
        }

        public override void Generate(uint[] array, int n = 0)
        {
            throw new NotImplementedException();
        }

        public override void GenerateLogNormal(float[] array, float mean, float stddev, int n = 0)
        {
            throw new NotImplementedException();
        }

        public override void GenerateLogNormal(double[] array, double mean, double stddev, int n = 0)
        {
            throw new NotImplementedException();
        }

        public override void Generate(ulong[] array, int n = 0)
        {
            throw new NotImplementedException();
        }

        public override void GenerateNormal(float[] array, float mean, float stddev, int n = 0)
        {
            throw new NotImplementedException();
        }

        public override void GenerateNormal(double[] array, float mean, float stddev, int n = 0)
        {
            throw new NotImplementedException();
        }

        public override void GenerateSeeds()
        {
            throw new NotImplementedException();
        }

        public override RandDirectionVectors32 GetDirectionVectors32(curandDirectionVectorSet set)
        {
            throw new NotImplementedException();
        }

        public override RandDirectionVectors64 GetDirectionVectors64(curandDirectionVectorSet set)
        {
            throw new NotImplementedException();
        }

        public override uint[] GetScrambleConstants32(int n)
        {
            throw new NotImplementedException();
        }

        public override ulong[] GetScrambleConstants64(int n)
        {
            throw new NotImplementedException();
        }

        public override int GetVersion()
        {
            throw new NotImplementedException();
        }

        public override void SetGeneratorOffset(ulong offset)
        {
            throw new NotImplementedException();
        }

        public override void SetGeneratorOrdering(curandOrdering order)
        {
            throw new NotImplementedException();
        }

        public override void SetQuasiRandomGeneratorDimensions(uint num_dimensions)
        {
            throw new NotImplementedException();
        }

        public override void SetStream(int streamId)
        {
            throw new NotImplementedException();
        }
    }
}
