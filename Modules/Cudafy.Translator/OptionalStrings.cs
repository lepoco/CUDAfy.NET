using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Cudafy.Translator
{
    internal class OptionalStrings
    {
        public const string get_global_id =
        @"__device__ int get_global_id(int dimension)
        {
            int result = 0;
            if (dimension == 0)
                result = blockIdx.x * blockDim.x + threadIdx.x;
            else if (dimension == 1)
                result = blockIdx.y * blockDim.y + threadIdx.y;
            else  if (dimension == 2)
                result = blockIdx.z * blockDim.z + threadIdx.z;
            return result;
        }";

        public const string get_local_id =
        @"__device__ int get_local_id(int dimension)
        {
            int result = 0;
            if (dimension == 0)
                result = threadIdx.x;
            else if (dimension == 1)
                result = threadIdx.y;
            else  if (dimension == 2)
                result = threadIdx.z;
            return result;
        }";

        public const string get_group_id =
        @"__device__ int get_group_id(int dimension)
        {
            int result = 0;
            if (dimension == 0)
                result = blockIdx.x;
            else if (dimension == 1)
                result = blockIdx.y;
            else  if (dimension == 2)
                result = blockIdx.z;
            return result;
        }";
        public const string get_local_size =
@"__device__ int get_local_size(int dimension)
        {
            int result = 0;
            if (dimension == 0)
                result = blockDim.x;
            else if (dimension == 1)
                result = blockDim.y;
            else  if (dimension == 2)
                result = blockDim.z;
            return result;
        }";
        public const string get_global_size =
@"__device__ int get_global_size(int dimension)
        {
            int result = 0;
            if (dimension == 0)
                result = blockDim.x * gridDim.x;
            else if (dimension == 1)
                result = blockDim.y * gridDim.y;
            else  if (dimension == 2)
                result = blockDim.z * gridDim.z;
            return result;
        }";
        public const string get_num_groups =
@"__device__ int get_num_groups(int dimension)
        {
            int result = 0;
            if (dimension == 0)
                result = gridDim.x;
            else if (dimension == 1)
                result = gridDim.y;
            else  if (dimension == 2)
                result = gridDim.z;
            return result;
        }";
        public const string popCount =
    @"#if __OPENCL_VERSION__ <= CL_VERSION_1_1
    int popcount(unsigned int x){
    int c = 0;
    for (; x > 0; x &= x -1) c++;
    return c;}
    #endif";

        public const string popCountll =
    @"#if __OPENCL_VERSION__ <= CL_VERSION_1_1
    int popcountll(long x){
    int c = 0;
    for (; x > 0; x &= x -1) c++;
    return c;}
    #else
    #define popcountll popcount
    #endif";
    }
}
