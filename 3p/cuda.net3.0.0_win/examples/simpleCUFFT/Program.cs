/*
 * Copyright 2008 Company for Advanced Supercomputing Solutions Ltd (GASS).
 * All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to GASS ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * GASS MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  GASS DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL GASS BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

using System;
using System.Collections.Generic;
using System.Text;

using GASS.CUDA;
using GASS.CUDA.Types;
using System.IO;
using GASS.CUDA.FFT;
using GASS.CUDA.FFT.Types;
using System.Runtime.InteropServices;

namespace simpleCUFFT
{
    class Program
    {
        static void Main(string[] args)
        {
            // Init and select 1st device.
            CUDA cuda = new CUDA(0, true);

            // load module
            //cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "simpleCUFFT.ptx"));
            CUfunction func = new CUfunction();// cuda.GetModuleFunction("ComplexPointwiseMulAndScale");

            // The filter size is assumed to be a number smaller than the signal size
            const int SIGNAL_SIZE = 50;
            const int FILTER_KERNEL_SIZE = 11;

            // Allocate host memory for the signal
            Float2[] h_signal = new Float2[SIGNAL_SIZE];
            // Initalize the memory for the signal
            Random r = new Random();
            for (int i = 0; i < SIGNAL_SIZE; ++i)
            {
                h_signal[i].x = r.Next() / (float)int.MaxValue;
                h_signal[i].y = 0;
            }

            // Allocate host memory for the filter
            Float2[] h_filter_kernel = new Float2[FILTER_KERNEL_SIZE];
            // Initalize the memory for the filter
            for (int i = 0; i < FILTER_KERNEL_SIZE; ++i)
            {
                h_filter_kernel[i].x = r.Next() / (float)int.MaxValue;
                h_filter_kernel[i].y = 0;
            }

            // Pad signal and filter kernel
            Float2[] h_padded_signal;
            Float2[] h_padded_filter_kernel;
            int new_size = PadData(h_signal, out h_padded_signal, SIGNAL_SIZE,
                                   h_filter_kernel, out h_padded_filter_kernel, FILTER_KERNEL_SIZE);

            // Allocate device memory for signal
            // Copy host memory to device
            CUdeviceptr d_signal = cuda.CopyHostToDevice<Float2>(h_padded_signal);

            // Allocate device memory for filter kernel
            // Copy host memory to device
            CUdeviceptr d_filter_kernel = cuda.CopyHostToDevice<Float2>(h_padded_filter_kernel);

            // CUFFT plan
            CUFFT fft = new CUFFT(cuda);
            cufftHandle handle = new cufftHandle();
            CUFFTResult fftres = CUFFTDriver.cufftPlan1d(ref handle, new_size, CUFFTType.C2C, 1);
            //fft.Plan1D(new_size, CUFFTType.C2C, 1);


            return;

            // Transform signal and kernel
            fft.ExecuteComplexToComplex(d_signal, d_signal, CUFFTDirection.Forward);
            fft.ExecuteComplexToComplex(d_filter_kernel, d_filter_kernel, CUFFTDirection.Forward);

            // Multiply the coefficients together and normalize the result
            // ComplexPointwiseMulAndScale<<<32, 256>>>(d_signal, d_filter_kernel, new_size, 1.0f / new_size);
            cuda.SetFunctionBlockShape(func, 256, 1, 1);
            cuda.SetParameter(func, 0, (uint)d_signal.Pointer);
            cuda.SetParameter(func, IntPtr.Size, (uint)d_filter_kernel.Pointer);
            cuda.SetParameter(func, IntPtr.Size * 2, (uint)new_size);
            cuda.SetParameter(func, IntPtr.Size * 2 + 4, 1.0f / new_size);
            cuda.SetParameterSize(func, (uint)(IntPtr.Size * 2 + 8));
            cuda.Launch(func, 32, 1);

            // Transform signal back
            fft.ExecuteComplexToComplex(d_signal, d_signal, CUFFTDirection.Inverse);

            // Copy device memory to host
            Float2[] h_convolved_signal = h_padded_signal;
            cuda.CopyDeviceToHost<Float2>(d_signal, h_convolved_signal);

            // Allocate host memory for the convolution result
            Float2[] h_convolved_signal_ref = new Float2[SIGNAL_SIZE];

            // Convolve on the host
            Convolve(h_signal, SIGNAL_SIZE,
                     h_filter_kernel, FILTER_KERNEL_SIZE,
                     h_convolved_signal_ref);

            // check result
            bool res = cutCompareL2fe(h_convolved_signal_ref, h_convolved_signal, 2 * SIGNAL_SIZE, 1e-5f);
            Console.WriteLine("Test {0}", (true == res) ? "PASSED" : "FAILED");

            //Destroy CUFFT context
            fft.Destroy();

            // cleanup memory
            cuda.Free(d_signal);
            cuda.Free(d_filter_kernel);
        }

        // Pad data
        private static int PadData(Float2[] signal, out Float2[] padded_signal, int signal_size,
                    Float2[] filter_kernel, out Float2[] padded_filter_kernel, int filter_kernel_size)
        {
            int minRadius = filter_kernel_size / 2;
            int maxRadius = filter_kernel_size - minRadius;
            int new_size = signal_size + maxRadius;

            // Pad signal
            Float2[] new_data = new Float2[new_size];
            Array.Copy(signal, new_data, signal_size);
            for (int i = 0; i < (new_size - signal_size)/2; i++)
            {
                new_data[signal_size + i] = new Float2();
            }
            padded_signal = new_data;

            // Pad filter
            new_data = new Float2[new_size];
            Array.Copy(filter_kernel, minRadius, new_data, 0, maxRadius);
            for (int i = 0; i < (new_size - filter_kernel_size)/2; i++)
            {
                new_data[maxRadius/2 + i] = new Float2();
            }
            Array.Copy(filter_kernel, (new_size - minRadius) / Marshal.SizeOf(typeof(Float2)), filter_kernel, 0, minRadius);
            padded_filter_kernel = new_data;

            return new_size;
        }

        ////////////////////////////////////////////////////////////////////////////////
        // Filtering operations
        ////////////////////////////////////////////////////////////////////////////////

        // Computes convolution on the host
        private static void Convolve(Float2[] signal, int signal_size,
                      Float2[] filter_kernel, int filter_kernel_size,
                      Float2[] filtered_signal)
        {
            int minRadius = filter_kernel_size / 2;
            int maxRadius = filter_kernel_size - minRadius;
            // Loop over output element indices
            for (int i = 0; i < signal_size; ++i)
            {
                filtered_signal[i].x = filtered_signal[i].y = 0;
                // Loop over convolution indices
                for (int j = -maxRadius + 1; j <= minRadius; ++j)
                {
                    int k = i + j;
                    if (k >= 0 && k < signal_size)
                        filtered_signal[i] = ComplexAdd(filtered_signal[i], ComplexMul(signal[k], filter_kernel[minRadius - j]));
                }
            }
        }

        private static Float2 ComplexAdd(Float2 n1, Float2 n2)
        {
            Float2 res = new Float2();
            res.x = n1.x + n2.x;
            res.y = n1.y + n2.y;

            return res;
        }

        private static Float2 ComplexMul(Float2 a, Float2 b)
        {
            Float2 res = new Float2();
            res.x = a.x * b.x - a.y * b.y;
            res.y = a.x * b.y + a.y * b.x;

            return res;
        }

        private static bool cutCompareL2fe(Float2[] reference, Float2[] data,
                int len, float epsilon)
        {
            float error = 0;
            float reff = 0;

            for (int i = 0; i < len / 2; ++i)
            {
                float diff = reference[i].x - data[i].x;
                error += diff * diff;
                reff += reference[i].x * reference[i].x;

                diff = reference[i].y - data[i].y;
                error += diff * diff;
                reff += reference[i].y * reference[i].y;
            }

            float normRef = (float)Math.Sqrt(reff);
            if (Math.Abs(reff) < 1e-7)
            {
                Console.WriteLine("ERROR, reference l2-norm is 0");

                return false;
            }
            float normError = (float)Math.Sqrt(error);
            error = normError / normRef;
            bool result = error < epsilon;
            if (!result)
            {
                Console.WriteLine("ERROR, l2-norm error {0} is greater than epsilon {1}", error, epsilon);
            }

            return result ? true : false;
        }
    }
}
