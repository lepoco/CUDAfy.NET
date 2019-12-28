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

using Cudafy;
using CudafyExamples.Arrays;
using CudafyExamples.Dummy;
using CudafyExamples.Complex;
using CudafyExamples.Misc;
using Cudafy.Translator;
namespace CudafyExamples
{
    class Program
    {

        /*
         * Choose the architecture that matches your version of the card
         * (backward compatibility works to a certain level)
         * 
         * SM_50 – Tesla/Quadro M series
         * SM_52 – Quadro M6000 , GeForce 900, GTX-970, GTX-980, GTX Titan X
         * SM_53 – Tegra (Jetson) TX1 / Tegra X1, Drive CX, Drive PX, Jetson Nano
         * SM_60 – Quadro GP100, Tesla P100, DGX-1 (Generic Pascal)
         * SM_61 – GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4, Discrete GPU on the NVIDIA Drive PX2
         * SM_62 – Integrated GPU on the NVIDIA Drive PX2, Tegra (Jetson) TX2
         * SM_70 – DGX-1 with Volta, Tesla V100, GTX 1180 (GV104), Titan V, Quadro GV100
         * SM_72 – Jetson AGX Xavier, Drive AGX Pegasus, Xavier NX
         * SM_75 – GTX/RTX Turing – GTX 1660 Ti, RTX 2060, RTX 2070, RTX 2080, Titan RTX, Quadro RTX 4000, Quadro RTX 5000, Quadro RTX 6000, Quadro RTX 8000, Quadro T1000/T2000, Tesla T4
         * SM_80 – RTX Ampere – RTX 3080
         */
        public static eArchitecture testArchitecture = eArchitecture.sm_50;

        static void Main(string[] args)
{
    try
    {
        CudafyModes.Target = eGPUType.Cuda;
        CudafyModes.DeviceId = 0;
        CudafyTranslator.Language = CudafyModes.Target == eGPUType.OpenCL ? eLanguage.OpenCL : eLanguage.Cuda;

        Console.WriteLine("===================================\n          CudafyExamples\n         Hybrid DSP Systems\nCopyright © Hybrid DSP Systems 2011\n===================================");
        Console.WriteLine("\n* VS 2019 & .NET 4.8\n* optimization by RapidDev\n");

        ConsoleHeader(1, "ArrayBasicIndexing");
        ArrayBasicIndexing.Execute();
        ConsoleHeader(2, "ArrayMultidimensions");
        ArrayMultidimensions.Execute();

        if (CudafyModes.Target != eGPUType.OpenCL)
        {
            ConsoleHeader(3, "Class examples");
            CudafyClassExamples.Execute();
            ConsoleHeader(4, "SIMDFunctions");
            SIMDFunctions.Execute();
            ConsoleHeader(5, "GlobalArrays");
            GlobalArrays.Execute();
            ConsoleHeader(6, "ComplexNumbersD");
            ComplexNumbersD.Execute();
            ConsoleHeader(7, "ComplexNumbersF");
            ComplexNumbersF.Execute();               
            ConsoleHeader(8, "DummyFunctions");
            DummyFunctions.Execute();

            ConsoleHeader(9, "TextInsertion");
            TextInsertion.Execute();


            ConsoleHeader(10, "Voting.Ballot");
            Voting.Ballot.Execute();
            ConsoleHeader(11, "Voting.SyncThreadCount");
            Voting.SyncThreadCount.Execute();

            ConsoleHeader(12, "FinanceTest");
            FinanceTest.Execute();
            ConsoleHeader(13, "Timing");
            Timing.Execute();
            ConsoleHeader(14, "PinnedAsyncIO");
            PinnedAsyncIO.Execute();
        }
        Console.WriteLine("Done!");
    }
    catch (Exception ex)
    {
        Console.WriteLine(ex);
    }
    Console.ReadKey();
}

static void ConsoleHeader(int id = -1, string name = "Unknown")
{
    Console.WriteLine("\r\nRUNNING TEST\n============\n#{0} of #14\n{1}", id, name);
}
}
}
