using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
//using System.Dynamic;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Cecil.Decompiler;
using Cecil.Decompiler.Languages;
namespace Cudafy
{
    class Program
    {
        static void Main(string[] args)
        {
            Assembly ass = Assembly.LoadFrom(@"D:\Nick Personal\Programming\GPU\Projects\Cudafy\Cudafy\bin\Debug\ESA.Dataflow.dll"); //(typeof(Program));
          //  ass.
            ////AssemblyDefinition.
            AssemblyDefinition ad = AssemblyFactory.GetAssembly(ass.Location);
            ModuleDefinitionCollection mdc = ad.Modules;
            StringBuilder sb = new StringBuilder();
            StringWriter streamWriter = new StringWriter(sb);
            foreach (ModuleDefinition mod in mdc)
            {
                Console.WriteLine(mod.Name);
                foreach (TypeDefinition type in mod.Types)
                {
                    Console.WriteLine(type.FullName);
                    if (type.Name == "ControlVector")
                    foreach (MethodDefinition md in type.Methods)
                    {

                        //foreach (CustomAttribute ca in md.CustomAttributes)
                        //{
                        //    if (ca.Constructor.DeclaringType.Name == "GPUFunctionAttribute")
                        //    {
                        if (md.Name == "Add")
                        {
                            Console.WriteLine(md.Name);
                            ILanguage lan = Cecil.Decompiler.Languages.CSharp.GetLanguage(Cecil.Decompiler.Languages.CSharpVersion.V3);
                            ILanguageWriter lanWriter = lan.GetWriter(new PlainTextFormatter(streamWriter));
                            lanWriter.Write(md);
                         
                            Console.WriteLine(sb.ToString());
                        }
                           // }
                        //}
                    }
                }
            }

            int size = 4;
            int[] myArray = new int[] { 0,1,2,3,4,5,6,7 };
            int[][] inputA = new int[size][];
            int[][] inputB = new int[size][];
            int[][] outputC = new int[size][];
            for (int i = 0; i < size; i++)
            {
                inputB[i] = new int[size];
                outputC[i] = new int[size];
                inputA[i] = new int[size];
                int cnt = i;
                for (int x = 0; x < size; x++)
                {
                    inputA[i][x] = cnt;
                    inputB[i][x] = cnt++;
                }
            }

            HCudafy cuda = new HCudafy();
            cuda.Cudafy(typeof(Program));

            Stopwatch sw = new Stopwatch();
            sw.Start();  
            int[] devMyArray = cuda.CopyToDevice(myArray);
            int[][] devA = cuda.CopyToDevice(inputA);
            int[][] devB = cuda.CopyToDevice(inputB);
            int[][] devC = cuda.Allocate(outputC);
            Dim3 grid = new Dim3(1);
            Dim3 block = new Dim3(size/1);
            cuda.Launch(grid, block, "doVecAdd", devA, devB, devC, 42, devMyArray);     
           
            cuda.CopyFromDevice(devC, outputC);
            sw.Stop();
            for (int i = 0; i < 4; i++)
            {
                for (int x = 0; x < 4; x++)
                    Console.Write("{0}\t", outputC[i][x]);
                Console.WriteLine();
            }

            int[] somestuff = new int[512];
            for (int y = 0; y < 512; y++)
                somestuff[y] = y * 10;
            int[] data = cuda.CopyToDevice(somestuff);
            int[] res = new int[512];
            cuda.CopyFromDevice(data, res);
            for (int y = 0; y < 512; y++)
                if (res[y] != somestuff[y])
                    throw new Exception();


            int[][] deviceArray2D = cuda.Allocate<int>(4, 8);
            int[] deviceArray = cuda.Allocate<int>(7);

            Console.WriteLine(sw.ElapsedMilliseconds + "ms");
            Console.WriteLine("Done");
            Console.ReadKey();
            return;
            #region scrap
            //Action<object> action = (object obj) =>
            //{
            //    Console.WriteLine("Task={0}, obj={1}, Thread={2}", Task.CurrentId, obj.ToString(), Thread.CurrentThread.ManagedThreadId);
            //};
            //Task t = new Task(action, "hello");
            //HThread ht = new HThread(action, "hello");
            
            //HGrid grid = new HGrid(
            //HCudafy.Launch(
            int side = 1024;
            int[][] myJaggedArray = new int[side][];
            for (int i = 0; i < side; i++)
            {
                myJaggedArray[i] = new int[side];
                int cnt = i;
                for (int x = 0; x < side; x++)
                    myJaggedArray[i][x] = cnt++;
            }

            int threads = Environment.ProcessorCount / 1;
          //  _barrier = new Barrier(threads);

            //Console.WriteLine("Before");
            //var po = new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount };
            //Parallel.For(0, side, po, i => Process(myJaggedArray[i]));
            myJaggedArray.AsParallel().WithDegreeOfParallelism(threads).ForAll(x => Process(x));
            //myJaggedArray.ToList().ForEach(x => Process(x));
            //Console.WriteLine("Between");
            //myJaggedArray.AsParallel().WithDegreeOfParallelism(threads).ForAll(x => Process(x));
            sw.Stop();
           // _barrier.Dispose();
            Console.WriteLine(sw.ElapsedMilliseconds+"ms");
            Console.WriteLine("Done");
            Console.ReadKey();
            #endregion
        }

        //public static void vecAdd(object o)
        //{
        //    HThreadParams hargs = o as HThreadParams;
        //    vecAddParams args = hargs.UserArgs as vecAddParams;
        //    doVecAdd(hargs.Thread, args.a, args.b, args.output);
        //}

        [GPUFunction]
        public static void doVecAdd(HThread thread, int[][] a, int[][] b, int[][] output, int myNumber, int[] array1d)
        {
#warning make this so you specify format and size
            int[] cache = thread.block.AllocateShared("cache", new int[thread.block.Dim.TotalSize]);
#warning change vars to be props of thread only
            int threadId = thread.threadIdx.x + thread.block.Idx.x * thread.block.Dim.x;
            for (int i = 0; i < a[threadId].Length; i++)
                output[threadId][i] = a[threadId][i] + b[threadId][i];
  
            cache[threadId] = threadId;
            thread.block.SyncThreads();
         
            if (threadId == 0)
            {
                Console.WriteLine("Thread {0} is done! My number is {1}", threadId, myNumber);
                int total = 0;
                for (int i = 0; i < thread.block.Dim.x; i++)
                    total += output[i][1];
               // for (int i = 0; i < cache.Length; i++)
                //    Console.WriteLine("{0} {1}", i, cache[i]);
                //Console.WriteLine("Thread {0} is done! Total is {1}", threadId, total);
                array1d.ToList().ForEach(v => Console.Write(v + ", "));
                Console.WriteLine();
            }
        }

        private static Barrier _barrier;

        public static void Process(object data)
        {
            Process(data as int[]);
        }

        public static double Process(int[] data)
        {

            double d = data.Average();
           // Console.WriteLine(string.Format("{0} {1}", data[0], d));

          _barrier.SignalAndWait();

            //Console.WriteLine(string.Format("{0} Post barrier", data[0]));
 
            return d;
        }



        static void TestThreads()
        {
            int side = 1024;
            int[][] myJaggedArray = new int[side][];
            for (int i = 0; i < side; i++)
            {
                myJaggedArray[i] = new int[side];
                int cnt = i;
                for (int x = 0; x < side; x++)
                    myJaggedArray[i][x] = cnt++;
            }
            _barrier = new Barrier(side);
            Stopwatch sw = new Stopwatch();
            sw.Start();
            Thread[] threads = new Thread[side];
            for (int i = 0; i < side; i++)
                threads[i] = new Thread(new ParameterizedThreadStart(Process));
            for (int i = 0; i < side; i++)
                threads[i].Start(myJaggedArray[i]);
            for (int i = 0; i < side; i++)
                threads[i].Join();
            sw.Stop();
            Console.WriteLine(sw.ElapsedMilliseconds + "ms");
            Console.WriteLine("Done");
            Console.ReadKey();
        }

        static void TestTasks()
        {
            int side = 1024;
            int[][] myJaggedArray = new int[side][];
            for (int i = 0; i < side; i++)
            {
                myJaggedArray[i] = new int[side];
                int cnt = i;
                for (int x = 0; x < side; x++)
                    myJaggedArray[i][x] = cnt++;
            }
            //_barrier = new Barrier(Environment.ProcessorCount, side);
            Task<double>[] tasks = new Task<double>[side];
            Stopwatch sw = new Stopwatch();
            sw.Start();
            for (int i = 0; i < side; i++)
            {
                tasks[i] = new Task<double>((stateObj) =>
                 {
                     int[] array = (int[])stateObj;

                     double d = Process(array);
                     return d;

                 }, myJaggedArray[i]);
            }

            foreach (Task t in tasks)
                t.Start();
            Task.WaitAll(tasks);
           // tasks.ToList().ForEach(t => Console.WriteLine(t.Result));
            sw.Stop();
            Console.WriteLine(sw.ElapsedMilliseconds + "ms");
            Console.WriteLine("Done");
            Console.ReadKey();
            //_barrier.Dispose();
        }
    }

    public class BarrierEx 
    {
        public BarrierEx(int count, int totalCount) 
        { 
            Count = count;
            TotalCount = totalCount;
            _localCount = Count;
        } 
        public int Count 
        { 
            get; 
            set; 
        }
        public int TotalCount
        {
            get;
            set;
        }

        private int _localCount;

        public void Wait()
        {
            --TotalCount;
            while (TotalCount > 0)
                Thread.Sleep(1);
        }
       
        public void SignalAndWait() 
        { 
            lock (this) 
            {
                while (TotalCount > 0)
                {
                    if (--Count > 0)
                    {
                        Console.WriteLine("Wait");
                        System.Threading.Monitor.Wait(this);
                    }
                    else
                    {
                        
                        
                        TotalCount -= _localCount;
                        Count = _localCount;
                        Console.WriteLine("here TotalCount {0}", TotalCount);
                    } 

                }
                System.Threading.Monitor.PulseAll(this);
                Console.WriteLine("Exiting lock Count {0} TotalCount {1}", Count, TotalCount);
                
                //TotalCount--;
                //if (--Count > 0) 
                //{ 
                //    System.Threading.Monitor.Wait(this); 
                //}
                //else
                //{
                //    System.Threading.Monitor.PulseAll(this);
                //} 
                //if (TotalCount > 0)
                //{
                //    System.Threading.Monitor.Wait(this);
                //} 
                //else 
                //{ 
                //    System.Threading.Monitor.PulseAll(this); 
                //} 
            } 
        } 
    }

    public class HThreadParams
    {
        public HThread Thread { get; set; }

        public GPUMethodArgs UserArgs { get; set; }
    }

    public abstract class GPUMethodArgs
    {

    }

    public class vecAddParams : GPUMethodArgs
    {
        public int[][] a { get; set; }
        public int[][] b { get; set; }
        public int[][] output { get; set; }

        //public int RandomValue;

        //public int RandomProperty  { get; set; }
    }

}
