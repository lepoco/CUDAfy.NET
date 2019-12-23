using Microsoft.Win32;
using System;
using System.Diagnostics;
using System.IO;

namespace Cudafy
{
    /// <summary>This utility class resolves path to nVidia's nvcc.exe and Microsoft's cl.exe.</summary>
    internal static class NvccExe
    {
        /// <summary>Get GPU Computing Toolkit 7.0 installation path.</summary>
        /// <remarks>Throws an exception if it's not installed.</remarks>
        static string getToolkitBaseDir()
        {
            //Just for now
            return @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2";
        }


        static readonly string toolkitBaseDir = getToolkitBaseDir();

        const string csNVCC = "nvcc.exe";

        /// <summary>Path to the nVidia's toolkit bin folder where nvcc.exe is located.</summary>
        public static string getCompilerPath()
        {
            return Path.Combine( toolkitBaseDir, "bin", csNVCC );
        }

        /// <summary>Path to the nVidia's toolkit's include folder.</summary>
        public static string getIncludePath()
        {
            return Path.Combine( toolkitBaseDir, @"include" );
        }

        /// <summary>Path to the Microsoft's visual studio folder where cl.exe is localed.</summary>
        public static string getClExeDirectory()
        {
            //Traditional method of searching by the registry
            string[] versionsToTry = new string[] { "12.0", "11.0" };
            RegistryKey localKey;
            if( Environment.Is64BitProcess )
                localKey = RegistryKey.OpenBaseKey( RegistryHive.LocalMachine, RegistryView.Registry32 );
            else
                localKey = Registry.LocalMachine;

            RegistryKey vStudio = localKey.OpenSubKey( @"SOFTWARE\Wow6432Node\Microsoft\VisualStudio" );
            if( null == vStudio )
                throw new CudafyCompileException( "nVidia GPU Toolkit error: visual studio was not found" );

            foreach( string ver in versionsToTry )
            {
                RegistryKey key = vStudio.OpenSubKey( ver );
                if( null == key )
                    continue;
                string InstallDir = key.GetValue( "InstallDir" ) as string;
                if( null == InstallDir )
                    continue;
                // C:\Program Files (x86)\Microsoft Visual Studio 12.0\Common7\IDE\

                InstallDir.TrimEnd( '\\', '/' );
                string clDir = Path.GetFullPath( Path.Combine( InstallDir, @"..\..\VC\bin" ) );

                if( Environment.Is64BitProcess )
                {
                    // In 64-bits processes we use a 64-bits compiler. If you'd like to always use the 32-bits one, remove this.
                    clDir = Path.Combine( clDir, "amd64" );
                }
                if( !Directory.Exists( clDir ) )
                    continue;

                string clPath = Path.Combine( clDir, "cl.exe" );
                if( File.Exists( clPath ) )
                    return clDir;
            }

            //Search using vswhere.exe
            Process getVS = new Process
            {
                StartInfo = {
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        FileName = "vswhere.exe",
                        Arguments = " -latest -property installationPath"
                }
            };
            getVS.Start();
            string vsPath = Path.GetFullPath(Path.Combine(getVS.StandardOutput.ReadLine(), @"VC\Tools\MSVC"));
            getVS.WaitForExit();

            string[] vsDirs = Directory.GetDirectories(vsPath);

            if (vsDirs.Length > 0)
                for (int i = vsDirs.Length; i > 0; i--)
                    if (File.Exists(Path.Combine(vsDirs[i - 1], @"bin\Hostx64\x64\cl.exe")))
                        return Path.Combine(vsDirs[i - 1], @"bin\Hostx64\x64\cl.exe");

            throw new CudafyCompileException( "nVidia GPU Toolkit error: cl.exe was not found" );
        }
    }
}