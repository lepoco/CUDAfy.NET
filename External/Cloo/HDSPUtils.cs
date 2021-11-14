using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;
namespace Cloo
{
    public class HDSPUtils
    {
        /// <summary>
        /// Gets the size of the type specified. Note that this differs from Marshal.SizeOf for System.Char (it returns 2 instead of 1).
        /// </summary>
        /// <param name="t">The type to get the size of.</param>
        /// <returns>Size of type in bytes.</returns>
        public static int SizeOf(Type t)
        {
            if (t == typeof(char))
                return 2;
            else
                return Marshal.SizeOf(t);
        }
    }
}
