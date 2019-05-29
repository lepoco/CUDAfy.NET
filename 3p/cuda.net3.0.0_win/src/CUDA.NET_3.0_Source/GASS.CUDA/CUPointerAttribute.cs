using System;
using System.Collections.Generic;
using System.Text;

namespace GASS.CUDA
{
    public enum CUPointerAttribute
    {
        Context = 1,        /**< The ::CUcontext on which a pointer was allocated or registered */
        MemoryType = 2,     /**< The ::CUmemorytype describing the physical location of a pointer */
        DevicePointer = 3,  /**< The address at which a pointer's memory may be accessed on the device */
        HostPointer = 4,    /**< The address at which a pointer's memory may be accessed on the host */
        P2PTokens = 5       /**< A pair of tokens for use with the nv-p2p.h Linux kernel interface */
    }

}
