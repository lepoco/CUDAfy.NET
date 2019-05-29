// (C) Hybrid DSP Systems 2011
namespace GASS.CUDA.FFT
{
    using System;

    /// <summary>
    /// Certain R2C and C2R transforms go much more slowly when FFTW memory
    /// layout and behaviour is required. The default is "best performance",
    /// which means not-compatible-with-fftw. Use the cufftSetCompatibilityMode
    /// API to enable exact FFTW-like behaviour.
    /// </summary>
    [Flags]
    public enum CUFFTCompatibility
    {
        /// <summary>
        /// </summary>
        Native = 0,
        /// <summary>
        ///      Inserts extra padding between packed in-place transforms for
        ///      batched transforms with power-of-2 size. (default)
        /// </summary>
        FFTW_Padding = 1,
        /// <summary>
        ///      Guarantees FFTW-compatible output for non-symmetric complex inputs
        ///      for transforms with power-of-2 size. This is only useful for
        ///      artificial (i.e. random) datasets as actual data will always be
        ///      symmetric if it has come from the real plane. If you don't
        ///      understand what this means, you probably don't have to use it.
        /// </summary>
        FFTW_Asymmetric = 2,
        /// <summary>
        ///     For convenience, enables all FFTW compatibility modes at once.
        /// </summary>
        FFTW_All = 3
    }
}
