using System;
using Cudafy.Types;
namespace Cudafy.Maths.BLAS
{
    public interface IGPGPUBLAS
    {
        //T[] Allocate<T>(T[] x);
        //T[] Allocate<T>(int x);
        //T[,] Allocate<T>(int x, int y);
        //void Free(object o);
        //void CopyFromDevice<T>(T[,] devArray, T[,] hostArray);
        //void CopyFromDevice<T>(T[] devArray, T[] hostArray);
        //void CopyToDevice<T>(T[,] hostArray, T[,] devArray);
        //void CopyToDevice<T>(T[] hostArray, T[] devArray);
        //void Dispose();

        double ASUM<T>(global::Cudafy.Types.ComplexD[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1);
        double ASUM<T>(global::Cudafy.Types.ComplexD[] vector, int n = 0, int row = 0, int incx = 1);
        float ASUM<T>(global::Cudafy.Types.ComplexF[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1);
        float ASUM<T>(global::Cudafy.Types.ComplexF[] vector, int n = 0, int row = 0, int incx = 1);
        double ASUM<T>(double[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1);
        double ASUM<T>(double[] vector, int n = 0, int row = 0, int incx = 1);
        float ASUM<T>(float[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1);
        float ASUM<T>(float[] vector, int n = 0, int row = 0, int incx = 1);
        void AXPY<T>(T alpha, object vectorx, object vectory, int n = 0, int row = 0, int incx = 1, int y = 0, int incy = 1);
        void COPY<T>(object src, object dst, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);
        T DOT<T>(T[] vectorx, T[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);
        T DOTC<T>(T[] vectorx, T[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);
        T NRM2<T>(T[] vectorx, int n = 0, int rowx = 0, int incx = 1);
        int IAMAX<T>(object devArray, int n = 0, int row = 0, int incx = 1);
        int IAMAX<T>(T[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1);
        int IAMIN<T>(object devArray, int n = 0, int row = 0, int incx = 1);
        int IAMIN<T>(T[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1);
        void SCAL<T>(T alpha, object vector, int n = 0, int row = 0, int incx = 1);
        void SCAL<T>(T alpha, T[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1);
        void ROT(float[] vectorx, float[] vectory, float sc, float ss, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0);
        void ROT(double[] vectorx, double[] vectory, double sc, double ss, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0);
        void ROT(ComplexF[] vectorx, ComplexF[] vectory, float sc, ComplexF ss, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0);
        void ROT(ComplexD[] vectorx, ComplexD[] vectory, float sc, ComplexD cs, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0);
        void ROTG(float[] host_sa, float[] host_sb, float[] host_sc, float[] host_ss);
        void ROTG(double[] host_da, double[] host_db, double[] host_dc, double[] host_ds);
        void ROTG(ComplexF[] host_ca, ComplexF[] host_cb, float[] host_sc, float[] host_ss);
        void ROTG(ComplexD[] host_ca, ComplexD[] host_cb, double[] host_dc, double[] host_ds);


    }
}
