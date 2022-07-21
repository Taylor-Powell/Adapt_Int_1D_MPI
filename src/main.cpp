/* -----------------------------------------------------------
 Taylor Powell                              September 29, 2021
 Adaptive Interpolation - Gaussian N-Point Adaptive Quadrature
 MPI and Eigen libraries required
 -------------------------------------------------------------
 Input:
 xl - left x bound
 xr - right x bound
 decimals - desired decimal precision for integral value
 f_x - function to be evaluated

 Output:
 Integral of function over interval

 Comments:
 Interval is discretized linearly (even partitions over
 integration range). Depending on the underlying function,
 this may not result in a corresponding speedup for the
 integration if portions of the integration range require more
 dense sampling

 -----------------------------------------------------------*/

#include <iostream>
#include <iomanip>
#include <functional>
#include <Eigen/Dense>
#include <mpi.h>
#include "int_funcs.h"

const double pi = 3.14159265358979323846; // 20 digits
const double e = 2.71828182845904523536; // 20 digits

const double xl = 0.0;      // left bound
const double xr = 1000; // right bound
const int decimals = 10;    // Desired # of correct decimals
const double eps = pow(10.0, -decimals); // tolerance
const int N = 10; // N-point Gauss Quadrature

// Function to be integrated
double f_x(double x)
{
    return x * cos(10.0 * x * x) / (x * x + 1.0);
}

int main(int argc, char** argv)
{
    int node, nproc;
    double time, maxtime, mintime, Qval;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &node);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Barrier(MPI_COMM_WORLD);

    time = MPI_Wtime();

    ///////////////////// ADAPTIVE GAUSS METHOD ////////////////////////

    // Initialize variables for Gauss Adaptive
    double Q = 0.0;
    double dx = (xr - xl) / nproc;

    // Structs used for the integrate function
    gauss_quad::vals x0, xf;
    x0.x = xl + node * dx;
    x0.fx = f_x(x0.x);
    xf.x = xl + (node + 1) * dx;
    xf.fx = f_x(xf.x);

    // Perform Integration
    Q = gauss_quad::integrate(f_x, x0, xf, eps / ((double)nproc), N);

    time = MPI_Wtime() - time;

    // Gather values 
    MPI_Reduce(&Q, &Qval, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time, &mintime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    if (node == 0) // Output on main node
    {
        std::cout << "f(x) = x cos(10 x^2) / (1 + x^2)" << std::endl; // Testing
        std::cout << "Integral of f(x) using Adaptive " << N << "-point Gauss Quadrature "
            << std::fixed << std::setprecision(3)
            << "from " << xl << " < x < " << xr << " is:" << std::endl
            << std::setprecision(std::min(decimals, 15)) << Qval 
            << std::endl << std::endl;

        std::cout << std::setprecision(2)
            << "Max time: " << 1000000.0 * maxtime << " microseconds" << std::endl
            << "Min time: " << 1000000.0 * mintime << " microseconds" << std::endl;
    }
    MPI_Finalize();

    return 0;
}