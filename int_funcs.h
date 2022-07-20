#ifndef INT_FUNCS_H_INCLUDED
#define INT_FUNCS_H_INCLUDED

#include <functional>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

struct vals
{
	double x, fx, sx;
};

namespace gauss_quad
{
	double integrate(function<double(double x)> f_x, vals x0, vals xf, double tol, int N);

	double gauss_int_comp(function<double(double x)> f_x, vals x0, vals xm, vals xf, double tol, int N, VectorXd x, VectorXd w);
	vals gauss_struct(function<double(double x)> f_x, vals x0, vals xf, VectorXd x, VectorXd w, int N);
	double Legendre_Gen(double x, int n);
	double Legendre_Gen_prime(double x, int n);
	VectorXd LegendreRootFinder(int n);
	double bisectionRootLeg(double xl, double xr, double eps, int n);
	VectorXd LegendreWeightFinder(VectorXd r, int n);
}

#endif