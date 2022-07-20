#ifndef INT_FUNCS_H_INCLUDED
#define INT_FUNCS_H_INCLUDED

#include <functional>
#include <Eigen/Dense>

namespace gauss_quad
{
	struct vals
	{
		double x, fx, sx;
	};
	
	double integrate(std::function<double(double x)> f_x, vals x0, vals xf, double tol, int N);

	double gauss_int_comp(std::function<double(double x)> f_x, vals x0, vals xm, vals xf, double tol, int N, Eigen::VectorXd x, Eigen::VectorXd w);
	vals gauss_struct(std::function<double(double x)> f_x, vals x0, vals xf, Eigen::VectorXd x, Eigen::VectorXd w, int N);
	double Legendre_Gen(double x, int n);
	double Legendre_Gen_prime(double x, int n);
	Eigen::VectorXd LegendreRootFinder(int n);
	double bisectionRootLeg(double xl, double xr, double eps, int n);
	Eigen::VectorXd LegendreWeightFinder(Eigen::VectorXd r, int n);
}

#endif