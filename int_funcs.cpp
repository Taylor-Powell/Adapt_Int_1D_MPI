#include <iostream>
#include <iomanip>
#include <functional>
#include <Eigen/Dense>
#include "int_funcs.h"

using namespace std;

const double pi = 3.14159265358979323846; // 20 digits

/*
	N-th order Gauss Quadrature integration over a vector of x-values. Built to be flexible on the order of
	the Legendre polynomials by solving for roots and weights numerically before computing the integral.
	If statement at the beginning is only relevant for this particular code to compare outputs and timing
	directly with other methods based on number of function calls.
*/
double gauss_quad::integrate(function<double(double x)> f_x, vals x0, vals xf, double tol, int N)
{
	Eigen::VectorXd x = gauss_quad::LegendreRootFinder(N);
	Eigen::VectorXd w = gauss_quad::LegendreWeightFinder(x, N);
	vals xm = gauss_quad::gauss_struct(f_x, x0, xf, x, w, N);
	return gauss_quad::gauss_int_comp(f_x, x0, xm, xf, tol, N, x, w);
}

// Returns Gauss N-point integration value
double gauss_quad::gauss_int_comp(function<double(double x)> f_x, vals x0, vals xm, vals xf, double tol, int N, Eigen::VectorXd x, Eigen::VectorXd w)
{
	vals xl = gauss_quad::gauss_struct(f_x, x0, xm, x, w, N);
	vals xr = gauss_quad::gauss_struct(f_x, xm, xf, x, w, N);
	double err_prev_curr = abs(xm.sx - xl.sx - xr.sx);

	if ((err_prev_curr < tol * 15.0) || (xf.x - x0.x < 1.0e-14))
	{
		if (abs(xf.fx - x0.fx) > 10000.0)
		{
			cout << "Possible singularity detected in the integration range near "
				<< "x = " << xm.x << endl
				<< "Exiting function...";
			exit(2);
		}
		return xl.sx + xr.sx;
	}

	// Return value of Gauss N-point intergration rule on interval
	return gauss_quad::gauss_int_comp(f_x, x0, xl, xm, tol / 2.0, N, x, w)
		+ gauss_quad::gauss_int_comp(f_x, xm, xr, xf, tol / 2.0, N, x, w);
}

// Function to purely return Gauss N-point integration from x0 to xf
gauss_quad::vals gauss_quad::gauss_struct(function<double(double x)> f_x, vals x0, vals xf, Eigen::VectorXd x, Eigen::VectorXd w, int N)
{
	vals xm;
	xm.x = (x0.x + xf.x) / 2.0;
	xm.fx = f_x(xm.x);
	double a, b, Qi;
	double Q = 0.0;
	a = (xf.x - x0.x) / 2.0;
	b = (xf.x + x0.x) / 2.0;
	for (int j = 0; j < N; j++)
		Q += w(j) * f_x(a * x(j) + b);
	xm.sx = Q * a;
	return xm;
}

/*
	Legendre polynomial generator using recursive definition
*/
double gauss_quad::Legendre_Gen(double x, int n)
{
	double Pn, Pnm1, Pnm2;
	Pn = 0.0;
	Pnm2 = 1.0;
	Pnm1 = x;
	for (int i = 2; i < n + 1; i++)
	{
		Pn = ((2.0 * i - 1.0) * x * Pnm1 - (i - 1.0) * Pnm2) / i;
		Pnm2 = Pnm1;
		Pnm1 = Pn;
	}
	return Pn;
}

/*
	For a known root of the Legendre polynomial, find the corresponding P'_n(x)
	Found using recurrence relation for the derivative of a Legendre polynomial
*/
double gauss_quad::Legendre_Gen_prime(double x, int n)
{
	double dPn, Pn, Pnm1, Pnm2;
	Pn = 0.0;
	Pnm2 = 1.0;
	Pnm1 = x;
	for (int i = 2; i < n + 1; i++)
	{
		Pn = ((2.0 * i - 1.0) * x * Pnm1 - (i - 1.0) * Pnm2) / i;
		Pnm2 = Pnm1;
		Pnm1 = Pn;
	}
	dPn = (x * Pn - Pnm2) * n / (x * x - 1.0);
	return dPn;
}

/*
	Brute force root-finding algorithm.
	Here, the number of roots inside the interval are known in advance, since
	the functions are Legendre polynomials on the range [-1, 1]. We also know
	the roots are symmetric about x=0, thus we narrow our search region and
	copy found roots to the other half of the domain. For odd Legendre
	polynomials, x=0 is a guarenteed root, thus we narrow our search interval
	one epsilon away from x=0 and add in the last root manually.
*/
Eigen::VectorXd gauss_quad::LegendreRootFinder(int n)
{
	double xl = -1.0;
	double xr = -0.000000001; // epsilon shift from x=0
	int rcount = 0;
	int N = 20 * n; // Parse regions 40 times more fine than number of roots in interval -- n/2 roots in [-1,0)
	double dx = (xr - xl) / N;
	Eigen::VectorXd roots(n); // x_i for Legendre roots
	Eigen::VectorXd droots(n);// w_i for weights in Gauss Quadrature
	Eigen::VectorXd x(N + 1); // fill array of x-values
	for (int i = 0; i < N + 1; i++)
		x(i) = xl + (double)i * dx;

	for (int i = 0; i < N; i++) // each subinterval
	{
		if (abs(Legendre_Gen(x(i), n)) < 1.0e-15) // allow epsilon accuracy
		{
			roots(n - rcount - 1) = -x(i); // reflect root to positive side
			roots(rcount) = x(i);
			rcount += 1;
		}
		else if (Legendre_Gen(x(i), n) * Legendre_Gen(x(i + 1), n) < 0.0)
		{
			roots(n - rcount - 1) = -bisectionRootLeg(x(i), x(i + 1), 1.0e-15, n);// reflect root to positive side
			roots(rcount) = -roots(n - rcount - 1);
			rcount += 1;
		}
	}
	if (n % 2 != 0) roots(rcount) = 0.0; // For odd Legendre polys, manually add root at x=0
	return roots;
}

/*
	Bisectional Root Finder. We forego a step counter, since we know the
	bisection method is guarenteed to converge, and the number of steps
	for a given accuracy is bounded, thereby eliminating the need for a
	step counter.
*/
double gauss_quad::bisectionRootLeg(double xl, double xr, double eps, int n)
{
	double fxl = Legendre_Gen(xl, n); // f(xl)
	double fxr = Legendre_Gen(xr, n); // f(xr)

	///////////// EXECUTE BISECTION METHOD /////////////
	double c, fc;
	c = (xl + xr) / 2.0;
	fc = Legendre_Gen(c, n);
	while (abs(xl - xr) >= eps) // Bisection Method
	{
		if (fc == 0.0)
			return c;
		else if ((fxl * fc < 0.0) && (fc * fxr > 0.0))
		{ // Root is in left interval
			xr = c;
			fxr = Legendre_Gen(xr, n);
		}
		else if ((fxl * fc > 0.0) && (fc * fxr < 0.0))
		{ // Root is in right interval
			xl = c;
			fxl = Legendre_Gen(xl, n);
		}
		c = (xl + xr) / 2.0;
		fc = Legendre_Gen(c, n);
	}
	return c;
	////////////////////////////////////////////////////
}

/*
	Shell function to find corresponding weights for Gauss Quadrature based
	on vector of roots of the corresponding n-th order Legendre Polynomials
*/
Eigen::VectorXd gauss_quad::LegendreWeightFinder(Eigen::VectorXd r, int n)
{
	Eigen::VectorXd w(n);
	for (int i = 0; i < n; i++)
		w(i) = 2.0 / ((1.0 - r(i) * r(i)) * pow(Legendre_Gen_prime(r(i), n), 2));
	return w;
}