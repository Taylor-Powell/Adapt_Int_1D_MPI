# Adaptive Single-Variable Integration Using MPI

## Purpose

This code was written as part of the effort that led to the conference paper titled _Project-Based Exploration of Cluster Computing and Parallelization Using Raspberry Pis_, available at [Link](https://ieeexplore.ieee.org/document/10025198).

The code employs N-point Gaussian quadrature to integrate over a vector of x-values for a given function. This form of adaptive integration ensures that the integration is both efficient and effective, similar to traditional quadrature methods for 'well-behaved' integration kernels. However, it is also efficient for integration kernels with rapid fluctuations that cause other methods to fail. This approach is combined with MPI in a purposely naïve way: the integration region is evenly subdivided across available nodes, and then adaptive quadrature is run on each node separately.

As shown in the paper, we purposefully chose integration kernels that exhibit rapid variation within the integration region, which causes uneven task distribution across the available nodes. 

<img width="400" height="161" alt="RasPi_Fig2" src="https://github.com/user-attachments/assets/114aaa1d-69db-4105-9d98-9b2cc3406b37" />
<img width="400" height="255" alt="Log Plot - Linear" src="https://github.com/user-attachments/assets/de101223-a177-454e-abde-4b716c5ef47e" /><img width="400" height="255" alt="Log Plot - Quadratic" src="https://github.com/user-attachments/assets/2fdf3b08-f1ce-4aac-994f-dc5fa3e145fd" />
<img width="400" height="255" alt="Log Plot - Cubic" src="https://github.com/user-attachments/assets/a0fc4b3f-6331-48cd-ab7f-bc175c43e29c" />

In the plots above, the set of functions $\{\sin(10x), \sin(10x^2), \sin(10x^3)\}$ was used to enforce this point. The first function, $\sin(10x)$, is well-behaved and oscillates at a fixed rate; thus, we see little variance in the log-log plot of the maximum and minimum time taken by particular nodes as the number of nodes is increased. This behavior is indicative of even task distribution across the computing cluster.

By contrast, both $\sin(10x^2)$ and $\sin(10x^3)$ generate increasingly rapid oscillations away from the origin; therefore, we see large deviations in their log-log plot between the maximum and minimum time taken by particular nodes.

## Gauss-Legendre Quadrature

Unlike the Riemann sum method discussed in the preceding section, here we will use a more robust integration routine with much higher accuracy: Gauss-Legendre quadrature. For standard Gauss-Legendre quadrature, we write an integral over the region $-1\le x\le1$ in terms of a series of weights ($w_i$) as,

$$\int_{-1}^1f(x)dx\simeq\sum_{i=1}^nw_if(x_i),$$

where $x_i$ are the $i$-th zeroes of the corresponding $n$-th order Legendre polynomials. For any region which is not integrated from $[-1,1]$, we can perform a change of integral simply by,

$$\begin{align}
\int_a^bf(x)dx&=\int_{-1}^1f\left(\frac{b-a}{2}x+\frac{a+b}{2}\right)\left(\frac{b-a}{2}\right)dx\\
&\simeq\frac{b-a}{2}\sum_{i=1}^nw_if\left(\frac{b-a}{2}x_i+\frac{a+b}{2}\right)
\end{align}$$

For this implementation of Gauss quadrature, we wrote the program to be flexible for any desired number of points for Gauss quadrature ($n$-point). Since the order of the Gauss rule changes the number and value of the weights used in the computation, we also wrote functions to quickly compute the roots of $n$-th order Legendre polynomials and compute the weights from them. The Legendre polynomials can be computed at a point quickly to arbitrary precision using the recurrence relation,

$$P_n(x)=\frac{(2n-1)xP_{n-1}(x)-(n-1)P_{n-2}(x)}{n},$$

where $P_0(x)=x$ and $P_1(x)=x$. Since the roots are symmetric in the interval $[-1,1]$ about $x=0$, and the number of roots is equal to the order of the Legendre polynomial, we use a modified brute-force algorithm. we search in the interval $[-1,-\epsilon]$ (with $\epsilon=10^{-9}$) for all roots. The interval is split into a region 40 times more fine than the number of roots on the interval (Legendre polynomials are generally well-behaved, and roots are decently spaced), and each of them is checked for a sign change. If a sign change is found, bisectional root-finding is run to a precision of $10^{-15}$. Then, for all computed roots, they are mirrored to the positive interval $[\epsilon,1]$. Finally, for odd-order Legendre polynomials, the guaranteed root at $x=0$ is added manually. 

Taking the roots found previously, each weight is computed using the relation,

$$w_i=\frac{2}{(1-x_i^2)[P_n'(x_i)]^2},$$

where the derivative of the Legendre polynomial is computed using the recurrence relation,

$$\frac{x^2-1}{n}\frac{d}{dx}P_n(x)=xP_n(x)-P_{n-1}(x)$$

And these weights are passed back to the Gauss quadrature function to be applied over the interval of integration.

## Adaptive Integration

Standard numerical integration techniques over a domain use discretize the interval and use the function values along each subinterval to approximate the value of the integral. Adaptive integration instead uses the behavior of the function to determine the number of subintervals needed recursively. If a function is found to be varying outside of allowed tolerances over a given interval, it is subdivided and checked again for the amount of variation. We follow a standard adaptive integration scheme:
1. Call integration function over $[x_L,x_R]$ with tolerance $\epsilon$.
2. Compute the approximate integral using Gauss $n$-point quadrature.
3. Break the integral into two halves and compute the approximate integral on each subinterval.
4. Check if the error is below tolerance. If not, recursively call the integration function on each subinterval with the acceptable tolerance halved ($\epsilon'=\epsilon/2$).
5. Return value to calling environment.

It just remains to develop a methodology for testing error on over the interval. We do this with two conditions:
1. Compare the value of the previous integral to the value of the integral approximated over each of the two subintervals. If their value differs by less than the desired tolerance, the error is satisfactory.
2. Check if the difference between $x_L$ and $x_R$ is approaching machine precision for subtraction. If so, the error is ``as good as it gets" for this interval.

These two conditions are compared with an _OR_ condition before the approximation is accepted and returned to the calling function.

For this computational scheme, the program may not need to uniformly sample a region to get an acceptable approximation if the function in one region varies more rapidly than in another area. For a concrete example, let's take a look at three sinusoidal functions given in the figure above: $\sin(10x)$, $\sin(10x^2)$, and $\sin(10x^3)$. The function $\sin(10x)$ is well-behaved and uniformly varying across the interval. Since sine has a linear dependence on $x$, its period of oscillation remains the same. By contrast, the other two functions have polynomial dependence on $x$; therefore, their periods shorten drastically as $x$ increases.

For an adaptive integration routine estimating the value of the integral of these functions, it will need to sample the right side of each function's interval more finely to generate a sufficiently accurate approximation for the area under the curve. This means, if we naïvely split the region evenly between the various threads and call adaptive integration, we would expect the threads working in regions further from the origin to take more computational time than those nearer to the origin. Thus, some threads may finish early and be idle while others are still working, which is an undesirable quality for a parallelized program. We may then quantify this discrepancy to demonstrate the loss of speedup.
