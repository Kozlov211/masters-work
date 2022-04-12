#include <cmath>
#include <iostream>
int main()
{
    // spot check for ν == 0
    double x = 4.35;
	double bes = std::cyl_bessel_i(0, x);
    std::cout << "I_0(" << x << ") = " << std::cyl_bessel_i(0, x) << '\n';
	double ans = 1 / (2 * 3.14159265359 * bes) * exp(x * cos(1));
	std::cout << ans << std::endl;
 
}
