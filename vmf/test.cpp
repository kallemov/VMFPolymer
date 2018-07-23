//g++ -o test test.cpp vmf_distribution.cpp newbessel.cc -I/usr/local/include -L/usr/local/lib -lntl -lm

//#include "bessel.hpp"
#include "newbessel.h"
#include "vmf_distribution.hpp"
#include <iostream>

using namespace std;

int main()
{
	// double tol = 1.e-16;

	// RR::SetPrecision(300);
  	
 //  	RR bessel, nu, nu_1,  x;
 //  	double dim = 3; //2
 //  	double dx;

 // 	nu   = to_RR(dim/2.0);
 // 	nu_1 = to_RR(dim/2.0 - 1.0);
  	
 //  	cout<< "Enter x = ";
	// cin >> dx;
 //  	x = to_RR(dx);

	// bessel = BesselI(nu, x);
	
	// double doub_Bessel = to_double(bessel);

	// cout.precision(20);
	// cout<< "Bessel function Is = " << doub_Bessel << endl;

	int N = 100;
	double * data = new double[N*3];
	double mu[3] = {1,0,0};
	double kappa = 20;

	for(int i=0;i<10;i++)
	{
		vmf_sampling(data, mu, kappa, N, 3);
		vmf_parameters_estimation(data, mu, kappa, N, 3);
		cout << "Estimated kappa = " << kappa <<endl;
		cout << "Estimated mu = " << mu[0] << " " << mu[1] << " " <<mu[2] << endl;
	}	


	return 0;
}