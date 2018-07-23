#include "movmf.hpp"
#include "cstdlib"
//#include "bessel.cpp"
#include "newbessel.h"

#define PI to_RR("3.1415926535897932384626433832795028841971693993751058209")
#define dPI to_double(PI)

movmf::movmf(int dim, int clusters, int vectors, double ** data)
{
	d = dim;
	k = clusters;
	n = vectors;

	data_points = data;

	alpha = new double [clusters];
	kappa = new double [clusters];

	//data_points = new double *[vectors];
	f = new double *[vectors];
	p   = new double *[vectors];
	for(int i=0;i<vectors; i++)
	{
	//	data_points[i] = new double[dim];
		f[i] = new double[clusters];
		p[i]   = new double[clusters];
	}	

	mu   = new double *[clusters];
	for(int i=0;i<clusters; i++)
		mu[i] = new double[dim];
	
	c = new double [clusters];
	//double ** r;
	clust = new int[vectors];

}

void movmf::initialize()
{
/*	alpha[0] = 0.3;
	alpha[1] = 0.7;

	mu[0][0] = 0; 
	mu[0][1] = 0.5;
	mu[0][2] = 0.5*sqrt(3);
	mu[1][0] = 0.5;
	mu[1][1] = 0.5*sqrt(3);
	mu[1][2] = 0;

	kappa[0] = 3;
	kappa[1] = 3;
	*/
	initialRandMeans();
	meansFromSpkmeans();

	double * clust_size = new double[k];
	for(int i=0;i<k;i++)
	{
		kappa[i] = 1;
		clust_size[i] = 0;
	}
	for(int i=0;i<n;i++)
		clust_size[clust[i]]++;

	for(int i=0;i<k;i++)
		alpha[i] = clust_size[i]/n;

	std::cout << "Kappa : ";
	printVec(kappa,k);
	std::cout << "Alpha : ";
	printVec(alpha,k);
	std::cout << "Mu : ";
	printMat(mu,k,d);

	//zero clust
	for(int i=0;i<n;i++)
		clust[i] = 0.0;
}


// Getting the initial random means
void movmf::initialRandMeans()
{
	// getting global mean
	double * sumv = new double[n];

	for(int j=0;j<d;j++)
		sumv[j] = 0.0;

	for(int j=0;j<d;j++)
		for(int i=0;i<n;i++)
			sumv[j] += data_points[i][j]; 

//	printVec(sumv, d);
	double * mu_0 = normalize(sumv);
//	printVec(mu_0, d);

	// perturbing global mean to get initial cluster centroids
	double perturb = 0.1;

	for(int h=0; h<k;h++)
	{
		double * randVec = new double[n];
		for(int i=0;i<n;i++)
			randVec[i] = ((double)rand()/(double)RAND_MAX) - 0.5;
		
		double randNorm = perturb * ((double)rand()/(double)RAND_MAX);

		double * smallRandVec = normalize(randVec);
		for(int i=0;i<d;i++)
		{
			smallRandVec[i] *= randNorm;
			mu[h][i] = mu_0[i] + randNorm*smallRandVec[i];		
		}	
		mu[h] = normalize(mu[h]);
		delete[] randVec;
	}

//	printMat(mu,k,d);
	delete[] sumv;
}

//Getting the means from spkmeans
void movmf::meansFromSpkmeans()
{
	double diff    = 1;
	double epsilon = 0.001;
	double value   = 100;
	int iteration = 1;

	double ** simMat = new double*[n];
	for(int i=0;i<n;i++)
	{
		simMat[i] = new double[k];
		for(int h=0;h<k;h++)
			simMat[i][h] = 0.0;
	}

	while (diff > epsilon && iteration < 10)
	{  
	//	display(['Iteration ',num2str(iteration)]);
		std::cout << "Iteration " << iteration << " ";
		std::cout << "diff = " << diff << " " << std::endl;

		iteration++;
		double oldvalue = value;

		for(int i=0;i<n;i++)
			for(int h=0;h<k;h++)
				simMat[i][h] = 0.0;
		// assign points to nearest cluster
		for(int i= 0;i<n;i++)
			for(int h=0;h<k;h++)
			{
				for(int j=0;j<d;j++)
					simMat[i][h] += data_points[i][j]*mu[h][j];
			}

	//	printMat(simMat,10,k);	

		double * simax = new double[n];
	//	int * clust = new int[n];
			
		for(int i=0;i<n;i++)
		{
			double temp_max = simMat[i][0];
			clust[i] = 0;
			for(int j=1;j<k;j++)
				if(temp_max < simMat[i][j])
				{
					temp_max = simMat[i][j];
					clust[i] = j;
				}	
			simax[i] = temp_max;	
		}	

		// compute objective function value
		value =0.0;
		for(int i=0;i<n;i++)
			value += simax[i];

		// compute cluster centroids
		for(int h=0;h<k;h++)
		{
			double * sumVec = new double[d];
			for(int j=0;j<d;j++)
				sumVec[j] = 0.0;

			for(int i=0;i<n;i++)
			{
				if(clust[i] == h)
				{
					for(int j=0;j<d;j++)
						sumVec[j] += data_points[i][j];
				}
			}	

			mu[h] = normalize(sumVec);
			delete[] sumVec;
		}
		printMat(mu, k, d);

		diff = abs(value - oldvalue);

		std::cout << "Value = " << value << " oldvalue = " << oldvalue << std::endl;

		

		delete[] simax;
	//	delete[] clust;
	}
	
	printVec(clust,n);
//	//display(clust);
//	figure;
//	subplot(2,1,1),plot(1:D,clust,'bo');
//	display('Initial iterations done');

//	Clust1 = clust;
}

double * movmf::normalize(double * vector)
{
	double * normalized = new double[d];

	double temp = 0.0;

	for(int i=0;i<d;i++)
		temp += vector[i]*vector[i];

	temp = sqrt(temp);
	for(int i=0;i<d;i++)
		normalized[i] = vector[i]/temp;

	return normalized;
}

void movmf::run()
{
	double diff      = 1;
	double epsilon   = 0.0001;
	double value     = 100;
	int iteration = 1;

	printVec(clust,n);

	while(diff > epsilon && iteration < 10)
	{
		std::cout << "Iteration " << iteration << " ";
		
		
		iteration++;
		double oldvalue = value;

		expectation_soft();
		//expectation_hard();
		
		value = 0.0;
		for(int i=0;i<n;i++)
			for(int h=0;h<k;h++)
				value += f[i][h];

		maximization();

		diff = abs(value - oldvalue);
		std::cout << "diff = " << diff << " " << std::endl;
		std::cout << "Value = " << value << " oldvalue = " << oldvalue << std::endl;
	}
	//printMat(p,n,k);

	printVec(clust,n);
}

void movmf::expectation_soft()
{
	calculate_norm_const();

	for (int i=0; i<n; i++)
	{
		for(int h=0; h<k; h++)
			f[i][h] = c[h] + log(alpha[h]) + mult(h,i);

		RR temp = to_RR(0.0);
		for(int h=0; h<k; h++)
			temp += exp(to_RR(f[i][h]));

		for(int h=0; h<k; h++)
			p[i][h] = f[i][h] - to_float(log(temp));

		double temp_max  = f[i][0];
		clust[i] = 0;
		for(int j=1;j<k;j++)
			if(temp_max < f[i][j])
			{
				temp_max = f[i][j];
				clust[i] = j;
			}
	}	
}

void movmf::expectation_hard()
{
	calculate_norm_const();

	for (int i=0; i<n; i++)
	{
		for(int h=0; h<k; h++)
			f[i][h] = c[h] + log(alpha[h]) + mult(h,i);

		for(int h=0; h<k; h++)
			p[i][h] = 0.0;

		double temp_max  = f[i][0];
		clust[i] = 0;
		for(int j=1;j<k;j++)
			if(temp_max < f[i][j])
			{
				temp_max = f[i][j];
				clust[i] = j;
			}	
				
		p[i][clust[i]] = 1.0;
	}	
}

void movmf::maximization()
{
	for(int h=0; h<k; h++)
	{
		//alpha update
		alpha[h] = 0.0;
		for(int i=0; i<n; i++)
			alpha[h] += exp(p[i][h]);
		alpha[h] = alpha[h]/n;
		//mu update
		for(int j=0; j<d; j++)
		{
			mu[h][j] = 0.0;
			for(int i=0; i<n; i++)
				mu[h][j] += data_points[i][j] * exp(p[i][h]); 
		}
		//r bar
		double norm_mu = 0.0;
		for(int j=0; j<d; j++)
			norm_mu += mu[h][j] * mu[h][j];
		double r_bar = sqrt(norm_mu)/(n * alpha[h]);
		//mu update
		//normalize(mu[h]);
		for(int j=0; j<d; j++)
			mu[h][j] /= sqrt(norm_mu);

		//kappa update
		kappa[h] = (r_bar * d - r_bar * r_bar * r_bar) / (1 - r_bar * r_bar);
	}
}

double movmf::mult(int h, int i)
{
	double result = 0.0;

	for(int l=0; l<d; l++)
		result += mu[h][l] * data_points[i][l];

	result *= kappa[h];

	return result;
}

void movmf::calculate_norm_const()
{
	RR::SetPrecision(300);
  	RR bessel, nu, x;
  	double dim = d; 
//  	std::cout << "half_dim = " << dim/2 <<std::endl;
  	nu = dim/2 -1;

//  	std::cout << " C(k): ";
	for(int h=0; h<k; h++)
	{
		c[h] = 0.0;

		x = kappa[h];
	//	std::cout << " x = " << x;
	//	std::cout << " nu = " << nu;

		bessel = BesselI(nu, x);
	//	std::cout << " bessel = " << bessel;		
		
		double logBessel = to_double(log(bessel));
	
		c[h] = (d/2-1)*log(kappa[h]) - (d/2) * log(2*dPI) - logBessel;
	
	//	std::cout << " c[h] = " << c[h];
	}	 
//	std::cout << std::endl;
}

void movmf::print()
{
	std::cout << "Clustering via mixture of von Mises-Fisher";
	std::cout << std::endl << "Alpha : ";
	for(int i=0; i<k;i++)
	{
		std::cout << alpha[i] << " ";
	}

	std::cout << std::endl << "Kappa : ";
	for(int i=0; i<k;i++)
	{
		std::cout << kappa[i] << " ";
	}

	std::cout << std::endl << "Mu : ";
	for(int i=0; i<k;i++)
	{	
		std::cout << std::endl << "i = " << i << " : ";
		for(int j=0;j<d;j++)
			std::cout << mu[i][j] << " ";
	}

	std::cout << std::endl << "Posterior : ";
	for(int h=0; h<k;h++)
	{	
		std::cout << std::endl << "k = " << h << " : ";
		for(int i=0;i<n && i<10;i++)
			std::cout << p[i][h] << " ";
	}
}

void movmf::printVec(double * vec, int size)
{
	std::cout << " Vector printing : "; 
	for(int i=0;i<size;i++)
		std::cout << vec[i] << " ";
	std::cout << std::endl;
}

void movmf::printVec(int * vec, int size)
{
	std::cout << " Vector printing : "; 
	for(int i=0;i<size;i++)
		std::cout << vec[i] << " ";
	std::cout << std::endl;
}

void movmf::printMat(double ** mat, int size1, int size2)
{
	std::cout << " Matrix printing : " << std::endl; 
	for(int i=0; i<size1;i++)
	{	
		//std::cout << std::endl << "i = " << i << " : ";
		for(int j=0;j<size2;j++)
			std::cout << mat[i][j] << " ";
		std::cout << std::endl;
	}
}

int main(int argc, char **argv)
{
	std::string filename = "dataset.dat";
	std::ifstream dataStream;
	dataStream.open(filename.c_str(), std::ios::in); // open data file
	if (!dataStream) {
		std::cerr << "Cannot open data file\n";
		exit(1);
	}
	std::istream * dataIn = &dataStream;				// make this the data stream

	int dimension;
	*dataIn >> dimension;
	
	int size;
	*dataIn >> size;

	double ** data = new double *[size];
	for(int i=0;i<size;i++)
	{	
		data[i] = new double[dimension];
		for(int j=0;j<dimension;j++)
			*dataIn >> data[i][j];
	}

	std::cout << "Size = " << size << std::endl;
	std::cout << "Dimension = " << dimension << std::endl;
	for(int i=0;i<size && i<10;i++)
	{	
		std::cout << "data[" << i << "] = "; 
		for(int j=0; j<dimension; j++)
			std::cout << data[i][j] << " ";
		std::cout << std::endl;
	}	
	
	for(int i=0;i<size;i++) //normalizing the data
	{
		double sqsum = 0;
		for(int j=0;j<dimension;j++)
			sqsum += data[i][j]*data[i][j];
		double norm = sqrt(sqsum);
		for(int j=0;j<dimension;j++)
			data[i][j] /= norm;
	}
/*
	for(int i=0;i<size;i++)
	{	
		std::cout << "data[" << i << "] = "; 
		for(int j=0; j<dimension; j++)
			std::cout << data[i][j] << " ";
		std::cout << std::endl;
	}
*/
	//number of expected clusters
	int clusters = 2;

	movmf obj(dimension, clusters, size, data);


	obj.initialize();
	obj.run();
	obj.print();

	return 0;
}





