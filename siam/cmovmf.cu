

#include "cmovmf.hpp"
#include "cstdlib"
//#include "bessel.cpp"
#include "newbessel.h"

#define PI to_RR("3.1415926535897932384626433832795028841971693993751058209")
#define dPI to_float(PI)

__global__ void cuda_expectation_soft(float * d_alpha, float * d_c, float * d_kappa, float * d_mu, float * d_x, float * d_p, int num_clusters, int dimension, float * d_clust, float *d_f)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float temp_sum = 0.0;
	for(int h=0; h<num_clusters; h++)
	{
		float dot_product = 0.0;
	//	float * f;

		for(int l=0; l<dimension; l++)
			dot_product += d_mu[h*dimension + l] * d_x[index * dimension + l];

		d_f[index * num_clusters + h] = d_c[h] + log(d_alpha[h]) + d_kappa[h] * dot_product;

	//	d_f[index * num_clusters + h] = index * num_clusters + h;
//		temp_sum += exp(d_f[index * num_clusters + h]);
	}
/*
	for(int h=0; h<num_clusters; h++)

		d_p[index*num_clusters + h] = d_f[index * num_clusters + h] - log(temp_sum);

	float temp_max  = d_f[index * num_clusters + 0];
	d_clust[index] = 0;
	for(int j=1;j<num_clusters;j++)
		if(temp_max < d_f[index * num_clusters + j])
		{
			temp_max = d_f[index * num_clusters + j];
			d_clust[index] = j;
		}
*/
//	__syncthreads();
}

movmf::movmf(int dim, int clusters, int vectors, float ** data)
{
	d = dim;
	k = clusters;
	n = vectors;

	data_points = data;

	alpha = new float [clusters];
	kappa = new float [clusters];

	//data_points = new float *[vectors];
	f = new float *[vectors];
	p   = new float *[vectors];
	for(int i=0;i<vectors; i++)
	{
	//	data_points[i] = new float[dim];
		f[i] = new float[clusters];
		p[i]   = new float[clusters];
	}	

	mu   = new float *[clusters];
	for(int i=0;i<clusters; i++)
		mu[i] = new float[dim];
	
	c = new float [clusters];
	//float ** r;
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

	float * clust_size = new float[k];
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
	float * sumv = new float[n];

	for(int j=0;j<d;j++)
		sumv[j] = 0.0;

	for(int j=0;j<d;j++)
		for(int i=0;i<n;i++)
			sumv[j] += data_points[i][j]; 

//	printVec(sumv, d);
	float * mu_0 = normalize(sumv);
//	printVec(mu_0, d);

	// perturbing global mean to get initial cluster centroids
	float perturb = 0.1;

	for(int h=0; h<k;h++)
	{
		float * randVec = new float[n];
		for(int i=0;i<n;i++)
			randVec[i] = ((float)rand()/(float)RAND_MAX) - 0.5;
		
		float randNorm = perturb * ((float)rand()/(float)RAND_MAX);

		float * smallRandVec = normalize(randVec);
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
	float diff    = 1;
	float epsilon = 0.001;
	float value   = 100;
	int iteration = 1;

	float ** simMat = new float*[n];
	for(int i=0;i<n;i++)
	{
		simMat[i] = new float[k];
		for(int h=0;h<k;h++)
			simMat[i][h] = 0.0;
	}

	while (diff > epsilon && iteration < 10)
	{  
	//	display(['Iteration ',num2str(iteration)]);
		std::cout << "Iteration " << iteration << " ";
		std::cout << "diff = " << diff << " " << std::endl;

		iteration++;
		float oldvalue = value;

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

		float * simax = new float[n];
	//	int * clust = new int[n];
			
		for(int i=0;i<n;i++)
		{
			float temp_max = simMat[i][0];
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
			float * sumVec = new float[d];
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

float * movmf::normalize(float * vector)
{
	float * normalized = new float[d];

	float temp = 0.0;

	for(int i=0;i<d;i++)
		temp += vector[i]*vector[i];

	temp = sqrt(temp);
	for(int i=0;i<d;i++)
		normalized[i] = vector[i]/temp;

	return normalized;
}

void movmf::run()
{	
	std::cout << "****************************    CUDA part       *************************************  " << std::endl;
	float diff      = 1;
	float epsilon   = 0.0001;
	float value     = 100;
	int iteration = 1;

	printVec(clust,n);

	// float* d_c, d_x, d_mu, d_kappa, d_p, d_alpha;
	// float* h_x, h_mu, h_p;

	h_x = (float *) malloc (n*d*sizeof(float));
	h_mu = (float *) malloc (k*d*sizeof(float));
	h_p = (float *) malloc (n*k*sizeof(float));
	h_f = (float *) malloc (n*k*sizeof(float));

	cudaMalloc((void**)&d_c, k*sizeof(float));
	cudaMalloc((void**)&d_x, n*d*sizeof(float));
	cudaMalloc((void**)&d_mu, k*d*sizeof(float));
	cudaMalloc((void**)&d_kappa, k*sizeof(float));
	cudaMalloc((void**)&d_p, n*k*sizeof(float));
	cudaMalloc((void**)&d_alpha, n*k*sizeof(float));
	cudaMalloc((void**)&d_clust, n*sizeof(float));
	cudaMalloc((void**)&d_f, n*k*sizeof(float));

	


	while(diff > epsilon && iteration < 20)
	{
		std::cout << "Iteration " << iteration << " ";
		
		
		iteration++;
		float oldvalue = value;

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

	for(int i=0;i<n;i++)
		for(int j=0;j<d;j++)
			h_x[i*d+j] = data_points[i][j];

	for(int h=0;h<k;h++)
		for(int j=0;j<d;j++)
			h_mu[h*d+j] = mu[h][j];
		
	for(int i=0;i<n;i++)
		for(int h=0;h<k;h++)
			h_p[i*k+h] = p[i][h];

	cudaMemcpy(d_alpha, alpha, k*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, k*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_kappa, kappa, k*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x, n*d*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mu, h_mu, k*d*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_p, h_p, n*k*sizeof(float), cudaMemcpyHostToDevice);

	int THREADS_PER_BLOCK = 1;//00;
	int BLOCKS = n;// /THREADS_PER_BLOCK;

	cuda_expectation_soft <<<BLOCKS,THREADS_PER_BLOCK>>> (d_alpha, d_c, d_kappa, d_mu, d_x, d_p, k, d, d_clust, d_f);

	cudaMemcpy(alpha, d_alpha, k*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, k*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(kappa, d_kappa, k*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_x, d_x, n*d*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_mu, d_mu, k*d*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_p, d_p, n*k*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_f, d_f, n*k*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(clust, d_clust, n*sizeof(float), cudaMemcpyDeviceToHost);

	for(int i=0;i<n;i++)
		for(int j=0;j<d;j++)
			data_points[i][j] = h_x[i*d+j];

	for(int h=0;h<k;h++)
		for(int j=0;j<d;j++)
			mu[h][j] = h_mu[h*d+j];
		
	for(int i=0;i<n;i++)
		for(int h=0;h<k;h++)
		{
			p[i][h] = h_p[i*k+h];
			f[i][h] = h_f[i*k+h];
		}	

	std::cout << " DEBUG ::: f from kernel = ";
	for(int i=0;i<n;i++)
		for(int h=0;h<k;h++)
			std::cout << h_f[i*k+h] << " ";
	std::cout <<std::endl;	

	for (int i=0; i<n; i++)
	{
	//	for(int h=0; h<k; h++)
	//		f[i][h] = c[h] + log(alpha[h]) + mult(h,i);

		float temp = 0.0;
		for(int h=0; h<k; h++)
			temp += exp(f[i][h]);

		for(int h=0; h<k; h++)
			p[i][h] = f[i][h] - log(temp);

		float temp_max  = f[i][0];
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

		float temp_max  = f[i][0];
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
		float norm_mu = 0.0;
		for(int j=0; j<d; j++)
			norm_mu += mu[h][j] * mu[h][j];
		float r_bar = sqrt(norm_mu)/(n * alpha[h]);
		//mu update
		//normalize(mu[h]);
		for(int j=0; j<d; j++)
			mu[h][j] /= sqrt(norm_mu);

		//kappa update
		kappa[h] = (r_bar * d - r_bar * r_bar * r_bar) / (1 - r_bar * r_bar);
	}
}

float movmf::mult(int h, int i)
{
	float result = 0.0;

	for(int l=0; l<d; l++)
		result += mu[h][l] * data_points[i][l];

	result *= kappa[h];

	return result;
}

void movmf::calculate_norm_const()
{
	RR::SetPrecision(300);
  	RR bessel, nu, x;
  	float dim = d; 
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
		
		float logBessel = to_float(log(bessel));
	
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
	std::cout << std::endl;
}

void movmf::printVec(float * vec, int size)
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

void movmf::printMat(float ** mat, int size1, int size2)
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

	float ** data = new float *[size];
	for(int i=0;i<size;i++)
	{	
		data[i] = new float[dimension];
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
		float sqsum = 0;
		for(int j=0;j<dimension;j++)
			sqsum += data[i][j]*data[i][j];
		float norm = sqrt(sqsum);
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

