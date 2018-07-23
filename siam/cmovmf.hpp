#ifndef MOVMF_HPP
#define MOVMF_HPP

#include <cmath>
#include <iostream>
#include <fstream>     			// file I/O

//#include <cstdlib>			// C standard library
//#include <cstdio>				// C I/O (for sscanf)
//#include <cstring>			// string manipulation

class movmf{
private:	
	int d;	//d - dimension of the Euclidean space. 
	int k;	//k - number of clusters (mixture of k vMF distributions)
	int n;	//n - number of data vectors

	float ** data_points;	//x set of data vectors (n x d)-array
	float * alpha;			//weights of the clusters (cluster size)
	float * kappa;			//concentration parameter (dispersion)
	float ** mu;			//mean direction  (k x d)-array
	float ** f;			//(f) vMF distribution (probability density function)  (k x n)-array   (objective function)
	float ** p;			//(p) pdistribution of the hidden variable  (k x n)-array
	float ** r;			//
	float *  c;			// c_d(kappa) normalizing constant
	int * clust;			//cluster indices

	float mult(int i, int j);
	void calculate_norm_const();
	void initialRandMeans();
	void meansFromSpkmeans();
	float * normalize(float * vector);
	void printVec(float * vec, int size);
	void printVec(int * vec, int size);
	void printMat(float ** mat, int size1, int size2);

	float *d_c, *d_x, *d_mu, *d_kappa, *d_p, *d_alpha, *d_f, *d_clust;
	float *h_x, *h_mu, *h_p, *h_f;

public:
	movmf(int dim, int clusters, int vectors, float ** data);
	void initialize();
	void initialize(float * alpha, float ** mu, float * kappa);
	void run();
	void print();
	float * getAlpha();
	float * getKappa();
	float ** getMu();	

protected:
	void expectation_soft();
	void expectation_hard();
	void maximization();

};

#endif