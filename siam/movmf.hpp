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

	double ** data_points;	//x set of data vectors (n x d)-array
	double * alpha;			//weights of the clusters (cluster size)
	double * kappa;			//concentration parameter (dispersion)
	double ** mu;			//mean direction  (k x d)-array
	double ** f;			//(f) vMF distribution (probability density function)  (k x n)-array   (objective function)
	double ** p;			//(p) pdistribution of the hidden variable  (k x n)-array
	double ** r;			//
	double *  c;			// c_d(kappa) normalizing constant
	int * clust;			//cluster indices

	double mult(int i, int j);
	void calculate_norm_const();
	void initialRandMeans();
	void meansFromSpkmeans();
	double * normalize(double * vector);
	void printVec(double * vec, int size);
	void printVec(int * vec, int size);
	void printMat(double ** mat, int size1, int size2);

	double *d_c, *d_x, *d_mu, *d_kappa, *d_p, *d_alpha, *d_f, *d_clust;
	double *h_x, *h_mu, *h_p, *h_f;

public:
	movmf(int dim, int clusters, int vectors, double ** data);
	void initialize();
	void initialize(double * alpha, double ** mu, double * kappa);
	void run();
	void print();
	double * getAlpha();
	double * getKappa();
	double ** getMu();	

protected:
	void expectation_soft();
	void expectation_hard();
	void maximization();

};

#endif