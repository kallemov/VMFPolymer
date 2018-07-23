#include "vmf_distribution.hpp"
#include "newbessel.h"

#include <cstdlib>
#include <math.h>
#include <iostream>

//#define EXP 2.71828182845904523536
//#define PI  3.1415926535897932384626433832795028841971
#define EXP 2.71828182845904523536028747135266249775724709369995
#define PI  3.14159265358979323846264338327950288419716939937510 

bool verbose_vmf=false;

double unifRand();
double unifRand(double a, double b);
double * vmf_generate3d(double kappa);
double  vmf_generate2d(double alpha, double kappa);
void vonmisesfisher3d(double * data, double * mu, double kappa, int N);
void vonmisesfisher2d(double * data, double * mu, double kappa, int N);
void vmf_sampling(double * data, double * mu, double kappa, int N, int dimension);
void vmf_parameters_estimation(double * data, double * mu, double &kappa, int N, int dimension);
double bessel(double ds, double dx);
double get_kappa(double R, int dimension);


#include <math.h>

#define EPSILON 0.000001

#define CROSS(dest, v1, v2){                 \
          dest[0] = v1[1] * v2[2] - v1[2] * v2[1]; \
          dest[1] = v1[2] * v2[0] - v1[0] * v2[2]; \
          dest[2] = v1[0] * v2[1] - v1[1] * v2[0];}

#define DOT(v1, v2) (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])

#define SUB(dest, v1, v2){       \
          dest[0] = v1[0] - v2[0]; \
          dest[1] = v1[1] - v2[1]; \
          dest[2] = v1[2] - v2[2];}

/*
 * A function for creating a rotation matrix that rotates a vector called
 * "from" into another vector called "to".
 * Input : from[3], to[3] which both must be *normalized* non-zero vectors
 * Output: mtx[3][3] -- a 3x3 matrix in colum-major form
 * Authors: Tomas Möller, John Hughes
 *          "Efficiently Building a Matrix to Rotate One Vector to Another"
 *          Journal of Graphics Tools, 4(4):1-4, 1999
 */
void fromToRotation(double from[3], double to[3], double mtx[3][3]) {
  double v[3];
  double e, h, f;

  CROSS(v, from, to);
  e = DOT(from, to);
  f = (e < 0)? -e:e;
  if (f > 1.0 - EPSILON)     /* "from" and "to"-vector almost parallel */
  {
    double u[3], v[3]; /* temporary storage vectors */
    double x[3];       /* vector most nearly orthogonal to "from" */
    double c1, c2, c3; /* coefficients for later use */
    int i, j;

    x[0] = (from[0] > 0.0)? from[0] : -from[0];
    x[1] = (from[1] > 0.0)? from[1] : -from[1];
    x[2] = (from[2] > 0.0)? from[2] : -from[2];

    if (x[0] < x[1])
    {
      if (x[0] < x[2])
      {
        x[0] = 1.0; x[1] = x[2] = 0.0;
      }
      else
      {
        x[2] = 1.0; x[0] = x[1] = 0.0;
      }
    }
    else
    {
      if (x[1] < x[2])
      {
        x[1] = 1.0; x[0] = x[2] = 0.0;
      }
      else
      {
        x[2] = 1.0; x[0] = x[1] = 0.0;
      }
    }

    u[0] = x[0] - from[0]; u[1] = x[1] - from[1]; u[2] = x[2] - from[2];
    v[0] = x[0] - to[0];   v[1] = x[1] - to[1];   v[2] = x[2] - to[2];

    c1 = 2.0 / DOT(u, u);
    c2 = 2.0 / DOT(v, v);
    c3 = c1 * c2  * DOT(u, v);

    for (i = 0; i < 3; i++) {
      for (j = 0; j < 3; j++) {
        mtx[i][j] =  - c1 * u[i] * u[j]
                     - c2 * v[i] * v[j]
                     + c3 * v[i] * u[j];
      }
      mtx[i][i] += 1.0;
    }
  }
  else  /* the most common case, unless "from"="to", or "from"=-"to" */
  {
#if 0
    /* unoptimized version - a good compiler will optimize this. */
    /* h = (1.0 - e)/DOT(v, v); old code */
    h = 1.0/(1.0 + e);      /* optimization by Gottfried Chen */
    mtx[0][0] = e + h * v[0] * v[0];
    mtx[0][1] = h * v[0] * v[1] - v[2];
    mtx[0][2] = h * v[0] * v[2] + v[1];

    mtx[1][0] = h * v[0] * v[1] + v[2];
    mtx[1][1] = e + h * v[1] * v[1];
    mtx[1][2] = h * v[1] * v[2] - v[0];

    mtx[2][0] = h * v[0] * v[2] - v[1];
    mtx[2][1] = h * v[1] * v[2] + v[0];
    mtx[2][2] = e + h * v[2] * v[2];
#else
    /* ...otherwise use this hand optimized version (9 mults less) */
    double hvx, hvz, hvxy, hvxz, hvyz;
    /* h = (1.0 - e)/DOT(v, v); old code */
    h = 1.0/(1.0 + e);      /* optimization by Gottfried Chen */
    hvx = h * v[0];
    hvz = h * v[2];
    hvxy = hvx * v[1];
    hvxz = hvx * v[2];
    hvyz = hvz * v[1];
    mtx[0][0] = e + hvx * v[0];
    mtx[0][1] = hvxy - v[2];
    mtx[0][2] = hvxz + v[1];

    mtx[1][0] = hvxy + v[2];
    mtx[1][1] = e + h * v[1] * v[1];
    mtx[1][2] = hvyz - v[0];

    mtx[2][0] = hvxz - v[1];
    mtx[2][1] = hvyz + v[0];
    mtx[2][2] = e + hvz * v[2];
#endif
  }
}


// Generate a random number between 0 and 1
// return a uniform number in [0,1].
double unifRand()
{
    return rand() / double(RAND_MAX);
}

// Generate a random number in a real interval.
// param a one end point of the interval
// param b the other end of the interval
// return a inform rand numberin [a,b].
double unifRand(double a, double b)
{
    return (b-a)*unifRand() + a;
}

double bessel(double ds, double dx)
{
	RR::SetPrecision(300);
  	
  	RR bessel, s, x;
 	s   = to_RR(ds);
  	x = to_RR(dx);

	bessel = BesselI(s, x);
	
	return to_double(bessel);
}

double get_kappa(double R, int dimension)
{
	//simple algorithm
	double kappa = R*(dimension - R*R)/(1-R*R);

	//Newton's iterations
	double dim = dimension;
	double s1 = dim/2.0;
 	double s2 = dim/2.0 - 1.0;
 	double kappa_old, bessel_1, bessel_2, A;

 	int iterations = 0;
 	while(iterations < 3) //((kappa/kappa_old - 1.0)>tolerance)
	{
    kappa_old = kappa;
    bessel_1 = bessel(s1, kappa_old);
    bessel_2 = bessel(s2, kappa_old);
    
		A = bessel_1/bessel_2;

		kappa = kappa_old - (A-R)/(1.0 - A*A - A*(dim-1.0)/kappa_old);
		iterations++;
	}

	return kappa;
}

void vmf_parameters_estimation(double * data, double * mu, double &kappa, int N, int dimension)
{
	//calcutate mu....
	for(int i=0;i<dimension;i++)
		mu[i] = 0.0;

	for(int i=0; i<N; i++)
		for(int dim=0;dim<dimension;dim++)
			mu[dim] += data[i*dimension + dim];

	double	norm_mu = 0.0;
	for(int i=0;i<dimension;i++)
		norm_mu += mu[i]*mu[i];
	norm_mu = sqrt(norm_mu);

	for(int i=0;i<dimension;i++)
		mu[i] = mu[i] /norm_mu;

	//calculate kappa...
	kappa = get_kappa(norm_mu/N, dimension);
}

void vmf_parameters_estimation(double * data, double * mu, double &kappa, int N, int dimension, double a_constraint)
{
  for(int i=N-1;i>0;i--)
    for(int icoor=0;icoor<dimension;icoor++)
      data[i*dimension+icoor] = (data[i*dimension+icoor] - data[(i-1)*dimension+icoor])/a_constraint;

  vmf_parameters_estimation(data, mu, kappa, N, dimension);
}

void vmf_sampling(double * data, double * mu, double kappa, int N, int dimension)
{
	if (dimension  != 2 && dimension != 3) 
    {
      if (verbose_vmf)
      {
      std::cout << "dimension = " << dimension << std::endl; 
    	std::cout << "Message from vmf_distribution.cpp: Invalid dimension value. Must be 2 or 3 !" << std::endl; 
    }
    	exit(1);
    }

	if (kappa < 0) 
    {
      if (verbose_vmf)
      {
    	std::cout << "Message from vmf_distribution.cpp: kappa must be >= 0" << std::endl; 
    	std::cout << "Message from vmf_distribution.cpp: Set kappa to be 0" << std::endl; 
    	}
      kappa = 0;
    }

  //normalize Mu
  double tmp=0;
  for(int i =0;i<dimension;i++)
      tmp += mu[i]*mu[i];
  tmp = sqrt(tmp);
  if (tmp >1e-06 )
    for(int i =0;i<dimension;i++)
        mu[i] = mu[i]/tmp;

  if(dimension == 3)
  { 
    if (verbose_vmf) std::cout << "**** 3D ****" << std::endl; 
    vonmisesfisher3d(data, mu, kappa, N);
  }  
  else if(dimension == 2)
  {
    if (verbose_vmf) std::cout << "**** 2D ****" << std::endl;  
    vonmisesfisher2d(data, mu, kappa, N);
  }  

}

void vmf_sampling(double * data, double * mu, double kappa, int N, int dimension, double a_constraint)
{
  vmf_sampling(data, mu, kappa,N,dimension);

  for(int i=0;i<N-1;i++)
    for(int icoor=0;icoor<dimension;icoor++)
      data[(i+1)*dimension+icoor] = data[i*dimension+icoor] + a_constraint*data[(i+1)*dimension+icoor];
}

double * vmf_generate3d(double kappa)
{
    double * vector = new double[3];

    double Y = unifRand();
    double tmp = pow(EXP, -kappa);
    tmp = tmp*(1-Y) +Y*pow(EXP, kappa);
    double W = log(tmp)/kappa;

    double theta;
    do{
        theta = unifRand(0,2*PI);
    }while (theta == 0);

    double v1 = cos(theta);
    double v2 = sin(theta);

    vector[0] = v1 * sqrt(1-W*W);
    vector[1] = v2 * sqrt(1-W*W);
    vector[2] = W;

    return vector;
}

void vonmisesfisher3d(double * data, double * mu, double kappa, int N)
{
    double z [3]= {0,0,1};
    double mat[3][3];
    fromToRotation(z, mu, mat);

    // for(int i=0;i<3;i++)
    // {
    //     for(int j=0;j<3;j++)
    //         std::cout << " " << mat[i][j];
    //     std::cout << std::endl;
    // }

    for(int i=0;i<N;i++)
    {
        double * temp;

        temp = vmf_generate3d(kappa);
        //if (verbose) std::cout << temp[0] << " " << temp[1] << " " << temp[2] << std::endl;
        for(int j=0;j<3;j++)
        {
            double dot = 0;    
            for(int k=0;k<3;k++)
                dot += mat[j][k]*temp[k];
            data[i*3+j] = dot;
        }
    }
}

double vmf_generate2d(double mu, double kappa) 
{
    /**
    Circular data distribution.
    
            mu is the mean angle, expressed in radians between 0 and 2*pi, and
            kappa is the concentration parameter, which must be greater than or
            equal to zero.  If kappa is equal to zero, this distribution reduces
            to a uniform random angle over the range 0 to 2*pi.
    
    */
    double a, b, c, f, r, theta, u1, u2, u3, z;

    if ((kappa<=1e-06)) {
        return ((2*PI)*unifRand());
    }
    a = (1.0+sqrt((1.0+((4.0*kappa)*kappa))));
    b = ((a-sqrt((2.0*a)))/(2.0*kappa));
    r = ((1.0+(b*b))/(2.0*b));

    while(1) {
        u1 = unifRand();
        z = cos(PI*u1);
        f = ((1.0+(r*z))/(r+z));
        c = (kappa*(r-f));
        u2 = unifRand();
        if ((!((u2>=(c*(2.0-c))) && (u2>(c*exp(1.0-c)))))) {
            break;
        }
    }
    u3 = unifRand();
    if ((u3>0.5)) {
        theta = (fmod(mu, (2*PI))+acos(f));
    }
    else {
        theta = (fmod(mu, (2*PI))-acos(f));
    }
    return theta;
}

void vonmisesfisher2d(double * data, double * mu, double kappa, int N)
{
    double alpha = atan2(mu[1],mu[0]);

    if (alpha <0)
      alpha +=2*PI;
    // std::cout << "Alpha = "<<alpha <<std::endl;

    double theta;

    for(int i=0;i<N;i++)
    {
        theta = vmf_generate2d(alpha,kappa);
        data[i*2] = cos(theta); 
        data[i*2+1] = sin(theta);
        // std::cout << "data =  " << data[i*2] << " " << data[i*2+1] << std::endl;
    }
}


/*
--------------------------------------------------------------------------------------------------------------------




%
% the following algorithm is following the modified Ulrich's algorithm 
% discussed by Andrew T.A. Wood in "SIMULATION OF THE VON MISES FISHER 
% DISTRIBUTION", COMMUN. STATIST 23(1), 1994.

% step 0 : initialize
b = (-2*kappa + sqrt(4*kappa^2 + (m-1)^2))/(m-1);
x0 = (1-b)/(1+b);
c = kappa*x0 + (m-1)*log(1-x0^2);

% step 1 & step 2
nnow = n; w = [];
%cnt = 0;
while(true)
    ntrial = max(round(nnow*1.2),nnow+10) ;
    Z = betarnd((m-1)/2,(m-1)/2,ntrial,1);
    U = rand(ntrial,1);
    W = (1-(1+b)*Z)./(1-(1-b)*Z);
    
    indicator = kappa*W + (m-1)*log(1-x0*W) - c >= log(U);
    if sum(indicator) >= nnow
        w1 = W(indicator);
        w = [w ;w1(1:nnow)];
        break;
    else
        w = [w ; W(indicator)];
        nnow = nnow-sum(indicator);
        %cnt = cnt+1;disp(['retrial' num2str(cnt) '.' num2str(sum(indicator))]);
    end
end

% step 3
V = UNIFORMdirections(m-1,n);
X = [repmat(sqrt(1-w'.^2),m-1,1).*V ;w'];

if muflag
    mu = mu / norm(mu);
    X = rotMat(mu)'*X;
end
end


function V = UNIFORMdirections(m,n)
% generate n uniformly distributed m dim'l random directions
% Using the logic: "directions of Normal distribution are uniform on sphere"

V = zeros(m,n);
nr = randn(m,n); %Normal random 
for i=1:n
    while 1
        ni=nr(:,i)'*nr(:,i); % length of ith vector
        % exclude too small values to avoid numerical discretization
        if ni<1e-10 
            % so repeat random generation
             nr(:,i)=randn(m,1);
        else
             V(:,i)=nr(:,i)/sqrt(ni);
            break;
        end
    end
end

end

function rot = rotMat(b,a,alpha)
% ROTMAT returns a rotation matrix that rotates unit vector b to a
%
%   rot = rotMat(b) returns a d x d rotation matrix that rotate
%   unit vector b to the north pole (0,0,...,0,1)
%
%   rot = rotMat(b,a ) returns a d x d rotation matrix that rotate
%   unit vector b to a
%
%   rot = rotMat(b,a,alpha) returns a d x d rotation matrix that rotate
%   unit vector b towards a by alpha (in radian)
%
%    See also .

% Last updated Nov 7, 2009
% Sungkyu Jung


[s1 s2]=size(b);
d = max(s1,s2);
b= b/norm(b);
if min(s1,s2) ~= 1 || nargin==0 , help rotMat, return, end  

if s1<=s2;    b = b'; end

if nargin == 1;
    a = [zeros(d-1,1); 1];
    alpha = acos(a'*b);
end

if nargin == 2;
    alpha = acos(a'*b);
end
if abs(a'*b - 1) < 1e-15; rot = eye(d); return, end
if abs(a'*b + 1) < 1e-15; rot = -eye(d); return, end

c = b - a * (a'*b); c = c / norm(c);
A = a*c' - c*a' ;

rot = eye(d) + sin(alpha)*A + (cos(alpha) - 1)*(a*a' +c*c');
end
*/
