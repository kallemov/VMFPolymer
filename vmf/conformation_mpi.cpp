//This is linear version of 2 order strong solution of langevin equation
// g++ newduhamel.cpp Duhamel.cpp ./dSFMT/dSFMT.c -DDSFMT_MEXP=44497 -L/usr/local/lib -lgmp -lmpfr -lmpfrcpp            

#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <math.h>
#include "./dSFMT/dSFMT.h"
#include "Duhamel.H"
#include <string>
#include "mpi.h"
using std::cout;
using std::cin;
using std::cerr;
using std::endl;
using std::string;

#include <assert.h>
#include <mpfrcpp/mpfrcpp.hpp>

/*
// #define GNUFPU
#ifdef GNUFPU
// enable SIGFPE trap for some gnu compiler systems
#define _gnu_source
#include <fpu_control.h>
#endif
*/

// templates for projection methods
void (*method)(  int, int, double*, double*, double*, double );
void rattle(     int, int, double*, double*, double*, double );
void ciccotti(   int, int, double*, double*, double*, double );
void newproject( int, int, double*, double*, double*, double );

// global parameters
bool set2fluidvel=true;
double gam, sigma, a_constraint;
double fluidvelzero, fluidwavelength, fluid_time_scale, fluid_coor_scale;
int precision;
bool verbose;
bool stattest = false;
int num_batches;
int case_fluid=3;
int init_conf=0;


//shift for resolutions
int resadd=0;

enum Centering { START=0, CENTER=1, END=2 };
// centering for x projection
Centering xcent = START;
// centering for v projection
Centering vcent = END;

// number of space dimensions in simulation
int Dim=3;
#define MAXDIM 3

// tolerances
double XTol = 1.e-16;
double VTol = 1.e-16;
int maxIterations = 100;

enum OnOff { OFF=0, ON=1 };
OnOff Projection = ON;
OnOff Stochastic = ON;

// if projection is off, set zero=1 to include the
// dx_dot_dv terms that would otherwise be zero.
int zero=0;

// starting seed of random number generator
int seed=0;

void set_parameters( int argc, char **argv,
 int &nodes, int &coarsesteps, double &endtime, int &paths,
 int &resolutions )
{
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD,&myid); 
  extern double gam, sigma, a_constraint;
  extern double fluidvelzero, fluidwavelength, fluid_time_scale, fluid_coor_scale;
  extern double XTol, VTol;
  extern int maxIterations, resadd, precision;
  extern int Dim, zero;
  extern bool verbose;
  extern bool stattest;
  extern int num_batches;
  extern int case_fluid, init_conf;

  static const char * projnames[] = {"RATTLE","CICCOTTI","NEW"};
  static const char * onoff[] = {"OFF","ON"};
  FILE *f=NULL;

  if (myid==0)
    cout << "argc=" << argc << endl;
  
  if( argc>=1 ) {
    if (myid==0)
      cout << "input file \"" << argv[0] << "\"" << endl;
    f = fopen(argv[0],"rb");
    if( f == NULL ) {
      cerr << "can't open \"" << argv[0] << "\"" << endl;
    }
  }
      
  if( f == NULL ) {
    f = fopen("input_order","rb");
    if( f == NULL ) {
      cerr << "can't open \"input\"" << endl;
      exit(0);
    }
  }
    

  char s[1000];
  char *c;
  static const char *delim = " =,\n\r";
  int anint;
  double adbl;
  while((c = fgets( s, 1000, f )) != NULL ) {
    char *t = strtok( s, delim );
    if( !strcmp(t,"xcentering") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&anint);
      switch(anint) {
        case 0: xcent=START; break;
        case 1: xcent=CENTER; break;
        case 2: xcent=END; break;
        default:
        cerr << "unknown value for X centering" << endl;
        exit(0);
      }
      if (myid==0)
	cout << "set X centering " << xcent << endl;
    } else if( !strcmp(t,"vcentering") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&anint);
      switch(anint) {
        case 0: vcent=START; break;
        case 1: vcent=CENTER; break;
        case 2: vcent=END; break;
        default:
        cerr << "unknown value for V centering" << endl;
        exit(0);
      }
      if (myid==0)
	cout << "set V centering " << vcent << endl;
    } else if( !strcmp(t,"xtol") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%lf",&XTol);
      if (myid==0)
	cout << "set XTol " << XTol << endl;
    } else if( !strcmp(t,"vtol") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%lf",&VTol);
      if (myid==0)
	cout << "set VTol " << VTol << endl;
    } else if( !strcmp(t,"maxiterations") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&maxIterations);
      if (myid==0)
	cout << "set max iterations " << maxIterations << endl;
    } else if( !strcmp(t,"stochastic") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&anint);
      Stochastic = OFF;
      if( anint ) Stochastic=ON;
      if (myid==0)
	cout << "set stochastic " << onoff[Stochastic] << endl;
    } else if( !strcmp(t,"dim") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&anint);
      Dim = anint;
      if (myid==0)
	cout << "set dimension " << Dim << endl;
      if( Dim < 1 || Dim > MAXDIM ) {
        cerr << "dimension has illegal value" << endl;
        exit(0);
      }
    } else if( !strcmp(t,"method") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&anint);
      switch(anint) {
        case 0: method=rattle; break;
        case 1: method=ciccotti; break;
        case 2: method=newproject; break;
        default:
        cerr << "unknown algorithm" << endl;
        exit(0);
      }
      if (myid==0)
	cout << "set projector method " << projnames[anint] << endl;
    } else if( !strcmp(t,"seed") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&anint);
      seed = anint;
      if (myid==0)
	cout << "set seed " << seed << endl;
    } else if( !strcmp(t,"zero") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&anint);
      zero = anint;
      if (myid==0)
	cout << "set zero " << zero << endl;
    } else if( !strcmp(t,"projection") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&anint);
      Projection = OFF;
      if( anint ) Projection=ON;
      if (myid==0)
	cout << "set projection " << onoff[Projection] << endl;
    } else if( !strcmp(t,"nodes") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&anint);
      nodes = anint;
      if (myid==0)
	cout << "set nodes " << nodes << endl;
    } else if( !strcmp(t,"gamma") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%lf",&adbl);
      gam = adbl;
      if (myid==0)
	cout << "set gamma " << gam << endl;
    } else if( !strcmp(t,"sigma") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%lf",&adbl);
      sigma = adbl;
      if (myid==0)
	cout << "set sigma " << sigma << endl;
    } else if( !strcmp(t,"a") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%lf",&adbl);
      a_constraint = adbl;
      if (myid==0)
	cout << "set a " << a_constraint << endl;
    } else if( !strcmp(t,"fluidv0") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%lf",&adbl);
      fluidvelzero = adbl;
      if (myid==0)
	cout << "set fluidvelzero " << fluidvelzero << endl;
    } else if( !strcmp(t,"fluidl") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%lf",&adbl);
      fluidwavelength = adbl;
      if (myid==0)
	cout << "set fluidwavelength " << fluidwavelength << endl;
    } else if( !strcmp(t,"fluidsclx") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%lf",&adbl);
      fluid_coor_scale = adbl;
      if (myid==0)
	cout << "set fluid_coor_scale " << fluid_coor_scale << endl;
    } else if( !strcmp(t,"fluidsclt") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%lf",&adbl);
      fluid_time_scale = adbl;
      if (myid==0)
	cout << "set fluid_time_scale " << fluid_time_scale << endl;
    } else if( !strcmp(t,"paths") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&anint);
      paths = anint;
      if (myid==0)
	cout << "set paths   " << paths << endl;
    } else if( !strcmp(t,"resolutions") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&anint);
      resolutions = anint;
      if (myid==0)
	cout << "set resolutions " << resolutions << endl;
    } else if( !strcmp(t,"endtime") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%lf",&adbl);
      endtime = adbl;
      if (myid==0)
	cout << "set endtime " << endtime << endl;
    } else if( !strcmp(t,"coarsesteps") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&anint);
      coarsesteps = anint;
      if (myid==0)
	cout << "set coarsesteps " << coarsesteps << endl;
    } else if( !strcmp(t,"resshift") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&anint);
      resadd = anint;
      if (myid==0)
	cout << "set resolution shift " << resadd << endl;
    } else if( !strcmp(t,"precision") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&precision);
      if (myid==0)
	cout << "set precision " << precision << endl;
    } else if( !strcmp(t,"verbose") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&anint);
      if(anint==0) verbose=false; else verbose=true;
    } else if( !strcmp(t,"fluid_field") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&case_fluid);
      if (myid==0)
	cout << "set fluid field " << case_fluid << endl;
    } else if( !strcmp(t,"init_config") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&init_conf);
      if (myid==0)
	cout << "initial configuration " << init_conf << endl;
    } else if( !strcmp(t,"stattest") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&anint);
      if(anint==0) stattest=false; else stattest=true;
      if ((paths < 2) && stattest)
	{ cerr << "number of pathes should be atleast 2 for stattest"<<endl; exit(2);}
    } else if( !strcmp(t,"batches") ) {
      t = strtok( NULL, delim );
      sscanf(t,"%d",&num_batches);
      if (myid==0)
	cout << "set number batches " << num_batches << endl;
      if ((num_batches < 2) && stattest)
	{ cerr << "number of batches should be atleast 2 for stattest"<<endl; exit(2);}
    } else {
      cerr << "ack" << endl;
    }
  }
}


#ifndef MAX
#define MAX(A,B) (((A)>(B))?(A):(B))
#endif

#ifndef kron
#define kron(A,B) (((A)==(B))?(1):(0))
#endif



inline double diffdot( double *x1, double *y1 )
{
  extern int Dim;
  double d=0;
  for( int i=0; i<Dim; i++ ) d += (x1[i]-x1[i-Dim])*(y1[i]-y1[i-Dim]);
  return d;
}


void rattle( int what, int N_part, double *x, double *v, double *x0, double dt )
{
  extern int Dim;
  static int is_initialized = 0;
  static double *xmod, dy[MAXDIM];
  int N_rod = N_part - 1;
  double asq = a_constraint*a_constraint;
  extern double XTol,VTol;
  extern int maxIterations;
  extern Centering xcent, vcent;

  if( ! is_initialized ) {
    xmod = new double[Dim*N_part];
    is_initialized = 1;
  }

  int found_one = 0;

  if( (what==0) || (what==2) ) {

    for( int i=0; i<Dim*N_part; i++ )
      xmod[i] = x[i];


    int iterations = 0;
    do {
       found_one = 0;
      for( int i=0; i<N_rod; i++ ) {
        double len = 0;
        for( int dir=0; dir<Dim; dir++ ) {
           double tmp = xmod[Dim*(i+1)+dir]-xmod[Dim*i+dir];
           len += tmp*tmp;
        }
        if( fabs(sqrt(len)/a_constraint - 1.) > XTol ) {
          iterations++;
          found_one = 1;
          double dot = 0;
          for( int dir=0; dir<Dim; dir++ ) {
            // director
            dy[dir] = 0.5*(2-xcent)*(x0[Dim*(i+1)+dir]-x0[Dim*i+dir])
                    + 0.5*xcent*(xmod[Dim*(i+1)+dir]-xmod[Dim*i+dir]);
            dot += (xmod[Dim*(i+1)+dir]-xmod[Dim*i+dir])*dy[dir];
          }
          double g = (len - asq)/(4.*dot);
          for( int dir=0; dir<Dim; dir++ ) {
           xmod[Dim*i+Dim+dir] -= g*dy[dir];
           xmod[Dim*i    +dir] += g*dy[dir];
          }
          break;
        }
      }
    } while( found_one && (iterations < maxIterations) );
  }

  if( what == 2 ) {

    for( int i=0; i<Dim*N_part; i++ ) {
      v[i] = v[i] + (xmod[i]-x[i])/dt;
      x[i] = xmod[i];
    }

  } else if( what == 0 ) {
    for( int i=0; i<Dim*N_part; i++ ) {
      x[i] = xmod[i];
    }
  }

  if( what == 1 || what == 2 ) {

    int iterations = 0;
    do {
      found_one = 0;
      for( int i=0; i<N_rod; i++ ) {
        double vxdot = 0;
        double xydot = 0;
        for( int dir=0; dir<Dim; dir++ ) {
          // director
          dy[dir] = 0.5*(2-vcent)*(x0[Dim*(i+1)+dir]-x0[Dim*i+dir])
                  + 0.5*vcent*(x[Dim*(i+1)+dir]-x[Dim*i+dir]);
          vxdot += (v[Dim*(i+1)+dir]-v[Dim*i+dir])*
                   (x[Dim*(i+1)+dir]-x[Dim*i+dir]);
          xydot += (x[Dim*(i+1)+dir]-x[Dim*i+dir])*dy[dir];
        }
        if( fabs(vxdot) > VTol ) {
          iterations++;
          found_one = 1;
          double k = vxdot/(2.*xydot);
          for( int dir=0; dir<Dim; dir++ ) {
            v[Dim*i+Dim+dir] -= k*dy[dir];
            v[Dim*i    +dir] += k*dy[dir];
          }
          break;
        }
      }
    } while( found_one && (iterations < maxIterations) );

  }
}

void ciccotti( int what, int N_part, double *x, double *v, double *x0, double dt )
{
  int N_rod = N_part-1;
  static double *xmod, *lambda, *dx, *b, *xdir;
  static int is_initialized = 0;
  static double **A;
  extern int Dim;
  extern double XTol;
  extern int maxIterations;

  if( ! is_initialized ) {
    A = new double*[4];
    for( int i=0; i<4; i++ ) A[i] = new double[N_rod];
    xdir           = new double[Dim*N_part];
    xmod           = new double[Dim*N_part];
    dx             = new double[Dim*N_rod];
    lambda         = new double[N_rod];
    b              = new double[N_rod];
    is_initialized = 1;
  }
  extern double a_constraint;
  double  maxerr = 0;

  // begin with forces directed along original coordinate lines

  for( int i=0; i<Dim*N_part; i++ ) xmod[i] = x[i];
  for( int i=0; i<N_rod; i++ ) lambda[i] = 0;

  if( (what==0) || (what==2) ) {

    // iterate until converged

    int    iterations = 0;
    do {

      // create matrix

      // A[0][i] = A[i][i-1] subdiagonal
      // A[1][i] = A[i][i]   diagonal
      // A[2][i] = A[i][i+1] superdiagonal
      // A[3][i] = A[i][i+2] superduper diagonal

      // xi = xi0 + lm * drm - l0 * dr0
      // dxi = dxi0 - lm drm + 2 l0 dr0 - lp drp
      // dxi.dxi = dxi0.dxi0 - 2 lm drm.dxi0 + 4 l0 dr0.dxi0 - 2 lp drp.dxi0
      //           + lm*lm drm.drm - 4 l0 lm dr0*drm + 2 lm lp drm*drp
      //            + 4 l0 l0 dr0.dr0 - 4 l0 lp dr0*drp + lp lp drp*drp
      // 2 drm*dxi0 lm - 4 dr0*dxi0 l0 + 2 drp*dxi0 lp =
      // ( d^2 - a^2 + lm^2 drm*drm + lp^2 drp*drp + 4 l0^2 dr0*dr0
      //         - 4 l0 lm dr0*drm + 2 lm lp drm*drp -4 lp l0 dr0*drp )
                 

      // GHM want START for NSTI
      // directors
      for( int i=0; i<Dim*N_part; i++ )
        xdir[i] =  0.5*(2-xcent)*x0[i] + 0.5*xcent*xmod[i];
      for( int i=0; i<Dim*N_rod; i++ )
        dx[i] = xdir[i+Dim] - xdir[i];


      for( int i=0; i<N_rod; i++ ) {
        A[0][i] = A[1][i] = A[2][i] = A[3][i] = 0;
        if( i!=0 )
          A[0][i] = 2.*diffdot( x+Dim*(i+1), xdir+Dim*i );
        A[1][i] = -4.*diffdot( x+Dim*(i+1), xdir+Dim*(i+1) );
        if( i!=N_rod-1 )
          A[2][i] = 2.*diffdot( x+Dim*(i+1), xdir+Dim*(i+2) );
        b[i] = diffdot( x+Dim*(i+1), x+Dim*(i+1) ) - a_constraint*a_constraint;
        b[i] += 4.*lambda[i]*lambda[i]*diffdot( xdir+Dim*(i+1), xdir+Dim*(i+1) );
        if( i != 0 ) {
          b[i] += lambda[i-1]*lambda[i-1]*diffdot( xdir+Dim*i, xdir+Dim*i );
          b[i] -= 4*lambda[i-1]*lambda[i]*diffdot( xdir+Dim*i, xdir+Dim*(i+1) );
        }
        if( i != N_rod-1 ) {
          b[i] += lambda[i+1]*lambda[i+1]*diffdot( xdir+Dim*(i+2), xdir+Dim*(i+2) );
          b[i] -= 4*lambda[i+1]*lambda[i]*diffdot( xdir+Dim*(i+2), xdir+Dim*(i+1) );
        }
        if( i !=0 && i != N_rod-1 ) {
          b[i] += 2*lambda[i+1]*lambda[i-1]*diffdot( xdir+Dim*(i+2), xdir+Dim*i );
        }
      }

      // solve for lambda by triangularizing with Givens reflections
      for( int i=0; i<N_rod-1; i++ ) {
        // treat rows i and i+1 to eliminate subdiagonal A[i+1][i]
        double x1 = A[1][i];    // A[i  ][i]
        double x2 = A[0][i+1];  // A[i+1][i]
        double mu = MAX(fabs(x1),fabs(x2));
        double s1 = x1/mu;
        double s2 = x2/mu;
        double p = mu*sqrt(s1*s1+s2*s2);
        if( x1 < 0 ) p = -p;
        double c = x1/p;
        double s = x2/p;
        double v = s/(1+c);
        double u1 = A[1][i];
        double u2 = A[0][i+1];
        A[1][i] = c*u1 + s*u2;
        A[0][i+1] = v*(u1 + A[1][i]) - u2;

        u1 = A[2][i];
        u2 = A[1][i+1];
        A[2][i] = c*u1 + s*u2;
        A[1][i+1] = v*(u1 + A[2][i]) - u2;

        u1 = A[3][i];
        u2 = A[2][i+1];
        A[3][i] = c*u1 + s*u2;
        A[2][i+1] = v*(u1 + A[3][i]) - u2;

        u1 = b[i];
        u2 = b[i+1];
        b[i] = c*u1 + s*u2;
        b[i+1] = v*(u1 + b[i]) - u2;
      }

      // now back substitution
      for( int i=N_rod-1; i>=0; i-- ) {
        lambda[i] = b[i];
        for( int j=2; j<4; j++ )
          if( i+j-1 < N_rod ) lambda[i] -= A[j][i]*lambda[i+j-1];
        lambda[i] /= A[1][i];
// cout << iterations << " " << i << " " << lambda[i] << endl;
      }
    
      // modify the vector on which forces are centered

      for( int i=0; i<N_part; i++ ) {
        for( int idir=0; idir<Dim; idir++ ) {
          xmod[Dim*i+idir] = x[Dim*i+idir];
          if( i > 0 )
            xmod[Dim*i+idir] += lambda[i-1]*dx[Dim*(i-1)+idir];
          if( i < N_part-1 )
            xmod[Dim*i+idir] -= lambda[i]*dx[Dim*i+idir];
        }
      } 

      maxerr = 0;
      for( int i=0; i<N_rod; i++ ) {
        double er = fabs(sqrt(diffdot(xmod+Dim*(i+1),xmod+Dim*(i+1)))/a_constraint-1.);
        maxerr = MAX(maxerr,er);
// cout << "rod " << i << " " << er << endl;
      }

      iterations++;
    } while( (maxerr > XTol) && (iterations < maxIterations) );


  }

  if( what == 2 ) {

    for( int i=0; i<Dim*N_part; i++ ) {
       v[i] += (xmod[i] - x[i])/dt;
       x[i] = xmod[i];
    }

  } else if( what == 0 ) {
    for( int i=0; i<Dim*N_part; i++ ) {
       x[i] = xmod[i];
    }
  }

  if ( what == 1 || what == 2 ) {

// v = lm dxm - lp dxp
// dv = - lm dxm + 2 l0 dx0 - lp dxp
// dxmod .dv = - lm dxm*dxmod + 2 l0 dx0*dxmod - lp dxp*dxmod

    // GHM want END for NSTI
    // director
    for( int i=0; i<Dim*N_part; i++ )
      xdir[i] =  0.5*(2-vcent)*x0[i] + 0.5*vcent*x[i];
    for( int i=0; i<Dim*N_rod; i++ )
      dx[i] = xdir[i+Dim]-xdir[i];
  
    // and a final step to make velocity obey required constraints.

    // A[0][i] = A[i][i-1] subdiagonal
    // A[1][i] = A[i][i]   diagonal
    // A[2][i] = A[i][i+1] superdiagonal
    // A[3][i] = A[i][i+2] superduper diagonal

    // vnew = v + lm dxdirm - l0 dxdir0
    // dvnew = dv - lm dxdirm +2 l0 dxdir0 - lp dxdirp
    // x.dvnew = 0 = dx.dv - lm dx.dxdirm + 2 l0 dx.dxdir0 - lp dx.dxdirp

    for( int i=0; i<N_rod; i++ ) {
      A[0][i] = A[1][i] = A[2][i] = A[3][i] = 0;
      A[1][i] = 2*diffdot( x+Dim*(i+1), xdir+Dim*(i+1) );
      if( i > 0 ) A[0][i] = -diffdot( x+Dim*(i+1), xdir+Dim*i );
      if( i < N_rod-1  ) A[2][i] = -diffdot( x+Dim*(i+1), xdir+Dim*(i+2) );
      b[i] = - diffdot( x+Dim*(i+1), v+Dim*(i+1) );
      lambda[i] = 0;
    }

    // solve for lambda by triangularizing with Givens reflections
    for( int i=0; i<N_rod-1; i++ ) {
      // treat rows i and i+1 to eliminate subdiagonal A[i+1][i]
      double x1 = A[1][i];    // A[i  ][i]
      double x2 = A[0][i+1];  // A[i+1][i]
      double mu = MAX(fabs(x1),fabs(x2));
      double s1 = x1/mu;
      double s2 = x2/mu;
      double p = mu*sqrt(s1*s1+s2*s2);
      if( x1 < 0 ) p = -p;
      double c = x1/p;
      double s = x2/p;
      double v = s/(1+c);
      double u1 = A[1][i];
      double u2 = A[0][i+1];
      A[1][i] = c*u1 + s*u2;
      A[0][i+1] = v*(u1 + A[1][i]) - u2;

      u1 = A[2][i];
      u2 = A[1][i+1];
      A[2][i] = c*u1 + s*u2;
      A[1][i+1] = v*(u1 + A[2][i]) - u2;

      u1 = A[3][i];
      u2 = A[2][i+1];
      A[3][i] = c*u1 + s*u2;
      A[2][i+1] = v*(u1 + A[3][i]) - u2;

      u1 = b[i];
      u2 = b[i+1];
      b[i] = c*u1 + s*u2;
      b[i+1] = v*(u1 + b[i]) - u2;
    }

    // now back substitution
    for( int i=N_rod-1; i>=0; i-- ) {
      lambda[i] = b[i];
      for( int j=2; j<4; j++ )
        if( i+j-1 < N_rod ) lambda[i] -= A[j][i]*lambda[i+j-1];
      lambda[i] /= A[1][i];
    }

    for( int i=0; i<N_part; i++ ) {
      for( int idir=0; idir<Dim; idir++ ) {
        if( i > 0 )
          v[Dim*i+idir] += lambda[i-1]*dx[Dim*(i-1)+idir];
        if( i < N_part-1 )
          v[Dim*i+idir] -= lambda[i]*dx[Dim*i+idir];
      }
    }
  }
}

// Per RATTLE, correct x then v.
// Use SHAKE's matrix form to get corrected x.  Assume
// the correction forces are aligned with the coordinates
// at the midpoint

void newproject( int what, int N_part, double *x, double *v, double *crap, double dt )
{
  int N_rod = N_part-1;
  static int is_initialized = 0;
  static double *w, *xmod, *dy, *lendx;
  static int    *s;
  extern double a_constraint;
  extern int Dim;
  double mx, tmp, Qe1[MAXDIM];
  extern double XTol;
  extern int maxIterations;

  if( !is_initialized ) {
    xmod  = new double[Dim*N_part];
    dy    = new double[Dim*N_rod];
    lendx = new double[N_rod];
    w     = new double[Dim*N_rod];
    s     = new int[N_rod];
    is_initialized = 1;
  }

  for( int i=0; i<Dim*N_part; i++ ) xmod[i] = x[i];

  if( (what==0) || (what==2) ) {
    int iterations=0;
    do {

      mx = 0;

      for( int i=0; i<Dim*N_rod; i++ )
       dy[i] = xmod[i+Dim]-xmod[i];

      // calculate Householder Q to transoform dy into e1
      //   ( stored as w[], and sign s[])
      // and calculate length of dx (lendx[])

      for( int i=0; i<N_rod; i++ ) {
        lendx[i] = 0;
        for( int idir=0; idir<Dim; idir++ ) {
          tmp = xmod[Dim*i+Dim+idir] - xmod[Dim*i+0+idir];
          lendx[i]  += tmp*tmp;
        }
        lendx[i] = sqrt(lendx[i]);
        mx = MAX(mx,fabs(lendx[i]/a_constraint-1.));
                    
        double len = 0;
        for( int idir=1; idir<Dim; idir++ )
          len += dy[Dim*i+idir]*dy[Dim*i+idir];
        double k = sqrt( len + dy[Dim*i+0]*dy[Dim*i+0] );
        s[i] = 1;
        if( dy[Dim*i] >= 0 ) s[i] = -1;
        if( s[i] < 0 ) k = -k;
        len = sqrt( len + (dy[Dim*i+0]-k)*(dy[Dim*i+0]-k) );
        w[Dim*i+0] = (dy[Dim*i+0]-k)/len;
        for( int idir=1; idir<Dim; idir++ ) {
          w[Dim*i+idir] =  dy[Dim*i+idir]/len;
        }
      }

      // correct x

      for( int i=0; i<N_rod; i++ ) {

        // make v = Q e_1
        double d = w[Dim*i+0];
        Qe1[0] = 1 - 2*d*w[Dim*i+0];
        for( int idir=1; idir<Dim; idir++ ) {
          Qe1[idir] =   - 2*d*w[Dim*i+idir];
        }

        for( int j=0; j<N_part; j++ ) {
          double c = (i+1);
          if( i >= j ) c -= N_part;
          c /= N_part;

          for( int k=0; k<Dim; k++ )
            xmod[Dim*j+k] += c*s[i]*(a_constraint-lendx[i])*Qe1[k];
        }
      }
      iterations++;

    } while( (mx > XTol) && (iterations < maxIterations) );


  }

  if( what == 2 ) {
    // update the velocity with the change in coordinate due to constraint
    // and set x

    // NOTE factor of 2 differs from Anderson

    for( int i=0; i<Dim*N_part; i++ ) {
      v[i] += 2*(xmod[i]-x[i])/dt;
      x[i] = xmod[i];
    }

  } else if( what == 0 ) {
    for( int i=0; i<Dim*N_part; i++ ) {
      x[i] = xmod[i];
    }
  }

  if( what == 1 ) {

    // re-make w[] and to correspond to Householder Q
    // that transforms dx into e1

    for( int i=0; i<Dim*N_rod; i++ )
       dy[i] = x[i+Dim]-x[i];

    for( int i=0; i<N_rod; i++ ) {
      double len = 0;
      for( int idir=1; idir<Dim; idir++ ) {
        len += dy[Dim*i+idir]*dy[Dim*i+idir];
      }
      double k = sqrt( len + dy[Dim*i+0]*dy[Dim*i+0] );
      if( dy[Dim*i] >= 0 ) k = -k;
      len = sqrt( len + (dy[Dim*i+0]-k)*(dy[Dim*i+0]-k) );
      w[Dim*i+0] = (dy[Dim*i+0]-k)/len;
      for( int idir=1; idir<Dim; idir++ ) {
        w[Dim*i+idir] =  dy[Dim*i+idir]/len;
      }
    }

    // do velocity projection

    for( int i=0; i<Dim*N_rod; i++ )
       dy[i] = v[i+Dim]-v[i];

    for( int i=0; i<N_rod; i++ ) {
      // e1T Q dv = e1T( dv - 2 w (w dot dv ))
      double w_dot_dv = 0;
      for( int idir=0; idir<Dim; idir++ ) {
        w_dot_dv += w[Dim*i+idir]*dy[Dim*i+idir];
      }

      // the thing we wish to be zero.
      double tmp = dy[Dim*i+0] - 2.*w_dot_dv*w[Dim*i+0];

      // Q e1
      double d = w[Dim*i+0];
      Qe1[0] = 1 - 2.*d*w[Dim*i+0];
      for( int idir=1; idir<Dim; idir++ ) {
        Qe1[idir] =   - 2.*d*w[Dim*i+idir];
      }

      for( int j=0; j<N_part; j++ ) {
        double c = (i+1.);
        if( i >= j ) c -= N_part;
        c /= ((double)N_part);

        for( int k=0; k<Dim; k++ )
          v[Dim*j+k] -= c*tmp*Qe1[k];
      }
    }
  }
}
       
// solve A lambda = g
// for A a special tridiagonal form made up of x.
// g is modified in this routine.

// if "re_use = 1" then A will not be reconstructed.

void Aband_lambda_g( int Npart, double *x, double *g, double *lambda,
	           int re_use )
{
  extern int Dim;
  static double **Aband;
  static double *c, *s;
  int Nrod = Npart-1;
  static int is_initialized = 0;
  if( ! is_initialized ) {
    Aband = new double*[Nrod];
    for( int irod=0; irod<Nrod; irod++ ) Aband[irod] = new double[5];
    c = new double[Nrod];
    s = new double[Nrod];
    is_initialized = 1;
  }

  // A[][] is a square matrix of size Nrod x Nrod
  // Aband[i][0] = A[i][i-1]
  // Aband[i][1] = A[i][i  ]
  // Aband[i][2] = A[i][i+1]
  // Aband[i][3] = A[i][i+2]

  if( ! re_use ) { // build from scratch

    for( int irod=0; irod<Nrod; irod++ ) {
      for( int k=0; k<5; k++ ) Aband[irod][k] = 0;
    }
    for( int irod=0; irod<Nrod; irod++ ) {
      for( int jrod=irod-1; jrod<=irod; jrod++ ) {
        if( jrod >= 0 ) {
          double dot = 0;
          for( int idir=0; idir<Dim; idir++ ) {
            dot += (x[Dim*(irod+1)+idir] - x[Dim*irod+idir])*
                   (x[Dim*(jrod+1)+idir] - x[Dim*jrod+idir]);
          }
          if( irod == jrod )  {
            Aband[irod][1] = -2.*dot;
          } else {
            Aband[irod][0] = dot;
            Aband[jrod][2] = dot;
          }
        }
      }
    }


    for( int irod=1; irod<Nrod; irod++ ) {
      double u1,u2,x1,x2,mu,s1,s2,p,v;
      x1 = Aband[irod-1][1];
      x2 = Aband[irod  ][0];
      mu = MAX(fabs(x1),fabs(x2));
      s1 = x1/mu;
      s2 = x2/mu;
      p  = mu*sqrt(s1*s1+s2*s2);
      if( x1 < 0 ) p = -p;
      c[irod] = x1/p;
      s[irod] = x2/p;
      v = s[irod]/(1+c[irod]);
      for( int j=0; j<=3; j++ ) {
        // irod+j-1 < Nrod
        if( j < (Nrod+1-irod) ) {
          u1 = Aband[irod-1][1+j];
          u2 = Aband[irod  ][ +j];
          Aband[irod-1][1+j] = c[irod]*u1 + s[irod]*u2;
          Aband[irod  ][ +j] = v*(u1 + Aband[irod-1][1+j])-u2;
        }
      }
    }
    // A now triangular.
  }

  // if "re_use = 1" the following is only code executed:

  // apply Givens rotations to rhs
  for( int irod=1; irod<Nrod; irod++ ) {
    double u1,u2,v;
    u1 = g[irod-1];
    u2 = g[irod  ];
    v = s[irod]/(1+c[irod]);
    g[irod-1] = c[irod]*u1 + s[irod]*u2;
    g[irod  ] = v*(u1 + g[irod-1])-u2;
  }


  // find Lambda by back substitution
  for( int irod=Nrod-1; irod>=0; irod-- ) {
    lambda[irod] = g[irod];
    for( int k=2; k<4; k++ ) {
      // irod+k-1<Nrod
      if( k < (Nrod+1-irod) )
        lambda[irod] -= Aband[irod][k]*lambda[irod+k-1];
    }
    lambda[irod] /= Aband[irod][1];
  }

}

void AInverse( int Npart, double *x1, double **Ai )
{
  static double* rhs;
  static double* lhs;
  static int is_initialized=0;
  if( !is_initialized ) {
    is_initialized = 1;
    lhs = new double[Npart-1];
    rhs = new double[Npart-1];
  }
  for( int i=0; i<Npart-1; i++ ) {
    for( int j=0; j<Npart-1; j++ ) rhs[j] = ((i==j)?1:0);
    Aband_lambda_g( Npart, x1, rhs, lhs, ((i==0)?0:1) );
    for( int j=0; j<Npart-1; j++ ) Ai[j][i] = lhs[j];
  }
}
// we should replace this with different incompressible flows,
// including Poiseuelle

// the prescribed fluid velocity u(x(t),t), and its material derivative
// Du/Dt = u_t(x(t),t) + (v.grad)u(x(t),t)

void get_fluid_velocity( int Npart, double *x, double *v, double *u, double *Du, double time)
{
   
  // u is prescribed fluid velocity
  extern int case_fluid;
  extern int Dim;
  extern double fluidvelzero, fluidwavelength, fluid_time_scale, fluid_coor_scale;
  if (case_fluid==0)
    {  
      //Poiseuille velocity field v=v_max*(1-(r/R)^2)
      double max_vel=1.5;
      double R=1.;

      if (Dim >2) 
	{
	  cerr << " The prescribed velocity fiels is inconsistent with this dimention" << endl;
	  cerr << " Please check function get_fluid velocity" << endl;
	  exit(1);
	}
      
      for (int irod=0; irod<Npart;irod++)
	{
	  int i0=irod*Dim;
	  u[i0] = max_vel * (1 - (x[i0+1]*x[i0+1]/R/R)) ;
	  u[i0+1] = 0;
	  
	  //advection term in derivative
	  Du[i0]   = -max_vel*2.*x[i0+1]*v[i0+1]/R/R;
	  Du[i0+1] = 0;
	}
    }
  if (case_fluid==1)
    {  
      for (int irod=0; irod<Npart;irod++){
	for( int idir=0; idir<Dim; idir++ ) { 
	  int i0=irod*Dim+idir;
	  u[i0] = fluidvelzero*cos(x[i0]/fluidwavelength);
	  Du[i0] = -v[i0]*fluidvelzero*sin(x[i0]/fluidwavelength)/fluidwavelength;
	}
      }
    }
  if (case_fluid==2)
    {  
      //BCG divergence free velocity field ksi = sin^2(PI*x) sin^2(PI*y) /PI
      //Ux = - sin^2(PI*x) sin(2*PI*y)
      //Uy =   sin(2*PI*x) sin^2(PI*y)
      if (Dim >2) 
	{
	  cerr << " The prescribed velocity fiels is inconsistent with this dimention" << endl;
	  cerr << " Please check function get_fluid velocity" << endl;
	  exit(1);
	}
      
      for (int irod=0; irod<Npart;irod++)
	{
	  int i0=irod*Dim;
	  double sqsin_x= sin(M_PI*x[i0])*sin(M_PI*x[i0]);
	  double sqsin_y= sin(M_PI*x[i0+1])*sin(M_PI*x[i0+1]);
	  double sin_2x= sin(2*M_PI*x[i0]);
	  double sin_2y= sin(2*M_PI*x[i0+1]);
	  
	  u[i0] = - sqsin_x * sin_2y;
	  u[i0+1] = sin_2x * sqsin_y;
	  
	  //advection term in derivative
	  Du[i0]   = -M_PI*(v[i0]*sin_2x*sin_2y +
			    2.*v[i0+1]*sqsin_x*cos(2.*M_PI*x[i0+1]));
	  Du[i0+1] =  M_PI*(2.*v[i0]*cos(2.*M_PI*x[i0])*sqsin_y +
			    v[i0+1]*sin_2x*sin_2y);
	}
    }
  if (case_fluid==3)
    {  

      //Leveque 1996 divergence free velocity field
      //U =  2 sin^2(PI*x) sin(2*PI*y) sin(2*PI*z)
      //V =  - sin(2*PI*x) sin^2(PI*y) sin(2*PI*z)
      //W =  - sin(2*PI*x) sin(2*PI*y) sin^2(PI*z)
      double v0=fluidvelzero;
      double v0_fac = fluidwavelength;
      double time_fac=fluid_time_scale;
      double pi_fac=M_PI*fluid_coor_scale;
      if (Dim != 3) 
	{
	  cerr << " The prescribed velocity fiels is inconsistent with this dimention" << endl;
	  cerr << " Please check function get_fluid velocity" << endl;
	  exit(1);
	}
      
      for (int irod=0; irod<Npart;irod++)
	{
	  int i0=irod*Dim;
	  double sqsin_x= sin(pi_fac*x[i0])*sin(pi_fac*x[i0]);
	  double sqsin_y= sin(pi_fac*x[i0+1])*sin(pi_fac*x[i0+1]);
	  double sqsin_z= sin(pi_fac*x[i0+2])*sin(pi_fac*x[i0+2]);
	  
	  double sin_2x= sin(2*pi_fac*x[i0]);
	  double sin_2y= sin(2*pi_fac*x[i0+1]);
	  double sin_2z= sin(2*pi_fac*x[i0+2]);
	  
	  u[i0]   = 2.*sqsin_x * sin_2y * sin_2z *v0_fac;
	  u[i0+1] =  - sin_2x * sqsin_y * sin_2z *v0_fac;
	  u[i0+2] =  - sin_2x * sin_2y * sqsin_z *v0_fac;
	}
      for (int irod=0; irod<Npart;irod++)
	{
	  int i0=irod*Dim;
	  double sqsin_x= sin(pi_fac*x[i0])*sin(pi_fac*x[i0]);
	  double sqsin_y= sin(pi_fac*x[i0+1])*sin(pi_fac*x[i0+1]);
	  double sqsin_z= sin(pi_fac*x[i0+2])*sin(pi_fac*x[i0+2]);
	  
	  double sin_2x= sin(2*pi_fac*x[i0]);
	  double sin_2y= sin(2*pi_fac*x[i0+1]);
	  double sin_2z= sin(2*pi_fac*x[i0+2]);
	  
	  //advection term in derivative
	  Du[i0]   = 4.*pi_fac*( v[i0]   * sin_2x  * sin_2y               * sin_2z * 0.5+
				 v[i0+1] * sqsin_x * cos(2.*pi_fac*x[i0+1]) * sin_2z +
				 v[i0+2] * sqsin_x * sin_2y               * cos(2.*pi_fac*x[i0+2]));
	  
	  Du[i0+1] =-2.*pi_fac*( v[i0]   * cos(2.*pi_fac*x[i0]) * sqsin_y * sin_2z +
				 v[i0+1] * sin_2x             * sin_2y  * sin_2z * 0.5 +
				 v[i0+2] * sin_2x             * sqsin_y * cos(2.*pi_fac*x[i0+2]));
	  
	  Du[i0+2] =-2.*pi_fac*( v[i0]   * cos(2.*pi_fac*x[i0]) * sin_2y               * sqsin_z +
				 v[i0+1] * sin_2x             * cos(2.*pi_fac*x[i0+1]) * sqsin_z +
				 v[i0+2] * sin_2x             * sin_2y               * sin_2z * 0.5);
	}
      
      for (int i=0; i<Npart*Dim;i++) Du[i] *= cos(time_fac*time)*v0_fac;
      
      for (int i=0; i<Npart*Dim;i++) Du[i] -= u[i]*sin(time_fac*time)*time_fac;
      
      for (int i=0; i<Npart*Dim;i++) u[i] *= cos(time_fac*time);
      for (int i=0; i<Npart*Dim;i++) u[i] += v0;
    } 
  if (case_fluid>3)
    {
      cerr << "Inconsistent fluid field"<<endl;
      exit(1);  
    }
  
}

void init_xv( double *x, double *v, int N_part )
{
  extern int Dim;
  
  extern int init_conf;
  extern bool set2fluidvel;

  for( int i=0; i<Dim*N_part; i++ ) x[i] = 0.;

  if (init_conf==0)
    {
      for( int ipart=1; ipart<N_part; ipart++ ) {
	for( int idir=0; idir<Dim; idir++ ) x[Dim*ipart+idir] = 0;
	x[Dim*ipart+0] = x[Dim*(ipart-1)+0] + a_constraint;
      }
    }
  
  if (init_conf==1)
    {
    
      for( int ipart=1; ipart<N_part; ipart++ ) {
	for( int idir=0; idir<Dim; idir++ ) {
	  x[Dim*ipart+idir] = x[Dim*(ipart-1)+idir] + a_constraint/sqrt((double)Dim);
	}
      }
    }

  if (init_conf==2)
    {
      //bended line

      double sqnumber = sqrt((double)(N_part-1));
      double node_position_x = 0.1;
      double node_position_y = 0.3;
      double node_position_z = 0.3;
      
      for( int ipart=0; ipart<N_part; ipart++ ) {
	x[Dim*ipart] = node_position_x; 
	x[Dim*ipart+1] = node_position_y;
	if (Dim>2)	x[Dim*ipart+2] = node_position_z;
 
	node_position_x += a_constraint*sqrt((double)ipart)/sqnumber;
	node_position_y += a_constraint*sqrt((double)(N_part-ipart-1))/sqnumber;
      }
    }
    if (init_conf==3) 
      { 
	//random coil
      
	x[0] = 0.1;
	if (Dim>1) x[1] = 0.3;
	if (Dim>2) x[2] = 0.3;
	//seed
	srand(2);

	for( int ipart=0; ipart<N_part; ipart++ ) {
	  int i0=(ipart-1)*Dim;
	  int i1=ipart*Dim;

	  double rand_angle1 = ((double)rand()/(double)RAND_MAX)*2.*3.1415;
	  double rand_angle2 = 0.;
	  if (Dim>2) rand_angle2 = ((double)rand()/(double)RAND_MAX)*2.*3.1415;
      
	  x[i1] = x[i0] + a_constraint*cos(rand_angle1)*cos(rand_angle2);
	  if (Dim>1) x[i1+1] = x[i0+1] + a_constraint*sin(rand_angle1)*cos(rand_angle2);
	  if (Dim>2) x[i1+2] = x[i0+2] + a_constraint*sin(rand_angle2);
	}
      }
    
  if (set2fluidvel)
    {
      // set particle velocity to fluid velocity
      double foo[Dim*N_part];
      get_fluid_velocity(N_part, x, foo, v, foo, 0.);
    }
  else
    {
      //set initial vel to zero
     for( int i0=0; i0<N_part*Dim; i0++ ) v[i0]=0.;
    }

  // project -- should leave x unchanged but modify v
  
  if( Projection == ON ) {
    (*method)( 2, N_part, x, v, x, 1.0 );  }
}

void step( int Npart,
	   double *x,
	   double *v,
	   double *W,
	   double *Z, double *Y, double *I, double *J, double *aux,
           double dt,
	   double time )
{
  extern double sigma, gam;
  extern OnOff Stochastic;
  extern OnOff Projection;
  extern int Dim;
  int Nrod = Npart-1;

  extern int zero;

  double *v0, *u0, *Du, *x0;
  double *g, **Ai, **LAi,*temp1, *temp2, *temp3;
  double *ef, *eg, *eLf, *eLg, *eGf, *eGLg, *eGGf;
  double *c1, *c2, *c3, *c4;
  double *v_dot_v, *v_dot_u, *x_dot_u, *x_dot_Du, *x_dot_v;

    v0 = new double[Dim*Npart];
    x0 = new double[Dim*Npart];
    u0 = new double[Dim*Npart];
    Du = new double[Dim*Npart];

    g = new double[Npart*Npart*Dim*Dim];
    temp1 = new double[Nrod];
    temp2 = new double[Nrod];
    temp3 = new double[Nrod];


    ef = new double[Dim*Npart];
    eg = new double[Dim*Npart];
    eLf = new double[Dim*Npart];
    eLg = new double[Dim*Npart];
    eGf = new double[Dim*Npart];
    eGLg = new double[Dim*Npart];
    eGGf = new double[Dim*Npart];

    c1 = new double[Nrod];//deterministic term from  derivative of z
    c2 = new double[Nrod];//Delta v and LAi 
    c3 = new double[Nrod];//Delta v _dot_ [Deltas x]
    c4 = new double[Nrod];//Delta x _dot_ [Deltas x]

    v_dot_v  = new double[Nrod];
    v_dot_u  = new double[Nrod];
    x_dot_u  = new double[Nrod];
    x_dot_Du = new double[Nrod];
    x_dot_v  = new double[Nrod];


    Ai = new double*[Nrod];
    LAi = new double*[Nrod];
    
    for( int i=0; i<Nrod; i++ ) {
      Ai[i] = new double[Nrod];
      LAi[i] = new double[Nrod];
    }


    
    for (int i=0;i<Nrod;i++)
      {
	for( int ii=0; ii<Nrod; ii++ ) {
	  Ai[i][ii] = 0;
	}
      }
  

    //    int nCoord = Dim*Npart;
    double gt=gam*dt;
    bool small=false;
    //    if (gt < 1e-4) small=true;
    double e = exp(-gt);
    double em1 = expm1(-gt);
    double em2 = expm1(-2*gt);

    for( int i=0; i<Dim*Npart; i++ ) {
      v0[i] = v[i];
      x0[i] = x[i];
      x[i] = 0.;
      v[i] = 0.;
    }



    // make A inverse for benefit of rho
    AInverse( Npart, x0, Ai );


    get_fluid_velocity( Npart, x0, v0, u0, Du, time);

    //store the scalar product of components
    for( int jrod=0; jrod<Nrod; jrod++ ) {
     
      v_dot_v[jrod]  = 0.;
      v_dot_u[jrod]  = 0.;
      x_dot_u[jrod]  = 0.;
      x_dot_Du[jrod] = 0.;
      x_dot_v[jrod]  = 0.;
     
     for( int jdir=0; jdir<Dim; jdir++ ) {
       int j0 = Dim*jrod+jdir;
       int j1 = j0 + Dim;
       double dx  = x0[j1] - x0[j0];
       double dv  = v0[j1] - v0[j0];
       double du  = u0[j1] - u0[j0];
       double dDu = Du[j1] - Du[j0];
       
       v_dot_v[jrod]   += dv*dv;
       v_dot_u[jrod]   += dv*du;
       x_dot_u[jrod]   += dx*du;
       x_dot_Du[jrod]  += dx*dDu;
       if (zero) 
       x_dot_v[jrod]   += dx*dv;
     }
   }
   
//g function evaluation*****************************

     for( int i0=0; i0<Npart*Npart*Dim*Dim; i0++ ) g[i0]=0.;

    for( int irod=0; irod<Nrod; irod++ ) {
      for( int idir=0; idir<Dim; idir++ ) {
      int i1= irod*Dim+idir;
      int i2= i1+Dim;

      for( int qrod=0; qrod<Nrod; qrod++ ) {
	for( int mdir=0; mdir<Dim; mdir++ ) {

	  int q1= qrod*Dim+mdir;
	  int q2= q1+Dim;
	  
	  double temp = sigma*(x0[i2]-x0[i1])*Ai[irod][qrod]*(x0[q2]-x0[q1]);
	
	  g[(irod*Dim+idir)*Npart*Dim+qrod*Dim+mdir]         += temp;
	  g[(irod*Dim+idir)*Npart*Dim+(qrod+1)*Dim+mdir]     -= temp;
	  g[((irod+1)*Dim+idir)*Npart*Dim+qrod*Dim+mdir]     -= temp;
	  g[((irod+1)*Dim+idir)*Npart*Dim+(qrod+1)*Dim+mdir] += temp;
	}
      }  
      g[(irod*Dim+idir)*Npart*Dim+irod*Dim+idir] += sigma;
      }
    }
    for( int idir=0; idir<Dim; idir++ ) g[(Nrod*Dim+idir)*Npart*Dim+Nrod*Dim+idir] += sigma;//last term adjustion.

    //LAi evaluation************************************
    double *dot_minus, *dot_center, *dot_plus;
    dot_minus = new double[Nrod];
    dot_center = new double[Nrod];
    dot_plus = new double[Nrod];

    for( int prod=0; prod<Nrod; prod++ ) {
      dot_minus[prod] =0.;
      dot_center[prod] =0.;
      dot_plus[prod] =0.;
      
      for( int pdir=0; pdir<Dim; pdir++ ) {
	int p1= prod*Dim+pdir;
	int p2= p1+Dim;
	int p3= p2+Dim;
	int p0= p1-Dim;
	dot_center[prod] += (x0[p2]-x0[p1])*(v0[p2]-v0[p1]);
	if (prod>0)  dot_minus[prod]  += (x0[p1]-x0[p0])*(v0[p2]-v0[p1])+(x0[p2]-x0[p1])*(v0[p1]-v0[p0]);
	if (prod<Nrod-1) dot_plus[prod]  +=(x0[p3]-x0[p2])*(v0[p2]-v0[p1])+(x0[p2]-x0[p1])*(v0[p3]-v0[p2]);
      }
    }	  
    for( int irod=0; irod<Nrod; irod++ ) {
      for( int jrod=0; jrod<Nrod; jrod++ ) {
	LAi[irod][jrod] =0.;
	for( int prod=0; prod<Nrod; prod++ ) {
	  if (zero)  LAi[irod][jrod] += 4*Ai[irod][prod]*dot_center[prod]*Ai[prod][jrod];
	  if (prod>0) LAi[irod][jrod] -= Ai[irod][prod]*dot_minus[prod]*Ai[prod-1][jrod];
	  if (prod <Nrod-1) LAi[irod][jrod] -= Ai[irod][prod]*dot_plus[prod]*Ai[prod+1][jrod];
	}
      }
    }
    delete[] dot_minus;
    delete[] dot_center;
    delete[] dot_plus;

    

//iteration begins************************************************


//coordinate x component*****************************************************************************************************************
  if(Stochastic==ON) {  
    for( int ipart=0; ipart<Npart; ipart++ ) {
      for( int idir=0; idir<Dim; idir++ ) {
        int i1 = Dim*ipart + idir;
	x[i1] = 0.;
	
    //Gf term
	double temp=0.;
	for( int alpha=0; alpha<Npart; alpha++ ) {
	  for( int ldir=0; ldir<Dim; ldir++ ) {
	    int l1=alpha*Dim+ldir;
	    temp +=g[(ipart*Dim+idir)*Npart*Dim+alpha*Dim+ldir]*(W[l1]-Z[l1])/gam;
	  }
	}
	
	x[i1] +=temp;
      }
    }
  }
    for( int irod=0; irod<Nrod; irod++ ) {

     //Lf term
      double temp = 0.;
      for( int jrod=0; jrod<Nrod; jrod++ ) {
	double fac;
	if (!small) fac=(v_dot_v[jrod]*em1*em1/(2.*gam*gam)
			 +x_dot_u[jrod]*(dt+em1/gam) 
			 +x_dot_v[jrod]*( em1/gam + dt*e));
	else fac = (v_dot_v[jrod]*dt*dt*0.5*(1.-gam*dt)
		    +x_dot_u[jrod]*dt*dt*0.5*(1.-gam*dt/3.) 
		    +x_dot_v[jrod]*dt*dt*gam*(-0.5+gam*dt/3.));
	temp += Ai[irod][jrod]*fac;
      }

      for( int idir=0; idir<Dim; idir++ ) {
        int i1 = Dim*irod + idir;
        int i2 = i1+Dim;
	
	x[i1] -=temp*(x0[i2]-x0[i1]);
	x[i2] +=temp*(x0[i2]-x0[i1]);
      }
    }
    

      //first order terms
    for( int i1=0; i1<Npart*Dim; i1++ ) 
      {
	if (!small) x[i1] += em1*(u0[i1]-v0[i1])/gam + dt*u0[i1] + x0[i1];
	else  x[i1] +=(u0[i1]-v0[i1])*dt*(-1.+gt*( 0.5 - gt*(1./6.)))
		      + dt*u0[i1] + x0[i1];
      }	
    
   
    // final correction (iterative) to put on contraint manifold.
    if( Projection == ON )
      (*method)( 0, Npart, x, (double*)NULL, x0, dt );
    
    
// now velocity**********************************************************************************************************************************

//initialization     
   for( int ipart=0; ipart<Npart*Dim; ipart++ ) {
     ef[ipart]   = 0.;
     eg[ipart]   = 0.;
     eLf[ipart]  = 0.;
     eLg[ipart]  = 0.;
     eGf[ipart]  = 0.;
     eGLg[ipart] = 0.;
     eGGf[ipart] = 0.;
   }



//e*f^0 evaluation *********************************
   for( int irod=0; irod<Nrod; irod++ ) {
     double temp=0.;
     for( int jrod=0; jrod<Nrod; jrod++ ) {
       double fac;
       if (!small) fac=((-e*em1/gam*v_dot_v[jrod] - e*gam*dt*x_dot_v[jrod]));
       else fac = e*dt*(-1.+ gam*dt*(0.5-gam*dt/6.))*v_dot_v[jrod] 
	          - e*gam*dt*x_dot_v[jrod];
       temp += Ai[irod][jrod]*fac;
     }

     for( int idir=0; idir<Dim; idir++ ) {
       int i1=irod*Dim+idir;
       int i2=i1+Dim;
       ef[i1] -= temp*(x0[i2]-x0[i1]);
       ef[i2] += temp*(x0[i2]-x0[i1]);
     }
   }
    
   for( int i1=0; i1<Npart*Dim; i1++ ) ef[i1] -= em1*u0[i1];



//e*(Lf)^0 evaluation *******************************

 //evaluation c1,c2,c3,c4
   for( int irod=0; irod<Nrod; irod++ ) {
     c1[irod]=0.;
     if (Stochastic==ON) {
       for( int qpart=0; qpart<Npart; qpart++ ) {
	 for( int mdir=0; mdir<Dim; mdir++ ) {
	   for( int kdir=0; kdir<Dim; kdir++ ) {
	     c1[irod] +=(g[((irod+1)*Dim+kdir)*Npart*Dim+qpart*Dim+mdir]-g[(irod*Dim+kdir)*Npart*Dim+qpart*Dim+mdir])
	       *(g[((irod+1)*Dim+kdir)*Npart*Dim+qpart*Dim+mdir]-g[(irod*Dim+kdir)*Npart*Dim+qpart*Dim+mdir]);
	   }
	 }
       }
       if (!small) c1[irod] *=em1*em1/gam/gam/2.;
       else c1[irod] *=dt*dt*(1.-gam*dt)*0.5;    
     }
     if (!small) c2[irod] = e*em1*em1/2./gam/gam*v_dot_v[irod] - (em1/gam+e*dt)*x_dot_u[irod] - e*(em1/gam+dt)*x_dot_v[irod];
     else c2[irod] = e*0.5*dt*dt*(1.- gam*dt)*v_dot_v[irod] - dt*dt*gam*(-0.5+gam*dt/3.)*x_dot_u[irod] 
	             - e*dt*dt*gam*(0.5-gam*dt/6.)*x_dot_v[irod];

     if (!small) c3[irod] = e*em1*em1/2./gam/gam*v_dot_v[irod] + e*(em1/gam+dt)*x_dot_u[irod] + e*(em1/gam+e*dt)*x_dot_v[irod];
     else c3[irod] = e*0.5*dt*dt*(1.- gam*dt)*v_dot_v[irod] + e*dt*dt*gam*(0.5-gam*dt/6.)*x_dot_u[irod] 
	    + dt*dt*gam*(-0.5+5.*gam*dt/6.)*x_dot_v[irod];
     
     if (!small) c4[irod] = e*(em1/gam+dt)*v_dot_v[irod] - (em1+e*gam*dt)*x_dot_u[irod] - e*gam*gam*dt*dt/2.*x_dot_v[irod]; 
     else c4[irod] = e*dt*dt*gam*(0.5-gam*dt/6.)*v_dot_v[irod] - dt*dt*gam*gam*(-0.5+gam*dt/3.)*x_dot_u[irod] 
	    - e*gam*gam*dt*dt/2.*x_dot_v[irod]; 
   }
    for( int jrod=0; jrod<Nrod; jrod++ ) {
      for( int p=-1; p<=1; p++ ) {
	int cp = ( p ? -1 : 2 );
	int jp = jrod + p;
	if( (jp >= 0) && (jp < Nrod) ) 
	  {
	    double dot_v=0.;
	    double dot_x=0.;
	    for( int idir=0; idir<Dim; idir++ ) {
	    int j0 = Dim*jrod + idir;
	    int j1 = j0 + Dim;
	    int jp0 = Dim*jp + idir;
	    int jp1 = jp0+Dim;
	    dot_v += (v0[j1]-v0[j0])*(x0[jp1]-x0[jp0]);
	    dot_x += (x0[j1]-x0[j0])*(x0[jp1]-x0[jp0]);
	    }
	    if ((jp==1)&&(zero==0)) dot_v=0.;
	    for( int mrod=0; mrod<Nrod; mrod++ ) 
	      c1[jrod] += Ai[jp][mrod]*cp*(2.*dot_v*c3[mrod]-dot_x*c4[mrod]);
	  }
      }
      if (!small) c1[jrod] += (em1/gam+dt)*x_dot_Du[jrod] 
		    + (2.*em2/gam-3.*em1/gam +e*dt)*v_dot_u[jrod] 
		    - e*(em1/gam+dt)*v_dot_v[jrod] 
		    + e*gam*dt*x_dot_u[jrod];
      else c1[jrod] += gam*dt*dt*(0.5-gam*dt/6.)*x_dot_Du[jrod] 
		    + gam*dt*dt*(3./2.-5./3.*gam*dt)*v_dot_u[jrod] 
	            - e*gam*dt*dt*(0.5-gam*dt/6.)*v_dot_v[jrod] 
		    + e*gam*dt*x_dot_u[jrod];
    }
    //evaluation (eLf)^0
    for( int irod=0; irod<Nrod; irod++ ) {
      double temp1=0.;
      double temp2=0.;
      for( int jrod=0; jrod<Nrod; jrod++ ) {
	temp1 +=Ai[irod][jrod]*c2[jrod];
	temp2 +=Ai[irod][jrod]*c1[jrod]  + LAi[irod][jrod]*c2[jrod];
      }

      for( int idir=0; idir<Dim; idir++ ) {
	int i1=irod*Dim+idir;
	int i2=i1+Dim;
	eLf[i1] -= temp1*(v0[i2]-v0[i1]) + temp2*(x0[i2]-x0[i1]);
	eLf[i2] += temp1*(v0[i2]-v0[i1]) + temp2*(x0[i2]-x0[i1]);
      }
    }

    for( int i1=0; i1<Npart*Dim; i1++ ) 
      if(!small) eLf[i1] += (em1/gam+dt)*Du[i1];
      else eLf[i1] += gam*dt*dt*(0.5-gam*dt/6.)*Du[i1];

    if (Stochastic==ON) {


      //e*g^0 evaluation *********************************
    for( int ipart=0; ipart<Npart; ipart++ ) {
      for( int idir=0; idir<Dim; idir++ ) {
	int i1=ipart*Dim+idir;
	double temp = 0.;
	for( int alpha=0; alpha<Npart; alpha++ ) {
	  for( int ldir=0; ldir<Dim; ldir++ ) {
	    int l1=alpha*Dim+ldir;
	    temp +=g[(ipart*Dim+idir)*Npart*Dim+alpha*Dim+ldir]*Z[l1];
	  }
	}
	eg[i1]=temp;
      }
    }
    
//(Lg)^0 evaluation *********************************
    for( int i1=0; i1<Nrod; i1++ ) 
      {
	temp1[i1]=0.;
	temp2[i1]=0.;
	temp3[i1]=0.;
      }
    for( int alpha=0; alpha<Nrod; alpha++ ) 
      {
      for( int mdir=0; mdir<Dim; mdir++ ) 
	{
	int m1=alpha*Dim + mdir;
	int m2=m1+Dim;
	temp1[alpha] += (x0[m2]-x0[m1])*((Z[m2]-Z[m1])-e*(W[m2]-W[m1]));
	temp2[alpha] += (v0[m2]-v0[m1])*((Z[m2]-Z[m1])-e*(W[m2]-W[m1]));
	temp3[alpha] += (x0[m2]-x0[m1])*((Z[m2]-Z[m1])-e*(W[m2]-W[m1]));
	}
      }

    for( int irod=0; irod<Nrod; irod++ ) 
      {
      for( int idir=0; idir<Dim; idir++ ) 
	{
	  int i1=Dim*irod+idir;
	  int i2=i1+Dim;
	  double sum1=0.;
	  double sum2=0.;
	  double sum3=0.;
	  for( int alpha=0; alpha<Nrod; alpha++ ) 
	    {
	      sum1 += Ai[irod][alpha]*temp1[alpha];
	      sum2 += Ai[irod][alpha]*temp2[alpha];
	      sum3 += LAi[irod][alpha]*temp3[alpha];
	    }
	  double temp_sum = sigma/gam*((v0[i2]-v0[i1])*sum1 
				       +(x0[i2]-x0[i1])*sum2 
				       +(x0[i2]-x0[i1])*sum3);

	  eLg[i1] -= temp_sum; 
	  eLg[i2] += temp_sum;
	}
      }

//e*(Gf)^0 evaluation *********************************
    for( int irod=0; irod<Nrod; irod++ ) {
      for( int jrod=0; jrod<Nrod; jrod++ ) {
	double temp1 = 0.;
	double temp2 = 0.;
	for( int alpha=0; alpha<Npart; alpha++ ) {
	  for( int mdir=0; mdir<Dim; mdir++ ) {
	    int m1=alpha*Dim + mdir;
	    double dot_v=0.;  
	    double dot_x=0.;  
	    for( int jdir=0; jdir<Dim; jdir++ ) {
	      int j1=jrod*Dim + jdir;
	      int j2=j1+Dim;
	      dot_v +=(v0[j2]-v0[j1])*(g[((jrod+1)*Dim+jdir)*Npart*Dim+alpha*Dim+mdir] - g[(jrod*Dim+jdir)*Npart*Dim+alpha*Dim+mdir]);
	      dot_x +=(x0[j2]-x0[j1])*(g[((jrod+1)*Dim+jdir)*Npart*Dim+alpha*Dim+mdir] - g[(jrod*Dim+jdir)*Npart*Dim+alpha*Dim+mdir]);
	    }
	    
	    temp1 +=dot_v*(W[m1]-Z[m1]);
	    temp2 +=dot_x*(dt*Z[m1]-Y[m1]);
	  }
	}
	double eGFterm=Ai[irod][jrod]*(2./gam*e*temp1-gam*temp2);
	for( int idir=0; idir<Dim; idir++ ) {
	  int i1=Dim*irod+idir;
	  int i2=i1+Dim;
	  eGf[i1] -=eGFterm*(x0[i2]-x0[i1]);
	  eGf[i2] +=eGFterm*(x0[i2]-x0[i1]);
	}
      }
    }

    //#define _CUTTED_
#ifndef _CUTTED_       
    
 //e*GLg term evaluation***************************************
    for( int i1=0; i1<Nrod; i1++ ) 
      {
	temp1[i1]=0.;
	temp2[i1]=0.;
      }
    //delta x term
    for( int alpha=0; alpha<Nrod; alpha++ ) 
      {
	for( int mdir=0; mdir<Dim; mdir++ ) 
	  {
	    for( int beta=0; beta<Npart; beta++ ) 
	      {
		for( int ldir=0; ldir<Dim; ldir++ ) 
		  {
		    int ab1 = (alpha*Dim + mdir)*Npart*Dim + beta*Dim +ldir;
		    int ab2 = ((alpha+1)*Dim + mdir)*Npart*Dim + beta*Dim +ldir;
		    
		    temp1[alpha] += (g[((alpha+1)*Dim+mdir)*Npart*Dim+beta*Dim+ldir] 
				    - g[(alpha*Dim+mdir)*Npart*Dim+beta*Dim+ldir])
		                    *((J[ab2]-I[ab2])-(J[ab1]-I[ab1]));
		  }
	      }
	  }
      }

    //GLAi term
    for( int prod=0; prod<Nrod; prod++ ) {

	for( int beta=0; beta<Npart; beta++ ) 
	  {
	    for( int ldir=0; ldir<Dim; ldir++ ) 
	      {
		double dot_minus =0.;
		double dot_center =0.;
		double dot_plus =0.;

		for( int pdir=0; pdir<Dim; pdir++ ) {
		  int p1= prod*Dim+pdir;
		  int p2= p1+Dim;
		  int p3= p2+Dim;
		  int p0= p1-Dim;
		  
		  dot_center += (x0[p2]-x0[p1])*(g[((prod+1)*Dim+pdir)*Npart*Dim+beta*Dim+ldir]
					       -g[(prod*Dim+pdir)*Npart*Dim+beta*Dim+ldir]);
		  
		  if (prod>0) dot_minus += (x0[p1]-x0[p0])*(g[((prod+1)*Dim+pdir)*Npart*Dim+beta*Dim+ldir]
							    -g[(prod*Dim+pdir)*Npart*Dim+beta*Dim+ldir])
			+(x0[p2]-x0[p1])*(g[(prod*Dim+pdir)*Npart*Dim+beta*Dim+ldir]
					  -g[((prod-1)*Dim+pdir)*Npart*Dim+beta*Dim+ldir]);
		  
		  if (prod<Nrod-1) dot_plus +=(x0[p3]-x0[p2])*(g[((prod+1)*Dim+pdir)*Npart*Dim+beta*Dim+ldir]
								     -g[(prod*Dim+pdir)*Npart*Dim+beta*Dim+ldir])
				     +(x0[p2]-x0[p1])*(g[((prod+2)*Dim+pdir)*Npart*Dim+beta*Dim+ldir]
						       -g[((prod+1)*Dim+pdir)*Npart*Dim+beta*Dim+ldir]);
		}	  
	      

		for( int alpha=0; alpha<Nrod; alpha++ ) 
		  {
		    for( int mdir=0; mdir<Dim; mdir++ ) 
		      {
			int m1=alpha*Dim + mdir;
			int m2=m1+Dim;
		
			int ab1 = (alpha*Dim + mdir)*Npart*Dim + beta*Dim +ldir;
			int ab2 = ((alpha+1)*Dim + mdir)*Npart*Dim + beta*Dim +ldir;

			temp2[prod] += 4*dot_center*Ai[prod][alpha]*(x0[m2]-x0[m1])
			  *((J[ab2]-I[ab2])-(J[ab1]-I[ab1]));
			if (prod>0) temp2[prod] -= dot_minus*Ai[prod-1][alpha]*(x0[m2]-x0[m1])
				      *((J[ab2]-I[ab2])-(J[ab1]-I[ab1]));;
			if (prod <Nrod-1) temp2[prod] -= dot_plus*Ai[prod+1][alpha]*(x0[m2]-x0[m1])
					    *((J[ab2]-I[ab2])-(J[ab1]-I[ab1]));;
		      }
		  }
	      }
	  }
    }

    //delta g term and all together        
    for( int irod=0; irod<Nrod; irod++ ) {
      for( int idir=0; idir<Dim; idir++ ) {
	int i1=Dim*irod+idir;
	int i2=i1+Dim;
	double sum1=0.;
	double sum2=0.;
 
	for( int alpha=0; alpha<Nrod; alpha++ ) {
	  for( int mdir=0; mdir<Dim; mdir++ ) {
	    int m1=alpha*Dim + mdir;
	    int m2=m1+Dim;
	    for( int beta=0; beta<Npart; beta++ ) {
	      for( int ldir=0; ldir<Dim; ldir++ ) {
		int ab1 = (alpha*Dim + mdir)*Npart*Dim + beta*Dim +ldir;
		int ab2 = ((alpha+1)*Dim + mdir)*Npart*Dim + beta*Dim +ldir;
		
		sum1 += (Ai[irod][alpha]*
			 (g[((irod+1)*Dim+idir)*Npart*Dim+beta*Dim+ldir]
			  -g[(irod*Dim+idir)*Npart*Dim+beta*Dim+ldir])
			 *(x0[m2]-x0[m1]))
		  *((J[ab2]-I[ab2])-(J[ab1]-I[ab1]));
		
	      }
	    }
	  }
	  sum2 +=Ai[irod][alpha]*(temp1[alpha]+temp2[alpha]);
	}
	double temp_sum = sigma/gam*(sum1+(x0[i2]-x0[i1])*sum2);
	eGLg[i1]  -= temp_sum;
	eGLg[i2]  += temp_sum;
      }
    }

//e*GGf term evaluation************************************
for( int jrod=0; jrod<Nrod; jrod++ ) temp1[jrod] = 0.;
     
 for( int jrod=0; jrod<Nrod; jrod++ ) {
    for( int alpha=0; alpha<Npart; alpha++ ) {
      for( int mdir=0; mdir<Dim; mdir++ ) {
        for( int beta=0; beta<Npart; beta++ ) {
          for( int ldir=0; ldir<Dim; ldir++ ) {
	    int ab=(alpha*Dim + mdir)*Npart*Dim + beta*Dim +ldir;
	    double dot_g=0.;  
	    for( int jdir=0; jdir<Dim; jdir++ ) {
	      dot_g +=(    g[((jrod+1)*Dim+jdir)*Npart*Dim+alpha*Dim+mdir] -
                           g[(jrod*Dim+jdir)*Npart*Dim+alpha*Dim+mdir]
                      )*(  g[((jrod+1)*Dim+jdir)*Npart*Dim+beta*Dim+ldir] -
                           g[(jrod*Dim+jdir)*Npart*Dim+beta*Dim+ldir]
                      );
	    }
	    temp1[jrod] +=dot_g*(I[ab]-aux[ab]);
	  }
	}
      }
    }
  }
  for( int irod=0; irod<Nrod; irod++ ) {
    double eGGfterm=0.;
    for( int jrod=0; jrod<Nrod; jrod++ ) {
      eGGfterm +=Ai[irod][jrod]*temp1[jrod];
    }
    eGGfterm *=2./gam;
    for( int idir=0; idir<Dim; idir++ ) {
      int i1=Dim*irod+idir;
      int i2=i1+Dim;
      eGGf[i1] -=eGGfterm*(x0[i2]-x0[i1]);
      eGGf[i2] +=eGGfterm*(x0[i2]-x0[i1]);
    }
  }
#endif    

  }
        
// FINAL EVALUATION FOR V(t+h)

   for( int i1=0; i1<Npart*Dim; i1++ ) {
     v[i1] = eGGf[i1]+eGLg[i1]+eGf[i1]+eLg[i1]+eLf[i1]+eg[i1]+ef[i1]+e*v0[i1];// final formula for v
   }
    



    // and correct again noniteratively to put on constraint manifold
    if( Projection == ON )
      (*method)( 1, Npart, x, v, x0, dt );
    /*
  cout.precision(10);
  cout <<"================================================"<<endl;
  for( int i=0; i<Dim*Npart; i++ ) {
    cout << "teststep " << i/Dim << "-"<< i%Dim
	 << "   x=" << x[i]  << "  v=" << v[i] << endl;
  }
    */

    // done
    delete[] temp3;
    delete[] temp2;
    delete[] temp1;
    delete[] v0;
    delete[] x0;
    delete[] u0;
    delete[] Du;
    delete[] ef;
    delete[] eg;
    delete[] eLf;
    delete[] eLg;
    delete[] eGf;
    delete[] eGLg;
    delete[] eGGf;
    delete[] c1;
    delete[] c2;
    delete[] c3;
    delete[] c4;
    delete[] v_dot_v;
    delete[] v_dot_u;
    delete[] x_dot_u;
    delete[] x_dot_Du;
    delete[] x_dot_v;

    for( int i=0; i<Nrod; i++ ) {
      delete [] Ai[i];
      delete [] LAi[i];
    }
    delete[] Ai;
    delete[] LAi;
    delete[] g;

    return;
}

int main( int argc, char** argv)
{
  extern int Dim, seed, precision;
  extern double gam;
  extern bool verbose,stattest;
  extern int num_batches;

  int rank, numprocs;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int nodes, coarsesteps, paths, resolutions;
  double endtime;


  set_parameters( argc-1, ++argv,
    nodes, coarsesteps, endtime, paths, resolutions );


  int finesteps = coarsesteps;
  for( int i=0; i<resolutions-1+resadd; i++ ) finesteps *= 2;
  if (rank==0)  cout << "fine steps " << finesteps << endl;

  double *v = new double[nodes*Dim];
  double *x = new double[nodes*Dim];
  double *W = new double[nodes * finesteps * Dim];
  double *Z = new double[nodes * finesteps * Dim];
  double *Y = new double[nodes * finesteps * Dim];
  double *I = new double[ nodes * nodes * finesteps * Dim * Dim];
  double *J = new double[ nodes * nodes * finesteps * Dim * Dim];
  double *aux = new double[ nodes * nodes * finesteps * Dim * Dim];

  double h_fine = endtime/finesteps;
  if (rank==0)
    cout << "h fine " << h_fine << endl;

  expectations ex;
  modelCoefficients mod;
  makeModel(ex,mod, h_fine, gam, verbose, precision);

  int N_part =nodes;
  double hfine=h_fine;
  int N_finestep=finesteps;
  int nres=resolutions;

  double **x_output_weak, **v_output_weak;
  double **x_output, **x_er0, **x_er1, **x_er2, **x_er_weak;
  double **v_output, **v_er0, **v_er1, **v_er2, **v_er_weak;

  double **x_stat, **v_stat, **varx, **varv;
  double **xerr_stat_w, **verr_stat_w;
  double **xerr_stat_s, **verr_stat_s;

  extern int Dim, seed;

  if (stattest)
    {
      
      x_stat = new double*[nres];
      v_stat = new double*[nres];
      for( int i=0; i<nres; i++ ) 
	{
	  x_stat[i] = new double[(paths/numprocs+1)*Dim*nodes];
	  v_stat[i] = new double[(paths/numprocs+1)*Dim*nodes];
	}
      
      if (rank==0)
	{
	  varx = new double*[num_batches];
	  varv = new double*[num_batches];
	  xerr_stat_w = new double*[num_batches];
	  verr_stat_w = new double*[num_batches];
	  xerr_stat_s = new double*[num_batches];
	  verr_stat_s = new double*[num_batches];

	  for( int i=0; i<num_batches; i++ ) 
	    {
	      varx[i] = new double[nres];
	      varv[i] = new double[nres];
	      xerr_stat_w[i] = new double[nres-1];
	      verr_stat_w[i] = new double[nres-1];
	      xerr_stat_s[i] = new double[nres-1];
	      verr_stat_s[i] = new double[nres-1];
	      for( int j=0; j<nres; j++ ) 
		{
		  varx[i][j] = 0.;
		  varv[i][j] = 0.;
		  if (j < nres-1)
		    {
		      xerr_stat_w[i][j] = 0;
		      verr_stat_w[i][j] = 0;
		      xerr_stat_s[i][j] = 0;
		      verr_stat_s[i][j] = 0;
		    }
		}
	    }
	}
    }

  x_output = new double*[nres];
  x_output_weak = new double*[nres];
  v_output_weak = new double*[nres];
  x_er0 = new double*[nres-1];
  x_er1 = new double*[nres-1];
  x_er2 = new double*[nres-1];
  x_er_weak = new double*[nres-1];
  v_er_weak = new double*[nres-1];
  v_output = new double*[nres];
  v_er0 = new double*[nres-1];
  v_er1 = new double*[nres-1];
  v_er2 = new double*[nres-1];
  for( int i=0; i<nres; i++ ) {
    x_output[i] = new double[Dim*N_part];
    v_output[i] = new double[Dim*N_part];
    x_output_weak[i] = new double[Dim*N_part];
    v_output_weak[i] = new double[Dim*N_part];
  }
  for( int i=0; i<nres-1; i++ ) {
    x_er_weak[i] = new double[Dim*N_part];
    v_er_weak[i] = new double[Dim*N_part];
    x_er0[i] = new double[Dim*N_part];
    x_er1[i] = new double[Dim*N_part];
    x_er2[i] = new double[Dim*N_part];
    v_er0[i] = new double[Dim*N_part];
    v_er1[i] = new double[Dim*N_part];
    v_er2[i] = new double[Dim*N_part];
  }
  
  //start for strong and weak
  std::fstream wox("duh-weak-output-x", std::ios_base::out );
  std::fstream wov("duh-weak-output-v", std::ios_base::out );
  std::fstream sox("duh-strong-output-x", std::ios_base::out );
  std::fstream sov("duh-strong-output-v", std::ios_base::out );
  std::fstream vov("duh-solution-variance", std::ios_base::out );

  for (int btch=0;btch < num_batches; btch++)
    {
  int count_path=0;
  int dof  = Dim*N_part;
  for( int i=0; i<nres; i++ ) 
    {
      for( int j=0; j<Dim*N_part; j++ ) 
	{
	  x_output[i][j] = 0.;
	  v_output[i][j] = 0.;
	  x_output_weak[i][j] = 0.;
	  v_output_weak[i][j] = 0.;
	
	  if (i < nres-1 )
	    {
	      x_er_weak[i][j] = 0.;
	      v_er_weak[i][j] = 0.;
	      x_er0[i][j] = 0.;
	      x_er1[i][j] = 0.;
	      x_er2[i][j] = 0.;
	      v_er0[i][j] = 0.;
	      v_er1[i][j] = 0.;
	      v_er2[i][j] = 0.;
	    }
	}
    }
    
  
  for( int ipath=rank; ipath<paths; ipath+=numprocs ) 
    {

      // seed random number
      dsfmt_gv_init_gen_rand(ipath+seed+btch);//seed for random number SFMT


      // compute random numbers
      Rs( dof, N_finestep, W, Z, Y, I, J, aux, mod );

      double h = hfine;
      int nstep = N_finestep;


      // initialize system
      init_xv( x, v, N_part );


      // compute trajectory


      int offset = 0;
      int bigoffset = 0;
      for( int n=0; n<nstep; n++ ) {

        // offset the R pointers to use a set of Dim*N_Part numbers
        // constructed for a different time interval.
        step( N_part, x, v, W+offset, Z+offset, Y+offset, I+bigoffset, J+bigoffset, aux+bigoffset, h, n*h );
        offset += dof;
        bigoffset += dof*dof;
      }
      
      // save results
      for( int i=0; i<Dim*N_part; i++ ) {
        x_output[0][i] = x[i];
        v_output[0][i] = v[i];
        x_output_weak[0][i] += x[i];
        v_output_weak[0][i] += v[i];
	if (stattest)
	  {
	    x_stat[0][count_path*Dim*N_part+i] = x[i];
	    v_stat[0][count_path*Dim*N_part+i] = v[i];
	  }
      }
      
      // coarsen random numbers
      coarsenRs( nstep, dof, W, Z, Y,I, J, aux, h, gam );

      // and coarsen other quantities
      h *= 2.;
      nstep /= 2;
      
      
      for( int ires=0; ires<resadd; ires++ ) {
	
	// coarsen random numbers
	coarsenRs( nstep, dof, W, Z, Y, I, J, aux, h, gam );
	
	// and coarsen other quantities
	h *= 2.;
	nstep /= 2;
	
      }
      
      for( int ires=1; ires<nres; ires++ ) {
	
	// initialize system
	init_xv( x, v, N_part );
	
	// compute trajectory
	
	
	int offset = 0;
	int bigoffset = 0;
	for( int n=0; n<nstep; n++ ) {
	  // offset the R pointers to use a set of Dim*N_Part numbers
	  // constructed for a different time interval.
	  step( N_part, x, v, W+offset, Z+offset, Y+offset, I+bigoffset, J+bigoffset, aux+bigoffset, h, n*h );
	  offset += dof;
	  bigoffset += dof*dof;
	}
	
	// save results
	for( int i=0; i<Dim*N_part; i++ ) {
	  x_output[ires][i] = x[i];
	  v_output[ires][i] = v[i];
	  x_output_weak[ires][i] += x[i];
	  v_output_weak[ires][i] += v[i];
	  if (stattest)
	    {
	      x_stat[ires][count_path*Dim*N_part+i] = x[i];
	      v_stat[ires][count_path*Dim*N_part+i] = v[i];
	    }
	}
	
	// coarsen random numbers
	coarsenRs( nstep, dof, W, Z, Y, I, J, aux, h, gam );
	
	// and coarsen other quantities
	h *= 2.;
	nstep /= 2;
	
      }
      for( int ires=0; ires<nres-1; ires++ ) {
	int beg=ires;
	if (resadd) beg=0;
	for( int i=0; i<Dim*N_part; i++ ) {
	  double xer = fabs(x_output[beg][i] - x_output[ires+1][i]);
	  x_er0[ires][i] = MAX( x_er0[ires][i], xer );
	  x_er1[ires][i] += xer;
	  x_er2[ires][i] += xer*xer;
	  
	  double ver = fabs(v_output[beg][i] - v_output[ires+1][i]);
	  v_er0[ires][i] = MAX( v_er0[ires][i], ver );
	  v_er1[ires][i] += ver;
	  v_er2[ires][i] += ver*ver;
	}
      }
      count_path++;
    } // end of paths


  cout << "Processor "<<rank<<" done "<<endl;
  MPI_Barrier(MPI_COMM_WORLD);

  for( int ires=0; ires<nres; ires++ ) 
    {
      for( int i=0; i<dof; i++ ) 
       {
	 double xx1  = x_output_weak[ires][i];
	 double vv1  = v_output_weak[ires][i];
	 
	 double temp1=0.;
	 double temp2=0.;
	 
	 MPI_Reduce(&xx1, &temp1, 1, MPI_DOUBLE, 
		    MPI_SUM, 0, MPI_COMM_WORLD);
	 MPI_Reduce(&vv1, &temp2, 1, MPI_DOUBLE, 
		    MPI_SUM, 0, MPI_COMM_WORLD);
	 
	 if (rank==0)
	   {
	     x_output_weak[ires][i] = temp1;
	     v_output_weak[ires][i] = temp2;
	   }
       }
   }
 
 for( int ires=0; ires<nres-1; ires++ ) 
   {
     for( int i=0; i<dof; i++ ) 
       {
	 double xxr0 = x_er0[ires][i];
	 double xxr1 = x_er1[ires][i];
	 double xxr2 = x_er2[ires][i];
	 double vvr0 = v_er0[ires][i];
	 double vvr1 = v_er1[ires][i];
	 double vvr2 = v_er2[ires][i];

	 double temp1=0.;
	 double temp2=0.;
	 double temp3=0.;
	 double temp4=0.;
	 double temp5=0.;
	 double temp6=0.;

	 MPI_Reduce(&xxr0, &temp1, 1, 
		    MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	 MPI_Reduce(&vvr0, &temp2, 1, 
		    MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	 MPI_Reduce(&xxr1, &temp3, 1, 
		    MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	 MPI_Reduce(&xxr2, &temp4, 1, 
		    MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	 MPI_Reduce(&vvr1, &temp5, 1, 
		    MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	 MPI_Reduce(&vvr2, &temp6, 1, 
		    MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	 if (rank==0)
	   {
	     x_er0[ires][i]  = temp1;
	     x_er1[ires][i]  = temp3;
	     x_er2[ires][i]  = temp4;

	     v_er0[ires][i]  = temp2;
	     v_er1[ires][i]  = temp5;
	     v_er2[ires][i]  = temp6;
	   }
       }
   }


  if (stattest)
    {
       
     double *x_gather, *v_gather;
      if (rank==0)
	{
	  x_gather = new double[(paths+numprocs)*dof];
	  v_gather = new double[(paths+numprocs)*dof];
	}
      for( int ires=0; ires<nres; ires++ ) 
	{
	  MPI_Gather(x_stat[ires],(paths/numprocs+1)*dof, MPI_DOUBLE, 
		     x_gather, (paths/numprocs+1)*dof, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	  MPI_Gather(v_stat[ires],(paths/numprocs+1)*dof, MPI_DOUBLE, 
		     v_gather, (paths/numprocs+1)*dof, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	      
	  if (rank==0)
	    {

	      for( int i=0; i<dof; i++ ) 
		{
		  double x_mean = x_output_weak[ires][i]/paths;
		  double v_mean = v_output_weak[ires][i]/paths;
		  for( int irank=0; irank<numprocs; irank++) 
		    for( int ipath=irank; ipath<paths; ipath+=numprocs ) 
		      {  
			int i0 = irank*(paths/numprocs+1)*dof+(ipath/numprocs)*dof + i;
			varx[btch][ires] +=(x_gather[i0]-x_mean)*(x_gather[i0]-x_mean);
			varv[btch][ires] +=(v_gather[i0]-v_mean)*(v_gather[i0]-v_mean);
		      }
		}
	      
	      varx[btch][ires] /=(paths-1)*dof;
	      varv[btch][ires] /=(paths-1)*dof;
	    }
	}
      if (rank==0)
	{
	  delete[] x_gather;
	  delete[] v_gather;
	}

    }

 if (rank==0)
   {
     for( int ires=0; ires<nres-1; ires++ ) {
       int beg=ires;
       if (resadd) beg=0;
       for( int i=0; i<Dim*N_part; i++ ) {
	 double xer = fabs( x_output_weak[beg][i] - x_output_weak[ires+1][i] )/paths;
	 double ver = fabs( v_output_weak[beg][i] - v_output_weak[ires+1][i] )/paths;
	 x_er_weak[ires][i] = xer;
	 v_er_weak[ires][i] = ver;
	 if (stattest)
	   {
	     xerr_stat_w[btch][ires] +=xer;
	     verr_stat_w[btch][ires] +=ver;
	   }
       }
       if (stattest)
	 {
	   xerr_stat_w[btch][ires] /=dof;
	   verr_stat_w[btch][ires] /=dof;
	 }
     }
     cout << "WEAK ERRORS AND CONVERGENCE ORDER" << endl;

     double h = hfine;
     static char xyz[3] = {'x','y','z'};
     for( int ires=0; ires<nres-1; ires++ ) {
       cout << "using (2^" << ires << ")h,(2^" << ires+1 << ")h" << endl;
       cout << "errors in 1 norm:" << endl;
       for( int i=0; i<Dim*N_part; i++ ) {
	 cout << "X  error " << i/Dim << xyz[i%Dim]
	      << " " << x_er_weak[ires][i]  << endl;
       }
       for( int i=0; i<Dim*N_part; i++ ) {
	 cout << "V  error " << i/Dim << xyz[i%Dim]
	      << " " << v_er_weak[ires][i]  << endl;
       }
       h *= 2.;
     }


     int nstep = N_finestep;

     wox <<"***************** batch="<<btch<< endl; 
     wov <<"***************** batch="<<btch<< endl; 
     sox <<"***************** batch="<<btch<< endl; 
     sov <<"***************** batch="<<btch<< endl; 
     vov <<"***************** batch="<<btch<< endl; 

     //  if (stattest) vov <<"variance of x v"<<endl; 
     //     sox <<"weak errors "<<endl; 
     //     sov <<"weak errors "<<endl; 
     for( int ires=0; ires<nres-1; ires++ ) {
       wox << nstep;
       wov << nstep;
       //sox << nstep;
       if (stattest) vov << nstep;
       for( int i=0; i<Dim*N_part; i++ ) {
	 wox << " " << x_er_weak[ires][i];
	 wov << " " << v_er_weak[ires][i];
       }
       if (stattest)
	 {
	   vov << " " << varx[btch][ires];
	   vov << " " << varv[btch][ires];
	 }
       wox << endl;
       wov << endl;
       if (stattest) vov << endl;
       nstep /= 2;
     }
     
     for( int ires=0; ires<nres-2; ires++ ) {
       cout << "richardson weak order using (2^"
	    << ires << ")h,(2^" << ires+1 << ")h,(2^" << ires+2  << ")h" << endl;
       cout << "rates for errors in 1 norm:" << endl;
       double avg=0;
       double r;
       for( int i=0; i<Dim*N_part; i++ ) {
	 cout << "X " << i/Dim << xyz[i%Dim]
	      << " " << (r=log(x_er_weak[ires+1][i]/x_er_weak[ires][i])/log(2.))
	      << endl;
	 avg += r;
       }
       avg /= (Dim*N_part);
       cout << "average for X: " << avg << endl;
       avg=0;
       for( int i=0; i<Dim*N_part; i++ ) {
	 cout << "V " << i/Dim << xyz[i%Dim]
	      << " " << (r=log(v_er_weak[ires+1][i]/v_er_weak[ires][i])/log(2.))
	      << endl;
	 avg += r;
       }
       avg /= (Dim*N_part);
       cout << "average for V: " << avg << endl;
     }
     
     cout << endl;
     cout << "STRONG ERRORS AND CONVERGENCE ORDER" << endl;

     h = hfine;
     //     static char xyz[3] = {'x','y','z'};
     for( int ires=0; ires<nres-1; ires++ ) {
       cout << "using (2^" << ires << ")h,(2^" << ires+1 << ")h" << endl;
       cout << "errors in norms Li > L2 > L1:" << endl;
       for( int i=0; i<Dim*N_part; i++ ) {
	 x_er1[ires][i] /= (double) paths;
	 x_er2[ires][i] /= (double) paths;
	 x_er2[ires][i] = sqrt( x_er2[ires][i] );
	 
	 cout << "X  error " << i/Dim << xyz[i%Dim]
	      << " " << x_er0[ires][i]
	      << " " << x_er2[ires][i]
	      << " " << x_er1[ires][i] << endl;
       }
       for( int i=0; i<Dim*N_part; i++ ) {
	 v_er1[ires][i] /= (double) paths;
	 v_er2[ires][i] /= (double) paths;
	 v_er2[ires][i] = sqrt( v_er2[ires][i] );
	 
	 if (stattest) 
	   {
	     xerr_stat_s[btch][ires] +=x_er2[ires][i];
	     verr_stat_s[btch][ires] +=v_er2[ires][i];
	   }
	 cout << "V  error " << i/Dim << xyz[i%Dim]
	      << " " << v_er0[ires][i]
	      << " " << v_er2[ires][i]
	      << " " << v_er1[ires][i] << endl;
       }
       h *= 2;
       if (stattest)
	 {
	   xerr_stat_s[btch][ires] /=dof;
	   verr_stat_s[btch][ires] /=dof;
	 }
     }
     
     nstep = N_finestep;
     
     //     sox <<endl; 
     //     sox <<"strong errors "<<endl; 
     //     sov <<endl; 
     //     sov <<"strong errors "<<endl; 
     for( int ires=0; ires<nres-1; ires++ ) {
       sox << nstep;
       sov << nstep;
       
       for( int i=0; i<Dim*N_part; i++ ) {
	 sox << " " << x_er2[ires][i];
	 sov << " " << v_er2[ires][i];
       }
       
       sox << endl;
       sov << endl;
       nstep /= 2;
     }
     
     for( int ires=0; ires<nres-2; ires++ ) {
       cout << "richardson strong order using (2^"
	    << ires << ")h,(2^" << ires+1 << ")h,(2^" << ires+2  << ")h" << endl;
       cout << "rates for errors in norms Li, L2, L1:" << endl;
       double avg0=0, avg1=0, avg2=0;
       double r0,r1,r2;
       for( int i=0; i<Dim*N_part; i++ ) {
	 cout << "X " << i/Dim << xyz[i%Dim]
	      << " " << (r0=log(x_er0[ires+1][i]/x_er0[ires][i])/log(2.))
	      << " " << (r2=log(x_er2[ires+1][i]/x_er2[ires][i])/log(2.))
	      << " " << (r1=log(x_er1[ires+1][i]/x_er1[ires][i])/log(2.))
	      << endl;
	 avg0 += r0;
	 avg1 += r1;
	 avg2 += r2;
       }
       avg0 /= (Dim*N_part);
       avg1 /= (Dim*N_part);
       avg2 /= (Dim*N_part);
       cout << "average for X: r(Li)=" << avg0 <<
	 " r(L2)=" << avg2 <<
	 " r(L1)=" << avg1 << endl;
       avg0=0, avg1=0, avg2=0;
       for( int i=0; i<Dim*N_part; i++ ) {
	 cout << "V " << i/Dim << xyz[i%Dim]
	      << " " << (r0=log(v_er0[ires+1][i]/v_er0[ires][i])/log(2.))
	      << " " << (r2=log(v_er2[ires+1][i]/v_er2[ires][i])/log(2.))
	      << " " << (r1=log(v_er1[ires+1][i]/v_er1[ires][i])/log(2.))
	      << endl;
	 avg0 += r0;
	 avg1 += r1;
	 avg2 += r2;
       }
       avg0 /= (Dim*N_part);
       avg1 /= (Dim*N_part);
       avg2 /= (Dim*N_part);
       cout << "average for V: r(Li)=" << avg0 <<
	 " r(L2)=" << avg2 <<
	 " r(L1)=" << avg1 << endl;
     }
   }
  
    }//err_batches
  if (rank==0)
    {
      if (stattest) 
	{
	  std::fstream varout("statistical-output", std::ios_base::out );
	  int nstep = N_finestep;
	  varout << "res" 
		 << " x_weak_err  " << "  st_dev" 
		 << " v_weak_err  " << "  st_dev" 
		 << " x_strong_err" << "  st_dev" 
		 << " v_strong_err" << "  st_dev"<<endl;
	  varout <<"-----------------------------------------" <<endl;

	  for( int ires=0; ires<nres-1; ires++ ) 
	    {
	      double xmean_w = 0.;
	      double vmean_w = 0.;
	      double xmean_s = 0.;
	      double vmean_s = 0.;
	      double xvar_w = 0.;
	      double vvar_w = 0.;
	      double xvar_s = 0.;
	      double vvar_s = 0.;
	      for( int btch=0; btch<num_batches; btch++ ) 
		{
		  xmean_w += xerr_stat_w[btch][ires];
		  vmean_w += verr_stat_w[btch][ires];
		  xmean_s += xerr_stat_s[btch][ires];
		  vmean_s += verr_stat_s[btch][ires];
		}
	      xmean_w /= num_batches;
	      vmean_w /= num_batches;
	      xmean_s /= num_batches;
	      vmean_s /= num_batches;

	      for( int btch=0; btch<num_batches; btch++ ) 
		{
		  xvar_w +=(xerr_stat_w[btch][ires] - xmean_w)*(xerr_stat_w[btch][ires] - xmean_w);
		  vvar_w +=(verr_stat_w[btch][ires] - vmean_w)*(verr_stat_w[btch][ires] - vmean_w);
		  xvar_s +=(xerr_stat_s[btch][ires] - xmean_s)*(xerr_stat_s[btch][ires] - xmean_s);
		  vvar_s +=(verr_stat_s[btch][ires] - vmean_s)*(verr_stat_s[btch][ires] - vmean_s);
		}

 	      int deg_freedom = num_batches-1;
	      xvar_w /= deg_freedom;
	      vvar_w /= deg_freedom;
	      xvar_s /= deg_freedom;
	      vvar_s /= deg_freedom;
	      
	      varout << nstep;
	      varout << "  " << xmean_w << "  " << sqrt(xvar_w/deg_freedom);
	      varout << "  " << vmean_w << "  " << sqrt(vvar_w/deg_freedom);
	      varout << "  " << xmean_s << "  " << sqrt(xvar_s/deg_freedom);
	      varout << "  " << vmean_s << "  " << sqrt(vvar_s/deg_freedom);
	      varout <<endl;
	      nstep/=2;
	    }
	  varout.close();
	}
    }

  MPI_Barrier(MPI_COMM_WORLD);

  wox.close();
  wov.close();
  sox.close();
  sov.close();
  vov.close();
  for( int i=nres-2; i>=0; i-- ) {
    delete[] v_er2[i];
    delete[] v_er1[i];
    delete[] v_er0[i];
    delete[] x_er2[i];
    delete[] x_er1[i];
    delete[] x_er0[i];
    delete[] x_er_weak[i];
    delete[] v_er_weak[i];
  }
  for( int i=nres-1; i>=0; i-- ) {
    delete[] v_output[i];
    delete[] x_output[i];
    delete[] x_output_weak[i];
    delete[] v_output_weak[i];
  }
  delete[] x_er_weak;
  delete[] v_er_weak;
  delete[] v_er2;
  delete[] v_er1;
  delete[] v_er0;
  delete[] v_output;
  delete[] x_er2;
  delete[] x_er1;
  delete[] x_er0;
  delete[] x_output;
  delete[] x_output_weak;
  delete[] v_output_weak;
  if (stattest)
    {
      
      for( int i=0; i<nres; i++ ) 
	{
	  delete[] x_stat[i];
	  delete[] v_stat[i];
	}
      delete[] x_stat;
      delete[] v_stat;
      
      if (rank==0)
	{
	  for( int i=0; i< num_batches; i++) 
	    {
	      delete[] varx[i];
	      delete[] varv[i];
	      delete[] xerr_stat_w[i];
	      delete[] verr_stat_w[i];
	      delete[] xerr_stat_s[i];
	      delete[] verr_stat_s[i];
	    }
	  delete[] varx;
	  delete[] varv;
	  delete[] xerr_stat_w;
	  delete[] verr_stat_w;
	  delete[] xerr_stat_s;
	  delete[] verr_stat_s;
	}
    }

  delete[] aux;
  delete[] I;
  delete[] J;
  delete[] W;
  delete[] Z;
  delete[] Y;
  delete[] x;
  delete[] v;

  MPI_Finalize();
  return 1;
}

