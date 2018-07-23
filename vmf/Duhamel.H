#ifndef __Duhamel__
#define __Duhamel__


class expectations {
  public:
  expectations()
  {
     // initialize because mpfrcpp will not set values if out of range
     eWW=0;
     eWZ=eZZ=0;
     eii=eiWW=eiWZ=eiZW=eiZZ=eiWY=eiYW=eiZY=eiYZ=eiYY=0;
     ejj=eij=ejWW=ejWZ=ejZW=ejZZ=ejWY=ejYW=ejZY=ejYZ=ejYY=0;
     eaa=eia=eja=eaWW=eaWZ=eaZW=eaZZ=eaWY=eaYW=eaZY=eaYZ=eaYY=0;
  }
  ~expectations(){}
  double eWW;
  double eWZ, eZZ;
  double eWY, eZY, eYY;
  double eii, eiWW, eiWZ, eiZW, eiZZ,eiWY,eiYW,eiZY,eiYZ,eiYY;
  double ejj, eij, ejWW, ejWZ, ejZW, ejZZ,ejWY,ejYW,ejZY,ejYZ,ejYY;
  double eaa, eia, eja, eaWW, eaWZ, eaZW, eaZZ,eaWY,eaYW,eaZY,eaYZ,eaYY;
};

class modelCoefficients{
  public:
  modelCoefficients()
  {
     // initialize because mpfrcpp will not set values if out of range
     a1=0;
     b1=b2=0;
     c1=c2=c3=0;
     d11=d12=d13=d21=d22=d23=d31=d32=d33=d4=d4sq=0;
     o11=o12=o13=o21=o22=o23=o31=o32=o33=o4=o5ab=o5ba=0;
     f11=f12=f13=f21=f22=f23=f31=f32=f33=f4=f5=f6=f6sq=0;
     D1=D2=D3=D4=D5=D6=D7=D7sq=0;
     O1=O2=O3=O4=O5=O6=O7=O8=0;
     F1=F2=F3=F4=F5=F6=F7=F8=F9=0;
  }
  ~modelCoefficients(){}
  double a1;
  double b1,b2;
  double c1,c2,c3;
  double d11,d12,d13,d21,d22,d23,d31,d32,d33,d4,d4sq;
  double o11,o12,o13,o21,o22,o23,o31,o32,o33,o4,o5ab,o5ba;
  double f11,f12,f13,f21,f22,f23,f31,f32,f33,f4,f5,f6, f6sq;
  double D1,D2,D3,D4,D5,D6,D7,D7sq;
  double O1,O2,O3,O4,O5,O6,O7,O8;
  double F1,F2,F3,F4,F5,F6,F7,F8,F9;
};    

void makeModel(expectations& ex,
	       modelCoefficients& mod,
	       double d_dt,
	       double gam,
	       bool m_verbose,
	       int a_precision);

void testRs( int Nstep, 
	     int Npart, 
	     double *W, 
	     double *Z, 
	     double *Y, 
	     double *I, 
	     double *J, 
	     double *aux,  
	     expectations& ex, 
	     bool equal );


double normaldeviate();

void Rs( int WDim, 
	 int Nstep, 
	 double *W, 
	 double *Z, 
	 double *Y, 
	 double *I, 
	 double *J, 
	 double *aux, 
	 modelCoefficients& mod);

void coarsenRs( int n_f, 
		int WDim, 
		double *W, 
		double *Z, 
		double *Y, 
		double *I, 
		double *J, 
		double *aux, 
		double h_f,
		double gam );
  
#endif // end of #ifndef __Duhamel_
