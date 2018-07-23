#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <string>
#include <assert.h>
#include <mpfrcpp/mpfrcpp.hpp>
#include "./dSFMT/dSFMT.h"
#include "Duhamel.H"

using std::cout;
using std::endl;
using mpfr::Real;
using namespace mpfr;



void QuadSolver(Precision p,
		Real a,
		Real b,
		Real &x1,
		Real &x2,
		int choice,
		bool verbose)
{
  SqrtClass sqc( p );
  Precision outputp(40);
  AbsClass abs(p);
  Real zerocut = 1e-200;
 
  Real Dis = b*b - a*a;
  if (verbose)
    cout << "Discriminant is " << Dis.toString(outputp) << endl;

  if (abs(Dis) < zerocut) Dis=0;

  Real x1sq= (b - sqc(Dis))/2;
  Real x2sq= (b + sqc(Dis))/2;
  if (verbose)
    {
      cout << "Roots are " << x1sq.toString(outputp) << " and " << x2sq.toString(outputp) << endl;

      cout << "first choice" << endl;
    }
  x1 = sqc(x1sq);
  x2 = a/x1/2;
  if (verbose)
    {
      cout << "x1 x2 are " << x1.toString(outputp) << " and " << x2.toString(outputp) << endl;
      cout << "square x2 is " << (x2*x2).toString(outputp) << endl;
      cout <<"----------------------"<<endl;
    }
  if (choice==2)
    {
      if (verbose)  
	cout << "second choice" << endl;


      x1 = sqc(x2sq);
      x2 = a/x1/2;
      if (verbose)
	{
	  cout << "x1 x2 are " << x1.toString(outputp) << " and " << x2.toString(outputp) << endl;
	  cout << "square x2 is " << (x2*x2).toString(outputp) << endl;
	  cout <<"----------------------" << endl;
	}
    }
}


void makeModel( expectations& ex,
		modelCoefficients& mod,
		double d_dt,
		double gam,
		bool m_verbose,
		int a_precision)
{
  Precision p(a_precision);

  Library.setPrecision( p );
  //Parameters.initializeByZero();
  //Parameters.setDefaultRoundMode(roundTowardInfinity);

  AbsClass abs(p);
  SqrtClass sqc( p );
  ExpClass ec( p );
  Expm1Class em1c( p );
  Precision outputp(35);
  

  Real zerocut = Real(1e-200);
  Real zero = Real(0.);

  Real gm = Real(gam);
  Real dt = Real(d_dt);
  Real gt = gm * dt;
  Real e = ec( -gt );
  Real e2 = e*e;
  Real e3 = e2*e;
  Real e4 = e2*e2;
  Real em1 = em1c( -gt );
  Real em2 = em1c( -2*gt );
  Real em3 = em1c( -3*gt );
  Real em4 = em1c( -4*gt );
  Real g2 = gm*gm;
  Real g3 = g2*gm;
  Real g4 = g2*g2;
  Real g5 = g3*g2;

  Real eWW = dt;
  ex.eWW = (double)eWW;


  if( m_verbose )
  cout << "eWW " << eWW.toString(outputp) << " " << ex.eWW <<  endl;
  

  Real a1 = sqc( eWW );

  mod.a1 = (double)a1;

  if( m_verbose ) {
  cout << "  a1 " << a1.toString(outputp) << " " << mod.a1 << endl;
  cout << "------------------------------" << endl;
  }

  Real eZZ = -em2/2/gm;
  Real eWZ = -em1/gm;
  ex.eWZ = (double)eWZ;
  ex.eZZ = (double)eZZ;

  if( m_verbose ) {
  cout << "eZZ " << eZZ.toString(outputp) << " " << ex.eZZ << endl;
  cout << "eWZ " << eWZ.toString(outputp) << " " << ex.eWZ << endl;
  }

  Real b1 = eWZ/a1;
  Real b2 = sqc( eZZ - b1*b1 );

  mod.b1 = (double)b1;
  mod.b2 = (double)b2;

  if( m_verbose ) {
  cout << "  b1 " << b1.toString(outputp) << " " << mod.b1 << endl;
  cout << "  b2 " << b2.toString(outputp) << " " << mod.b2 << endl;
  cout << "------------------------------" << endl;
  }

  Real eYY = -em2/g3/4 + dt*(dt-1/gm)/gm/2;
  Real eWY = dt/gm + em1/g2;
  Real eZY = dt/2/gm + em2/4/g2;
  ex.eWY = (double)eWY;
  ex.eYY = (double)eYY;
  ex.eZY = (double)eZY;

  if( m_verbose ) {
  cout << "eYY " << eYY.toString(outputp) << " " << ex.eYY << endl;
  cout << "eWZ " << eWY.toString(outputp) << " " << ex.eWY << endl;
  cout << "eWZ " << eZY.toString(outputp) << " " << ex.eZY << endl;
  }

  Real c1 = eWY/a1;
  Real c2 = (eZY - b1*c1 )/b2;
  Real c3 = sqc( eYY - c1*c1 -c2*c2);

  mod.c1 = (double)c1;
  mod.c2 = (double)c2;
  mod.c3 = (double)c3;

  if( m_verbose ) {
  cout << "  c1 " << c1.toString(outputp) << " " << mod.c1 << endl;
  cout << "  c2 " << c2.toString(outputp) << " " << mod.c2 << endl;
  cout << "  c3 " << c3.toString(outputp) << " " << mod.c3 << endl;
  cout << "------------------------------" << endl;
  }

  Real eii  = -em2/4/g2 - e2*dt/2/gm;
  Real eiWW = -em1/g2 - e*dt/gm;
  Real eiWZ = -em2/4/g2 - e2*dt/2/gm;
  Real eiZW = em1*em1/2/g2;
  Real eiZZ = em3/3/g2 - em2/2/g2;
  Real eiWY = dt/4/g2 + e2*dt/4/g2 + em2/4/g3;
  Real eiYW = em1/g3 - 3*em2/4/g3 + dt*(-2*e+1)/2/g2;
  Real eiZY = -em3/9/g3 + em2/4/g3 + dt/6/g2;
  Real eiYZ = em2/2/g3 - 4*em3/9/g3 + dt*(1-3*e2)/6/g2;
  Real eiYY = 5*em3/27/g4 - em2/4/g4 - dt*(7-9*e2)/36/g3 + dt*dt/6/g2;

  ex.eii  = (double)eii;
  ex.eiWW = (double)eiWW;
  ex.eiWZ = (double)eiWZ;
  ex.eiZW = (double)eiZW;
  ex.eiZZ = (double)eiZZ;
  ex.eiWY = (double)eiWY;
  ex.eiYW = (double)eiYW;
  ex.eiZY = (double)eiZY;
  ex.eiYZ = (double)eiYZ;
  ex.eiYY = (double)eiYY;

  if( m_verbose ) {
  cout << "eii  " << eii.toString(outputp) << " " << ex.eii << endl;
  cout << "eiWW " << eiWW.toString(outputp) << " " << ex.eiWW << endl;
  cout << "eiWZ " << eiWZ.toString(outputp) << " " << ex.eiWZ << endl;
  cout << "eiZW " << eiZW.toString(outputp) << " " << ex.eiZW << endl;
  cout << "eiZZ " << eiZZ.toString(outputp) << " " << ex.eiZZ << endl;
  cout << "eiWY " << eiWY.toString(outputp) << " " << ex.eiWY << endl;
  cout << "eiYW " << eiYW.toString(outputp) << " " << ex.eiYW << endl;
  cout << "eiZY " << eiZY.toString(outputp) << " " << ex.eiZY << endl;
  cout << "eiYZ " << eiYZ.toString(outputp) << " " << ex.eiYZ << endl;
  cout << "eiYY " << eiYY.toString(outputp) << " " << ex.eiYY << endl;
  }

 
  Real d11 = eiWW/(a1*a1);
  Real d12 = (eiWZ-a1*b1*d11)/(a1*b2);
  Real d21 = (eiZW-a1*b1*d11)/(a1*b2);
  Real d22 = (eiZZ-b1*b1*d11-b1*b2*(d12+d21))/(b2*b2);
  Real d13 = (eiWY-a1*c1*d11-a1*c2*d12)/(a1*c3);
  Real d31 = (eiYW-a1*c1*d11-a1*c2*d21)/(a1*c3);
  Real d23 = (eiZY-b1*c1*d11-b1*c2*d12-b1*c3*d13-b2*c1*d21-b2*c2*d22)/(b2*c3);
  Real d32 = (eiYZ-b1*c1*d11-b1*c2*d21-b1*c3*d31-b2*c1*d12-b2*c2*d22)/(b2*c3);
  Real d33 = (eiYY-c1*c1*d11-c2*c2*d22-c1*c2*(d12+d21)-c1*c3*(d13+d31)-c2*c3*(d23+d32))/(c3*c3);


  Real b = eii-d11*d11-d12*d12-d13*d13-d21*d21-d22*d22-d23*d23-d31*d31-d32*d32-d33*d33;

  Real a = -d11*d11-d12*d21-d13*d31-d21*d12-d22*d22-d23*d32-d31*d13-d32*d23-d33*d33;
  if( m_verbose ) 
    cout << "d4 " << " b= " << b.toString(outputp)<< "    a= " << a.toString(outputp)<<endl;
 
  Real d4sq = b/2;

  if (!isnan(d4sq))
    if( m_verbose ) 
      if (d4sq < zero) cout << "Negative square for coefficient d4 = " << d4sq.toString(outputp) << endl;
  Real d4 = sqc( d4sq );
  if (abs(d4sq) < zerocut) d4 = zero;

  if (abs(d11) < zerocut) mod.d11=0.; else  mod.d11 = (double)d11;
  if (abs(d12) < zerocut) mod.d12=0.; else  mod.d12 = (double)d12;
  if (abs(d13) < zerocut) mod.d13=0.; else  mod.d13 = (double)d13;
  if (abs(d21) < zerocut) mod.d21=0.; else  mod.d21 = (double)d21;
  if (abs(d22) < zerocut) mod.d22=0.; else  mod.d22 = (double)d22;
  if (abs(d23) < zerocut) mod.d23=0.; else  mod.d23 = (double)d23;

  //d31 is always zero?
  //  if (!isnan(d31)) if (abs(d31) < zerocut) mod.d31=0.; else  mod.d31 = (double)d31;
  mod.d31=0.;

  if (abs(d32) < zerocut) mod.d32=0.; else  mod.d32 = (double)d32;
  if (abs(d33) < zerocut) mod.d33=0.; else  mod.d33 = (double)d33;
  if (abs(d4) < zerocut) mod.d4=0.;   else  mod.d4  = (double)d4;
  if (abs(d4sq) < zerocut) mod.d4sq=0.;else  mod.d4sq  = (double)d4sq;

  if( m_verbose ) {
  cout << "  d11  " << d11.toString(outputp) << " " << mod.d11 << endl;
  cout << "  d12  " << d12.toString(outputp) << " " << mod.d12 << endl;
  cout << "  d21  " << d21.toString(outputp) << " " << mod.d21 << endl;
  cout << "  d22  " << d22.toString(outputp) << " " << mod.d22 << endl;
  cout << "  d13  " << d13.toString(outputp) << " " << mod.d13 << endl;
  cout << "  d31  " << d31.toString(outputp) << " " << mod.d31 << endl;
  cout << "  d23  " << d23.toString(outputp) << " " << mod.d23 << endl;
  cout << "  d32  " << d32.toString(outputp) << " " << mod.d32 << endl;
  cout << "  d33  " << d33.toString(outputp) << " " << mod.d33 << endl;
  cout << "  d4   " << d4.toString(outputp) << " " << mod.d4 << endl;
  cout << "------------------------------" << endl;
  }

  Real eij = em1*em1/2/g2;
  Real ejj  = em2/4/g2 + dt/2/gm;
  Real ejWW = em1/g2 + dt/gm;
  Real ejWZ = em1*em1/2/g2;
  Real ejZW = em2/4/g2 + dt/2/gm;
  Real ejZZ = em3/6/g2 - em1/2/g2;
  Real ejWY = dt/2/g2 + em1/g3 - em2/4/g3;
  Real ejYW = -em2/4/g3 + dt*(-1/g2+dt/gm)/2;
  Real ejZY = em1/2/g3 - em3/18/g3 + dt/3/g2;
  Real ejYZ = -5*em3/36/g3 + em1/4/g3 + dt*(2-3*e)/6/g2;
  Real ejYY = -em1/4/g4 + 7*em3/108/g4 - dt*(10-9*e)/18/g3 + dt*dt/3/g2;

  ex.eij  = (double)eij;
  ex.ejj  = (double)ejj;
  ex.ejWW = (double)ejWW;
  ex.ejWZ = (double)ejWZ;
  ex.ejZW = (double)ejZW;
  ex.ejZZ = (double)ejZZ;
  ex.ejWY = (double)ejWY;
  ex.ejYW = (double)ejYW;
  ex.ejZY = (double)ejZY;
  ex.ejYZ = (double)ejYZ;
  ex.ejYY = (double)ejYY;

  if( m_verbose ) {
  cout << "ejj  " << ejj.toString(outputp) << " " << ex.ejj << endl;
  cout << "eij  " << eij.toString(outputp) << " " << ex.eij << endl;
  cout << "ejWW " << ejWW.toString(outputp) << " " << ex.ejWW << endl;
  cout << "ejWZ " << ejWZ.toString(outputp) << " " << ex.ejWZ << endl;
  cout << "ejZW " << ejZW.toString(outputp) << " " << ex.ejZW << endl;
  cout << "ejZZ " << ejZZ.toString(outputp) << " " << ex.ejZZ << endl;
  cout << "ejWY " << ejWY.toString(outputp) << " " << ex.ejWY << endl;
  cout << "ejYW " << ejYW.toString(outputp) << " " << ex.ejYW << endl;
  cout << "ejZY " << ejZY.toString(outputp) << " " << ex.ejZY << endl;
  cout << "ejYZ " << ejYZ.toString(outputp) << " " << ex.ejYZ << endl;
  cout << "ejYY " << ejYY.toString(outputp) << " " << ex.ejYY << endl;
  }

  Real o11 = ejWW/(a1*a1);
  Real o12 = (ejWZ-a1*b1*o11)/(a1*b2);
  Real o21 = (ejZW-a1*b1*o11)/(a1*b2);
  Real o22 = (ejZZ-b1*b1*o11-b1*b2*(o12+o21))/(b2*b2);
  Real o13 = (ejWY-a1*c1*o11-a1*c2*o12)/(a1*c3);
  Real o31 = (ejYW-a1*c1*o11-a1*c2*o21)/(a1*c3);
  Real o23 = (ejZY-b1*c1*o11-b1*c2*o12-b1*c3*o13-b2*c1*o21-b2*c2*o22)/(b2*c3);
  Real o32 = (ejYZ-b1*c1*o11-b1*c2*o21-b1*c3*o31-b2*c1*o12-b2*c2*o22)/(b2*c3);
  Real o33 = (ejYY-c1*c1*o11-c2*c2*o22-c1*c2*(o12+o21)-c1*c3*(o13+o31)-c2*c3*(o23+o32))/(c3*c3);


  b = eij-d11*o11-d12*o12-d13*o13-d21*o21-d22*o22-d23*o23-d31*o31-d32*o32-d33*o33;
  a = -d11*o11-d12*o21-d13*o31-d21*o12-d22*o22-d23*o32-d31*o13-d32*o23-d33*o33;
  if( m_verbose ) 
    cout << "o4 " << " b= " << b.toString(outputp)<< "    a= " << a.toString(outputp)<<endl;
  Real o4 = b/d4/2;

  Real o5ab;
  Real o5ba;
  b = ejj-o11*o11-o12*o12-o13*o13-o21*o21-o22*o22-o23*o23-o31*o31-o32*o32-o33*o33-2*o4*o4;
  a = -o11*o11-o12*o21-o13*o31-o21*o12-o22*o22-o23*o32-o31*o13-o32*o23-o33*o33+2*o4*o4;
  if( m_verbose ) 
    cout << "o5 " <<  " b= " << b.toString(outputp)<< "    a= " << a.toString(outputp)<<endl;


  QuadSolver(p,a,b,o5ab,o5ba,1,m_verbose);

  if (abs(o11) < zerocut) mod.o11=0.; else mod.o11 = (double)o11;
  if (abs(o12) < zerocut) mod.o12=0.; else mod.o12 = (double)o12;
  //o13 is always zero?
  if (abs(o13) < zerocut) mod.o13=0.; else mod.o13 = (double)o13;
  //  mod.o13 = 0.;
  if (abs(o21) < zerocut) mod.o21=0.; else mod.o21 = (double)o21;
  if (abs(o22) < zerocut) mod.o22=0.; else mod.o22 = (double)o22;
  if (abs(o23) < zerocut) mod.o23=0.; else mod.o23 = (double)o23;
  if (abs(o31) < zerocut) mod.o31=0.; else mod.o31 = (double)o31;
  if (abs(o32) < zerocut) mod.o32=0.; else mod.o32 = (double)o32;
  if (abs(o33) < zerocut) mod.o33=0.; else mod.o33 = (double)o33;
  if (abs(o4) < zerocut) mod.o4=0.;   else mod.o4  = (double)o4;

  if (abs(o5ab) < zerocut) mod.o5ab=0.;   else mod.o5ab  = (double)o5ab;
  if (abs(o5ba) < zerocut) mod.o5ba=0.;   else mod.o5ba  = (double)o5ba;

  if( m_verbose ) {
  cout << "  o11  " << o11.toString(outputp) << " " << mod.o11 << endl;
  cout << "  o12  " << o12.toString(outputp) << " " << mod.o12 << endl;
  cout << "  o21  " << o21.toString(outputp) << " " << mod.o21 << endl;
  cout << "  o22  " << o22.toString(outputp) << " " << mod.o22 << endl;
  cout << "  o13  " << o13.toString(outputp) << " " << mod.o13 << endl;
  cout << "  o31  " << o31.toString(outputp) << " " << mod.o31 << endl;
  cout << "  o23  " << o23.toString(outputp) << " " << mod.o23 << endl;
  cout << "  o32  " << o32.toString(outputp) << " " << mod.o32 << endl;
  cout << "  o33  " << o33.toString(outputp) << " " << mod.o33 << endl;
  cout << "  o4   " << o4.toString(outputp) << " " << mod.o4 << endl;
  cout << "  o5ab " << o5ab.toString(outputp) << " " << mod.o5ab << endl;
  cout << "  o5ba " << o5ba.toString(outputp) << " " << mod.o5ba << endl;
  cout << "------------------------------" << endl;
  }

  Real eaa = em2*em2/8/g2;
  Real eia = em3/3/g2 - em2/2/g2;
  Real eja  = em3/6/g2 - em1/2/g2;
  Real eaWW = em1*em1/2/g2;
  Real eaWZ = em3/3/g2 - em2/2/g2;
  Real eaZW = em3/6/g2 - em1/2/g2;
  Real eaZZ = em2*em2/8/g2;
  Real eaWY = em2/4/g3 - em3/9/g3 + dt/6/g2;
  Real eaYW = em1/4/g3 - 5*em3/36/g3 + dt*(2-3*e)/6/g2;
  Real eaZY = em2/8/g3 - em4/32/g3 + dt/8/g2;
  Real eaYZ = em2/8/g3 - 3*em4/32/g3 + dt*(1-2*e2)/8/g2;
  Real eaYY = (em2/gm + 2*dt)*(em2/gm + 2*dt)/32/g2;

  ex.eaa  = (double)eaa;
  ex.eia  = (double)eia;
  ex.eja  = (double)eja;
  ex.eaWW = (double)eaWW;
  ex.eaWZ = (double)eaWZ;
  ex.eaZW = (double)eaZW;
  ex.eaZZ = (double)eaZZ;
  ex.eaWY = (double)eaWY;
  ex.eaYW = (double)eaYW;
  ex.eaZY = (double)eaZY;
  ex.eaYZ = (double)eaYZ;
  ex.eaYY = (double)eaYY;

  if( m_verbose ) {
  cout << "eja  " << eja.toString(outputp) << " " << ex.eja << endl;
  cout << "eia  " << eia.toString(outputp) << " " << ex.eia << endl;
  cout << "eaWW " << eaWW.toString(outputp) << " " << ex.eaWW << endl;
  cout << "eaWZ " << eaWZ.toString(outputp) << " " << ex.eaWZ << endl;
  cout << "eaZW " << eaZW.toString(outputp) << " " << ex.eaZW << endl;
  cout << "eaZZ " << eaZZ.toString(outputp) << " " << ex.eaZZ << endl;
  cout << "eaWY " << eaWY.toString(outputp) << " " << ex.eaWY << endl;
  cout << "eaYW " << eaYW.toString(outputp) << " " << ex.eaYW << endl;
  cout << "eaZY " << eaZY.toString(outputp) << " " << ex.eaZY << endl;
  cout << "eaYZ " << eaYZ.toString(outputp) << " " << ex.eaYZ << endl;
  cout << "eaYY " << eaYY.toString(outputp) << " " << ex.eaYY << endl;
  }

  Real f11 = eaWW/(a1*a1);
  Real f12 = (eaWZ-a1*b1*f11)/(a1*b2);
  Real f21 = (eaZW-a1*b1*f11)/(a1*b2);
  Real f22 = (eaZZ-b1*b1*f11-b1*b2*(f12+f21))/(b2*b2);
  Real f13 = (eaWY-a1*c1*f11-a1*c2*f12)/(a1*c3);
  Real f31 = (eaYW-a1*c1*f11-a1*c2*f21)/(a1*c3);
  Real f23 = (eaZY-b1*c1*f11-b1*c2*f12-b1*c3*f13-b2*c1*f21-b2*c2*f22)/(b2*c3);
  Real f32 = (eaYZ-b1*c1*f11-b1*c2*f21-b1*c3*f31-b2*c1*f12-b2*c2*f22)/(b2*c3);
  Real f33 = (eaYY-c1*c1*f11-c2*c2*f22-c1*c2*(f12+f21)-c1*c3*(f13+f31)-c2*c3*(f23+f32))/(c3*c3);

  b = eia-d11*f11-d12*f12-d13*f13-d21*f21-d22*f22-d23*f23-d31*f31-d32*f32-d33*f33;
  a = -d11*f11-d12*f21-d13*f31-d21*f12-d22*f22-d23*f32-d31*f13-d32*f23-d33*f33;
  if( m_verbose ) 
    cout << "f4 " << " b= " << b.toString(outputp)<< "    a= " << a.toString(outputp)<<endl;
  Real f4 = b/d4/2;

  b = eja-o11*f11-o12*f12-o13*f13-o21*f21-o22*f22-o23*f23-o31*f31-o32*f32-o33*f33-2*o4*f4;
  a = -o11*f11-o12*f21-o13*f31-o21*f12-o22*f22-o23*f32-o31*f13-o32*f23-o33*f33+2*o4*f4;
  if( m_verbose ) 
    cout << "f5 " << " b= " << b.toString(outputp)<< "    a= " << a.toString(outputp)<<endl;

    Real c = o5ab*o5ab-o5ba*o5ba;
    Real f5ab = (b*o5ab+a*o5ba)/c;
    Real f5 = f5ab;

  b = eaa-f11*f11-f12*f12-f13*f13-f21*f21-f22*f22-f23*f23-f31*f31-f32*f32-2*f33*f33-2*f4*f4-2*f5*f5;
  a = -f11*f11-f12*f21-f13*f31-f21*f12-f22*f22-f23*f32-f31*f13-f32*f23+2*f33*f33+2*f4*f4+2*f5*f5;
  if( m_verbose ) 
    cout << "f6 " << " b= " << b.toString(outputp)<< "    a= " << a.toString(outputp)<<endl;

  Real f6sq = b/2;
  if( m_verbose )
    if (f6sq < zero)    cout << "Negative square for coefficient f6 = " << f6sq.toString(outputp) << endl;
  Real f6 = sqc( f6sq );
  if (abs(f6sq) < zerocut) f6=zero;

  if (abs(f11) < zerocut) mod.f11=0.; else mod.f11 = (double)f11;
  if (abs(f12) < zerocut) mod.f12=0.; else mod.f12 = (double)f12;
  if (abs(f13) < zerocut) mod.f13=0.; else mod.f13 = (double)f13;
  if (abs(f21) < zerocut) mod.f21=0.; else mod.f21 = (double)f21;
  if (abs(f22) < zerocut) mod.f22=0.; else mod.f22 = (double)f22;
  if (abs(f23) < zerocut) mod.f23=0.; else mod.f23 = (double)f23;
  if (abs(f31) < zerocut) mod.f31=0.; else mod.f31 = (double)f31;
  if (abs(f32) < zerocut) mod.f32=0.; else mod.f32 = (double)f32;
  //f33 is always zero?
  if (abs(f33) < zerocut) mod.f33=0.; else mod.f33 = (double)f33;
  //mod.f33=0.;
  if (abs(f4) < zerocut) mod.f4=0.;   else mod.f4  = (double)f4;
  if (abs(f5) < zerocut) mod.f5=0.;   else mod.f5  = (double)f5;
  if (abs(f6) < zerocut) mod.f6=0.;   else mod.f6  = (double)f6;
  if (abs(f6sq) < zerocut) mod.f6sq=0.;else mod.f6sq  = (double)f6sq;

  if( m_verbose ) {
  cout << "  f11  " << f11.toString(outputp) << " " << mod.f11 << endl;
  cout << "  f12  " << f12.toString(outputp) << " " << mod.f12 << endl;
  cout << "  f21  " << f21.toString(outputp) << " " << mod.f21 << endl;
  cout << "  f22  " << f22.toString(outputp) << " " << mod.f22 << endl;
  cout << "  f13  " << f13.toString(outputp) << " " << mod.f13 << endl;
  cout << "  f31  " << f31.toString(outputp) << " " << mod.f31 << endl;
  cout << "  f23  " << f23.toString(outputp) << " " << mod.f23 << endl;
  cout << "  f32  " << f32.toString(outputp) << " " << mod.f32 << endl;
  cout << "  f33  " << f33.toString(outputp) << " " << mod.f33 << endl;
  cout << "  f4   " << f4.toString(outputp) << " " << mod.f4 << endl;
  cout << "  f5   " << f5.toString(outputp) << " " << mod.f5 << endl;
  cout << "  f6   " << f6.toString(outputp) << " " << mod.f6 << endl;
  cout << "------------------------------" << endl;
  }


  // same
 Real det = a1*b2*c3;
  det *= -8*det*det*det;

  Real a12 = a1*a1;
  Real a13 = a1*a12;
  Real a14 = a12*a12;
  Real b12 = b1*b1;
  Real b22 = b2*b2;
  Real b23 = b22*b2;
  Real b24 = b22*b22;
  Real c22 = c2*c2;
  Real c32 = c3*c3;
  Real c33 = c32*c3;
  Real c34 = c32*c32;
  Real bc = (b2*c1-b1*c2);
  Real bc2 = bc*bc;

  Real Ai11 = -4*a12*b24*c34;
  Real Ai12 = 0;
  Real Ai13 = 0;
  Real Ai14 = 0;
  Real Ai15 = 0;
  Real Ai16 = 0;
  Real Ai21 = -4*a12*b12*b22*c34;
  Real Ai22 = 8*a13*b1*b22*c34;
  Real Ai23 = -4*a14*b22*c34;
  Real Ai24 = 0;
  Real Ai25 = 0;
  Real Ai26 = 0;
  Real Ai31 = -4*a12*b22*bc2*c32;
  Real Ai32 = -8*a13*b22*bc*c2*c32;
  Real Ai33 = -4*a14*b22*c22*c32;
  Real Ai34 = 8*a13*b23*bc*c32;
  Real Ai35 = 8*a14*b23*c2*c32;
  Real Ai36 = -4*a14*b24*c32;
  Real Ai41 = -8*a12*b1*b22*bc*c33;
  Real Ai42 = 8*a13*b22*(b2*c1 - 2*b1*c2)*c33;
  Real Ai43 = 8*a14*b22*c2*c33;
  Real Ai44 = 8*a13*b1*b23*c33;
  Real Ai45 = -8*a14*b23*c33;
  Real Ai46 = 0;
  Real Ai51 = 8*a12*b23*bc*c33;
  Real Ai52 = 8*a13*b23*c2*c33;
  Real Ai53 = 0;
  Real Ai54 = -8*a13*b24*c33;
  Real Ai55 = 0;
  Real Ai56 = 0;
  Real Ai61 = 8*a12*b1*b23*c34;
  Real Ai62 = -8*a13*b23*c34;
  Real Ai63 = 0;
  Real Ai64 = 0;
  Real Ai65 = 0;
  Real Ai66 = 0;

  Real D1 = (Ai11*(2*eiWW)    + Ai12*(eiWZ+eiZW) + Ai13*(2*eiZZ) +
             Ai14*(eiYW+eiWY) + Ai15*(eiYZ+eiZY) + Ai16*(2*eiYY) )/det;
  Real D2 = (Ai21*(2*eiWW)    + Ai22*(eiWZ+eiZW) + Ai23*(2*eiZZ) +
             Ai24*(eiYW+eiWY) + Ai25*(eiYZ+eiZY) + Ai26*(2*eiYY) )/det;
  Real D3 = (Ai31*(2*eiWW)    + Ai32*(eiWZ+eiZW) + Ai33*(2*eiZZ) +
             Ai34*(eiYW+eiWY) + Ai35*(eiYZ+eiZY) + Ai36*(2*eiYY) )/det;

  Real D4 = (Ai41*(2*eiWW)    + Ai42*(eiWZ+eiZW) + Ai43*(2*eiZZ) +
             Ai44*(eiYW+eiWY) + Ai45*(eiYZ+eiZY) + Ai46*(2*eiYY) )/det;
  Real D5 = (Ai51*(2*eiWW)    + Ai52*(eiWZ+eiZW) + Ai53*(2*eiZZ) +
             Ai54*(eiYW+eiWY) + Ai55*(eiYZ+eiZY) + Ai56*(2*eiYY) )/det;
  Real D6 = (Ai61*(2*eiWW)    + Ai62*(eiWZ+eiZW) + Ai63*(2*eiZZ) +
             Ai64*(eiYW+eiWY) + Ai65*(eiYZ+eiZY) + Ai66*(2*eiYY) )/det;

  Real D7sq = eii - ( 2*D1*D1 + 2*D2*D2 + 2*D3*D3 + D4*D4 + D5*D5 + D6*D6 );
  if( m_verbose )
    if ( D7sq < zero) cout << "Negative square for coefficient D7 = " << D7sq.toString(outputp) << endl;
  Real D7 = sqc(D7sq);
  if (abs(D7sq) < zerocut) D7=zero;


  if (abs(D1) < zerocut) mod.D1=0.; else mod.D1 = (double)D1;
  if (abs(D2) < zerocut) mod.D2=0.; else mod.D2 = (double)D2;
  if (abs(D3) < zerocut) mod.D3=0.; else mod.D3 = (double)D3;
  if (abs(D4) < zerocut) mod.D4=0.; else mod.D4 = (double)D4;
  if (abs(D5) < zerocut) mod.D5=0.; else mod.D5 = (double)D5;
  if (abs(D6) < zerocut) mod.D6=0.; else mod.D6 = (double)D6;
  if (abs(D7) < zerocut) mod.D7=0.; else mod.D7 = (double)D7;
  if (abs(D7sq) < zerocut) mod.D7sq=0.; else mod.D7sq = (double)D7sq;

  if( m_verbose ) {
  cout << "  D1   " << D1.toString(outputp) << " " << mod.D1 << endl;
  cout << "  D2   " << D2.toString(outputp) << " " << mod.D2 << endl;
  cout << "  D3   " << D3.toString(outputp) << " " << mod.D3 << endl;
  cout << "  D4   " << D4.toString(outputp) << " " << mod.D4 << endl;
  cout << "  D5   " << D5.toString(outputp) << " " << mod.D5 << endl;
  cout << "  D6   " << D6.toString(outputp) << " " << mod.D6 << endl;
  cout << "  D7   " << D7.toString(outputp) << " " << mod.D7 << endl;
  cout << "------------------------------" << endl;
  }

  Real O1 = (Ai11*(2*ejWW)    + Ai12*(ejWZ+ejZW) + Ai13*(2*ejZZ) +
             Ai14*(ejYW+ejWY) + Ai15*(ejYZ+ejZY) + Ai16*(2*ejYY) )/det;
  Real O2 = (Ai21*(2*ejWW)    + Ai22*(ejWZ+ejZW) + Ai23*(2*ejZZ) +
             Ai24*(ejYW+ejWY) + Ai25*(ejYZ+ejZY) + Ai26*(2*ejYY) )/det;
  Real O3 = (Ai31*(2*ejWW)    + Ai32*(ejWZ+ejZW) + Ai33*(2*ejZZ) +
             Ai34*(ejYW+ejWY) + Ai35*(ejYZ+ejZY) + Ai36*(2*ejYY) )/det;

  Real O4 = (Ai41*(2*ejWW)    + Ai42*(ejWZ+ejZW) + Ai43*(2*ejZZ) +
             Ai44*(ejYW+ejWY) + Ai45*(ejYZ+ejZY) + Ai46*(2*ejYY) )/det;
  Real O5 = (Ai51*(2*ejWW)    + Ai52*(ejWZ+ejZW) + Ai53*(2*ejZZ) +
             Ai54*(ejYW+ejWY) + Ai55*(ejYZ+ejZY) + Ai56*(2*ejYY) )/det;
  Real O6 = (Ai61*(2*ejWW)    + Ai62*(ejWZ+ejZW) + Ai63*(2*ejZZ) +
             Ai64*(ejYW+ejWY) + Ai65*(ejYZ+ejZY) + Ai66*(2*ejYY) )/det;

  Real O7 = (eij -( 2*D1*O1 + 2*D2*O2 + 2*D3*O3 + D4*O4 + D5*O5 + D6*O6 ))/
        D7;
  //O8 is always zero ?
  Real O8sq = ejj - ( 2*O1*O1 + 2*O2*O2 + 2*O3*O3 + O4*O4 + O5*O5 +  O6*O6 + O7*O7 );
  //if ((double) O8sq <0) cout << "Negative square for coefficient O8 = " << O8sq.toString(outputp) << endl;
  //Real O8 = sqc(O8sq);
  Real O8=0;

  if (abs(O1) < zerocut) mod.O1=0.; else mod.O1 = (double)O1;
  if (abs(O2) < zerocut) mod.O2=0.; else mod.O2 = (double)O2;
  if (abs(O3) < zerocut) mod.O3=0.; else mod.O3 = (double)O3;
  if (abs(O4) < zerocut) mod.O4=0.; else mod.O4 = (double)O4;
  if (abs(O5) < zerocut) mod.O5=0.; else mod.O5 = (double)O5;
  if (abs(O6) < zerocut) mod.O6=0.; else mod.O6 = (double)O6;
  if (abs(O7) < zerocut) mod.O7=0.; else mod.O7 = (double)O7;
  //if ((double) O8sq<0) mod.O8=0.; else if (fabs((double)O8) < zerocut) mod.O8=0.; else mod.O8 = (double)O8;
  mod.O8=0.;

  if( m_verbose ) {
  cout << "  O1   " << O1.toString(outputp) << " " << mod.O1 << endl;
  cout << "  O2   " << O2.toString(outputp) << " " << mod.O2 << endl;
  cout << "  O3   " << O3.toString(outputp) << " " << mod.O3 << endl;
  cout << "  O4   " << O4.toString(outputp) << " " << mod.O4 << endl;
  cout << "  O5   " << O5.toString(outputp) << " " << mod.O5 << endl;
  cout << "  O6   " << O6.toString(outputp) << " " << mod.O6 << endl;
  cout << "  O7   " << O7.toString(outputp) << " " << mod.O7 << endl;
  cout << "  O8   " << O8.toString(outputp) << " " << mod.O8 << endl;
  cout << "  O8sq " << O8sq.toString(outputp) << endl;
  cout << "------------------------------" << endl;
  }

  Real F1 = (Ai11*(2*eaWW)    + Ai12*(eaWZ+eaZW) + Ai13*(2*eaZZ) +
             Ai14*(eaYW+eaWY) + Ai15*(eaYZ+eaZY) + Ai16*(2*eaYY) )/det;
  Real F2 = (Ai21*(2*eaWW)    + Ai22*(eaWZ+eaZW) + Ai23*(2*eaZZ) +
             Ai24*(eaYW+eaWY) + Ai25*(eaYZ+eaZY) + Ai26*(2*eaYY) )/det;
  Real F3 = (Ai31*(2*eaWW)    + Ai32*(eaWZ+eaZW) + Ai33*(2*eaZZ) +
             Ai34*(eaYW+eaWY) + Ai35*(eaYZ+eaZY) + Ai36*(2*eaYY) )/det;
  // exactly zero?
  Real F4 = (Ai41*(2*eaWW)    + Ai42*(eaWZ+eaZW) + Ai43*(2*eaZZ) +
             Ai44*(eaYW+eaWY) + Ai45*(eaYZ+eaZY) + Ai46*(2*eaYY) )/det;
  // exactly zero?
  Real F5 = (Ai51*(2*eaWW)    + Ai52*(eaWZ+eaZW) + Ai53*(2*eaZZ) +
             Ai54*(eaYW+eaWY) + Ai55*(eaYZ+eaZY) + Ai56*(2*eaYY) )/det;
  // exactly zero?
  Real F6 = (Ai61*(2*eaWW)    + Ai62*(eaWZ+eaZW) + Ai63*(2*eaZZ) +
             Ai64*(eaYW+eaWY) + Ai65*(eaYZ+eaZY) + Ai66*(2*eaYY) )/det;
  Real F7 = (eia -( 2*D1*F1 + 2*D2*F2 + 2*D3*F3 + D4*F4 + D5*F5 + D6*F6))/
        D7;
  Real F8num = (eja -( 2*F1*O1 + 2*F2*O2 + 2*F3*O3 + F4*O4 + F5*O5 + F6*O6 +
         F7*O7));
  // cout << " F8 numerator " << F8num.toString(outputp) << endl;
  //Real F8 = (eja -( 2*F1*O1 + 2*F2*O2 + 2*F3*O3 + F4*O4 + F5*O5 + F6*O6 + F7*O7))/O8;
  Real F8=0;
  Real F9sq = eaa -( 2*F1*F1 + 2*F2*F2 + 2*F3*F3 + F4*F4 +
         F5*F5 + F6*F6 + F7*F7 + F8*F8);
  // if ((double)F9sq < 0) cout << "Negative square for coefficient F9 = " << F9sq.toString(outputp) << endl;
  Real F9 = sqc(F9sq);

  if (abs(F1) < zerocut) mod.F1=0.; else mod.F1 = (double)F1;
  if (abs(F2) < zerocut) mod.F2=0.; else mod.F2 = (double)F2;
  //  if (fabs((double)F3) < zerocut) mod.F3=0.; else mod.F3 = (double)F3;
  mod.F3 = 0.;
  // if (fabs((double)F4) < zerocut) mod.F4=0.; else mod.F4 = (double)F4;
  mod.F4 = 0.;
  //  if (fabs((double)F5) < zerocut) mod.F5=0.; else mod.F5 = (double)F5;
  mod.F5 = 0.;
  if (abs(F6) < zerocut) mod.F6=0.; else mod.F6 = (double)F6;
  //  if (fabs((double)F7) < zerocut) mod.F7=0.; else mod.F7 = (double)F7;
  mod.F7 = 0.;
  mod.F8 = 0;//(double)F8;
  //if ((double) F9sq<0) mod.F9=0.; else if (fabs((double)F9) < zerocut) mod.F9=0.; else mod.F9 = (double)F9;
  mod.F9=0.;

  if( m_verbose ) {
  cout << "  F1   " << F1.toString(outputp) << " " << mod.F1 << endl;
  cout << "  F2   " << F2.toString(outputp) << " " << mod.F2 << endl;
  cout << "  F3   " << F3.toString(outputp) << " " << mod.F3 << endl;
  cout << "  F4   " << F4.toString(outputp) << " " << mod.F4 << endl;
  cout << "  F5   " << F5.toString(outputp) << " " << mod.F5 << endl;
  cout << "  F6   " << F6.toString(outputp) << " " << mod.F6 << endl;
  cout << "  F7   " << F7.toString(outputp) << " " << mod.F7 << endl;
  cout << "  F8   " << F8.toString(outputp) << " " << mod.F8 << endl;
  cout << "  F9   " << F9.toString(outputp) << " " << mod.F9 << endl;
  cout << "------------------------------" << endl;
  }
}
  
// Box-Muller transform to make gaussian normal deviates
// from uniform deviates.

double normaldeviate()
{
  static int got_one = 0;
  static double extra_value;
  double v1, v2, tmp, rsq;

  if( got_one ) {
    got_one = 0;
    return extra_value;
  } else {
    do {
      // get 2 uniform deviates on [-1,1]
      // genrand_real1 returns number in [0,1]
      v1 = 2.*dsfmt_gv_genrand_close_open()-1.;
      v2 = 2.*dsfmt_gv_genrand_close_open()-1.;
      rsq = v1*v1 + v2*v2;
    } while( rsq==0 || rsq >= 1 );
    tmp = sqrt( - 2.*log(rsq)/rsq );
    extra_value = tmp*v2;
    got_one = 1;
    return tmp*v1;
  }
}
  
// build a list of W,Y,Z variables.
// N is number of particles times number of fine time steps
// dt is fine time step
// order:
//    d = dimension 0,1,2=D-1
//    p = particle number 0,...,P-1
//    s = step number 0,...,S-1
// W[ DPs + Dp + d ] = random number for dim d, part p, step s

void Rs( int WDim, int Nstep, double *W, double *Z, double *Y, double *I, double *J, double *aux, modelCoefficients& mod )
{

  // order in this way so dimension 0 sequence
  // independent of Dim, everything else equal.

  double *u1lst = new double[WDim];
  double *u2lst = new double[WDim];
  double *u3lst = new double[WDim];
  double *q1lst = new double[WDim*WDim];
  double *q2lst = new double[WDim*WDim];
  double *q3lst = new double[WDim*WDim];
  for( int nstep=0; nstep<Nstep; nstep++ ) {
    for( int alpha=0; alpha<WDim; alpha++ ) {
      int ibg = (WDim)*nstep + alpha;
      double u1 = u1lst[alpha] = normaldeviate();
      double u2 = u2lst[alpha] = normaldeviate();
      double u3 = u3lst[alpha] = normaldeviate();

      W [ibg] = mod.a1*u1;
      Z[ibg] = mod.b1*u1 + mod.b2*u2;
      Y[ibg] = mod.c1*u1 + mod.c2*u2 + mod.c3*u3;

    }

    for( int alpha=0; alpha<WDim; alpha++ ) 
      for( int beta=0; beta<WDim; beta++ ) 
	{
	  int iab = WDim*alpha + beta;
	  q1lst[iab] = normaldeviate();
	  q2lst[iab] = normaldeviate();
	  q3lst[iab] = normaldeviate();
	}
    for( int alpha=0; alpha<WDim; alpha++ ) {
      double u1a = u1lst[alpha];
      double u2a = u2lst[alpha];
      double u3a = u3lst[alpha];

      for( int beta=0; beta<WDim; beta++ ) {
	int ib = WDim*WDim*nstep + WDim*alpha + beta;
 	int iab = WDim*alpha + beta;

	double q1ab = q1lst[iab];
	double q2ab = q2lst[iab];
	double q3ab = q3lst[iab];

        if( alpha != beta ) {
          double u1b = u1lst[beta];
          double u2b = u2lst[beta];
          double u3b = u3lst[beta];

	  int iba = WDim*beta + alpha;
	  double q1ba = q1lst[iba];
	  double q2ba = q2lst[iba];
	  double q3ba = q3lst[iba];

	  //here I  is an integral of the type "int_0^t dW(t1) int_0^t1 exp(gam*t2) dW(t2)"
	  //     J  is an integral of the type "int_0^t exp(gam*t1) dW(t1) int_0^t1 dW(t2)"
	  //    aux is an integral of the type "int_0^t exp(gam*t1) dW(t1) int_0^t1 exp(gam*t2) dW(t2)"

          I[ib] = mod.d11*u1a*u1b + mod.d12*u1a*u2b + mod.d13*u1a*u3b
	      + mod.d21*u2a*u1b + mod.d22*u2a*u2b + mod.d23*u2a*u3b
	      + mod.d31*u3a*u1b + mod.d32*u3a*u2b + mod.d33*u3a*u3b
	    + mod.d4*(q1ab-q1ba);
          J[ib] = mod.o11*u1a*u1b + mod.o12*u1a*u2b + mod.o13*u1a*u3b
                + mod.o21*u2a*u1b + mod.o22*u2a*u2b + mod.o23*u2a*u3b
                + mod.o31*u3a*u1b + mod.o32*u3a*u2b + mod.o33*u3a*u3b
	    + mod.o4*(q1ab-q1ba) + mod.o5ab*q2ab + mod.o5ba*q2ba;
          aux[ib] = mod.f11*u1a*u1b + mod.f12*u1a*u2b + mod.f13*u1a*u3b
                  + mod.f21*u2a*u1b + mod.f22*u2a*u2b + mod.f23*u2a*u3b
                  + mod.f31*u3a*u1b + mod.f32*u3a*u2b + mod.f33*u3a*u3b
	    + mod.f4*(q1ab-q1ba) + mod.f5*(q2ab-q2ba) + mod.f6*(q3ab-q3ba);
        } else {

	  I[ib] = mod.D1*u1a*u1a + mod.D2*u2a*u2a + mod.D3*u3a*u3a
                + mod.D4*u3a*u2a + mod.D5*u3a*u1a + mod.D6*u2a*u1a
                + mod.D7*q1ab
                - mod.D1 - mod.D2 - mod.D3;

          J[ib] = mod.O1*u1a*u1a + mod.O2*u2a*u2a + mod.O3*u3a*u3a
                + mod.O4*u3a*u2a + mod.O5*u3a*u1a + mod.O6*u2a*u1a
                + mod.O7*q1ab + mod.O8*q2ab
                - mod.O1 - mod.O2 - mod.O3;

          aux[ib] = mod.F1*u1a*u1a + mod.F2*u2a*u2a + mod.F3*u3a*u3a
                  + mod.F4*u3a*u2a + mod.F5*u3a*u1a + mod.F6*u2a*u1a
                  + mod.F7*q1ab + mod.F8*q2ab + mod.F9*q3ab
                  - mod.F1 - mod.F2 - mod.F3;
        }
      }
    }
  }
  delete[] q3lst;
  delete[] q2lst;
  delete[] q1lst;
  delete[] u3lst;
  delete[] u2lst;
  delete[] u1lst;
}


void testRs( int Nstep, int WDim, double *W, double *Z, double *Y, double *I, double *J, double *aux,  expectations& ex, bool equal )
{
  int N = Nstep;

  if (equal) cout << "test alpha not equal beta -------------------------------" << endl; 
  else cout << "test alpha equal beta -------------------------------" << endl;

  cout << N << " points " << endl;

  double my_ii = 0;
  double my_iWW = 0;
  double my_iZW = 0;
  double my_iWZ = 0;
  double my_iZZ = 0;
  double my_iWY = 0;
  double my_iYW = 0;
  double my_iZY = 0;
  double my_iYZ = 0;
  double my_iYY = 0;

  double my_jj = 0;
  double my_ji = 0;
  double my_jWW = 0;
  double my_jZW = 0;
  double my_jWZ = 0;
  double my_jZZ = 0;
  double my_jWY = 0;
  double my_jYW = 0;
  double my_jZY = 0;
  double my_jYZ = 0;
  double my_jYY = 0;

  double my_aa = 0;
  double my_ia = 0;
  double my_ja = 0;
  double my_aWW = 0;
  double my_aZW = 0;
  double my_aWZ = 0;
  double my_aZZ = 0;
  double my_aWY = 0;
  double my_aYW = 0;
  double my_aZY = 0;
  double my_aYZ = 0;
  double my_aYY = 0;

  double test_zero    = 0.;
  double my_aa_zero   = 0.;
  double my_ia_zero   = 0.;
  double my_ja_zero   = 0.;
  double my_ii_zero   = 0.; 
  double my_jj_zero   = 0.;
  double my_ji_zero   = 0.;


  int cnt=0;
  for( int nstep=0; nstep<Nstep; nstep++ ) {
    for( int alpha=0; alpha<WDim; alpha++ ) {
      int ia = nstep*WDim + alpha;
      int iaa = nstep*WDim*WDim + alpha*WDim + alpha;
      if (equal){
	for( int beta=0; beta<WDim; beta++ ) {
	  int ib = nstep*WDim + beta;
	  if( alpha != beta ) {
	    int iab = nstep*WDim*WDim + alpha*WDim + beta;
	    int iba = nstep*WDim*WDim + beta*WDim + alpha;

	    my_ii  += I[iab]*I[iab]; 
	    my_iWW += I[iab]*W[ia]*W[ib];
	    my_iZW += I[iab]*Z[ia]*W[ib];
	    my_iWZ += I[iab]*W[ia]*Z[ib];
	    my_iZZ += I[iab]*Z[ia]*Z[ib];
	    my_iWY += I[iab]*W[ia]*Y[ib];
	    my_iYW += I[iab]*Y[ia]*W[ib];
	    my_iZY += I[iab]*Z[ia]*Y[ib];
	    my_iYZ += I[iab]*Y[ia]*Z[ib];;
	    my_iYY += I[iab]*Y[ia]*Y[ib];

	    my_jj  += J[iab]*J[iab];
	    my_ji  += J[iab]*I[iab];
	    my_jWW += J[iab]*W[ia]*W[ib];
	    my_jZW += J[iab]*Z[ia]*W[ib];
	    my_jWZ += J[iab]*W[ia]*Z[ib];
	    my_jZZ += J[iab]*Z[ia]*Z[ib];
	    my_jWY += J[iab]*W[ia]*Y[ib];
	    my_jYW += J[iab]*Y[ia]*W[ib];
	    my_jZY += J[iab]*Z[ia]*Y[ib];
	    my_jYZ += J[iab]*Y[ia]*Z[ib];;
	    my_jYY += J[iab]*Y[ia]*Y[ib];
	    
	    my_aa  += aux[iab]*aux[iab];
	    my_ia  += aux[iab]*I[iab];
	    my_ja  += aux[iab]*J[iab];
	    my_aWW += aux[iab]*W[ia]*W[ib];
	    my_aZW += aux[iab]*Z[ia]*W[ib];
	    my_aWZ += aux[iab]*W[ia]*Z[ib];
	    my_aZZ += aux[iab]*Z[ia]*Z[ib];
	    my_aWY += aux[iab]*W[ia]*Y[ib];
	    my_aYW += aux[iab]*Y[ia]*W[ib];
	    my_aZY += aux[iab]*Z[ia]*Y[ib];
	    my_aYZ += aux[iab]*Y[ia]*Z[ib];;
	    my_aYY += aux[iab]*Y[ia]*Y[ib];

	    my_aa_zero  += aux[iab]*aux[iba];
	    my_ia_zero  += aux[iab]*I[iba];
	    my_ja_zero  += aux[iab]*J[iba];
	    my_ii_zero  += I[iab]*I[iba]; 
	    my_jj_zero  += J[iab]*J[iba];
	    my_ji_zero  += J[iab]*I[iba];

	    test_zero += normaldeviate();
	    
	    cnt++;
	  }
      }
      } else 
	{
          my_ii  += I[iaa]*I[iaa];
          my_iWW += I[iaa]*W[ia]*W[ia];
          my_iZW += I[iaa]*Z[ia]*W[ia];
          my_iZZ += I[iaa]*Z[ia]*Z[ia];
	  my_iYW += I[iaa]*Y[ia]*W[ia];
	  my_iYZ += I[iaa]*Y[ia]*Z[ia];;
	  my_iYY += I[iaa]*Y[ia]*Y[ia];
	  
          my_jj  += J[iaa]*J[iaa];
          my_ji  += J[iaa]*I[iaa];
          my_jWW += J[iaa]*W[ia]*W[ia];
          my_jZW += J[iaa]*Z[ia]*W[ia];
          my_jZZ += J[iaa]*Z[ia]*Z[ia];
	  my_jYW += J[iaa]*Y[ia]*W[ia];
	  my_jYZ += J[iaa]*Y[ia]*Z[ia];;
	  my_jYY += J[iaa]*Y[ia]*Y[ia];
	  
          my_aa  += aux[iaa]*aux[iaa];
          my_ia  += aux[iaa]*I[iaa];
          my_ja  += aux[iaa]*J[iaa];
          my_aWW += aux[iaa]*W[ia]*W[ia];
          my_aZW += aux[iaa]*Z[ia]*W[ia];
          my_aZZ += aux[iaa]*Z[ia]*Z[ia];
	  my_aYW += aux[iaa]*Y[ia]*W[ia];
	  my_aYZ += aux[iaa]*Y[ia]*Z[ia];;
	  my_aYY += aux[iaa]*Y[ia]*Y[ia];


          cnt++;
	}
    }
  }
  my_ji  /= ((double)cnt);
  my_ii  /= ((double)cnt);
  my_jj  /= ((double)cnt);
  my_aa  /= ((double)cnt);
  my_ia  /= ((double)cnt);
  my_ja  /= ((double)cnt);

  my_iWW /= ((double)cnt);
  my_iZW /= ((double)cnt);
  my_iWZ /= ((double)cnt);
  my_iZZ /= ((double)cnt);
  my_iWY /= ((double)cnt);
  my_iYW /= ((double)cnt);
  my_iZY /= ((double)cnt);
  my_iYZ /= ((double)cnt);
  my_iYY /= ((double)cnt);

 
  my_jWW /= ((double)cnt);
  my_jZW /= ((double)cnt);
  my_jWZ /= ((double)cnt);
  my_jZZ /= ((double)cnt);
  my_jWY /= ((double)cnt);
  my_jYW /= ((double)cnt);
  my_jZY /= ((double)cnt);
  my_jYZ /= ((double)cnt);
  my_jYY /= ((double)cnt);

  my_aWW /= ((double)cnt);
  my_aZW /= ((double)cnt);
  my_aWZ /= ((double)cnt);
  my_aZZ /= ((double)cnt);
  my_aWY /= ((double)cnt);
  my_aYW /= ((double)cnt);
  my_aZY /= ((double)cnt);
  my_aYZ /= ((double)cnt);
  my_aYY /= ((double)cnt);

  test_zero  /= ((double)cnt);
  my_aa_zero  /= ((double)cnt);
  my_ia_zero  /= ((double)cnt);
  my_ja_zero  /= ((double)cnt);
  my_ii_zero  /= ((double)cnt); 
  my_jj_zero  /= ((double)cnt);
  my_ji_zero  /= ((double)cnt);



  //testing W,Z,Y correlations
  
  double my_ZZ = 0;
  double my_WW = 0;
  double my_ZW = 0;
  double my_YW = 0;
  double my_YZ = 0;
  double my_YY = 0;
  
  
  for( int i=0; i<N*WDim; i++ ) {
    my_ZZ += Z[i]*Z[i];
    my_WW += W[i]*W[i];
    my_ZW += Z[i]*W[i];
    my_YW += Y[i]*W[i];
    my_YZ += Y[i]*Z[i];
    my_YY += Y[i]*Y[i];
    }
  my_ZZ /= ((double)(N*WDim));
  my_WW /= ((double)(N*WDim));
  my_ZW /= ((double)(N*WDim));
  my_YW /= ((double)(N*WDim));
  my_YZ /= ((double)(N*WDim));
  my_YY /= ((double)(N*WDim));
  
  cout << "expectations that should not be zero" << endl;
  cout << "WW " << my_WW << " (vs " << ex.eWW << ")" << endl;
  cout << "ZZ " << my_ZZ << " (vs " << ex.eZZ << ")" << endl;
  cout << "ZW " << my_ZW << " (vs " << ex.eWZ << ")" << endl;
  cout << "YW " << my_YW << " (vs " << ex.eWY << ")" << endl;
  cout << "YZ " << my_YZ << " (vs " << ex.eZY << ")" << endl;
  cout << "YY " << my_YY << " (vs " << ex.eYY << ")" << endl;
  cout << "scaled WW " << my_WW/ex.eWW << " (" << 100*(my_WW/ex.eWW-1) << "\%)" << endl;
  cout << "scaled ZZ " << my_ZZ/ex.eZZ << " (" << 100*(my_ZZ/ex.eZZ-1) << "\%)" << endl;
  cout << "scaled ZW " << my_ZW/ex.eWZ << " (" << 100*(my_ZW/ex.eWZ-1) << "\%)" << endl;
  cout << "scaled YW " << my_YW/ex.eWY << " (" << 100*(my_YW/ex.eWY-1) << "\%)" << endl;
  cout << "scaled YZ " << my_YZ/ex.eZY << " (" << 100*(my_YZ/ex.eZY-1) << "\%)" << endl;
  cout << "scaled YY " << my_YY/ex.eYY << " (" << 100*(my_YY/ex.eYY-1) << "\%)" << endl;
  
  cout << "----------------------------------" << endl;
  /*
  double my_eZZ[MAXDIM][MAXDIM];
  double my_eWW[MAXDIM][MAXDIM];
  double my_eZW[MAXDIM][MAXDIM];
  double my_eWZ[MAXDIM][MAXDIM];
  
  for( int idir=0; idir<Dim; idir++ )
    for( int jdir=0; jdir<Dim; jdir++ ) {
      my_eZZ[idir][jdir] = 0;
      my_eWW[idir][jdir] = 0;
      my_eZW[idir][jdir] = 0;
      my_eWZ[idir][jdir] = 0;
	
      
      cnt=0;
      for( int i=0; i<N*Dim; i++ ) {
	int idir = i%Dim;
	int ipart=i/Dim;
	for( int j=0; j<=0; j++ ) {
	  int jdir = j%Dim;
	    int jpart=j/Dim;
	    if( ipart != jpart ) {
	      cnt++;
	      my_eZZ[idir][jdir] += Z[i]*Z[j];
	      my_eWW[idir][jdir] += W[i]*W[j];
	      my_eZW[idir][jdir] += Z[i]*W[j];
	      my_eWZ[idir][jdir] += W[i]*Z[j];
	    }
	  }
      }
      for( int idir=0; idir<Dim; idir++ ) {
	for( int jdir=0; jdir<Dim; jdir++ ) {
	  my_eZZ[idir][jdir] /= ((double)cnt);
	  my_eWW[idir][jdir] /= ((double)cnt);
	  my_eZW[idir][jdir] /= ((double)cnt);
	  my_eWZ[idir][jdir] /= ((double)cnt);
	  }
      }
      cout << "expectations that should be zero" << endl;
      cout << "ZZ ";
      for( int idir=0; idir<Dim; idir++ )
	for( int jdir=0; jdir<Dim; jdir++ )
	    cout << " " << my_eZZ[idir][jdir];
      cout << endl;
      cout << "WW ";
      for( int idir=0; idir<Dim; idir++ )
	for( int jdir=0; jdir<Dim; jdir++ )
	    cout << " " << my_eWW[idir][jdir];
      cout << endl;
      cout << "ZW ";
      for( int idir=0; idir<Dim; idir++ )
	for( int jdir=0; jdir<Dim; jdir++ )
	  cout << " " << my_eZW[idir][jdir];
      cout << endl;
      cout << "WZ ";
      for( int idir=0; idir<Dim; idir++ )
	for( int jdir=0; jdir<Dim; jdir++ )
	  cout << " " << my_eWZ[idir][jdir];
      cout << endl;
      cout << "scaled ZZ ";
      for( int idir=0; idir<Dim; idir++ )
	for( int jdir=0; jdir<Dim; jdir++ )
	  cout << " " << my_eZZ[idir][jdir]/ex.eZZ;
      cout << endl;
      cout << "scaled WW ";
      for( int idir=0; idir<Dim; idir++ )
	for( int jdir=0; jdir<Dim; jdir++ )
	  cout << " " << my_eWW[idir][jdir]/ex.eWW;
      cout << endl;
      cout << "scaled ZW ";
      for( int idir=0; idir<Dim; idir++ )
	for( int jdir=0; jdir<Dim; jdir++ )
	  cout << " " << my_eZW[idir][jdir]/ex.eWZ;
      cout << endl;
      cout << "scaled WZ ";
      for( int idir=0; idir<Dim; idir++ )
	for( int jdir=0; jdir<Dim; jdir++ )
	  cout << " " << my_eWZ[idir][jdir]/ex.eWZ;
      cout << endl;
    }
  
  cout << "----------------------------------" << endl;
  
  */  
  if (equal) {
    
    
    // next 20 or so lines only for alpha!=beta
    
    cout << "II  " << my_ii << " (vs " << ex.eii << ")" << endl;
    cout << "IWW " << my_iWW << " (vs " << ex.eiWW << ")" << endl;
    cout << "IWZ " << my_iWZ << " (vs " << ex.eiWZ << ")" << endl;
    cout << "IZW " << my_iZW << " (vs " << ex.eiZW << ")" << endl;
    cout << "IZZ " << my_iZZ << " (vs " << ex.eiZZ << ")" << endl;
    cout << "IWY " << my_iWY << " (vs " << ex.eiWY << ")" << endl;
    cout << "IYW " << my_iYW << " (vs " << ex.eiYW << ")" << endl;
    cout << "IZY " << my_iZY << " (vs " << ex.eiZY << ")" << endl;
    cout << "IYZ " << my_iYZ << " (vs " << ex.eiYZ << ")" << endl;
    cout << "IYY " << my_iYY << " (vs " << ex.eiYY << ")" << endl;
    
    cout << "JJ  " << my_jj << " (vs " << ex.ejj << ")" << endl;
    cout << "JI  " << my_ji << " (vs " << ex.eij << ")" << endl;
    cout << "JWW " << my_jWW << " (vs " << ex.ejWW << ")" << endl;
    cout << "JWZ " << my_jWZ << " (vs " << ex.ejWZ << ")" << endl;
    cout << "JZW " << my_jZW << " (vs " << ex.ejZW << ")" << endl;
    cout << "JZZ " << my_jZZ << " (vs " << ex.ejZZ << ")" << endl;
    cout << "JWY " << my_jWY << " (vs " << ex.ejWY << ")" << endl;
    cout << "JYW " << my_jYW << " (vs " << ex.ejYW << ")" << endl;
    cout << "JZY " << my_jZY << " (vs " << ex.ejZY << ")" << endl;
    cout << "JYZ " << my_jYZ << " (vs " << ex.ejYZ << ")" << endl;
    cout << "JYY " << my_jYY << " (vs " << ex.ejYY << ")" << endl;
    
    cout << "AA  " << my_aa << " (vs " << ex.eaa << ")" << endl;
    cout << "AJ  " << my_ia << " (vs " << ex.eia << ")" << endl;
    cout << "AI  " << my_ja << " (vs " << ex.eja << ")" << endl;
    cout << "AWW " << my_aWW << " (vs " << ex.eaWW << ")" << endl;
    cout << "AZW " << my_aZW << " (vs " << ex.eaZW << ")" << endl;
    cout << "AWZ " << my_aWZ << " (vs " << ex.eaWZ << ")" << endl;
    cout << "AZZ " << my_aZZ << " (vs " << ex.eaZZ << ")" << endl;
    cout << "AWY " << my_aWY << " (vs " << ex.eaWY << ")" << endl;
    cout << "AYW " << my_aYW << " (vs " << ex.eaYW << ")" << endl;
    cout << "AZY " << my_aZY << " (vs " << ex.eaZY << ")" << endl;
    cout << "AYZ " << my_aYZ << " (vs " << ex.eaYZ << ")" << endl;
    cout << "AYY " << my_aYY << " (vs " << ex.eaYY << ")" << endl;

    
    
    double r;
    r=my_ii/ex.eii;             cout << "scaled II  " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_iWW/ex.eiWW;           cout << "scaled IWW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_iZW/ex.eiZW;           cout << "scaled IZW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_iWZ/ex.eiWZ;           cout << "scaled IWZ " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_iZZ/ex.eiZZ;           cout << "scaled IZZ " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_iWY/ex.eiWY;           cout << "scaled IWY " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_iYW/ex.eiYW;           cout << "scaled IYW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_iZY/ex.eiZY;           cout << "scaled IZY " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_iYZ/ex.eiYZ;           cout << "scaled IYZ " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_iYY/ex.eiYY;           cout << "scaled IYY " << r << "(" << 100*(r-1) << "\%)" << endl;
    
    r=my_jj/ex.ejj;             cout << "scaled JJ  " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_ji/ex.eij;             cout << "scaled JI  " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_jWW/ex.ejWW;           cout << "scaled JWW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_jZW/ex.ejZW;           cout << "scaled JZW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_jWZ/ex.ejWZ;           cout << "scaled JWZ " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_jZZ/ex.ejZZ;           cout << "scaled JZZ " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_jWY/ex.ejWY;           cout << "scaled JWY " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_jYW/ex.ejYW;           cout << "scaled JYW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_jZY/ex.ejZY;           cout << "scaled JZY " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_jYZ/ex.ejYZ;           cout << "scaled JYZ " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_jYY/ex.ejYY;           cout << "scaled JYY " << r << "(" << 100*(r-1) << "\%)" << endl;
    
    r=my_aa/ex.eaa;             cout << "scaled AA  " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_ja/ex.eja;             cout << "scaled AJ  " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_ia/ex.eia;             cout << "scaled AI  " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_aWW/ex.eaWW;           cout << "scaled AWW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_aZW/ex.eaZW;           cout << "scaled AZW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_aWZ/ex.eaWZ;           cout << "scaled AWZ " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_aZZ/ex.eaZZ;           cout << "scaled AZZ " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_aWY/ex.eaWY;           cout << "scaled AWY " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_aYW/ex.eaYW;           cout << "scaled AYW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_aZY/ex.eaZY;           cout << "scaled AZY " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_aYZ/ex.eaYZ;           cout << "scaled AYZ " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_aYY/ex.eaYY;           cout << "scaled AYY " << r << "(" << 100*(r-1) << "\%)" << endl;

    cout << "--------------------------------------" << endl;
    cout << "Expectations that should be zero" << endl;
    cout << "Test zero is equal " <<  test_zero << endl;

    cout << "AA  " << my_aa_zero << endl;
    cout << "AI  " << my_ia_zero << endl;
    cout << "AJ  " << my_ja_zero << endl;
    cout << "II  " << my_ii_zero << endl;
    cout << "JJ  " << my_jj_zero << endl;
    cout << "JI  " << my_ji_zero << endl;



    
  }else {
    
    cout << "II  " << my_ii << " (vs " << ex.eii << ")" << endl;
    cout << "IWW " << my_iWW << " (vs " << 2*ex.eiWW << ")" << endl;
    cout << "IZW " << my_iZW << " (vs " << (ex.eiZW+ex.eiWZ) << ")" << endl;
    cout << "IZZ " << my_iZZ << " (vs " << 2*ex.eiZZ << ")" << endl;
    cout << "IYW " << my_iYW << " (vs " << (ex.eiYW+ex.eiWY) << ")" << endl;
    cout << "IYZ " << my_iYZ << " (vs " << (ex.eiYZ+ex.eiZY) << ")" << endl;
    cout << "IYY " << my_iYY << " (vs " << 2*ex.eiYY << ")" << endl;
    
    cout << "JJ  " << my_jj << " (vs " << ex.ejj << ")" << endl;
    cout << "JI  " << my_ji << " (vs " << ex.eij << ")" << endl;
    cout << "JWW " << my_jWW << " (vs " << 2*ex.ejWW << ")" << endl;
    cout << "JZW " << my_jZW << " (vs " << (ex.ejZW+ex.ejWZ) << ")" << endl;
    cout << "JZZ " << my_jZZ << " (vs " << 2*ex.ejZZ << ")" << endl;
    cout << "JYW " << my_jYW << " (vs " << (ex.ejYW+ex.ejWY) << ")" << endl;
    cout << "JYZ " << my_jYZ << " (vs " << (ex.ejYZ+ex.ejZY) << ")" << endl;
    cout << "JYY " << my_jYY << " (vs " << 2*ex.ejYY << ")" << endl;
    
    cout << "AA  " << my_aa << " (vs " << ex.eaa << ")" << endl;
    cout << "AJ  " << my_ia << " (vs " << ex.eia << ")" << endl;
    cout << "AI  " << my_ja << " (vs " << ex.eja << ")" << endl;
    cout << "AWW " << my_aWW << " (vs " << 2*ex.eaWW << ")" << endl;
    cout << "AZW " << my_aZW << " (vs " << (ex.eaZW+ex.eaWZ) << ")" << endl;
    cout << "AZZ " << my_aZZ << " (vs " << 2*ex.eaZZ << ")" << endl;
    cout << "AYW " << my_aYW << " (vs " << (ex.eaYW+ex.eaWY) << ")" << endl;
    cout << "AYZ " << my_aYZ << " (vs " << (ex.eaYZ+ex.eaZY) << ")" << endl;
    cout << "AYY " << my_aYY << " (vs " << 2*ex.eaYY << ")" << endl;
  
    double r;
    r=my_ii/ex.eii;             cout << "scaled II  " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_iWW/(2*ex.eiWW);       cout << "scaled IWW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_iZW/(ex.eiZW+ex.eiWZ); cout << "scaled IZW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_iZZ/(2*ex.eiZZ);       cout << "scaled IZZ " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_iYW/(ex.eiYW+ex.eiWY); cout << "scaled IYW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_iYZ/(ex.eiYZ+ex.eiZY); cout << "scaled IYZ " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_iYY/(2*ex.eiYY);       cout << "scaled IYY " << r << "(" << 100*(r-1) << "\%)" << endl;

    r=my_jj/ex.ejj;             cout << "scaled JJ  " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_ji/ex.eij;             cout << "scaled JI  " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_jWW/(2*ex.ejWW);       cout << "scaled JWW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_jZW/(ex.ejZW+ex.ejWZ); cout << "scaled JZW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_jZZ/(2*ex.ejZZ);       cout << "scaled JZZ " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_jYW/(ex.ejYW+ex.ejWY); cout << "scaled JYW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_jYZ/(ex.ejYZ+ex.ejZY); cout << "scaled JYZ " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_jYY/(2*ex.ejYY);       cout << "scaled JYY " << r << "(" << 100*(r-1) << "\%)" << endl;
    
    r=my_aa/ex.eaa;             cout << "scaled AA  " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_ja/ex.eja;             cout << "scaled AJ  " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_ia/ex.eia;             cout << "scaled AI  " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_aWW/(2*ex.eaWW);       cout << "scaled AWW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_aZW/(ex.eaZW+ex.eaWZ); cout << "scaled AZW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_aZZ/(2*ex.eaZZ);       cout << "scaled AZZ " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_aYW/(ex.eaYW+ex.eaWY); cout << "scaled AYW " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_aYZ/(ex.eaYZ+ex.eaZY); cout << "scaled AYZ " << r << "(" << 100*(r-1) << "\%)" << endl;
    r=my_aYY/(2*ex.eaYY);       cout << "scaled AYY " << r << "(" << 100*(r-1) << "\%)" << endl;
    
  }
  cout << "***********end of the current resolution*****************************" << endl;
}


void coarsenRs( int n_f, int WDim, double *W, double *Z, double *Y, double *I, double *J, double *aux, double h_f, double gam )
{
  // n_f is the number of fine steps

  double e=exp(-gam*h_f);
  double e2=exp(-2*gam*h_f);
  int n_c = n_f/2;
  int dof = WDim;

  int c_base = 0;
  int f_base1 = 0;
  int f_base2 = dof;

  int c_sq_base = 0;
  int f_sq_base1 = 0;
  int f_sq_base2 = dof*dof;

  for( int i_c=0; i_c<n_c; i_c++ ) {
    for( int alpha=0; alpha<dof; alpha++ ) {
      for( int beta=0; beta<dof; beta++ ) {
	double W_beta_1 = W[f_base1+beta];
        double rZ_beta_1 = Z[f_base1+beta];
        double W_alpha_2 = W[f_base2+alpha];
        double rZ_alpha_2 = Z[f_base2+alpha];
        double i1_f = I[f_sq_base1 + dof*alpha + beta];
        double i2_f = I[f_sq_base2 + dof*alpha + beta];
        double j1_f = J[f_sq_base1 + dof*alpha + beta];
        double j2_f = J[f_sq_base2 + dof*alpha + beta];
        double aux1_f = aux[f_sq_base1 + dof*alpha + beta];
        double aux2_f = aux[f_sq_base2 + dof*alpha + beta];

        I[c_sq_base + dof*alpha + beta] = e*i1_f + i2_f + e*W_alpha_2*rZ_beta_1;
	      
        J[c_sq_base + dof*alpha + beta] = e*j1_f + j2_f + rZ_alpha_2*W_beta_1;
 
	aux[c_sq_base + dof*alpha + beta] = e2*aux1_f + aux2_f + e*rZ_alpha_2*rZ_beta_1;
      }
    }
    c_sq_base  += dof*dof;
    f_sq_base1 += 2*dof*dof;
    f_sq_base2 += 2*dof*dof;

    for( int ivar=0; ivar<dof; ivar++ ) {

      // the two fine-h W terms
      double rW1_f = W[f_base1+ivar];
      double rW2_f = W[f_base2+ivar];
      // the two fine-h Z terms
      double rZ1_f = Z[f_base1+ivar];
      double rZ2_f = Z[f_base2+ivar];
      // the two fine-h Y terms
      double rY1_f = Y[f_base1+ivar];
      double rY2_f = Y[f_base2+ivar];

      // build the coarse-h (2* fine-h) R,W terms
      W[c_base+ivar] = rW1_f + rW2_f;
      Z[c_base+ivar] = e*rZ1_f + rZ2_f ;
      Y[c_base+ivar] = e*rY1_f + rY2_f + h_f*rZ2_f;
      
    }
    c_base  += dof;
    f_base1 += 2*dof;
    f_base2 += 2*dof;
  }
}
