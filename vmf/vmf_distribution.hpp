#ifndef VMF_DISTRIBUTION_HPP
#define VMF_DISTRIBUTION_HPP

void vmf_parameters_estimation(double * data, double * mu, double &kappa, int N, int dimension);
void vmf_parameters_estimation(double * data, double * mu, double &kappa, int N, int dimension, double a_constraint);

void vmf_sampling(double * data, double * mu, double kappa, int N, int dimension);
void vmf_sampling(double * data, double * mu, double kappa, int N, int dimension, double a_constraint);

#endif