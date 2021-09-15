#ifndef Helicity_hh
#define Helicity_hh

#include <iostream>



/// Code containing the partial wave decomposition
/// for 2-body nuclear potential in helicity formalism.
/// This is the implementation of eq 4.20 to eq 4.24 from
/// K. Erkelenz, R. Alzetta, and K. Holinde, Nucl. Phys. A 176, 413 (1971).
/// as well as the gerenaliztion for tensor potentials.
/// Implemented by Antoine Belley
/// Date : 09/2021

namespace Helicity{
  int phase(int x);
  // PWD of scalar 2-body potentials
  double AIntegrand(double p, double pp, int J, int l, std::function<double(double,double,double)> potential);
  double A(double p, double pp, int J, int l, std::function<double(double,double,double)> potential, int size);
  double central_force_decomposition(double p, double pp, int S, int L, int Lp, int J,  std::function<double(double,double,double)> potential, int size);
  double spin_spin_decomposition(double p, double pp, int S, int L, int Lp, int J,  std::function<double(double,double,double)> potential, int size);
  double spin_orbit_decomposition(double p, double pp, int S, int L, int Lp, int J,  std::function<double(double,double,double)> potential, int size);
  double sigma_L_decomposition(double p, double pp, int S, int L, int Lp, int J,  std::function<double(double,double,double)> potential, int size);
  double tensor_decomposition(double p, double pp, int S, int L, int Lp, int J,  std::function<double(double,double,double)> potential, int size);
  double sigma_k_decomposition(double p, double pp, int S, int L, int Lp, int J,  std::function<double(double,double,double)> potential, int size);
  double scalar_partial_wave_decomposition(double p, double pp, int S, int L, int Lp, int J,  std::function<double(double,double,double)> potential, int size, std::string potential_type);
  // Generalization to tensor potentials
  double binomial_coefficient(int n, int k);
  double JacobiPolynomials(int n, int a, int b, double z);
  double wigner_small_d_matrix(int j, int m1, int m2, double theta);
  double integrate_op_rotation (double p, double pp, std::function<double(double, double, double,int,int,int,int,int)> op_fm, int Jp, int operator_index, double lam1, double lam2, double lamp1, double lamp2, double Lam, double Lamp);
  double projection_to_j_basis(double p, double pp, int J, int Jp, double lam1, double lam2, double lamp1, double lamp2, double Lam, double Lamp,  std::function<double(double, double, double, int,int,double,double,double,double)> op, int operator_rank);
  double overlap_lsj(int S, int L, int J, double lam1, double lam2, double Lam);
  double tensor_partial_wave_decomposition(double p, double pp, int L, int Lp, int S, int Sp, int J, int Jp,  std::function<double(double, double, double, int,int,double,double,double,double)> op, int operator_rank);
  //Helicity functions, useful to write operators in helicity formalism
  double helicity_expectation_value_identity(double lam1, double lam2,  double z);
  double helicity_expectation_value_x(double lam1, double lam2, double z);
  double helicity_expectation_value_y(double lam1, double lam2, double z);
  double helicity_expectation_value_z(double lam1, double lam2, double z);
  double helicity_expectation_value_sigma(double lam1, double lam2, double z, int m);
  double identity_helicity_representaition(double lam1, double lam2, double lam3, double lam4, double z);
  double sigma_dot_sigma(double lam1, double lam2, double lam3, double lam4, double z);
  double sigma_plus_sigma_dot_q(double lam1, double lam2, double lam3, double lam4, double p_out, double p_in, double z);
  double sigma_plus_sigma_dot_pout_plus_pin(double lam1, double lam2, double lam3, double lam4, double p_out, double p_in, double z);
  double sigma_minus_sigma_dot_q(double lam1, double lam2, double lam3, double lam4, double p_out, double p_in, double z);
  double sigma_minus_sigma_dot_pout_plus_pin(double lam1, double lam2, double lam3, double lam4, double p_out, double p_in, double z);
  double sigma_cross_sigma_dot_q(double lam1, double lam2, double lam3, double lam4, double p_out, double p_in, double z);
  double sigma_cross_sigma_dot_pout_plus_pin(double lam1, double lam2, double lam3, double lam4, double p_out, double p_in, double z);
  double sigma_dot_q_sigma_dot_q(double lam1, double lam2, double lam3, double lam4, double p_out, double p_in, double z);
  double test_operator_helicity_basis(double p, double pp, double z, int operator_index, int operator_rank, double lam1, double lam2, double lamp1, double lamp2);
}

#endif