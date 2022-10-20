#ifndef PWD_h
#define PWD_h 1

#include "Helicity.hh"
#include <iostream>
#include <unordered_map>
#include <gsl/gsl_integration.h>
#include <unordered_map>
#include <map>

class PWD
{
  protected:
    int momentum_mesh_size;
    gsl_integration_glfixed_table *momentum_mesh;
    int angular_mesh_size;
    gsl_integration_glfixed_table *angular_mesh;
    double max_momentum;

    const char *types[6] = {"central", "spin-spin", "spin-orbit", "sigma_L", "tensor", "sigma_k"};
    std::function<double(double, double, double)> potential_central;
    std::function<double(double, double, double)> potential_spin_spin;
    std::function<double(double, double, double)> potential_spin_orbit;
    std::function<double(double, double, double)> potential_sigma_L;
    std::function<double(double, double, double)> potential_tensor;
    std::function<double(double, double, double)> potential_sigma_k;
    std::map<std::string, std::function<double(double, double, double)>> potential_list = {{"central", potential_central},
                                                                                           {"spin-spin", potential_spin_spin},
                                                                                           {"spin-orbit", potential_spin_orbit},
                                                                                           {"sigma_L", potential_sigma_L},
                                                                                           {"tensor", potential_tensor},
                                                                                           {"sigma_k", potential_sigma_k}};
    bool central_on = false;
    bool spin_spin_on = false;
    bool spin_orbit_on = false;
    bool sigma_L_on = false;
    bool tensor_on = false;
    bool sigma_k_on = false;
    std::map<std::string, bool> potential_bool = {{"central", central_on},
                                                  {"spin-spin", spin_spin_on},
                                                  {"spin-orbit", spin_orbit_on},
                                                  {"sigma_L", sigma_L_on},
                                                  {"tensor", tensor_on},
                                                  {"sigma_k", sigma_k_on}};
    std::unordered_map<uint64_t, double> AList;

  public:
    void initializeMomentumMesh(int size);
    void setMomentumMesh(gsl_integration_glfixed_table *t, int size);
    int getMomentumMeshSize();
    gsl_integration_glfixed_table *getMomentumMesh();
     void freeMomentumMesh();
     
    void initializeAngularMesh(int size);
    void setAngularMesh(gsl_integration_glfixed_table *momentum_mesh, int size);
    int getAngularMeshSize();
    gsl_integration_glfixed_table *getAngularMesh();
    void freeAngularMesh();

    void setMaxMomentum(double max);
    double getMaxMomentum();

    void setPotential(std::function<double(double, double, double)> potential_func,  std::string potential_type);

    void calcA(int e2max, int lmax);
    void clearA();

    double getW(double p, double pp, int index_p, int index_pp, int S, int L, int Lp, int J);

    PWD();
};




class PWDNonLocal : public PWD
{
  private:
    std::function<double(double, double)> regulator;

  public:
    void setRegulator(double regulator_cutoff, int regulator_power);
    double getW(double p, double pp, int index_p, int index_pp, int S, int L, int Lp, int J);
    PWDNonLocal();
};

double AIntegrand(double p, double pp, int J, int l, std::function<double(double,double,double)> potential);
double A(double p, double pp, int J, int l, std::function<double(double, double, double)> potential, gsl_integration_glfixed_table *t, int z_size);
uint64_t AHash(int index_p, int index_pp, int J, int l, int type);
void AUnHash(uint64_t key, uint64_t &index_p, uint64_t &index_pp, uint64_t &J, uint64_t &l, uint64_t &type);
void PrecalculateA(int e2max, std::function<double(double, double, double)> potential, int lmax, int type, double max_momentum, gsl_integration_glfixed_table *t_momentum, gsl_integration_glfixed_table *t_z, int n_momentum_points, int n_z_points, std::unordered_map<uint64_t, double>& AList);
double GetA(int index_p, int index_pp, int J, int l, int type, std::unordered_map<uint64_t, double> &AList);
double central_force_decomposition(double p, double pp, int index_p, int index_pp, int S, int L, int Lp, int J, std::unordered_map<uint64_t, double> &AList);
double spin_spin_decomposition(double p, double pp, int index_p, int index_pp, int S, int L, int Lp, int J, std::unordered_map<uint64_t, double> &AList);
double spin_orbit_decomposition(double p, double pp, int index_p, int index_pp, int S, int L, int Lp, int J, std::unordered_map<uint64_t, double> &AList);
double sigma_L_decomposition(double p, double pp, int index_p, int index_pp, int S, int L, int Lp, int J,  std::unordered_map<uint64_t, double> &AList);
double tensor_decomposition(double p, double pp, int index_p, int index_pp, int S, int L, int Lp, int J, std::unordered_map<uint64_t, double> &AList);
double sigma_k_decomposition(double p, double pp, int index_p, int index_pp, int S, int L, int Lp, int J, std::unordered_map<uint64_t, double> &AList);
double regulator_local(double q, double regulator_cutoff, int regulator_power);
double regulator_nonlocal(double p, double pp, double regulator_cutoff, int regulator_power);

#endif