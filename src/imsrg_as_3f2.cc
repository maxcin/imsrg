// Copyright (C) 2025 Matthias Heinz, Ragnar Stroberg, and collaborators
//
// This file is part of NuIMSRG.
//
// NuIMSRG is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// NuIMSRG is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with NuIMSRG.  If not, see <https://www.gnu.org/licenses/>.

/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
///                                                  ____                                         ///
///        _________________           _____________/   /\               _________________        ///
///       /____/_____/_____/|         /____/_____/ /___/  \             /____/_____/_____/|       ///
///      /____/_____/__G_ /||        /____/_____/|/   /\  /\           /____/_____/____ /||       ///
///     /____/_____/__+__/|||       /____/_____/|/ G /  \/  \         /____/_____/_____/|||       ///
///    |     |     |     ||||      |     |     |/___/   /\  /\       |     |     |     ||||       ///
///    |  I  |  M  |     ||/|      |  I  |  M  /   /\  /  \/  \      |  I  |  M  |     ||/|       ///
///    |_____|_____|_____|/||      |_____|____/ + /  \/   /\  /      |_____|_____|_____|/||       ///
///    |     |     |     ||||      |     |   /___/   /\  /  \/       |     |     |     ||||       ///
///    |  S  |  R  |     ||/|      |  S  |   \   \  /  \/   /        |  S  |  R  |  G  ||/|       ///
///    |_____|_____|_____|/||      |_____|____\ __\/   /\  /         |_____|_____|_____|/||       ///
///    |     |     |     ||||      |     |     \   \  /  \/          |     |     |     ||||       ///
///    |     |  +  |     ||/       |     |  +  |\ __\/   /           |     |  +  |  +  ||/        ///
///    |_____|_____|_____|/        |_____|_____|/\   \  /            |_____|_____|_____|/         ///
///                                               \___\/                                          ///
///                                                                                               ///
///           imsrg++ : Interface for performing standard IMSRG calculations.                     ///
///                     Usage is imsrg++  option1=value1 option2=value2 ...                       ///
///                     To get a list of options, type imsrg++ help                               ///
///                                                                                               ///
///                                                      - Ragnar Stroberg 2016                   ///
///                                                                                               ///
/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

// #include <_stdio.h>
#include <cstdlib>
#include <vector>
#include "HFMBPT.hh"
#include "ModelSpace.hh"
#include "ReadWrite.hh"
#ifdef USE_MPI
#include <mpi.h>
#endif

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "Commutator.hh"
#include "IMSRG.hh"
#include "Operator.hh"
#include "Parameters.hh"
#include "PhysicalConstants.hh"
#include "imsrg_util.hh"
#include "version.hh"

struct OpFromFile {
  std::string file2name, file3name, opname;
  int j, p, t, r;  // J rank, parity, dTz, particle rank
};

void CheckFile(std::string file_path);
std::vector<OpFromFile> ParseOpsFromFile(const std::vector<std::string>& opsfromfile);
void SetAndCheckScratch(ReadWrite& rw, std::string scratch);
ModelSpace BuildModelSpace(
    int emax,
    int E3max,
    double hw,
    int A,
    Parameters& parameters,
    std::string reference,
    std::string valence_space,
    std::string custom_valence_space,
    std::string flowfile,
    std::string intfile,
    std::string occ_file,
    double betaCM);

Operator
ReadOneBodyDensity(ReadWrite& rw, ModelSpace& modelspace, std::string in_dir, std::string in_iso);
Operator
ReadTwoBodyDensity(ReadWrite& rw, ModelSpace& modelspace, std::string in_dir, std::string in_iso);
Operator GetTwoBodyDensity(ModelSpace& modelspace, Operator& rho1B, Operator& lambda);
void CheckDensity(Operator& rho, ModelSpace& modelspace, int targetMass, int targetP);
double TraceOneBody(Operator& rho);
std::vector<double> TraceTwoBody(Operator& rho);
arma::mat DiagonalizeRho(Operator& rho, Operator& Hbare, ModelSpace& modelspace);

void UpdateOccupations(
    ModelSpace& modelspace,
    Operator& rho,
    double cut,
    std::string generator_occ,
    bool vs_cut,
    bool vs_calc,
    int maxPcore,
    int maxNcore,
    int maxPvalence,
    int maxNvalence);
Operator UpdateTwoBodyTrace(
    ModelSpace& modelspace,
    Operator& rho,
    Operator& lambda,
    int targetMass,
    int targetZ);

Operator BuildHamiltonian(
    ReadWrite& rw,
    ModelSpace& modelspace,
    std::string in_dir,
    std::string in_iso,
    int filee1max,
    int filee2max,
    int filel1max,
    double BetaCM,
    double hwBetaCM);

Operator BuildHamiltonian(
    ReadWrite& rw,
    ModelSpace& modelspace,
    std::string inputtbme,
    int file2e1max,
    int file2e2max,
    int file2lmax,
    std::string fmt2,
    std::string input3bme,
    int file3e1max,
    int file3e2max,
    int file3e3max,
    std::string fmt3,
    std::string input3bme_type,
    std::string no2b_precision,
    double BetaCM,
    double hwBetaCM);

HFMBPT SolveHF(
    Operator& Hbare,
    bool freeze_occupations,
    std::string basis,
    std::string NAT_order,
    bool order_NAT_by_energy,
    bool use_NAT_occupations,
    bool use_HF_reference_in_NAT);

// HFMBPT ReadHF(Operator& HNO, Operator& rho);

Operator GetNormalOrderedHamiltonian(
    Operator& Hbare,
    HFMBPT& hf,
    std::string basis,
    double BetaCM,
    double hwBetaCM);

void EvaluateMBPT2(Operator& HNO);
void CheckMethod(std::string method);
void GetTruncatedOccupations(ModelSpace& ms, std::map<index_t, double>& newref, double cut);
void GetTruncatedOccupationsVS(
    ModelSpace& ms,
    std::map<index_t, double>& newref,
    double cut,
    int maxPcore,
    int maxNcore,
    int maxPvalence,
    int maxNvalence);

void SwapDegenerate(Operator& H, HFMBPT& hf, int maxCoreindex, int maxValenceIndex, ModelSpace& ms);

// Operator GetLambda(ModelSpace& ms, Operator& rho);

double GetHod1B(Operator& Hs, ModelSpace& modelspace);

void OutputHelper(Operator& H, ModelSpace& ms);

int main(int argc, char** argv) {
  // Default parameters, and everything passed by command line args.
  std::cout << "######  imsrg++ build version: " << version::BuildVersion() << std::endl;

  int world_size = 1;
  int my_rank = 0;
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif

  Parameters parameters(argc, argv);
  if (parameters.help_mode) return 0;

  std::string inputtbme = parameters.s("2bme");
  std::string input3bme = parameters.s("3bme");
  std::string input3bme_type = parameters.s("3bme_type");
  std::string no2b_precision = parameters.s("no2b_precision");
  std::string reference = parameters.s("reference");
  std::string valence_space = parameters.s("valence_space");
  std::string custom_valence_space = parameters.s("custom_valence_space");
  std::string basis = parameters.s("basis");
  std::string method = parameters.s("method");
  std::string flowfile = parameters.s("flowfile") + "_Rank" + std::to_string(my_rank);
  std::string intfile = parameters.s("intfile") + "_Rank" + std::to_string(my_rank);
  std::string core_generator = parameters.s("core_generator");
  std::string valence_generator = parameters.s("valence_generator");
  std::string fmt2 = parameters.s("fmt2");
  std::string fmt3 = parameters.s("fmt3");
  std::string input_op_fmt = parameters.s("input_op_fmt");
  std::string denominator_delta_orbit = parameters.s("denominator_delta_orbit");
  std::string LECs = parameters.s("LECs");
  std::string scratch = parameters.s("scratch");
  std::string valence_file_format = parameters.s("valence_file_format");
  std::string occ_file = parameters.s("occ_file");
  std::string physical_system = parameters.s("physical_system");
  std::string denominator_partitioning = parameters.s("denominator_partitioning");
  std::string NAT_order = parameters.s("NAT_order");
  // New paramters
  // std::string occ_dir = parameters.s("custom_occ_dir");
  std::string occ_inp = parameters.s("custom_occ_input");
  std::string generator_occ = parameters.s("generator_occ");
  std::string active_space = parameters.s("active_space");

  bool use_brueckner_bch = parameters.s("use_brueckner_bch") == "true";
  bool nucleon_mass_correction = parameters.s("nucleon_mass_correction") == "true";
  bool relativistic_correction = parameters.s("relativistic_correction") == "true";
  bool IMSRG3 = parameters.s("IMSRG3") == "true";
  bool imsrg3_n7 = parameters.s("imsrg3_n7") == "true";
  bool reduced_232_impl = parameters.s("reduced_232_impl") == "true";
  bool imsrg3_mp4 = parameters.s("imsrg3_mp4") == "true";
  bool imsrg3_at_end = parameters.s("imsrg3_at_end") == "true";
  bool imsrg3_no_qqq = parameters.s("imsrg3_no_qqq") == "true";
  bool write_omega = parameters.s("write_omega") == "true";
  bool freeze_occupations = parameters.s("freeze_occupations") == "true";
  bool discard_no2b_from_3n = parameters.s("discard_no2b_from_3n") == "true";
  bool hunter_gatherer = parameters.s("hunter_gatherer") == "true";
  bool goose_tank = parameters.s("goose_tank") == "true";
  bool discard_residual_input3N = parameters.s("discard_residual_input3N") == "true";
  bool use_NAT_occupations = (parameters.s("use_NAT_occupations") == "true") ? true : false;
  bool order_NAT_by_energy = (parameters.s("order_NAT_by_energy") == "true") ? true : false;
  bool store_3bme_pn = (parameters.s("store_3bme_pn") == "true");
  bool only_2b_eta = (parameters.s("only_2b_eta") == "true");
  bool only_2b_omega = (parameters.s("only_2b_omega") == "true");
  bool only_2b_omega_at_end = (parameters.s("only_2b_omega_at_end") == "true");
  bool perturbative_triples = (parameters.s("perturbative_triples") == "true");
  bool brueckner_restart = false;
  bool write_HO_ops = parameters.s("write_HO_ops") == "true";  // added by Antoine Belley
  bool write_HF_ops = parameters.s("write_HF_ops") == "true";  // added by Antoine Belley
  bool use_HF_reference_in_NAT = parameters.s("use_HF_reference_in_NAT") == "true";
  bool use_HF_valence_in_NAT = parameters.s("use_HF_valence_in_NAT") == "true";
  // New Parameters
  // bool use_mr = parameters.s("use_mr") == "true";
  bool vs_cut = parameters.s("vs_cut") == "true";
  bool vs_calc = parameters.s("vs_calc") == "true";
  bool use_hf_density = parameters.s("use_hf_density") == "true";

  int eMax = parameters.i("emax");
  int lmax = parameters.i("lmax");  // so far I only use this with atomic systems.
  int E3max = parameters.i("e3max");
  int lmax3 = parameters.i("lmax3");
  int targetMass = parameters.i("A");
  int file2e1max = parameters.i("file2e1max");
  int file2e2max = parameters.i("file2e2max");
  int file2lmax = parameters.i("file2lmax");
  int file3e1max = parameters.i("file3e1max");
  int file3e2max = parameters.i("file3e2max");
  int file3e3max = parameters.i("file3e3max");
  int atomicZ = parameters.i("atomicZ");
  int emax_unocc = parameters.i("emax_unocc");
  int eMax_imsrg = parameters.i("emax_imsrg");
  int e2Max_imsrg = parameters.i("e2max_imsrg");
  int e3Max_imsrg = parameters.i("e3max_imsrg");
  int eMax_3body_imsrg = parameters.i("emax_3body_imsrg");
  int imsrg3_commutator_depth = parameters.i("imsrg3_commutator_depth");
  // New parameters
  int maxPcore = parameters.i("maxPcore");
  int maxNcore = parameters.i("maxNcore");
  int maxPvalence = parameters.i("maxPvalence");
  int maxNvalence = parameters.i("maxNvalence");
  //  if ( not ( eMax_imsrg==-1 and e2Max_imsrg==-1 and e3Max_imsrg==-1 ) )
  //  {
  //    if ( eMax_imsrg==-1 ) eMax_imsrg = eMax;
  //    if ( e2Max_imsrg==-1 ) e2Max_imsrg = 2*eMax_imsrg;
  //    if ( e3Max_imsrg==-1 ) e3Max_imsrg = std::min( E3max, 3*eMax_imsrg);
  //  }
  ////  if (e2Max_imsrg==-1 and eMax_imsrg != -1) e2Max_imsrg = 2*eMax_imsrg;
  ////  if (e3Max_imsrg==-1 and eMax_imsrg != -1) e3Max_imsrg = std::min(E3max, 3*eMax_imsrg);

  double hw = parameters.d("hw");
  double smax = parameters.d("smax");
  double ode_tolerance = parameters.d("ode_tolerance");
  double dsmax = parameters.d("dsmax");
  double ds_0 = parameters.d("ds_0");
  double domega = parameters.d("domega");
  double omega_norm_max = parameters.d("omega_norm_max");
  double denominator_delta = parameters.d("denominator_delta");
  double BetaCM = parameters.d("BetaCM");
  double hwBetaCM = parameters.d("hwBetaCM");
  double eta_criterion = parameters.d("eta_criterion");
  double hw_trap = parameters.d("hw_trap");
  double dE3max = parameters.d("dE3max");
  double OccNat3Cut = parameters.d("OccNat3Cut");
  double threebody_threshold = parameters.d("threebody_threshold");
  // New parameters
  double density_cut = parameters.d("density_cut");
  // double denominator_delta_as = parameters.d("denominator_delta_as");
  // double alt_eta_criterion = parameters.d("alt_eta_criterion");
  // double alt_smax = parameters.d("alt_smax");

  std::vector<std::string> opnames = parameters.v("Operators");
  std::vector<std::string> opsfromfile = parameters.v("OperatorsFromFile");
  std::vector<std::string> opnamesPT1 = parameters.v("OperatorsPT1");
  std::vector<std::string> opnamesRPA = parameters.v("OperatorsRPA");
  std::vector<std::string> opnamesTDA = parameters.v("OperatorsTDA");

  std::vector<Operator> ops;
  std::vector<std::string> spwf = parameters.v("SPWF");

  using PhysConst::DARWIN_FOLDY;
  using PhysConst::NEUTRON_RCH2;
  using PhysConst::PROTON_RCH2;

  // test 2bme file
  CheckFile(inputtbme);
  // test 3bme file
  CheckFile(input3bme);

  // unpack the awkward input format for reading an operator from file, and put it into a struct.
  // the format should look like OpName^j_t_p_r^/path/to/2bfile^/path/to/3bfile  if particle rank of
  // Op is 2-body, then 3bfile is not needed.
  const std::vector<OpFromFile> opsfromfile_unpacked = ParseOpsFromFile(opsfromfile);

  ReadWrite rw;
  SetAndCheckScratch(rw, scratch);

  // std::cout << "Setting custom occupation profile from " << occ_dir << ", using input " << occ_inp
  //           << std::endl;

  ModelSpace modelspace = BuildModelSpace(
      eMax,
      E3max,
      hw,
      targetMass,
      parameters,
      reference,
      valence_space,
      custom_valence_space,
      flowfile,
      intfile,
      occ_file,
      BetaCM);

  if (hwBetaCM < 0) hwBetaCM = modelspace.GetHbarOmega();

  HFMBPT* hf_ptr = nullptr;
  // HFMBPT* dummy_ptr = nullptr;
  Operator HNO;
  // Operator lambda;

  //------------- AS SETUP -------------//
  if (use_hf_density) {
    std::cout << "-----AS - USING HF DENSITIES-----" << std::endl;
    Operator Hbare = BuildHamiltonian(
        rw,
        modelspace,
        inputtbme,
        file2e1max,
        file2e2max,
        file2lmax,
        fmt2,
        input3bme,
        file3e1max,
        file3e2max,
        file3e3max,
        fmt3,
        input3bme_type,
        no2b_precision,
        BetaCM,
        hwBetaCM);

    HFMBPT hf = SolveHF(
        Hbare,
        freeze_occupations,
        basis,
        NAT_order,
        order_NAT_by_energy,
        use_NAT_occupations,
        use_HF_reference_in_NAT);

    hf_ptr = &hf;

    HNO = GetNormalOrderedHamiltonian(Hbare, hf, basis, BetaCM, hwBetaCM);
    imsrg_util::WriteSPWaveFunctions(spwf, hf, intfile);
    EvaluateMBPT2(HNO);

    std::cout << "-----AS - TRANSFORMING DENSITIES AND UPDATING OCCUPATIONS-----" << std::endl;

    // All we need here is to correctly define the active space for decoupling.
    Operator rho1B = Operator(modelspace, 0, 0, 0, 1);
    for (int i = 0; i < modelspace.GetNumberOrbits(); ++i) {
      rho1B.OneBody(i, i) = modelspace.GetOrbit(i).occ;
    }
    UpdateOccupations(
        modelspace,
        rho1B,
        density_cut,
        generator_occ,
        vs_cut,
        vs_calc,
        maxPcore,
        maxNcore,
        maxPvalence,
        maxNvalence);
  }
  // else {
  //   std::cout << "-----AS - READING OPERATORS-----" << std::endl;
  //   // Reading operators in HO basis
  //   Operator rho1B = ReadOneBodyDensity(rw, modelspace, occ_dir, occ_inp);
  //   lambda = ReadTwoBodyDensity(rw, modelspace, occ_dir, occ_inp);
  //   Operator Hbare = BuildHamiltonian(
  //       rw,
  //       modelspace,
  //       occ_dir,
  //       occ_inp,
  //       file2e1max,
  //       file2e2max,
  //       file2lmax,
  //       BetaCM,
  //       hwBetaCM);

  //   std::cout << "-----AS - GENERATING TWO-BODY DENSITY-----" << std::endl;

  //   // Constructing the full density matrix, incl. the full 2B density
  //   Operator rho2B = GetTwoBodyDensity(modelspace, rho1B, lambda);
  //   Operator rho = rho1B + rho2B;

  //   // Checking densities are correct by calculating the traces
  //   CheckDensity(rho, modelspace, targetMass, atomicZ);

  //   std::cout << "-----AS - TRANSFORMING DENSITIES AND UPDATING OCCUPATIONS-----" << std::endl;
  //   // // Creating a dummy HFMBPT object for basis transformations
  //   HFMBPT dummy = HFMBPT(Hbare);
  //   dummy_ptr = &dummy;

  //   // Diagonalizing rho1B (going to NAT basis) where the cuts make sense.
  //   dummy.C_HO2NAT = DiagonalizeRho(rho1B, Hbare, modelspace);
  //   SwapDegenerate(
  //       Hbare,
  //       dummy,
  //       std::max(maxPcore, maxNcore),
  //       std::max(maxPvalence, maxNvalence),
  //       modelspace);
  //   rho1B = dummy.TransformHOToNATBasis(rho1B);

  //   // double t = 0.;
  //   // std::vector<double> can;
  //   // std::vector<double> ens;
  //   // for (int i = 0; i < modelspace.GetNumberOrbits(); ++i) {
  //   //   ens.push_back(rho1B.OneBody(i, i));
  //   //   if (i < 6) {
  //   //     can.push_back(1.);
  //   //   } else if (i > 7) {
  //   //     can.push_back(0.);
  //   //   } else {
  //   //     can.push_back(1. / 3.);
  //   //   }
  //   //   rho1B.OneBody(i, i) = (1 - t) * can[i] + t * ens[i];
  //   // }

  //   // Creating the density in original NAT basis with cut occupations based on the cut and
  //   updating
  //   // the modelspace

  //   std::cout << modelspace.GetTargetMass() << std::endl;
  //   UpdateOccupations(
  //       modelspace,
  //       rho1B,
  //       density_cut,
  //       generator_occ,
  //       vs_cut,
  //       vs_calc,
  //       maxPcore,
  //       maxNcore,
  //       maxPvalence,
  //       maxNvalence);
  //   for (int i = 0; i < modelspace.GetNumberOrbits(); ++i) {
  //     Orbit& oi = modelspace.GetOrbit(i);
  //     rho1B.OneBody(i, i) = oi.occ;
  //   }

  //   // We want to NO with respect to the cut density, so...
  //   dummy.C_HO2NAT = dummy.C_HO2NAT.t();
  //   rho1B = dummy.TransformHOToNATBasis(rho1B);  // ... we transform the cut density back to HO
  //   basis dummy.C_HO2NAT = DiagonalizeRho(rho1B, Hbare, modelspace);  // ... diagonalize the cut
  //   density SwapDegenerate(
  //       Hbare,
  //       dummy,
  //       std::max(maxPcore, maxNcore),
  //       std::max(maxPvalence, maxNvalence),
  //       modelspace);
  //   rho1B = dummy.TransformHOToNATBasis(
  //       rho1B);  //... actually transform the cut density to its own NAT basis
  //   lambda = dummy.TransformHOToNATBasis(lambda);  // .. transform lambda to the NAT basis

  //   // Constructing a new renonrmalized 2B density based on the cut 1B density
  //   rho2B = GetTwoBodyDensity(modelspace, rho1B, lambda);
  //   rho = rho1B + rho2B;
  //   rho2B = UpdateTwoBodyTrace(
  //       modelspace,
  //       rho,
  //       lambda,
  //       targetMass,
  //       atomicZ);  // Here lambda is updated so that the trace conditions are satisfied and a new
  //                  // full two body density is constructed. At the moment I do this even if cut is
  //                  0,
  //                  // just to make sure NO works. What to do with lambda
  //   rho = rho1B + rho2B;
  //   CheckDensity(rho, modelspace, targetMass, atomicZ);

  //   std::cout << "-----AS - TRANSFORMING THE HAMILTONAIAN AND NO wrt CUT DENSITY-----" <<
  //   std::endl;

  //   // Transform the Hamiltonian to the NAT basis
  //   HNO = dummy.TransformHOToNATBasis(Hbare);
  //   HNO = HNO.DoNormalOrdering(lambda);
  //   double CM_0body = -1 * BetaCM * 1.5 * hwBetaCM;
  //   HNO += CM_0body;

  //   EvaluateMBPT2(HNO);

  //   if (use_mr) {
  //     Commutator::SetMR(use_mr);
  //     Commutator::SetRho2B(lambda);  // I have to double check if this needs to be lambda or rho.
  //   }
  // }

  CheckMethod(method);

  if (only_2b_omega) {
    std::cout << " Restricting the Magnus operator Omega to be 2b." << std::endl;
    Commutator::SetOnly2bOmega(only_2b_omega);
  }

  // We may want to use a smaller model space for the IMSRG evolution than we used for the HF step.
  // This is most effective when using natural orbitals or when including 3-body operators.
  //  ModelSpace modelspace_imsrg = ( reference=="default" ?
  //  ModelSpace(eMax_imsrg,e2Max_imsrg,e3Max_imsrg,valence_space) :
  //  ModelSpace(eMax_imsrg,e2Max_imsrg,e3Max_imsrg,reference,valence_space) );
  ModelSpace modelspace_imsrg = modelspace;
  if ((eMax_imsrg != -1) or (e2Max_imsrg != -1) or (e3Max_imsrg != -1) or (eMax_3body_imsrg != -1)) {
    if (eMax_imsrg == -1) eMax_imsrg = eMax;
    if (e2Max_imsrg == -1) e2Max_imsrg = 2 * eMax_imsrg;
    if (e3Max_imsrg == -1) e3Max_imsrg = std::min(E3max, 3 * eMax_imsrg);
    if (eMax_3body_imsrg == -1) eMax_3body_imsrg = eMax_imsrg;

    //     ModelSpace modelspace_imsrg = modelspace;
    std::cout << "Truncating modelspace for IMSRG calculation: emax e2max e3max  ->  " << eMax_imsrg
              << " " << e2Max_imsrg << " " << e3Max_imsrg << std::endl;
    modelspace_imsrg.SetEmax(eMax_imsrg);
    modelspace_imsrg.SetE2max(e2Max_imsrg);
    modelspace_imsrg.SetE3max(e3Max_imsrg);
    modelspace_imsrg.SetEmax3Body(eMax_3body_imsrg);
    modelspace_imsrg.Init(eMax_imsrg, reference, valence_space);
    //  if (emax_unocc>0) modelspace_imsrg.SetEmaxUnocc(emax_unocc);
    if (physical_system == "atomic")
      modelspace_imsrg.InitSingleSpecies(eMax_imsrg, reference, valence_space);
    if (occ_file != "none" and occ_file != "")
      modelspace_imsrg.Init_occ_from_file(eMax_imsrg, valence_space, occ_file);
    //     if (physical_system == "atomic") modelspace_imsrg.InitSingleSpecies(eMax_imsrg,
    //     eMax_imsrg, e3Max_imsrg, reference, valence_space); if (occ_file != "none" and occ_file
    //     !=
    //     "" )
    //     modelspace_imsrg.Init_occ_from_file(eMax_imsrg,e2Max_imsrg,e3Max_imsrg,valence_space,occ_file);

    // If the occupations in modelspace were different from the naive filling, we want to keep
    // those.
    std::map<index_t, double> hole_map;
    for (auto& i_new : modelspace_imsrg.all_orbits) {
      Orbit& oi_new = modelspace_imsrg.GetOrbit(i_new);
      index_t i_old = modelspace.GetOrbitIndex(oi_new.n, oi_new.l, oi_new.j2, oi_new.tz2);
      Orbit& oi_old = modelspace.GetOrbit(i_old);
      hole_map[i_new] = oi_old.occ;
    }
    modelspace_imsrg.SetReference(hole_map);

    HNO = HNO.Truncate(modelspace_imsrg);
    if (IMSRG3) {
      HNO.ThreeBody.SwitchToPN_and_discard();
    }

    //     modelspace = modelspace_imsrg;  // this could cause some confusion later on...
    //    hf.PrintSPEandWF();
  } else {
    HNO.SetModelSpace(modelspace_imsrg);
    if (IMSRG3) {
      HNO.ThreeBody.SwitchToPN_and_discard();
    }
  }

  const int emax_reference =
      std::max(modelspace_imsrg.e_fermi.at(-1), modelspace_imsrg.e_fermi.at(1));
  const int emax_3b_reference = std::min(modelspace_imsrg.GetEMax3Body(), emax_reference);
  const int e3max_reference = std::min(3 * emax_3b_reference, modelspace_imsrg.GetE3max());
  const std::string emax_ref_string = "_e_" + std::to_string(emax_reference);
  const std::string emax3b_ref_string = "_e_" + std::to_string(emax_3b_reference);
  const std::string e3max_ref_string = "_E3_" + std::to_string(e3max_reference);

  // After truncating, get the perturbative energies again to see how much things changed.
  if (eMax_imsrg != eMax) {
    std::cout << "Perturbative estimates of gs energy:" << std::endl;
    double EMP2 = HNO.GetMP2_Energy();
    double EMP2_3B = HNO.GetMP2_3BEnergy();
    std::cout << "EMP2 = " << EMP2 << std::endl;
    std::cout << "EMP2_3B = " << EMP2_3B << std::endl;
    std::array<double, 3> Emp_3 = HNO.GetMP3_Energy();
    double EMP3 = Emp_3[0] + Emp_3[1] + Emp_3[2];
    std::cout << "E3_pp = " << Emp_3[0] << "  E3_hh = " << Emp_3[1] << " E3_ph = " << Emp_3[2]
              << "   EMP3 = " << EMP3 << std::endl;
    std::cout << "To 3rd order, E = " << HNO.ZeroBody + EMP2 + EMP3 + EMP2_3B << std::endl;
  }

  if (method == "MP3") {
    HNO.PrintTimes();
    return 0;
  }

  IMSRGSolver imsrgsolver(HNO);
  imsrgsolver.SetReadWrite(rw);
  imsrgsolver.SetMethod(method);
  imsrgsolver.SetDenominatorPartitioning(denominator_partitioning);
  imsrgsolver.SetEtaCriterion(eta_criterion);
  imsrgsolver.GetGenerator().SetOnly2bEta(only_2b_eta);
  imsrgsolver.GetGenerator().SetOccCut(density_cut);
  imsrgsolver.GetGenerator().SetGeneratorOcc(generator_occ);
  imsrgsolver.GetGenerator().SetMaxCoreIdx(std::max(maxPcore, maxNcore));
  imsrgsolver.max_omega_written = 1000;
  imsrgsolver.SetHunterGatherer(hunter_gatherer);
  imsrgsolver.SetPerturbativeTriples(perturbative_triples);
  imsrgsolver.SetSmax(smax);
  imsrgsolver.SetFlowFile(flowfile);
  imsrgsolver.SetDs(ds_0);
  imsrgsolver.SetDsmax(dsmax);
  imsrgsolver.SetDenominatorDelta(denominator_delta);
  imsrgsolver.SetdOmega(domega);
  imsrgsolver.SetOmegaNormMax(omega_norm_max);
  imsrgsolver.SetODETolerance(ode_tolerance);
  imsrgsolver.SetGeneratorOcc(generator_occ);
  imsrgsolver.SetCut(density_cut);
  if (denominator_delta_orbit != "none")
    imsrgsolver.SetDenominatorDeltaOrbit(denominator_delta_orbit);
  if (core_generator.find("imaginary") != std::string::npos or
      core_generator.find("wegner") != std::string::npos) {
    if (ds_0 > 1e-2) {
      ds_0 = 1e-4;
      dsmax = 1e-2;
      imsrgsolver.SetDs(ds_0);
      imsrgsolver.SetDsmax(dsmax);
    }
  }

  ModelSpace ms2(modelspace_imsrg);

  if (!vs_calc) {
    // --- non debug ---
    imsrgsolver.SetGenerator(core_generator);
    imsrgsolver.GetGenerator().SetMode("Step1");
    imsrgsolver.GetGenerator().SetDenominatorDeltaAS(0);
    imsrgsolver.Solve();

    if (method == "magnus") smax *= 2;
    imsrgsolver.GetGenerator().SetMode("Step2");
    modelspace_imsrg.ResetFirstPass();
    imsrgsolver.SetSmax(smax);
    imsrgsolver.Solve();

    if (method == "magnus") smax *= 2;
    imsrgsolver.GetGenerator().SetMode("Step3");
    modelspace_imsrg.ResetFirstPass();
    imsrgsolver.SetSmax(smax);
    imsrgsolver.Solve();

    if (method == "magnus") smax *= 2;
    imsrgsolver.GetGenerator().SetMode("Step4");
    modelspace_imsrg.ResetFirstPass();
    imsrgsolver.SetSmax(smax);
    imsrgsolver.Solve();

    if (method == "magnus") smax *= 2;
    imsrgsolver.GetGenerator().SetMode("Step5");
    // imsrgsolver.GetGenerator().SetDenominatorDeltaAS(denominator_delta_as);
    // if (alt_eta_criterion > 0.) {
    //   imsrgsolver.SetEtaCriterion(alt_eta_criterion);
    // }
    // if (alt_smax > 0.) {
    //   smax = imsrgsolver.GetS() + alt_smax;
    //   std::cout << smax << std::endl;
    // }
    modelspace_imsrg.ResetFirstPass();
    imsrgsolver.SetSmax(smax);
    imsrgsolver.Solve();

    HNO = imsrgsolver.GetH_s();

    // std::cout << cnt << std::endl;
    for (auto& h : modelspace_imsrg.holes) {
      std::cout << h << ", ";
    }
    std::cout << std::endl;
    for (auto& h : modelspace_imsrg.core) {
      std::cout << h << ", ";
    }
    std::cout << std::endl;
    for (auto& h : modelspace_imsrg.valence) {
      std::cout << h << ", ";
    }
    std::cout << std::endl;

    std::string flow = "SR";
    std::ostringstream oss;
    oss << "/home/vernik/Projects/MR_IMSRG/output/" << "HoutHO_ens_" << occ_inp << "_"
        << "s" << smax << "_" << flow + "_" << BetaCM << std::setprecision(2) << "_" + active_space;
    std::string fname = oss.str();
    std::cout << "Undoing NO for MR-Interaction:";
    std::cout << "A=" << modelspace_imsrg.GetAref() << " Z=" << modelspace_imsrg.GetZref()
              << std::endl;
    std::cout << "Before doing so, the spes are " << std::endl;
    for (auto i : modelspace_imsrg.all_orbits)
      std::cout << "  " << i << " : " << HNO.OneBody(i, i) << std::endl;
    if (use_hf_density) {
      HNO = HNO.UndoNormalOrdering();
    }
    // else {
    //   HNO = HNO.UndoNormalOrdering(lambda);
    // }
    for (auto i : modelspace_imsrg.all_orbits)
      std::cout << "  " << i << " : " << HNO.OneBody(i, i) << std::endl;
    // rw.WriteTokyoFull(HNO, fname + "_full.snt");
    rw.WriteTokyoTruncated(HNO, fname + ".snt", std::max(maxPvalence, maxNvalence) + 1);
    rw.Write_me1j(fname + "_HF.me1j", HNO, eMax, lmax);
    rw.Write_me2jp(fname + "_HF.me2jp", HNO, eMax, 2 * eMax, lmax);
    // Operator HsHO;
    // if (use_hf_density) {
    //   HFMBPT hf = *hf_ptr; // dangerous
    //   HsHO = hf.TransformFromHFBasis(HNO);
    // } else {
    //   HFMBPT dummy = *dummy_ptr;
    //   dummy.C_HO2NAT = dummy.C_HO2NAT.t();
    //   HsHO = dummy.TransformHOToNATBasis(HNO);
    //   dummy.C_HO2NAT = dummy.C_HO2NAT.t();
    // }
    // rw.Write_me1j(fname + ".me1j", HsHO, eMax, lmax);
    // rw.Write_me2jp(fname + ".me2jp", HsHO, eMax, 2 * eMax, lmax);
    // OutputHelper(HNO, modelspace_imsrg);
    // std::cout << "Writing .snt" << std::endl;
    // modelspace.valence = modelspace.core;
    // rw.WriteTokyoFull(HsHO, fname + "_full.snt");
    // rw.WriteTokyoTruncated(HsHO, fname + ".snt", std::max(maxPvalence, maxNvalence) + 1);
  }
  // Not yet needed for 3f2...
  // else {  // if (!vs_calc)
  //   if (core_generator == "as-atan") {
  //     std::cout << "Generator is as-atan, doing VS calculations so switching to atan." <<
  //     std::endl; core_generator = "atan";
  //   }
  //   imsrgsolver.SetGenerator(core_generator);
  //   imsrgsolver.Solve();

  //   int nsteps = modelspace.valence.size() > 0 ? 2 : 1;
  //   if (nsteps > 1 and valence_space != reference)  // two-step decoupling, do core first
  //   {
  //     if (method == "magnus") smax *= 2;

  //     imsrgsolver.SetGenerator(valence_generator);
  //     std::cout << "Setting generator to " << valence_generator << std::endl;
  //     modelspace_imsrg.ResetFirstPass();
  //     if (valence_generator.find("imaginary") != std::string::npos or
  //         valence_generator.find("wegner") != std::string::npos) {
  //       if (ds_0 > 1e-2) {
  //         ds_0 = 1e-4;
  //         dsmax = 1e-2;
  //         imsrgsolver.SetDs(ds_0);
  //         imsrgsolver.SetDsmax(dsmax);
  //       }
  //     }
  //     imsrgsolver.SetSmax(smax);
  //     imsrgsolver.Solve();
  //   }

  //   bool renormal_order = false;
  //   if (modelspace_imsrg.valence.size() > 0) {
  //     renormal_order = modelspace.holes.size() != modelspace_imsrg.core.size();
  //     if (!renormal_order) {
  //       for (auto c : modelspace_imsrg.core) {
  //         if ((find(modelspace_imsrg.holes.begin(), modelspace_imsrg.holes.end(), c) ==
  //              modelspace_imsrg.holes.end()) ||
  //             (std::abs(1 - modelspace_imsrg.GetOrbit(c).occ) > 1e-6)) {
  //           renormal_order = true;
  //           break;
  //         }
  //       }
  //     }
  //   }

  //   if (renormal_order) {
  //     HNO = imsrgsolver.GetH_s();
  //     std::cout << "Undoing NO wrt A=" << modelspace_imsrg.GetAref()
  //               << " Z=" << modelspace_imsrg.GetZref() << std::endl;
  //     std::cout << "Before doing so, the spes are " << std::endl;
  //     for (auto i : modelspace_imsrg.all_orbits)
  //       std::cout << "  " << i << " : " << HNO.OneBody(i, i) << std::endl;
  //     for (auto& i : modelspace.holes) {
  //       std::cout << i << ", ";
  //     }
  //     std::cout << "\n";

  //     ms2.SetReference(modelspace_imsrg.core);

  //     if (use_hf_density) {
  //       HNO = HNO.UndoNormalOrdering();
  //     }
  //     // else {
  //     //   HNO = HNO.UndoNormalOrdering(lambda);
  //     // }
  //     HNO.SetModelSpace(ms2);
  //     std::cout << "Doing NO wrt A=" << ms2.GetAref() << " Z=" << ms2.GetZref()
  //               << "  norbits = " << ms2.GetNumberOrbits() << std::endl;
  //     for (auto& i : ms2.holes) {
  //       std::cout << i << ", ";
  //     }
  //     std::cout << "\n";
  //     std::cout << HNO.ZeroBody << std::endl;
  //     HNO = HNO.DoNormalOrdering();  // Probably not using lambda here because we are NO wrt to a
  //                                    // closed core.
  //     std::cout << HNO.ZeroBody << std::endl;

  //     imsrgsolver.FlowingOps[0] = HNO;
  //     rw.WriteTokyo(imsrgsolver.GetH_s(), intfile + "_VSAS.snt", "");
  //   }

  //   if (modelspace_imsrg.valence.size() > 0) {
  //     rw.WriteTokyo(imsrgsolver.GetH_s(), intfile + ".snt", "");
  //   } else {
  //     // single ref. just print the zero body pieces out. (maybe check if its magnus?)
  //     std::cout << "Energy = " << std::setprecision(6) << imsrgsolver.GetH_s().ZeroBody <<
  //     std::endl;
  //   }
  // }

  // if ((method == "magnus") || (method == "magnus_backoff")) {
  //   /// if method is magnus, we didn't do this already. So we need to unpack any operators from
  //   /// file.

  //   for (auto& opff : opsfromfile_unpacked) {
  //     opnames.push_back(opff.opname + "_FROMFILE");
  //   }

  //   int count_from_file = 0;

  //   if (opnames.size() > 0) std::cout << "transforming operators" << std::endl;

  //   for (size_t i = 0; i < opnames.size(); ++i) {
  //     auto opname = opnames[i];
  //     std::cout << i << ": " << opname << " " << std::endl;

  //     Operator op;

  //     if (opname.find("_FROMFILE") != std::string::npos) {
  //       // OpFromFile& opff = opsfromfile_unpacked[count_from_file];
  //       //  std::cout << "reading " << opff.opname << " with " << opff.j << " " << opff.t << " "
  //       <<
  //       //  opff.p << " " << opff.r << "  from file " << opff.file2name << std::endl;
  //       // op = Operator(modelspace, opff.j, opff.t, opff.p, opff.r );
  //       // if (opff.r>2)
  //       // {
  //       //   op.ThreeBody.SetMode("no2b");
  //       //   op.ThreeBody.Allocate();
  //       // }
  //       // if ( input_op_fmt == "navratil" )
  //       // {
  //       //   rw.Read2bCurrent_Navratil( opff.file2name, op );
  //       // }
  //       // else if ( input_op_fmt == "miyagi" )
  //       // {
  //       //   if (opff.file2name != "")
  //       //   {
  //       //       Operator optmp = rw.ReadOperator2b_Miyagi( opff.file2name, modelspace );
  //       //       op.TwoBody = optmp.TwoBody;
  //       //   }
  //       //   if ( opff.r>2 and opff.file3name != "")  rw.Read_Darmstadt_3body( opff.file3name, op,
  //       //   file3e1max,file3e2max,file3e3max);
  //       // }
  //       // else if ( input_op_fmt == "heinz")
  //       // {
  //       //   if ( opff.r>2 and opff.file3name != "")
  //       //   {

  //       //     op.ThreeBody.SetMode("no2b");
  //       //     if (no2b_precision == "half")  op.ThreeBody.SetMode("no2bhalf");

  //       //     op.ThreeBody.ReadFile( {opff.file3name}, {file3e1max, file3e2max, file3e3max,
  //       //     file3e1max} );
  //       //     // rw.Read_Darmstadt_3body( opff.file3name, op,  file3e1max,file3e2max,file3e3max);
  //       //   }
  //       // }
  //       // count_from_file++;
  //       // opname = opff.opname; // Get rid of the _FROMFILE bit.
  //     } else {
  //       op = imsrg_util::OperatorFromString(modelspace, opname);
  //     }
  //     //      Operator op = imsrg_util::OperatorFromString( modelspace, opname );

  //     // Added by Antoine Belley
  //     // if (write_HO_ops) {
  //     //   std::cout << "writing HO tensor files " << std::endl;
  //     //   if (valence_file_format == "tokyo") {
  //     //     rw.WriteTensorTokyo(intfile + opnames[i] + "_HO_2b.snt", op);
  //     //   } else {
  //     //     op.OneBody.print();
  //     //     rw.WriteTensorOneBody(intfile + opnames[i] + "_HO_1b.op", op, opnames[i]);
  //     //     rw.WriteTensorTwoBody(intfile + opnames[i] + "_HO_2b.op", op, opnames[i]);
  //     //   }
  //     // }

  //     // if (basis == "oscillator" or opname == "OccRef") {
  //     //   op = op.DoNormalOrdering(lambda);
  //     // } else if (basis == "HF") {
  //     //   std::cout << "Normal ordering the operator wrt to a deformed density..." << std::endl;
  //     //   dummy.C.print();
  //     //   op = dummy.TransformToHFBasis(op).DoNormalOrdering(lambda);
  //     // } else if (basis == "NAT") {
  //     //   op = dummy.TransformHOToNATBasis(op).DoNormalOrdering();
  //     //   // Operator op_2b = hf.TransformHOToNATBasis(op);
  //     //   // op_2b.SetParticleRank(2);

  //     //   // op = hf.GetNormalOrdered3BOperator(op) + op_2b.DoNormalOrdering();
  //     // }

  //     std::cout << "Transforming to HF basis and normal ordering the operator wrt to a deformed "
  //                  "density..."
  //               << std::endl;
  //     if (use_hf_density) {
  //       HFMBPT hf = *hf_ptr;
  //       op = hf.TransformFromHFBasis(op).DoNormalOrdering();
  //     }
  //     // else {
  //     //   HFMBPT dummy = *dummy_ptr;
  //     //   op = dummy.TransformHOToNATBasis(op).DoNormalOrdering(lambda);
  //     // }

  //     std::cout << std::setprecision(24) << "   HF: " << op.ZeroBody << std::endl;
  //     std::cout << opname << "_HF: " << op.ZeroBody << std::endl;

  //     if ((eMax_imsrg != -1) or (e2Max_imsrg != -1) or (e3Max_imsrg) != -1) {
  //       //     ModelSpace modelspace_imsrg = modelspace;
  //       std::cout << "Truncating modelspace for IMSRG calculation: emax e2max e3max  ->  "
  //                 << eMax_imsrg << " " << e2Max_imsrg << " " << e3Max_imsrg << std::endl;
  //       op = op.Truncate(modelspace_imsrg);
  //     }

  //     // Added by Antoine Belley
  //     // if (write_HF_ops) {
  //     //   std::cout << "writing HF tensor files " << std::endl;
  //     //   if (valence_file_format == "tokyo") {
  //     //     rw.WriteTensorTokyo(intfile + opnames[i] + "_HF_2b.snt", op);
  //     //   } else {
  //     //     rw.WriteTensorOneBody(intfile + opnames[i] + "_HF_1b.op", op, opnames[i]);
  //     //     rw.WriteTensorTwoBody(intfile + opnames[i] + "_HF_2b.op", op, opnames[i]);
  //     //   }
  //     // }

  //     op = imsrgsolver.Transform(op);

  //     // Unclear whether we should do NO2B here as well...
  //     // std::cout << "Before renormal ordering Op(5,4) is " << std::setprecision(10) <<
  //     // op.OneBody(5,4) << std::endl;
  //     bool renormal_order = true;  // (U) I added this here, make sure it makes sense.
  //     if (renormal_order) {
  //       if (IMSRG3) {
  //         rw.Write_me1j(
  //             intfile + opname + "_ENO" + emax_ref_string + ".me1j",
  //             op,
  //             emax_reference,
  //             emax_reference);
  //         rw.Write_me2jp(
  //             intfile + opname + "_ENO" + emax_ref_string + ".me2jp",
  //             op,
  //             emax_reference,
  //             2 * emax_reference,
  //             emax_reference);
  //         rw.Write_me3jp(
  //             intfile + opname + "_ENO" + emax3b_ref_string + e3max_ref_string + ".me3jp",
  //             op,
  //             emax_3b_reference,
  //             2 * emax_3b_reference,
  //             e3max_reference);
  //         rw.Write_NaiveVS3B(intfile + opname + ".vs3b", op);
  //         op.SetNumberLegs(4);
  //         op.SetParticleRank(2);
  //       }
  //       op = op.UndoNormalOrdering(lambda);
  //       if (vs_calc) {
  //         op.SetModelSpace(ms2);
  //       }
  //       op = op.DoNormalOrdering();
  //       if ((op.GetJRank() == 0) && (op.GetTRank() == 0) && (op.GetParity() == 0)) {
  //         rw.Write_NaiveVS1B(intfile + opname + ".vs1b", op);
  //         rw.Write_NaiveVS2B(intfile + opname + ".vs2b", op);
  //         rw.Write_me1j(
  //             intfile + opname + "_coreNO" + emax_ref_string + ".me1j",
  //             op,
  //             emax_reference,
  //             emax_reference);
  //         rw.Write_me2jp(
  //             intfile + opname + "_coreNO" + emax_ref_string + ".me2jp",
  //             op,
  //             emax_reference,
  //             2 * emax_reference,
  //             emax_reference);
  //       }
  //     }
  //     //      std::cout << " (" << ops[i].ZeroBody << " ) " << std::endl;
  //     std::cout << "   IMSRG: " << op.ZeroBody << std::endl;
  //     std::cout << opname << "_IMSRG: " << op.ZeroBody << std::endl;
  //     std::string flow = use_mr ? "MR" : "SR";
  //     std::ostringstream oss;
  //     oss << "/home/vernik/Projects/MR_IMSRG/output/" << "OpoutHO_ens_" << occ_inp << "_"
  //         << "s" << smax << "_" << flow + "_" << BetaCM << std::setprecision(2)
  //         << "_" + active_space;
  //     std::string fname = oss.str();
  //     // std::cout << "Undoing the NO, transforming back to HO basis..." << std::endl;
  //     if (use_hf_density) {
  //       op = op.UndoNormalOrdering();
  //     } else {
  //       op = op.UndoNormalOrdering(lambda);
  //     }
  //     rw.WriteTokyoTruncated(
  //         op,
  //         fname + "_" + opname + "_HF.snt",
  //         std::max(maxPvalence, maxNvalence) + 1);
  //     if (use_hf_density) {
  //       HFMBPT hf = *hf_ptr;
  //       op = hf.TransformFromHFBasis(op);
  //     } else {
  //       HFMBPT dummy = *dummy_ptr;
  //       dummy.C_HO2NAT = dummy.C_HO2NAT.t();
  //       op = dummy.TransformHOToNATBasis(op);
  //       dummy.C_HO2NAT = dummy.C_HO2NAT.t();
  //     }

  //     rw.WriteTokyoTruncated(
  //         op,
  //         fname + "_" + opname + ".snt",
  //         std::max(maxPvalence, maxNvalence) + 1);

  //     //      rw.WriteOperatorHuman(ops[i],intfile+opnames[i]+"_step2.op");
  //     //      std::cout << "After renormal ordering Op(5,4) is " << std::setprecision(10) <<
  //     //      op.OneBody(5,4) << std::endl;

  //     // std::cout << "      " << op.GetJRank() << " " << op.GetTRank() << " " << op.GetParity()
  //     //           << "   " << op.GetNumberLegs() << std::endl;
  //     // if (((op.GetJRank() + op.GetTRank() + op.GetParity()) < 1) and (op.GetNumberLegs() % 2 ==
  //     // 0)) {
  //     //   std::cout << "writing scalar files " << std::endl;
  //     //   if (valence_file_format == "tokyo") {
  //     //     rw.WriteTokyo(op, intfile + opname + ".snt", "op");
  //     //   } else {
  //     //     rw.WriteNuShellX_op(op, intfile + opname + ".int");
  //     //   }
  //     // } else if (op.GetNumberLegs() % 2 == 1)  // odd number of legs -> this is a dagger
  //     // operator
  //     // {
  //     //   //      rw.WriteNuShellX_op(ops[i],intfile+opnames[i]+".int"); // do this for now.
  //     later
  //     //   make
  //     //   //      a *.dag format.
  //     //   rw.WriteDaggerOperator(op, intfile + opname + ".dag", opname);
  //     // } else {
  //     //   std::cout << "writing tensor files " << std::endl;
  //     //   if (valence_file_format == "tokyo") {
  //     //     rw.WriteTensorTokyo(intfile + opname + "_2b.snt", op);
  //     //   } else {
  //     //     rw.WriteTensorOneBody(intfile + opname + "_1b.op", op, opname);
  //     //     rw.WriteTensorTwoBody(intfile + opname + "_2b.op", op, opname);
  //     //   }
  //     // }

  //   }  // for opnames

  // }  // if method == "magnus"

  //   if (write_omega) {
  //     std::string scratch = rw.GetScratchDir();
  //     imsrgsolver.FlushOmegaToScratch();

  //     bool filesucess = false;
  //     if (basis == "HF") {
  //       filesucess = hf.C.save(intfile + "C.mat");
  //     } else if (basis == "NAT") {
  //       filesucess = hf.C_HO2NAT.save(intfile + "C.mat");
  //     }

  //     if (filesucess == false) {
  //       std::cout << "Couldn't save HF coefficient matrix." << std::endl;
  //     }
  //   }

  HNO.PrintTimes();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  return EXIT_SUCCESS;
}

void CheckFile(std::string path_to_file) {
  if ((path_to_file != "none") && (path_to_file != "")) {
    if (!std::ifstream(path_to_file).good()) {
      std::cout << "ERROR: " << "Trouble reading " << path_to_file << " ." << std::endl;
      exit(1);
    }
  }
}

std::vector<OpFromFile> ParseOpsFromFile(const std::vector<std::string>& opsfromfile) {
  std::vector<OpFromFile> opsfromfile_unpacked;

  // If we're reading in other operators, make sure those are ok too
  for (auto& tag : opsfromfile) {
    std::istringstream ss(tag);
    std::string opname, qnumbers, f2name, f3name = "";

    OpFromFile opff;

    getline(ss, opname, '^');
    getline(ss, qnumbers, '^');
    getline(ss, f2name, '^');
    if (!ss.eof()) getline(ss, f3name, '^');
    opff.opname = opname;
    opff.file2name = f2name;
    opff.file3name = f3name;

    ss.str(qnumbers);
    ss.clear();
    std::string tmp;
    getline(ss, tmp, '_');
    std::istringstream(tmp) >> opff.j;
    getline(ss, tmp, '_');
    std::istringstream(tmp) >> opff.t;
    getline(ss, tmp, '_');
    std::istringstream(tmp) >> opff.p;
    getline(ss, tmp, '_');
    std::istringstream(tmp) >> opff.r;

    std::cout << "Parsed operator from file.\n"
              << "opname = " << opff.opname << "\n"
              << "J = " << opff.j << "\n"
              << "Tz = " << opff.t << "\n"
              << "Parity = " << opff.p << "\n"
              << "Rank = " << opff.r << "\n"
              << "file2 = " << opff.file2name << "\n"
              << "file3 = " << opff.file3name << std::endl;

    // now make sure the files exist before we add them to the list.
    CheckFile(opff.file2name);
    CheckFile(opff.file3name);

    // if the files look good, then add it to the list
    opsfromfile_unpacked.push_back(opff);
  }

  return opsfromfile_unpacked;
}

void SetAndCheckScratch(ReadWrite& rw, std::string scratch) {
  // Test whether the scratch directory exists and we can write to it.
  // This is necessary because otherwise you get garbage for transformed operators and it's
  // not obvious what went wrong.
  rw.SetScratchDir(scratch);
  if ((scratch == "/dev/null") || (scratch == "/dev/null/")) {
    std::cout << "ERROR: Scratch set to " << scratch << " ." << std::endl;
    exit(EXIT_FAILURE);
  } else {
    const std::string testfilename = scratch + "/_this_is_a_test_delete_me_";
    std::ofstream testout(testfilename);
    testout << "PASSED" << std::endl;
    testout.close();

    // now read it back.
    std::ifstream testin(testfilename);
    std::string checkpassed;
    testin >> checkpassed;
    if ((checkpassed != "PASSED") || (!testout.good()) || (!testin.good())) {
      std::cout << "ERROR: Failed test write to scratch directory " << scratch << " ." << std::endl;
      exit(EXIT_FAILURE);
    }
    testin.close();
    remove(testfilename.c_str());
  }
}

ModelSpace BuildModelSpace(
    int emax,
    int E3max,
    double hw,
    int A,
    Parameters& parameters,
    std::string reference,
    std::string valence_space,
    std::string custom_valence_space,
    std::string flowfile,
    std::string intfile,
    std::string occ_file,
    double betaCM) {
  if (custom_valence_space !=
      "")  // if a custom space is defined, the input valence_space is just used as a name
  {
    if (valence_space == "")  // if no name is given, then just name it "custom"
    {
      valence_space = "custom";
      flowfile = parameters.DefaultFlowFile();
      intfile = parameters.DefaultIntFile();
    }
    valence_space = custom_valence_space;
  }
  std::cout << emax << " " << reference << " " << valence_space << " " << std::endl;
  ModelSpace modelspace =
      (reference == "default" ? ModelSpace(emax, valence_space)
                              : ModelSpace(emax, reference, valence_space));

  modelspace.com_beta_ = betaCM;

  modelspace.SetE3max(E3max);

  if ((occ_file != "none") && (occ_file != "")) {
    modelspace.Init_occ_from_file(emax, valence_space, occ_file);
  }

  modelspace.SetHbarOmega(hw);
  modelspace.SetTargetMass(A);

  return modelspace;
}

Operator
ReadOneBodyDensity(ReadWrite& rw, ModelSpace& modelspace, std::string in_dir, std::string in_iso) {
  Operator rho = Operator(modelspace, 0, 0, 0, 1);

  std::string dm_me1j = in_dir + "/" + "dm_ens_HO_" + in_iso + ".me1j";
  // std::string dm_me1j = in_dir + "/" + "dm_HO_" + in_iso + ".me1j";
  rw.Read_me1j(dm_me1j, rho, modelspace.Emax, modelspace.Lmax);

  return rho;
}

Operator
ReadTwoBodyDensity(ReadWrite& rw, ModelSpace& modelspace, std::string in_dir, std::string in_iso) {
  Operator rho = Operator(modelspace, 0, 0, 0, 2);

  std::string dm_me2j = in_dir + "/" + "dm_ens_HO_" + in_iso + ".me2jp";
  // std::string dm_me2j = in_dir + "/" + "dm_HO_" + in_iso + ".me2jp";
  rw.Read_me2jp(dm_me2j, rho, modelspace.Emax, modelspace.E2max, modelspace.Lmax);

  return rho;
}

Operator GetTwoBodyDensity(ModelSpace& modelspace, Operator& rho1B, Operator& lambda) {
  Operator rho2B = Operator(modelspace, lambda.rank_J, lambda.rank_T, lambda.parity, 2);
  for (int p = 0; p < modelspace.GetNumberOrbits(); ++p) {
    int twojp = modelspace.GetOrbit(p).j2;
    for (int q = 0; q < modelspace.GetNumberOrbits(); ++q) {
      int twojq = modelspace.GetOrbit(q).j2;
      for (int r = 0; r < modelspace.GetNumberOrbits(); ++r) {
        int twojr = modelspace.GetOrbit(r).j2;
        for (int s = 0; s < modelspace.GetNumberOrbits(); ++s) {
          int twojs = modelspace.GetOrbit(s).j2;

          int Jmin = std::max(abs(twojp - twojq), abs(twojr - twojs)) / 2;
          int Jmax = std::min((twojp + twojq), (twojr + twojs)) / 2;

          for (int J = Jmin; J <= Jmax; J++) {
            double rr1 = rho1B.OneBody(p, r) * rho1B.OneBody(q, s);
            double rr2 = rho1B.OneBody(p, s) * rho1B.OneBody(q, r);
            double ph = AngMom::phase(J - (twojp + twojq) / 2);

            double normpq = 1.;
            double normrs = 1.;
            (p == q) ? normpq = std::sqrt(1 + pow(-1, J)) / 2 : 1.;
            (r == s) ? normrs = std::sqrt(1 + pow(-1, J)) / 2 : 1.;

            double me = normpq * normrs * (rr1 - ph * rr2);
            me += normpq * normrs * lambda.TwoBody.GetTBME_J(J, p, q, r, s);
            if (abs(me) > 1e-8) {
              rho2B.TwoBody.SetTBME_J(J, p, q, r, s, me);
            }
          }
        }
      }
    }
  }
  return rho2B;
}

void CheckDensity(Operator& rho, ModelSpace& modelspace, int targetMass, int targetZ) {
  double trace1B = TraceOneBody(rho);
  std::vector<double> traces2B = TraceTwoBody(rho);

  std::cout << "\nChecking densities..." << std::endl;

  double trace2B = traces2B[0];
  double traceP = traces2B[1];
  double traceN = traces2B[2];
  double traceM = traces2B[3];

  int trace_1B_val = static_cast<int>(std::round(trace1B));
  int trace_2B_val = static_cast<int>(std::round(trace2B));
  int trace_P_val = static_cast<int>(std::round(traceP));
  int trace_N_val = static_cast<int>(std::round(traceN));
  int trace_M_val = static_cast<int>(std::round(traceM));

  // std::cout << "Two-body trace: " << traces2B[0] << std::endl;
  // std::cout << "Two-body P trace: " << traces2B[1] << std::endl;
  // std::cout << "Two-body N trace: " << traces2B[2] << std::endl;
  // std::cout << "Two-body M trace: " << traces2B[3] << std::endl;

  if (trace_1B_val != targetMass || std::abs(trace_1B_val - trace1B) > 1e-2) {
    std::cerr << "Particle-number mismatch in 1B density trace!\n"
              << "Got: " << trace1B << ", expected: " << targetMass << std::endl;
    std::exit(EXIT_FAILURE);
  } else if (
      trace_2B_val != targetMass * (targetMass - 1) || std::abs(trace_2B_val - trace2B) > 1e-1) {
    std::cerr << "Particle-number mismatch in 2B density trace!\n"
              << "Got: " << trace2B << ", expected: " << targetMass * (targetMass - 1) << std::endl;
    std::exit(EXIT_FAILURE);
  } else if (trace_P_val != targetZ * (targetZ - 1) || std::abs(trace_P_val - traceP) > 1e-1) {
    std::cerr << "Particle-number mismatch in 2B P density trace!\n"
              << "Got: " << traceP << ", expected: " << targetZ * (targetZ - 1) << std::endl;
    std::exit(EXIT_FAILURE);
  } else if (
      trace_N_val != (targetMass - targetZ) * ((targetMass - targetZ) - 1) ||
      std::abs(trace_N_val - traceN) > 1e-1) {
    std::cerr << "Particle-number mismatch in 2B N density trace!\n"
              << "Got: " << traceN
              << ", expected: " << (targetMass - targetZ) * ((targetMass - targetZ) - 1)
              << std::endl;
    std::exit(EXIT_FAILURE);
  } else if (
      trace_M_val != 2 * targetZ * (targetMass - targetZ) || std::abs(trace_M_val - traceM) > 1e-1) {
    std::cerr << "Particle-number mismatch in 2B M density trace!\n"
              << "Got: " << traceM << ", expected: " << 2 * targetZ * (targetMass - targetZ)
              << std::endl;
    std::exit(EXIT_FAILURE);
  } else {
    std::cout << "Traces are correct:\n "
              << "\t1B: " << trace1B << "\n"
              << "\t2B: " << trace2B << "\n"
              << "\tP:  " << traceP << "\n"
              << "\tN:  " << traceN << "\n"
              << "\tM:  " << traceM << "\n"
              << std::endl;
  }
}

double TraceOneBody(Operator& rho) {
  double tr = 0.;
  for (int i = 0; i < rho.modelspace->GetNumberOrbits(); ++i) {
    int j2 = rho.modelspace->GetOrbit(i).j2;
    tr += (j2 + 1) * rho.OneBody(i, i);
  }
  return tr;
}

std::vector<double> TraceTwoBody(Operator& rho) {
  double tr2B = 0.;
  double trP = 0.;
  double trN = 0.;
  double trM = 0.;

  for (int i = 0; i < rho.modelspace->GetNumberOrbits(); ++i) {
    Orbit& oi = rho.modelspace->GetOrbit(i);
    int ji = oi.j2;
    for (int j = 0; j < rho.modelspace->GetNumberOrbits(); ++j) {
      Orbit& oj = rho.modelspace->GetOrbit(j);
      int jj = oj.j2;
      int tz = (oi.tz2 + oj.tz2) / 2;
      int Jmax = (ji + jj) / 2;
      int Jmin = abs(ji - jj) / 2;
      for (int J = Jmin; J <= Jmax; ++J) {
        double pref = (2 * J + 1);
        double tr = pref * rho.TwoBody.GetTBME_J(J, i, j, i, j);
        tr2B += tr;
        if (tz == 1)
          trN += tr;
        else if (tz == -1)
          trP += tr;
        else
          trM += tr;
      }
    }
  }
  return std::vector<double>{tr2B, trP, trN, trM};
}

Operator UpdateTwoBodyTrace(
    ModelSpace& modelspace,
    Operator& rho,
    Operator& lambda,
    int targetMass,
    int targetP) {
  std::vector<double> tracesLambda = TraceTwoBody(lambda);
  double traceLambda2B = tracesLambda[0];
  double traceLambdaP = tracesLambda[1];
  double traceLambdaN = tracesLambda[2];
  double traceLambdaM = tracesLambda[3];
  if (abs(traceLambda2B) < 1e-6) {
    return GetTwoBodyDensity(modelspace, rho, lambda);
  }
  std::vector<double> traces2B = TraceTwoBody(rho);
  double trace2B = traces2B[0];
  double traceP = traces2B[1];
  double traceN = traces2B[2];
  double traceM = traces2B[3];
  // std::cout << "Two-body trace: " << traces2B[0] << std::endl;
  // std::cout << "Two-body P trace: " << traces2B[1] << std::endl;
  // std::cout << "Two-body N trace: " << traces2B[2] << std::endl;
  // std::cout << "Two-body M trace: " << traces2B[3] << std::endl;
  // std::cout << "Two-body trace Lambda: " << tracesLambda[0] << std::endl;
  // std::cout << "Two-body P trace Lambda: " << tracesLambda[1] << std::endl;
  // std::cout << "Two-body N trace Lambda: " << tracesLambda[2] << std::endl;
  // std::cout << "Two-body M trace Lambda: " << tracesLambda[3] << std::endl;

  int targetN = targetMass - targetP;

  double missing2B = targetMass * (targetMass - 1) - trace2B;
  double missingP = targetP * (targetP - 1) - traceP;
  double missingN = targetN * (targetN - 1) - traceN;
  double missingM = 2 * targetP * targetN - traceM;

  // // std::cout << "Missing trace 2B: " << missing2B << std::endl;
  // std::cout << "Missing trace N: " << (1 + missingN / traceLambdaN) << std::endl;
  // std::cout << "Missing trace P: " << (1 + missingP / traceLambdaP) << std::endl;
  // std::cout << "Missing trace M: " << (1 + missingM / traceLambdaM) << std::endl;

  for (int ch1 = 0; ch1 < lambda.modelspace->GetNumberTwoBodyChannels(); ++ch1) {
    TwoBodyChannel& tbc1 = lambda.modelspace->GetTwoBodyChannel(ch1);
    arma::mat& tbme = lambda.TwoBody.GetMatrix(ch1);
    if (tbc1.Tz == 1 &&
        abs(missingN) > 1e-2) {  // To avoid instabilities instabilities! What happens if there is
                                 // a lot missing but the trace is small?
      tbme *= (1 + missingN / traceLambdaN);
    } else if (tbc1.Tz == 0 && abs(missingM) > 1e-2) {
      tbme *= (1 + missingM / traceLambdaM);
    } else if (tbc1.Tz == -1 && abs(missingP) > 1e-2) {
      tbme *= (1 + missingP / traceLambdaP);
    }
  }
  return GetTwoBodyDensity(modelspace, rho, lambda);
}

void UpdateOccupations(
    ModelSpace& modelspace,
    Operator& rho,
    double cut,
    std::string generator_occ,
    bool vs_cut,
    bool vs_calc,
    int maxPcore,
    int maxNcore,
    int maxPvalence,
    int maxNvalence) {
  arma::mat occs = rho.OneBody.diag();
  std::map<index_t, double> new_reference;

  std::cout << "Printing holes before updating occupations..." << std::endl;
  for (index_t h : modelspace.holes) {
    Orbit& oh = modelspace.GetOrbit(h);
    std::cout << h << " (" << oh.occ << "), ";  // This can be misleading as it only prints defaults
                                                // IMSRG occs != density occ even before cutting.
  }
  std::cout << std::endl;

  std::map<index_t, double> new_occs;
  for (auto& i : modelspace.all_orbits) {
    new_occs[i] = occs[i];
  }

  if (vs_cut) {
    GetTruncatedOccupationsVS(
        modelspace,
        new_occs,
        cut,
        maxPcore,
        maxNcore,
        maxPvalence,
        maxNvalence);
  } else if (cut && (generator_occ == "occ")) {  // If we make a cut and we do Occ perscription
    GetTruncatedOccupations(modelspace, new_occs, cut);
  }

  std::cout << std::fixed << std::setw(3) << "i" << ": " << std::setw(3) << "n" << " "
            << std::setw(3) << "l" << " " << std::setw(3) << "2j" << " " << std::setw(3) << "2tz"
            << "   " << std::setw(12) << "2j + 1" << " " << std::setw(12) << "untruncated"
            << " " << std::setw(12) << "truncated" << std::setw(12) << "Space" << std::endl;

  for (auto& i : modelspace.all_orbits) {
    Orbit& oi = modelspace.GetOrbit(i);
    double old_occ = oi.occ * (oi.j2 + 1);
    oi.occ = new_occs[i];
    new_reference[i] = oi.occ;
    std::string incl = "P (core)";
    if (oi.tz2 == -1 and i <= maxPvalence and i > maxPcore and vs_calc) {
      incl = "P (valence)";
    } else if (oi.tz2 == 1 and i <= maxNvalence and i > maxNcore and vs_calc) {
      incl = "P (valence)";
    } else if (i > std::max(maxPvalence, maxNvalence)) {
      incl = "Q";
    }
    // std::cout << "Untruncated: " << occs[i] << std::endl;
    // std::cout << "Truncated: " << new_occs[i] << std::endl;
    std::cout << std::fixed << std::setw(3) << i << ": " << std::setw(3) << oi.n << " "
              << std::setw(3) << oi.l << " " << std::setw(3) << oi.j2 << " " << std::setw(3)
              << oi.tz2 << "   " << std::setw(12) << std::setprecision(6) << oi.j2 + 1 << " "
              << std::setw(12) << occs[i] * (oi.j2 + 1) << " " << std::setw(12)
              << new_occs[i] * (oi.j2 + 1) << std::setw(12) << incl << std::endl;
  }

  modelspace.SetReferenceMR(new_reference, maxPcore, maxNcore, maxPvalence, maxNvalence, vs_calc);
  // modelspace.SetReference(new_reference);

  std::cout << "Printing holes after updating occupations..." << std::endl;
  for (index_t h : modelspace.holes) {
    Orbit& oh = modelspace.GetOrbit(h);
    std::cout << h << " (" << oh.occ << "), ";
  }
  std::cout << std::endl;
  std::cout << "Where the core is: " << std::endl;
  for (index_t h : modelspace.core) {
    Orbit& oh = modelspace.GetOrbit(h);
    std::cout << h << "(" << oh.occ << "), ";
  }
  std::cout << std::endl;
  std::cout << "Where the valence space is: " << std::endl;
  for (index_t h : modelspace.valence) {
    Orbit& oh = modelspace.GetOrbit(h);
    std::cout << h << "(" << oh.occ << "), ";
  }
  std::cout << std::endl;

  double numP = 0;
  double numN = 0;
  for (auto& i : modelspace.proton_orbits) {
    Orbit& oi = modelspace.GetOrbit(i);
    numP += oi.occ * (oi.j2 + 1);
  }
  for (auto& i : modelspace.neutron_orbits) {
    Orbit& oi = modelspace.GetOrbit(i);
    numN += oi.occ * (oi.j2 + 1);
  }
  std::cout << "Particle check: P: " << numP << " N: " << numN << std::endl;
}

void GetTruncatedOccupations(
    ModelSpace& modelspace,
    std::map<index_t, double>& new_occs,
    double cut) {
  std::map<index_t, double> max_occ;  // in units of absoulte particle number
  std::map<index_t, bool> avail_p_orbits;
  std::map<index_t, bool> avail_n_orbits;
  double rem_p_occ = 0;
  double rem_n_occ = 0;
  for (auto& i : modelspace.proton_orbits) {
    Orbit& oi = modelspace.GetOrbit(i);
    if ((oi.j2 + 1) * new_occs[i] < cut) {
      avail_p_orbits[i] = false;
      // max_occ[i] = 0;
      rem_p_occ += (oi.j2 + 1) * new_occs[i];
      new_occs[i] = 0;
    } else {
      avail_p_orbits[i] = true;
      max_occ[i] = oi.j2 + 1;
    }
  }

  for (auto& i : modelspace.neutron_orbits) {
    Orbit& oi = modelspace.GetOrbit(i);
    if ((oi.j2 + 1) * new_occs[i] < cut) {
      avail_n_orbits[i] = false;
      // max_occ[i] = 0;
      rem_n_occ += (oi.j2 + 1) * new_occs[i];
      new_occs[i] = 0;
    } else {
      avail_n_orbits[i] = true;
      max_occ[i] = oi.j2 + 1;
    }
  }

  std::cout << "Particles to redistribute: P: " << rem_p_occ << " N: " << rem_n_occ << std::endl;

  while (rem_p_occ > 1e-10) {
    int num_avail =
        std::count_if(avail_p_orbits.begin(), avail_p_orbits.end(), [](const auto& entry) {
          return entry.second;
        });
    if (num_avail == 0) {
      exit(EXIT_FAILURE);
    }
    double to_split = rem_p_occ / num_avail;
    for (auto& i : new_occs) {
      if (avail_p_orbits[i.first]) {
        int j2 = modelspace.GetOrbit(i.first).j2;
        double max_add = max_occ[i.first] - (j2 + 1) * i.second;
        if (to_split < max_add) {
          new_occs[i.first] += to_split / (j2 + 1.);
          rem_p_occ -= to_split;
        } else {
          new_occs[i.first] += max_add / (j2 + 1.);
          rem_p_occ -= max_add;
          avail_p_orbits[i.first] = false;
        }
      }
    }
  }

  while (rem_n_occ > 1e-10) {
    int num_avail =
        std::count_if(avail_n_orbits.begin(), avail_n_orbits.end(), [](const auto& entry) {
          return entry.second;
        });
    if (num_avail == 0) {
      exit(EXIT_FAILURE);
    }
    double to_split = rem_n_occ / num_avail;
    for (auto& i : new_occs) {
      if (avail_n_orbits[i.first]) {
        int j2 = modelspace.GetOrbit(i.first).j2;
        double max_add = max_occ[i.first] - (j2 + 1) * i.second;
        if (to_split < max_add) {
          new_occs[i.first] += to_split / (j2 + 1.);
          rem_n_occ -= to_split;
        } else {
          new_occs[i.first] += max_add / (j2 + 1.);
          rem_n_occ -= max_add;
          avail_n_orbits[i.first] = false;
        }
      }
    }
  }
}

void GetTruncatedOccupationsVS(
    ModelSpace& modelspace,
    std::map<index_t, double>& new_occs,
    double cut,
    int maxPcore,
    int maxNcore,
    int maxPvalence,
    int maxNvalence) {
  std::map<index_t, double> max_occ;  // in units of absoulte particle number
  std::map<index_t, std::string> avail_p_orbits;
  std::map<index_t, std::string> avail_n_orbits;
  double rem_p_occ = 0;
  double rem_n_occ = 0;
  for (auto& i : modelspace.proton_orbits) {
    Orbit& oi = modelspace.GetOrbit(i);
    max_occ[i] = oi.j2 + 1;
    if (i > maxPcore) {
      avail_p_orbits[i] = "valence";
      // max_occ[i] = 0;
      if (i > maxPvalence) {
        rem_p_occ += (oi.j2 + 1) * new_occs[i];
        new_occs[i] = 0;
        avail_p_orbits[i] = "excluded";
      }
    } else {
      avail_p_orbits[i] = "core";
    }
  }

  for (auto& i : modelspace.neutron_orbits) {
    Orbit& oi = modelspace.GetOrbit(i);
    max_occ[i] = oi.j2 + 1;
    if (i > maxNcore) {
      avail_n_orbits[i] = "valence";
      // max_occ[i] = 0;
      if (i > maxNvalence) {
        rem_n_occ += (oi.j2 + 1) * new_occs[i];
        new_occs[i] = 0;
        avail_n_orbits[i] = "excluded";
      }
    } else {
      avail_n_orbits[i] = "core";
    }
  }

  std::cout << "Particles to redistribute (VS): P: " << rem_p_occ << " N: " << rem_n_occ
            << std::endl;

  bool PcoreFull = false;
  bool NcoreFull = false;
  std::string filling = "core";

  while (rem_p_occ > 1e-10) {
    int num_avail =
        std::count_if(avail_p_orbits.begin(), avail_p_orbits.end(), [&filling](const auto& entry) {
          return entry.second == filling;
        });
    if (num_avail == 0) {
      if (PcoreFull) {
        exit(EXIT_FAILURE);
      }
      PcoreFull = true;
      filling = "valence";
      continue;
    }
    double to_split = rem_p_occ / num_avail;
    for (auto& i : new_occs) {
      if (avail_p_orbits[i.first] == filling) {
        int j2 = modelspace.GetOrbit(i.first).j2;
        double max_add = max_occ[i.first] - (j2 + 1) * i.second;
        if (to_split < max_add) {
          new_occs[i.first] += to_split / (j2 + 1.);
          rem_p_occ -= to_split;
        } else {
          new_occs[i.first] += max_add / (j2 + 1.);
          rem_p_occ -= max_add;
          avail_p_orbits[i.first] = "full";
        }
      }
    }
  }

  filling = "core";
  while (rem_n_occ > 1e-10) {
    int num_avail =
        std::count_if(avail_n_orbits.begin(), avail_n_orbits.end(), [&filling](const auto& entry) {
          return entry.second == filling;
        });
    if (num_avail == 0) {
      if (NcoreFull) {
        exit(EXIT_FAILURE);
      }
      NcoreFull = true;
      filling = "valence";
      continue;
    }
    double to_split = rem_n_occ / num_avail;
    for (auto& i : new_occs) {
      if (avail_n_orbits[i.first] == filling) {
        int j2 = modelspace.GetOrbit(i.first).j2;
        double max_add = max_occ[i.first] - (j2 + 1) * i.second;
        if (to_split < max_add) {
          new_occs[i.first] += to_split / (j2 + 1.);
          rem_n_occ -= to_split;
        } else {
          new_occs[i.first] += max_add / (j2 + 1.);
          rem_n_occ -= max_add;
          avail_n_orbits[i.first] = "full";
        }
      }
    }
  }
}

Operator BuildHamiltonian(
    ReadWrite& rw,
    ModelSpace& modelspace,
    std::string in_dir,
    std::string in_iso,
    int file2e1max,
    int file2e2max,
    int file2lmax,
    double BetaCM,
    double hwBetaCM) {
  Operator H = Operator(modelspace, 0, 0, 0, 2);
  H.SetHermitian();

  std::cout << "Reading Hint files..." << std::endl;

  std::string hint_me1j = in_dir + "/" + "HintHO_ens_" + in_iso + ".me1j";
  std::string hint_me2j = in_dir + "/" + "HintHO_ens_" + in_iso + ".me2jp";
  // std::string hint_me1j = in_dir + "/" + "HintHO_" + in_iso + ".me1j";
  // std::string hint_me2j = in_dir + "/" + "HintHO_" + in_iso + ".me2jp";
  rw.Read_me1j(hint_me1j, H, modelspace.Emax, modelspace.Lmax);
  rw.Read_me2jp(
      hint_me2j,
      H,
      file2e1max,
      file2e2max,
      file2lmax);  // I have to double check that file.. args are the correct ones

  if (std::abs(BetaCM) > 1e-3) {
    if (hwBetaCM < 0) hwBetaCM = modelspace.GetHbarOmega();
    std::ostringstream hcm_opname;
    hcm_opname << "HCM_" << hwBetaCM;
    H += BetaCM * imsrg_util::OperatorFromString(modelspace, hcm_opname.str());
  }

  return H;
}

Operator BuildHamiltonian(
    ReadWrite& rw,
    ModelSpace& modelspace,
    std::string inputtbme,
    int file2e1max,
    int file2e2max,
    int file2lmax,
    std::string fmt2,
    std::string input3bme,
    int file3e1max,
    int file3e2max,
    int file3e3max,
    std::string fmt3,
    std::string input3bme_type,
    std::string no2b_precision,
    double BetaCM,
    double hwBetaCM) {
  int particle_rank = input3bme == "none" ? 2 : 3;
  Operator Hbare = Operator(modelspace, 0, 0, 0, particle_rank);
  Hbare.SetHermitian();

  std::cout << "Reading interactions..." << std::endl;

  if (inputtbme != "none") {
    if (fmt2 == "me2j")
      rw.ReadBareTBME_Darmstadt(inputtbme, Hbare, file2e1max, file2e2max, file2lmax);
    rw.File2N = inputtbme;
    std::cout << "done reading 2N" << std::endl;
  }

  // Read in the 3-body file
  if (Hbare.particle_rank >= 3) {
    rw.Set3NFormat(fmt3);
    if (input3bme_type == "full") {
      rw.Read_Darmstadt_3body(input3bme, Hbare, file3e1max, file3e2max, file3e3max);
    }
    if (input3bme_type == "no2b") {
      Hbare.ThreeBody.SetMode("no2b");
      if (no2b_precision == "half") {
        Hbare.ThreeBody.SetMode("no2bhalf");
      }
      Hbare.ThreeBody.ReadFile({input3bme}, {file3e1max, file3e2max, file3e3max, file3e1max});
    }
    rw.File3N = input3bme;
    std::cout << "done reading 3N" << std::endl;
  }

  Hbare += imsrg_util::Trel_Op(modelspace);

  // Add a Lawson center of mass term. If hwBetaCM is specified, use that frequency, otherwise use
  // the basis frequency
  if (std::abs(BetaCM) > 1e-3) {
    if (hwBetaCM < 0) hwBetaCM = modelspace.GetHbarOmega();
    std::ostringstream hcm_opname;
    hcm_opname << "HCM_" << hwBetaCM;
    Hbare += BetaCM * imsrg_util::OperatorFromString(modelspace, hcm_opname.str());
  }

  return Hbare;
}

// Operator GetLambda(ModelSpace& ms, Operator& rho) {
//   std::cout << "Constructing new lambda..." << std::endl;
//   Operator lambda = Operator(ms);
//   int nchan = ms.GetNumberTwoBodyChannels();
//   for (int ch = 0; ch < nchan; ++ch) {
//     arma::mat& LAM = lambda.TwoBody.GetMatrix(ch);
//     arma::mat& RHO2 = rho.TwoBody.GetMatrix(ch);
//     TwoBodyChannel& tbc = ms.GetTwoBodyChannel(ch);
//     for (int ibra = 0; ibra < tbc.GetNumberKets(); ++ibra) {
//       Ket& bra = tbc.GetKet(ibra);
//       for (int iket = 0; iket < tbc.GetNumberKets(); ++iket) {
//         Ket& ket = tbc.GetKet(iket);
//         LAM(ibra, iket) = RHO2(ibra, iket) - rho.OneBody(bra.p, ket.p) * rho.OneBody(bra.q,
//         ket.q)
//         +
//                           rho.OneBody(bra.p, ket.q) * (bra.q, ket.p);
//       }
//     }
//   }
//   return lambda;
// }

HFMBPT SolveHF(
    Operator& Hbare,
    bool freeze_occupations,
    std::string basis,
    std::string NAT_order,
    bool order_NAT_by_energy,
    bool use_NAT_occupations,
    bool use_HF_reference_in_NAT) {
  std::cout << "Creating HF" << std::endl;
  HFMBPT hf(Hbare);  // HFMBPT inherits from HartreeFock, so this works for HF and NAT bases.

  if (!freeze_occupations) hf.UnFreezeOccupations();
  std::cout << "Solving" << std::endl;

  if (basis != "oscillator") {
    hf.Solve();
    if (basis == "NAT") {
      // we want to use the natural orbital basis
      // for backwards compatibility: order_NAT_by_energy overrides NAT_order
      if (order_NAT_by_energy) NAT_order = "energy";

      hf.UseNATOccupations(use_NAT_occupations);
      hf.OrderNATBy(NAT_order);

      //  GetNaturalOrbitals() calls GetDensityMatrix(), which computes the 1b density matrix up to
      //  MBPT2 using the NO2B Hamiltonian in the HF basis, obtained with GetNormalOrderedH(). Then
      //  it calls DiagonalizeRho() which diagonalizes the density matrix, yielding the natural
      //  orbital basis.
      hf.GetNaturalOrbitals();
      if (use_HF_reference_in_NAT) {
        hf.UseHFForHoleAndValenceStates();
      }
    }
  }

  std::cout << basis << " Single particle energies and wave functions:" << std::endl;
  hf.PrintSPEandWF();
  std::cout << std::endl;

  return hf;
}

Operator GetNormalOrderedHamiltonian(
    Operator& Hbare,
    HFMBPT& hf,
    std::string basis,
    double BetaCM,
    double hwBetaCM) {
  const int hno_particle_rank = 2;
  double CM_0body =
      -1 * BetaCM * 1.5 *
      hwBetaCM;  // This is just the zero-body piece. The other stuff was added earlier.
  std::cout << BetaCM << " " << hwBetaCM << " " << basis << std::endl;
  if (basis == "HF") {
    double NO0B = hf.GetNormalOrderedH(hno_particle_rank).ZeroBody;
    std::cout << NO0B + CM_0body << std::endl;
    return hf.GetNormalOrderedH(hno_particle_rank) + CM_0body;
  } else if (basis == "NAT") {
    return hf.GetNormalOrderedHNAT(hno_particle_rank) + CM_0body;

  } else if (basis == "oscillator") {
    return Hbare.DoNormalOrdering() + CM_0body;
  }

  exit(EXIT_FAILURE);
}

void EvaluateMBPT2(Operator& HNO) {
  std::cout << "Hbare 0b = " << std::setprecision(8) << HNO.ZeroBody << std::endl;
  std::cout << "Perturbative estimates of gs energy:" << std::endl;
  double EMP2 = HNO.GetMP2_Energy();
  double EMP2_3B = HNO.GetMP2_3BEnergy();
  std::cout << "EMP2 = " << EMP2 << std::endl;
  std::cout << "EMP2_3B = " << EMP2_3B << std::endl;
  // std::array<double,3> Emp_3 = HNO.GetMP3_Energy();
  // double EMP3 = Emp_3[0]+Emp_3[1]+Emp_3[2];
  // std::cout << "E3_pp = " << Emp_3[0] << "  E3_hh = " << Emp_3[1] << " E3_ph = " << Emp_3[2] <<
  // "   EMP3 = " << EMP3 << std::endl; std::cout << "To 3rd order, E = " << HNO.ZeroBody + EMP2 +
  // EMP3 + EMP2_3B << std::endl;
  std::cout << "done with pert stuff" << std::endl;
}

void CheckMethod(std::string method) {
  std::cout << "method = " << method << std::endl;
  const std::vector<std::string> supported_methods = {"magnus", "flow"};
  const std::vector<std::string> operator_evolution_supported_methods = {"magnus"};
  bool valid_method = false;
  bool op_evolution_supported = false;
  for (const auto& x : supported_methods) {
    if (method == x) {
      valid_method = true;
    }
  }
  if (!valid_method) {
    std::cout << "ERROR: Method is unsupported." << std::endl;
    exit(EXIT_FAILURE);
  }
  for (const auto& x : operator_evolution_supported_methods) {
    if (method == x) {
      op_evolution_supported = true;
    }
  }
  if (!op_evolution_supported) {
    std::cout << "WARNING: Evolving other operators is not supported with this method." << std::endl;
  }
}

arma::mat DiagonalizeRho(Operator& rho, Operator& Hbare, ModelSpace& modelspace) {
  arma::mat C = arma::mat(modelspace.norbits, modelspace.norbits, arma::fill::eye);
  arma::vec Occ = arma::vec(modelspace.norbits, arma::fill::zeros);
  for (auto& it : Hbare.OneBodyChannels) {
    arma::uvec orbvec(std::vector<index_t>(it.second.begin(), it.second.end()));
    arma::mat rho_ch = rho.OneBody.submat(orbvec, orbvec);
    arma::mat vec;
    arma::vec eig;
    bool success = false;
    success = arma::eig_sym(
        eig,
        vec,
        rho_ch);  // eigenvalues of rho (i.e. occupations) are in ascending order
    if (not success) {
      std::cout << "Error in diagonalization of density matrix" << std::endl;
      std::cout << "Density Matrix:" << std::endl;
      rho_ch.print();
      exit(0);
    }
    // vec.print();
    // int dim = std::min(vec.n_cols, vec.n_rows);
    // bool zerosOnDiag = true;
    // for (int d = 0; d < dim; ++d) {
    //   if (std::abs(vec(d, d)) > 1e-8) {
    //     zerosOnDiag = false;
    //     break;
    //   }
    // }
    // if (zerosOnDiag) {
    //   vec.eye();
    // }
    // std::cout << "--\n";
    // vec.print();
    // std::cout << "---------\n";
    //    Occ(orbvec_d) = eig; // assigning with descending indexes produces NAT orbits ordered
    //    by descending occupation. C_HF2NAT.submat(orbvec, orbvec_d) = vec; // NAT orbits
    //    ordered by descending occupation

    Occ(orbvec) = arma::reverse(eig);  // reverse so NAT orbits ordered by descending occupation.
    C.submat(orbvec, orbvec) =
        arma::reverse(vec, 1);  // "1" means reverse elements in each row
                                //    std::cout << "1-body channel ";
                                //    for ( auto x : it.first ) std::cout << x << " ";
                                //    std::cout << "  : " << std::endl << Occ(orbvec) << std::endl;
    //    std::cout << " C submat: " << std::endl << C_HF2NAT.submat(orbvec, orbvec ) << std::endl;
  }

  return C;
  // Choose ordering and phases so that C_HF2NAT looks as close to the identity as possible
  // NB: Rather than C_HF2NAT being close to the identity, we really want the orbits for a given
  // (l,j,tz) ordered according to decreasing occupation, so that a later emax_imsrg truncation is
  // reasonable. -SRS Jan 2021.
  //  ReorderHFMBPTCoefficients();
  //  std::cout << " line " << __LINE__ << "   Occ = " << std::endl << Occ << std::endl;
}

void SwapDegenerate(Operator& H, HFMBPT& hf, int maxCoreIndex, int maxValenceIndex, ModelSpace& ms) {
  std::map<int, int> swapMap = {};
  std::map<std::vector<int>, std::vector<int>> JMap = {};

  for (int i = maxCoreIndex + 1; i < ms.GetNumberOrbits(); ++i) {
    Orbit& o = ms.GetOrbit(i);
    JMap[{o.l, o.j2, o.tz2}].push_back(i);
  }

  Operator HTrafo = hf.TransformHOToNATBasis(H).DoNormalOrdering();

  for (auto& js : JMap) {
    int minIndex = js.second[0];
    int minEIndex = js.second[0];
    if (minIndex > maxValenceIndex) continue;
    for (auto& i : js.second) {
      if (i < maxValenceIndex) continue;
      if (HTrafo.OneBody(i, i) < HTrafo.OneBody(minEIndex, minEIndex)) {
        minEIndex = i;
      }
    }
    if (minIndex != minEIndex) {
      swapMap[minIndex] = minEIndex;  // One might also sort other values if you have more empty
                                      // statee of the same channel in the P space
    }
  }

  for (auto& idx : swapMap) {
    std::cout << "Swapping: " << idx.first << ": " << HTrafo.OneBody(idx.first, idx.first)
              << " with " << idx.second << ": " << HTrafo.OneBody(idx.second, idx.second)
              << std::endl;
    hf.C_HO2NAT.swap_cols(idx.first, idx.second);
  }

  // for (int i = 0; i <= maxPIndex; ++i) {
  //   if (abs(coeffs(i, i)) > 1e-8) continue;
  //   Orbit& oi = ms.GetOrbit(i);
  //   for (int j = maxPIndex + 1; j < ms.GetNumberOrbits(); ++j) {
  //     Orbit oj = ms.GetOrbit(j);
  //     if (oi.l == oj.l and oi.j2 == oj.j2 and oi.tz2 == oi.tz2 and abs(1 - coeffs(i, j)) < 1e-8)
  //     {
  //       swapMap[i] = j;
  //     }
  //   }
  // }
}

// Dummy function to do something I want at the end
void OutputHelper(Operator& H, ModelSpace& ms) {
  for (int ch = 0; ch < H.nChannels; ch++) {
    TwoBodyChannel& c = ms.GetTwoBodyChannel(ch);
    std::cout << ch << " " << (2 * c.J + 1) << " " << c.Tz << " " << c.parity << std::endl;
    std::string fname = "mat_" + std::to_string(ch) + ".txt";
    H.TwoBody.GetMatrix(ch).save(fname, arma::raw_ascii);
    if (ch == 1) {
      for (int iket = 0; iket < c.GetNumberKets(); ++iket) {
        Ket& ket = c.GetKet(iket);
        std::cout << ket.p << " " << ket.q << std::endl;
      }
    }
  }
}
