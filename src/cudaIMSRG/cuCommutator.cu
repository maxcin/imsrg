#include "cuCommutator.hh"
#include <ModelSpace.hh>

// #ifndef CUCO_BITWISE_DOUBLE
// #define CUCO_BITWISE_DOUBLE
// CUCO_DECLARE_BITWISE_COMPARABLE(double)  
// #endif

#define COOT_DONT_USE_OPENCL
#define COOT_USE_CUDA
#define COOT_DEFAULT_BACKEND CUDA_BACKEND
#include <bandicoot>

// __device__ void cuCommutator::initialize_MppMhh()
// {
//   Mpp.modelspace = modelspace;
//   Mpp.J = 0;
//   Mpp.P = 0;
//   Mpp.T = 0;

//   Mhh.modelspace = modelspace;
//   Mhh.J = 0;
//   Mhh.P = 0;
//   Mhh.T = 0;

//   Mpp.allocate();
//   Mhh.allocate();

// }

// __device__ void cuCommutator::destroy_MppMhh()
// {
//   Mpp.deallocate();
//   Mhh.deallocate();
// }

__global__ void kernel_setupComm(cuCommutator* C)
{
  C->setupComm();
}

__global__ void kernel_cleanComm(cuCommutator* C)
{
  C->cleanComm();
}

void cuCommSetup(cuCommutator* C)
{
  kernel_setupComm<<<1,1>>>(C);
}

void cuCommClean(cuCommutator* C)
{
  kernel_cleanComm<<<1,1>>>(C);
}


__device__ void cuCommutator::setupComm()
{
  // initialize_MppMhh();
  int Nch_cc = modelspace->Nchannels_cc;
  cudaMalloc(&Z_cc, sizeof(Matrix)*Nch_cc);
}

__device__ void cuCommutator::cleanComm()
{
  // destroy_MppMhh();
  cudaFree(Z_cc);
}

__global__ void kernel_Z_cc_mem_ptr(cuCommutator* Comm, int ich_cc, double* mem_ptr)
{
  // printf("Assigning Z_cc for channel %d / %d  at address %p" , ich_cc, Comm->modelspace->Nchannels_cc, mem_ptr);
  
  int nkets_cc = Comm->modelspace->Nkets_channel_cc[ich_cc];
  // printf(" number of elements %d\n", nkets_cc*2*nkets_cc);
  Comm->Z_cc[ich_cc].nrows = nkets_cc;
  Comm->Z_cc[ich_cc].ncols = 2*nkets_cc;
  Comm->Z_cc[ich_cc].data = mem_ptr;
}

// 110ss could (in principle) be split up by performing atomic operations on Z.ZeroBody for the outer loop
//That is overkill for a reference implementation

__global__ void kernel_cuComm110ss(cuOperator* X, cuOperator* Y, cuOperator* Z)
{
    Matrix& X1 = X->OneBody;
    Matrix& Y1 = Y->OneBody;
    double z0 = 0;

    cuModelSpace* modelspace = X->modelspace;
    int* j2 = modelspace->j2;
    double* occ = modelspace->occ;

    for(int a = 0; a < modelspace->Norbits; ++a)
    {
      for(int b = 0; b < modelspace->Norbits; ++b)
      {
        z0 += (j2[a] + 1) * occ[a] * (1 - occ[b]) * (X1(a, b) * Y1(b, a) - Y1(a, b) * X1(b, a));
      }
    }
    Z->ZeroBody += z0;
}


//This can be very much optimized using techniques from i.e. https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
//but keep it simple for now and only start to get more complicated once it is worth it

//perform reduction with single thread in each channel (<- this is bottleneck for sure)
//Then perform atomic sum on Zero Body element of Z

//For me this gives problems on a 1070. Enable compiler flag -arch=sm_60 to force at least compute capability 6.0 (adds atomicAdd for doubles)
__global__ void kernel_cuComm220ss(int ich, cuOperator* X, cuOperator* Y, cuOperator* Z)
{
    cuModelSpace& modelspace = *(X->modelspace);

    double* occ = modelspace.occ;

    int Nkets = modelspace.Nkets_channel[ich];

    Matrix& X2 = X->TwoBody->MatEl[ich];
    Matrix& Y2 = Y->TwoBody->MatEl[ich];

    double E_ch = 0.0;

    int J = modelspace.Jchannel[ich];
    
    for(int ab = 0; ab < Nkets; ++ab)
    {
      int ab_global = modelspace.GetGlobalIndex(ich, ab);
      int a = modelspace.Ket_p[ab_global];
      int b = modelspace.Ket_q[ab_global];

      double na = occ[a];
      double nb = occ[b];
      for(int cd = 0; cd < Nkets; ++cd)
      {
        int cd_global = modelspace.GetGlobalIndex(ich, cd);
        int c = modelspace.Ket_p[cd_global];
        int d = modelspace.Ket_q[cd_global];

        double ncbar = 1-occ[c];
        double ndbar = 1-occ[d];
        
        E_ch += (2*J+1) * (X2(ab,cd)*Y2(cd,ab)-Y2(ab,cd)*X2(cd,ab))*na*nb*ncbar*ndbar;
      }
    }

    atomicAdd( &(Z->ZeroBody), E_ch);
}

//Outer loop gets absorbed into multiple kernel launches
__global__ void kernel_cuComm220ss_new(int ich, cuOperator* X, cuOperator* Y, cuOperator* Z)
{
    int ab = threadIdx.x;

    cuModelSpace& modelspace = *(X->modelspace);

    double* occ = modelspace.occ;

    int Nkets = modelspace.Nkets_channel[ich];

    Matrix& X2 = X->TwoBody->MatEl[ich];
    Matrix& Y2 = Y->TwoBody->MatEl[ich];

    double E_ch = 0.0;

    int J = modelspace.Jchannel[ich];
    
    if(ab < Nkets)
    {
      int ab_global = modelspace.GetGlobalIndex(ich, ab);
      int a = modelspace.Ket_p[ab_global];
      int b = modelspace.Ket_q[ab_global];

      double na = occ[a];
      double nb = occ[b];
      for(int cd = 0; cd < Nkets; ++cd)
      {
        int cd_global = modelspace.GetGlobalIndex(ich, cd);
        int c = modelspace.Ket_p[cd_global];
        int d = modelspace.Ket_q[cd_global];

        double ncbar = 1-occ[c];
        double ndbar = 1-occ[d];
        
        E_ch += (2*J+1) * (X2(ab,cd)*Y2(cd,ab)-Y2(ab,cd)*X2(cd,ab))*na*nb*ncbar*ndbar;
      }
    }

    atomicAdd( &(Z->ZeroBody), E_ch);
}



//111ss implemented to calculate one element per thread (this is likely inefficient because many zeros are there)
__global__ void kernel_cuComm111ss(cuOperator* X, cuOperator* Y, cuOperator* Z)
{
    Matrix& X1 = X->OneBody;
    Matrix& Y1 = Y->OneBody;
    Matrix& Z1 = Z->OneBody;
    int i = threadIdx.x;
    int j = blockIdx.x;

    int Norbits = X->modelspace->Norbits;
    if(i < Norbits and j < Norbits)
    {
      for(int a = 0; a < X->modelspace->Norbits; ++a)
        Z1(i, j) += X1(i, a) * Y1(a, j) - Y1(i, a) * X1(a, j);
    }
    
}


//122 commutator. This is taken nearly directly from the "slow" version of the CPU code
//Hopefully GPU takes care of this one
//There may be complicated branching divergence due to the OneBody channel checks
//That can likely be optimized in modelspace if necessary
__global__ void kernel_cuComm122ss(int ich, cuOperator* X, cuOperator* Y, cuOperator* Z)
{
    int ij = blockIdx.x * blockDim.x + threadIdx.x;
    int kl = blockIdx.y * blockDim.y + threadIdx.y;
    
    cuModelSpace* modelspace = X->modelspace;
    
    int* qn_l = modelspace->l;
    int* j2 = modelspace->j2;
    int* tz2 = modelspace->tz2;

    int Nkets = modelspace->Nkets_channel[ich];
    int Norbits = modelspace->Norbits;
    
    cuTwoBodyME& X2 = * X->TwoBody;
    cuTwoBodyME& Y2 = * Y->TwoBody;
    cuTwoBodyME& Z2 = * Z->TwoBody;

    Matrix& X1 = X->OneBody;
    Matrix& Y1 = Y->OneBody;

    if(ij < Nkets and kl < Nkets and kl >= ij)
    {
      int ij_global = modelspace->globalIndex[ich][ij];
      int kl_global = modelspace->globalIndex[ich][kl];

      int i = modelspace->Ket_p[ij_global];
      int j = modelspace->Ket_q[ij_global];

      int k = modelspace->Ket_p[kl_global];
      int l = modelspace->Ket_q[kl_global];

      int J = modelspace->Jchannel[ich];

      double zijkl = 0;
      // X1 Y2
      //for (size_t a : X.OneBodyChannels.at({oi.l, oi.j2, oi.tz2}))
      for(int a = 0; a < Norbits; ++a)
      {
        if(qn_l[a] == qn_l[i] and j2[a] == j2[i] and tz2[a] == tz2[i])
        zijkl += X1(i, a) * Y2.GetTBME_J(J, a, j, k, l);
      }
      //for (size_t a : X.OneBodyChannels.at({oj.l, oj.j2, oj.tz2}))
      for(int a = 0; a < Norbits; ++a)
      {
        if(qn_l[a] == qn_l[j] and j2[a] == j2[j] and tz2[a] == tz2[j])
        zijkl += X1(j, a) * Y2.GetTBME_J(J, i, a, k, l);
      }
      // X2 Y1
      //for (size_t a : Y.OneBodyChannels.at({ok.l, ok.j2, ok.tz2}))
      for(int a = 0; a < Norbits; ++a)
      {
        if(qn_l[a] == qn_l[k] and j2[a] == j2[k] and tz2[a] == tz2[k])
        zijkl += X2.GetTBME_J(J, i, j, a, l) * Y1(a, k);
      }
      //for (size_t a : Y.OneBodyChannels.at({ol.l, ol.j2, ol.tz2}))
      for(int a = 0; a < Norbits; ++a)
      {
        if(qn_l[a] == qn_l[l] and j2[a] == j2[l] and tz2[a] == tz2[l])
        zijkl += X2.GetTBME_J(J, i, j, k, a) * Y1(a, l);
      }

      // Y1 X2
      //for (size_t a : Y.OneBodyChannels.at({oi.l, oi.j2, oi.tz2}))
      for(int a = 0; a < Norbits; ++a)
      {
        if(qn_l[a] == qn_l[i] and j2[a] == j2[i] and tz2[a] == tz2[i])
        zijkl -= Y1(i, a) * X2.GetTBME_J(J, a, j, k, l);
      }
      //for (size_t a : Y.OneBodyChannels.at({oj.l, oj.j2, oj.tz2}))
      for(int a = 0; a < Norbits; ++a)
      {
        if(qn_l[a] == qn_l[j] and j2[a] == j2[j] and tz2[a] == tz2[j])
        zijkl -= Y1(j, a) * X2.GetTBME_J(J, i, a, k, l);
      }
      // Y2 X1
      //for (size_t a : X.OneBodyChannels.at({ok.l, ok.j2, ok.tz2}))
      for(int a = 0; a < Norbits; ++a)
      {
        if(qn_l[a] == qn_l[k] and j2[a] == j2[k] and tz2[a] == tz2[k])
        zijkl -= Y2.GetTBME_J(J, i, j, a, l) * X1(a, k);
      }
      //for (size_t a : X.OneBodyChannels.at({ol.l, ol.j2, ol.tz2}))
      for(int a = 0; a < Norbits; ++a)
      {
        if(qn_l[a] == qn_l[l] and j2[a] == j2[l] and tz2[a] == tz2[l])
        zijkl -= Y2.GetTBME_J(J, i, j, k, a) * X1(a, l);
      }

      // Need to normalize here, because AddToTBME expects a normalized TBME.
      if (i == j)
        zijkl /= sqrt(2.0);
      if (k == l)
        zijkl /= sqrt(2.0);

      Z2.AddToTBME(ich, i,j, k,l, zijkl);
    }
}



//MppMhh implementation, fairly simple implementation assuming that operators are either hermitian/antihermitian
//Start a kernel for each channel separately
__global__ void kernel_cuConstructScalarMpp_Mhh(int ich, cuOperator* X, cuOperator* Y, cuCommutator* Comm)
{
    cuTwoBodyME& Mpp = * Comm->Mpp;
    cuTwoBodyME& Mhh = * Comm->Mhh;
    
    int ij = blockIdx.x * blockDim.x + threadIdx.x;
    int kl = blockIdx.y * blockDim.y + threadIdx.y;
    
    cuModelSpace* modelspace = X->modelspace;
    double* occ = modelspace->occ;
    int Nkets = modelspace->Nkets_channel[ich];
    
    Matrix& X_mat = X->TwoBody->MatEl[ich];
    Matrix& Y_mat = Y->TwoBody->MatEl[ich];
    
    int hX = X->hermitian ? +1 : -1;
    int hY = Y->hermitian ? +1 : -1;
    
    if(ij < Nkets and kl < Nkets)
    {
        for(int ab = 0; ab < Nkets; ++ab)
        {
            int iket_ab_global = modelspace->globalIndex[ich][ab];
            int a = modelspace->Ket_p[iket_ab_global];
            int b = modelspace->Ket_q[iket_ab_global];
    
            //Column major order means same coloumn = close in memory => row index is ab 
            // double TBME = X_mat(ij,ab) * Y_mat(ab,kl) - Y_mat(ij,ab) * X_mat(ab,kl);
            double TBME = hX*X_mat(ab,ij) * Y_mat(ab,kl) - hY*Y_mat(ab,ij) * X_mat(ab,kl);
            
    
            double occ_factor = 1.0;
    
            if(occ[a] > 1e-9 or occ[b] > 1e-9) occ_factor = (1-occ[a])*(1-occ[b]);
    
            Mpp.MatEl[ich](ij, kl) += occ_factor * TBME;
    
            if(occ[a] > 1e-9 and occ[b] > 1e-9)
            {
              Mhh.MatEl[ich](ij,kl) += occ[a]*occ[b]*TBME;
            }
        }
    }

}


//After calculating Mpp_Mhh we need to add it to the TwoBody parts
__global__ void kernel_cu222add_pp_hhss(int ich, cuOperator* Z, cuCommutator* Comm)
{
    cuTwoBodyME& Mpp = * Comm->Mpp;
    cuTwoBodyME& Mhh = * Comm->Mhh;

    int ij = blockIdx.x * blockDim.x + threadIdx.x;
    int kl = blockIdx.y * blockDim.y + threadIdx.y;

    Matrix& Mpp_mat = Mpp.MatEl[ich];
    Matrix& Mhh_mat = Mhh.MatEl[ich];
    Matrix& Z_mat = Z->TwoBody->MatEl[ich];

    cuModelSpace* modelspace = Z->modelspace;
    int Nkets = modelspace->Nkets_channel[ich];
    
    if(ij < Nkets and kl < Nkets)
    {
      Z_mat(ij,kl) += Mpp_mat(ij,kl);
      Z_mat(ij,kl) -= Mhh_mat(ij,kl);
    }
}

__global__ void kernel_cu221add(cuOperator* Z, cuCommutator* Comm)
{
    int i = threadIdx.x;
    int j = blockIdx.x;

    int Norbits = Z->modelspace->Norbits;

    Matrix& Z1 = Z->OneBody;
    cuTwoBodyME& Mpp = * Comm->Mpp;
    cuTwoBodyME& Mhh = * Comm->Mhh;

    int* l = Z->modelspace->l;
    int* j2 = Z->modelspace->j2;
    int* tz2 = Z->modelspace->tz2;
    double* occ = Z->modelspace->occ;

    int hZ = Z->hermitian ? 1 : -1;

    if(j < i) return;
    if(i < Norbits and j < Norbits)
    {
        if(l[i] != l[j] or j2[i] != j2[j] or tz2[i] != tz2[j]) return;
        double zij = 0.0;
        //for (auto &c : Z.modelspace->all_orbits)
        for(int c = 0; c<Norbits; ++c)
        {
          double nc = occ[c];
          double nbarc = 1.0 - nc;

          int Jmin = max(abs(j2[c]-j2[i]),abs(j2[c]-j2[j])) / 2;
          int Jmax = (j2[c] + min(j2[i], j2[j])) / 2;

          if (abs(nc) > 1e-9)
          {
            for (int J = Jmin; J <= Jmax; J++)
            {
              zij += (2 * J + 1) * nc * Mpp.GetTBME_J(J, c, i, c, j);
            }
          }
          if (abs(nbarc) > 1e-9)
          {
            for (int J = Jmin; J <= Jmax; J++)
            {
              zij += (2 * J + 1) * nbarc * Mhh.GetTBME_J(J, c, i, c, j);
            }
          }
        }

        Z1(i, j) += zij / (j2[i] + 1.0);
        if(i != j) 
          Z1(j, i) += hZ * zij / (j2[i] + 1.0);
    }
}


//121 commutator
//Again just do each one body matrix element separately (this needs to be revisited with some unified concept of allowing multiple threads to 
// cooperate on a single one-body ME result)
__global__ void kernel_cuComm121ss(cuOperator* X, cuOperator* Y, cuOperator* Z)
{
    int i = threadIdx.x;
    int j = blockIdx.x;

    int hZ = Z->hermitian ? 1 : -1;

    cuModelSpace& modelspace =  *(X->modelspace);
    int Norbits = modelspace.Norbits;

    int* l = Z->modelspace->l;
    int* j2 = Z->modelspace->j2;
    int* tz2 = Z->modelspace->tz2;
    double* occ = Z->modelspace->occ;

    double zij = 0;

    if(l[i] != l[j] or j2[i] != j2[j] or tz2[i] != tz2[j]) return;
    if(j < i) return;
    //for (auto &a : Z.modelspace->holes) // C++11 syntax
    for(int a = 0; a<Norbits; ++a)
    {
      if(occ[a] < 1e-9) continue; //a is hole
      //Orbit &oa = Z.modelspace->GetOrbit(a);

      //for (auto b : X.GetOneBodyChannel(oa.l, oa.j2, oa.tz2))
      for(int b = 0; b<Norbits; ++b)
      {
        if(l[a] != l[b] or j2[a] != j2[b] or tz2[a] != tz2[b]) continue;

        double nanb = occ[a] * (1 - occ[b]);
        if (abs(nanb) < 1e-6)
          continue;

        double ybiaj = Y->TwoBody->GetTBMEmonopole(b, i, a, j);
        double yaibj = Y->TwoBody->GetTBMEmonopole(a, i, b, j);
        zij += (j2[b] + 1) * nanb * X->OneBody(a, b) * ybiaj;
        zij -= (j2[a] + 1) * nanb * X->OneBody(b, a) * yaibj;

        double xbiaj = X->TwoBody->GetTBMEmonopole(b, i, a, j);
        double xaibj = X->TwoBody->GetTBMEmonopole(a, i, b, j);
        zij -= (j2[b] + 1) * nanb * Y->OneBody(a, b) * xbiaj;
        zij += (j2[a] + 1) * nanb * Y->OneBody(b, a) * xaibj;
      }
    }
    Z->OneBody(i, j) += zij;
    if (j != i)
      Z->OneBody(j, i) += hZ * zij;
}

__global__ void kernel_cuPandyaTransformSingleChannel(int ich_cc, cuOperator* X, cuOperator* Y, double* Xt_bar_cc, double* Y_bar_cc)
{
    int bra_cc= blockIdx.x * blockDim.x + threadIdx.x; //2*nph
    int ket_cc = blockIdx.y * blockDim.y + threadIdx.y; //nkets_cc

    cuModelSpace* modelspace = X->modelspace;

    int* j2 = modelspace->j2;
    double* occ = modelspace->occ;

    int nph = modelspace->Nkets_channel_cc_hh_ph[ich_cc];
    int nkets_cc = modelspace->Nkets_channel_cc[ich_cc];

    int J_cc = modelspace->Jchannel_cc[ich_cc];

    int hY = Y->hermitian ? 1 : -1;

    Matrix X2_CC_ph;
    Matrix Y2_CC_ph;
    // Y_bar_cc has dimension (2*nph , nKets_CC)
    // X_bar_t_cc has dimension (nKets_CC, 2*nph )
    Y2_CC_ph.nrows = 2*nph;
    Y2_CC_ph.ncols = nkets_cc;
    Y2_CC_ph.data = Y_bar_cc;

    X2_CC_ph.ncols = 2*nph;
    X2_CC_ph.nrows = nkets_cc;
    X2_CC_ph.data = Xt_bar_cc;


    if(bra_cc < 2*nph and ket_cc < nkets_cc)
    {
      int bra_shift = bra_cc >= nph ? nph : 0; //account for both orderings
      int bra_global_index = modelspace->Kets_cc_hh_ph[ich_cc][bra_cc-bra_shift];
      int ket_global_index = modelspace->globalIndex_cc[ich_cc][ket_cc];

      int a = bra_cc >= nph ? modelspace->Ket_q[bra_global_index] :modelspace->Ket_p[bra_global_index];
      int b = bra_cc >= nph ? modelspace->Ket_p[bra_global_index] :modelspace->Ket_q[bra_global_index];

      double na_nb_factor = occ[a] - occ[b];

      int c = modelspace->Ket_p[ket_global_index];
      int d = modelspace->Ket_q[ket_global_index];

      double normfactor = 1.0;
      if (a == d)
        normfactor *= sqrt(2.0);
      if (c == b)
        normfactor *= sqrt(2.0);

      int jmin = max(abs(j2[a] - j2[d]), abs(j2[c] - j2[b]))/2;
      int jmax = min(j2[a] + j2[d], j2[c] + j2[b])/2;
      double Xbar = 0;
      double Ybar = 0;
      int dJ_std = 1;
      if (a == d or c == b)
      {
        dJ_std = 2;
        jmin += jmin % 2;
      }
      for (int J_std = jmin; J_std <= jmax; J_std += dJ_std)
      {
        // uint64_t sixj_hash = modelspace->SixJHash(j2[a]*0.5, j2[b]*0.5, J_cc, j2[c]*0.5, j2[d]*0.5, J_std);
        // double sixj = SixJs.find(sixj_hash)->second;
        double sixj = modelspace->SixJ(j2[a], j2[b], J_cc, j2[c], j2[d], J_std);
        if (abs(sixj) < 1e-8)
         continue;

        // Since we want the same element of two different operators, we use GetTBME_J_norm_twoOps
        // which does all the phase/index lookup stuff once and then just accesses the two matrices.
        double xcbad = 0;
        double yadcb = 0;
        X->TwoBody->GetTBME_J_norm_twoOps( * Y->TwoBody,  J_std, c, b, a, d, xcbad, yadcb);
        Xbar -= (2 * J_std + 1) * sixj * xcbad;
        Ybar -= (2 * J_std + 1) * sixj * yadcb * hY;
      }
      X2_CC_ph(ket_cc, bra_cc ) = Xbar * normfactor * na_nb_factor;
      Y2_CC_ph(bra_cc, ket_cc) = Ybar * normfactor;

    }

}

__global__ void kernel_cuAddInversePandya(int ich, cuOperator* Z, cuCommutator* Comm)
{
    int ibra = blockIdx.x * blockDim.x + threadIdx.x;
    int iket = blockIdx.y * blockDim.y + threadIdx.y;

    cuModelSpace* modelspace = Z->modelspace;
    int* j2 = modelspace->j2;
    int* l_qn = modelspace->l;
    int* tz2 = modelspace->tz2;

    int J = modelspace->Jchannel[ich];

    Matrix& Zmat = Z->TwoBody->MatEl[ich];
    int hZ = Z->hermitian ? 1 : -1;

    int Nkets = modelspace->Nkets_channel[ich];

    if(ibra < Nkets and iket < Nkets and iket >= ibra)
    {
      int ibra_global = modelspace->GetGlobalIndex(ich, ibra);
      int iket_global = modelspace->GetGlobalIndex(ich, iket);
      //bra
      int i = modelspace->Ket_p[ibra_global];
      int j = modelspace->Ket_q[ibra_global];

      //ket
      int k = modelspace->Ket_p[iket_global];
      int l = modelspace->Ket_q[iket_global];

      double commij = 0;
      double commji = 0;

      int parity_cc = (l_qn[i] + l_qn[l]) % 2;
      int Tz_cc = abs(tz2[i] - tz2[l]) / 2;
      int Jpmin = max(abs((j2[i] - j2[l])), abs((j2[k] - j2[j]))) / 2;
      int Jpmax = min(int(j2[i] + j2[l]), int(j2[k] + j2[j])) / 2;
      for (int Jprime = Jpmin; Jprime <= Jpmax; ++Jprime)
      {
        double sixj = modelspace->SixJ(j2[i], j2[j], J, j2[k], j2[l], Jprime);
        if (abs(sixj) < 1e-8)
          continue;
        int ch_cc = modelspace->GetTwoBodyChannelIndex(Jprime, parity_cc, Tz_cc);
        int nkets_cc = modelspace->Nkets_channel_cc[ch_cc];//tbc_cc.GetNumberKets();
        int indx_il = modelspace->GetLocalIndex_cc(ch_cc, min(i,l), max(i,l));//tbc_cc.GetLocalIndex(std::min(i, l), std::max(i, l)) + (i > l ? nkets_cc : 0);
        int indx_kj = modelspace->GetLocalIndex_cc(ch_cc, min(k,j), max(k,j));//tbc_cc.GetLocalIndex(std::min(j, k), std::max(j, k)) + (k > j ? nkets_cc : 0);
        if( i > l ) indx_il += nkets_cc;
        if( k > j ) indx_kj += nkets_cc;
        // printf("Want to access channel_cc %d with %d matrix elements\n", ch_cc, Comm->Z_cc[ch_cc].nrows*Comm->Z_cc[ch_cc].ncols);
        double me1 = Comm->Z_cc[ch_cc](indx_il, indx_kj); 
        commij -= (2 * Jprime + 1) * sixj * me1;
      }

      if (k == l)
      {
        commji = commij;
      }
      else if (i == j)
      {
        commji = ((j2[i] + j2[j] + j2[k] + j2[l])/2) % 2 == 0 ? commij : -commij;
      }
      else
      {
        // now loop over the cross coupled TBME's
        parity_cc = (l_qn[i] + l_qn[k]) % 2;
        Tz_cc = abs(tz2[i] - tz2[k]) / 2;
        Jpmin = max(abs((j2[j] - j2[l])), abs((j2[k] - j2[i]))) / 2;
        Jpmax = min((j2[j] + j2[l]), int(j2[k] + j2[i])) / 2;
        for (int Jprime = Jpmin; Jprime <= Jpmax; ++Jprime)
        {
          double sixj = modelspace->SixJ(j2[j], j2[i], J, j2[k], j2[l], Jprime);
          if (abs(sixj) < 1e-8)
            continue;
          int ch_cc = modelspace->GetTwoBodyChannelIndex(Jprime, parity_cc, Tz_cc);
          // TwoBodyChannel_CC &tbc_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_cc);
          int nkets_cc = modelspace->Nkets_channel_cc[ch_cc];// tbc_cc.GetNumberKets();
          // int indx_ik = tbc_cc.GetLocalIndex(std::min(i, k), std::max(i, k)) + (i > k ? nkets_cc : 0);
          // int indx_lj = tbc_cc.GetLocalIndex(std::min(l, j), std::max(l, j)) + (l > j ? nkets_cc : 0);
          int indx_ik = modelspace->GetLocalIndex_cc(ch_cc, min(i,k), max(i,k));
          int indx_lj = modelspace->GetLocalIndex_cc(ch_cc, min(l,j), max(l,j));
          if( i > k ) indx_ik += nkets_cc;
          if( l > j ) indx_lj += nkets_cc;
          // we always have i<=k so we should always flip Z_jlki = (-1)^{i+j+k+l} Z_iklj
          // the phase we get from that flip combines with the phase from Pij, to give the phase included below
          double me1 = Comm->Z_cc[ch_cc](indx_ik, indx_lj);
          commji -= (2 * Jprime + 1) * sixj * me1;
        }
      }
   
      int bra_dpq = i == j ? 1 : 0;
      int ket_dpq = k == l ? 1 : 0;
      double norm = bra_dpq == ket_dpq ? 1 + bra_dpq : sqrt(2.0);
      int phase = ( (j2[k]+j2[l])/2 - J )% 2 == 0 ? 1 : -1;
      double zijkl = -(commij - phase * commji) / norm;

      Zmat(ibra, iket) += zijkl;
      if (ibra != iket)
        Zmat(iket, ibra) += hZ * zijkl;
    }
}

