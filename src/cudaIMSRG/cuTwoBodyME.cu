#include "cuTwoBodyME.hh"

__global__ void cuTBMEallocate(cuTwoBodyME* TBME)
{
    int nchannels = TBME->modelspace->Nchannels;
    cudaMalloc(&TBME->MatEl, sizeof(Matrix)*nchannels); //Here is assumed scalar
}

__global__ void cuTBMEdeallocate(cuTwoBodyME* TBME)
{
    cudaFree(TBME->MatEl);
}

__global__ void cuTBME_setMatEl(cuTwoBodyME* TBME, int ich, int Nbras, int Nkets, double* mem_ptr)
{
    // printf("Assigning TBME on GPU channel : %d / %d\n", ich, TBME->modelspace->Nchannels);
    Matrix& Mat = TBME->MatEl[ich];
    Mat.nrows = Nbras;
    Mat.ncols = Nkets;
    Mat.data = mem_ptr;
}

__global__ void cuTBMEassignhermitian(cuTwoBodyME* TBME, bool hermitian, bool antihermitian)
{
    TBME->hermitian = hermitian;
    TBME->antihermitian = antihermitian;
}

// __device__ void cuTwoBodyME::allocate()
// {
//     int nchannels = modelspace->Nchannels;
//     cudaMalloc(&MatEl, sizeof(Matrix)*nchannels); //Here is assumed scalar
//     for(int ich = 0; ich < nchannels; ++ich)
//     {
//         int nkets = modelspace->Nkets_channel[ich];
//         MatEl[ich].allocate(nkets, nkets);
//     }
//     allocated = true;
// }

// __device__ void cuTwoBodyME::deallocate()
// {
//     int nchannels = modelspace->Nchannels;
//     for(int ich = 0; ich < nchannels; ++ich)
//     {
//         MatEl[ich].deallocate();
//     }

//     cudaFree(MatEl);
//     allocated = false;
// }

__device__ double cuTwoBodyME::GetTBME_J(int j, int a, int b, int c, int d) const
{
    int p = (modelspace->l[a]+modelspace->l[b]) % 2;
    int tz = (modelspace->tz2[a]+modelspace->tz2[b]) / 2;
    int ch = modelspace->GetTwoBodyChannelIndex(j , p ,tz);
    return GetTBME(ch, a,b,c,d);
}

__device__ double cuTwoBodyME::GetTBME(int ch, int a, int b, int c, int d) const
{
    double norm = 1;
    if( a==b ) norm *= sqrt(2.0);
    if( c==d ) norm *= sqrt(2.0);
    return norm* GetTBME_norm(ch, a,b,c,d);
}

__device__ double cuTwoBodyME::GetTBME_J_norm(int j, int a, int b, int c, int d) const
{
    int p = (modelspace->l[a]+modelspace->l[b]) % 2;
    int tz = (modelspace->tz2[a]+modelspace->tz2[b]) / 2;
    int ch = modelspace->GetTwoBodyChannelIndex(j , p ,tz);
    return GetTBME_norm(ch, a,b,c,d);
}

__device__ double cuTwoBodyME::GetTBME_norm(int ch, int a, int b, int c, int d) const
{
    if(not allocated) return 0;
    int bra_ind = modelspace->GetLocalIndex(ch, min(a,b), max(a,b));
    int ket_ind = modelspace->GetLocalIndex(ch, min(c,d), max(c,d));

    if( bra_ind < 0 or ket_ind < 0 or bra_ind > modelspace->Nkets_channel[ch] or ket_ind > modelspace->Nkets_channel[ch])
        return 0;

    double phase = 1;
    if( a>b ) phase *= modelspace->phase(ch, a,b);
    if( c>d ) phase *= modelspace->phase(ch, c,d);

    return phase * MatEl[ch](bra_ind, ket_ind);
}

__device__ void cuTwoBodyME::SetTBME_J(int j, int a, int b, int c, int d, double tbme)
{
    int p = (modelspace->l[a]+modelspace->l[b]) % 2;
    int tz = (modelspace->tz2[a]+modelspace->tz2[b]) / 2;
    int ch = modelspace->GetTwoBodyChannelIndex(j , p ,tz);
   SetTBME(ch, a,b,c,d, tbme);
}

__device__ void  cuTwoBodyME::SetTBME(int ch, int a, int b, int c, int d, double tbme)
{
    if(not allocated) return;
    int bra_ind = modelspace->GetLocalIndex(ch, min(a,b), max(a,b));
    int ket_ind = modelspace->GetLocalIndex(ch, min(c,d), max(c,d));

    double phase = 1;
    if(a>b) phase *= modelspace->phase(ch, a,b);
    if(c>d) phase *= modelspace->phase(ch, c,d);

    MatEl[ch](bra_ind, ket_ind) = phase* tbme;
    if(hermitian) MatEl[ch](ket_ind, bra_ind) = phase*tbme;
    if(antihermitian) MatEl[ch](ket_ind, bra_ind) = -phase*tbme;
}

__device__ void cuTwoBodyME::AddToTBME_J(int j, int a, int b, int c, int d, double tbme)
{
    int p = (modelspace->l[a]+modelspace->l[b]) % 2;
    int tz = (modelspace->tz2[a]+modelspace->tz2[b]) / 2;
    int ch = modelspace->GetTwoBodyChannelIndex(j , p ,tz);
    AddToTBME(ch, a,b,c,d, tbme);
}

__device__ void cuTwoBodyME::AddToTBME(int ch, int a, int b, int c, int d, double tbme)
{
    if(not allocated) return;
    int bra_ind = modelspace->GetLocalIndex(ch, min(a,b), max(a,b));
    int ket_ind = modelspace->GetLocalIndex(ch, min(c,d), max(c,d));

    double phase = 1;
    if(a>b) phase *= modelspace->phase(ch, a,b);
    if(c>d) phase *= modelspace->phase(ch, c,d);

    MatEl[ch](bra_ind, ket_ind) += phase* tbme;
    if(bra_ind != ket_ind)
    {
        if(hermitian) MatEl[ch](ket_ind, bra_ind) += phase*tbme;
        if(antihermitian) MatEl[ch](ket_ind, bra_ind) += -phase*tbme;
    }
    
}

__device__ double cuTwoBodyME::GetTBMEmonopole(int a, int b, int c, int d) const
{

  double norm = 1;
  if (a==b) norm *= sqrt(2.0);
  if (c==d) norm *= sqrt(2.0);
  return norm * GetTBMEmonopole_norm( a,b,c,d);
}

__device__ double cuTwoBodyME::GetTBMEmonopole_norm(int a, int b, int c, int d) const
{
   double mon = 0;

   int* l = modelspace->l;
   int* tz2 = modelspace->tz2;
   int* j2 = modelspace->j2;

   int Tzab = (tz2[a] + tz2[b])/2;
   int parityab = (l[a] + l[b])%2;
   int Tzcd = (tz2[c] + tz2[d])/2;
   int paritycd = (l[c] + l[d])%2;

//   if (Tzab != Tzcd or parityab != paritycd) return 0;
   if (  (parityab + paritycd + P)%2 !=0 ) return 0;
   if ( abs( Tzab - Tzcd) > T ) return 0;

   int jmin = abs(j2[a] - j2[b])/2;
   int jmax = (j2[a] + j2[b])/2;
   
   for (int J=jmin;J<=jmax;++J)
   {
      int ich = modelspace->GetTwoBodyChannelIndex(J,parityab,Tzab);
      mon += (2*J+1) * GetTBME_norm(ich,a,b,c,d);
   }
   mon /= (j2[a] +1)*(j2[b]+1);
   return mon;
}


__device__ void cuTwoBodyME::GetTBME_J_norm_twoOps(cuTwoBodyME& OtherTBME, int J, int a, int b, int c, int d, double& tbme_this, double& tbme_other)
{
   int* l = modelspace->l;
   int* tz2  = modelspace->tz2;
   int parity_bra = (l[a]+l[b])%2;
   int parity_ket = (l[c]+l[d])%2;
   int Tz_bra = (tz2[a]+tz2[b])/2;
   int Tz_ket = (tz2[c]+tz2[d])/2;
   tbme_this  =0;
   tbme_other =0;
   if ( (P+parity_bra+parity_ket)%2 > 0) return;
   if ( std::abs(Tz_bra-Tz_ket)!=T) return;

   int ich = modelspace->GetTwoBodyChannelIndex(J,parity_bra,Tz_bra);

   int bra_ind = modelspace->GetLocalIndex(ich, min(a,b), max(a,b));
   int ket_ind = modelspace->GetLocalIndex(ich, min(c,d), max(c,d));
   if (bra_ind < 0 or ket_ind < 0 or bra_ind > modelspace->Nkets_channel[ich] or ket_ind > modelspace->Nkets_channel[ich] )
   {
     return;
   }

   double phase = 1;
   if (a>b) phase *= modelspace->phase(ich, a,b);
   if (c>d) phase *= modelspace->phase(ich, c,d);

   tbme_this =  phase * MatEl[ich](bra_ind, ket_ind);
   tbme_other =  phase * OtherTBME.MatEl[ich](bra_ind, ket_ind);
}