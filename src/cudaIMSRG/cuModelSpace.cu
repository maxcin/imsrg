#include "cuModelSpace.hh"
#include <stdio.h>
//Collection of functions to transfer everything to the gpu
//Modelspace can handle its own memory making the final code a bit cleaner BUT would be a mess 
//if we modify the modelspace data and would like to have it on the host again
//(There surely is a better way of doing this)

#include <thrust/swap.h>

__global__ void cuMS_allocate(cuModelSpace* ms)
{
    //One-body
    cudaMalloc(&ms->n, sizeof(int)*ms->Norbits);
    cudaMalloc(&ms->l, sizeof(int)*ms->Norbits);
    cudaMalloc(&ms->j2, sizeof(int)*ms->Norbits);
    cudaMalloc(&ms->tz2, sizeof(int)*ms->Norbits);
    cudaMalloc(&ms->cvq, sizeof(int)*ms->Norbits);

    cudaMalloc(&ms->occ, sizeof(double)*ms->Norbits);

    //Two-body basics
    cudaMalloc(&ms->Ket_p, sizeof(int)*ms->Nkets_modelspace);
    cudaMalloc(&ms->Ket_q, sizeof(int)*ms->Nkets_modelspace);
    cudaMalloc(&ms->Jchannel, sizeof(int)*ms->Nchannels);
    cudaMalloc(&ms->Tzchannel, sizeof(int)*ms->Nchannels);
    cudaMalloc(&ms->Pchannel, sizeof(int)*ms->Nchannels);
    cudaMalloc(&ms->Nkets_channel, sizeof(int)*ms->Nchannels);

    //Two-body double pointers
    cudaMalloc(&ms->localIndex, sizeof(int*)*ms->Nchannels);
    cudaMalloc(&ms->globalIndex, sizeof(int*)*ms->Nchannels);

    //For each channel there is a map for localIndex and gobalIndex
    //Here we can only allocate the localIndex
    for(int nch = 0; nch < ms->Nchannels; ++nch)
    {
        cudaMalloc(&ms->localIndex[nch], sizeof(int)*ms->Nkets_modelspace);
    }

    //Two-Body cc
    cudaMalloc(&ms->Jchannel_cc, sizeof(int)*ms->Nchannels_cc);
    cudaMalloc(&ms->dTzchannel_cc, sizeof(int)*ms->Nchannels_cc);
    cudaMalloc(&ms->Pchannel_cc, sizeof(int)*ms->Nchannels_cc);
    cudaMalloc(&ms->Nkets_channel_cc, sizeof(int)*ms->Nchannels_cc);
    cudaMalloc(&ms->Nkets_channel_cc_hh_ph, sizeof(int)*ms->Nchannels_cc);

    cudaMalloc(&ms->localIndex_cc, sizeof(int*)*ms->Nchannels_cc);
    cudaMalloc(&ms->globalIndex_cc, sizeof(int*)*ms->Nchannels_cc);
    cudaMalloc(&ms->Kets_cc_hh_ph, sizeof(int*)*ms->Nchannels_cc);

    for(int ich_cc = 0; ich_cc < ms->Nchannels_cc; ++ich_cc)
    {
        cudaMalloc(&ms->localIndex_cc[ich_cc], sizeof(int)*ms->Nkets_modelspace);
    }

    // cudaMalloc(&ms->SixJCache_112112, sizeof(double)*ms->N_SixJ);

    // printf("Allocated memory for modelspace\n");
}


__global__ void cuMS_deallocate(cuModelSpace* ms)
{
    // printf("Freeing memory for modelspace\n");
    //One-body
    cudaFree(ms->n);
    cudaFree(ms->l);
    cudaFree(ms->j2);
    cudaFree(ms->tz2);
    cudaFree(ms->cvq);

    cudaFree(ms->occ);

    //Two-body basics
    cudaFree(ms->Ket_p);
    cudaFree(ms->Ket_q);
    cudaFree(ms->Jchannel);
    cudaFree(ms->Tzchannel);
    cudaFree(ms->Pchannel);
    cudaFree(ms->Nkets_channel);

    for(int ich = 0; ich<ms->Nchannels; ++ich)
    {
        cudaFree(ms->localIndex[ich]);
        cudaFree(ms->globalIndex[ich]);
    }
    cudaFree(ms->localIndex);
    cudaFree(ms->globalIndex);

    cudaFree(ms->Jchannel_cc);
    cudaFree(ms->dTzchannel_cc);
    cudaFree(ms->Pchannel_cc);
    cudaFree(ms->Nkets_channel_cc);
    cudaFree(ms->Nkets_channel_cc_hh_ph);

    for(int ich_cc = 0; ich_cc<ms->Nchannels; ++ich_cc)
    {
        cudaFree(ms->localIndex_cc[ich_cc]);
        cudaFree(ms->globalIndex_cc[ich_cc]);
        cudaFree(ms->Kets_cc_hh_ph[ich_cc]);
    }
    cudaFree(ms->localIndex_cc);
    cudaFree(ms->globalIndex_cc);
    cudaFree(ms->Kets_cc_hh_ph);

    // cudaFree(ms->SixJCache_112112);

    // printf("Freed memory for modelspace\n");
}

__global__ void cuMS_Memcpy_Orbital(cuModelSpace* ms, int i, int n, int l, int j2, int tz2, double occ, int cvq)
{
    ms->n[i] = n;
    ms->l[i] = l;
    ms->j2[i] = j2;
    ms->tz2[i] = tz2;
    ms->occ[i] = occ;
    ms->cvq[i] = cvq;
}

__global__ void cuMS_Memcpy_TBC(cuModelSpace* ms, int ich, int nkets_channel, int J, int Tz, int P)
{
    ms->Nkets_channel[ich] = nkets_channel;
    ms->Jchannel[ich] = J;
    ms->Tzchannel[ich] = Tz;
    ms->Pchannel[ich] = P;
}

__global__ void cuMS_Memcpy_TBC_CC(cuModelSpace* ms, int ich_cc, int nkets_channel, int nkets_channel_cc_hh_ph, int J, int Tz, int P)
{
    ms->Nkets_channel_cc[ich_cc] = nkets_channel;
    ms->Nkets_channel_cc_hh_ph[ich_cc] = nkets_channel_cc_hh_ph;
    ms->Jchannel_cc[ich_cc] = J;
    ms->dTzchannel_cc[ich_cc] = Tz;
    ms->Pchannel_cc[ich_cc] = P;
}


__global__ void cuMS_construct_Ketpq(cuModelSpace* ms)
{
    for(int p = 0; p<ms->Norbits; ++p)
    {
        for(int q = 0; q<ms->Norbits; ++q)
        {
            if(q < p) continue;
            int Ket_index = ms->Index2(p,q);
            ms->Ket_p[Ket_index] = p;
            ms->Ket_q[Ket_index] = q;
        }
    }
}

__global__ void cuMS_construct_localIndex_map(cuModelSpace* ms)
{
    int ich = threadIdx.x;
    int J = ms->Jchannel[ich];
    int Tz = ms->Tzchannel[ich];
    int parity = ms->Pchannel[ich];

    //Allocate global index map to fill it up at the same time
    cudaMalloc(&ms->globalIndex[ich], sizeof(int)*ms->Nkets_channel[ich]);

    int iket_local = 0;

    for(int iket = 0; iket < ms->Nkets_modelspace; ++iket)
    {
        int p = ms->Ket_p[iket];
        int q = ms->Ket_q[iket];
        
        //Check if this ket participates in this channel
        bool participates = true;
        
        if ((p==q) and (J%2 != 0)) participates = false; // Pauli principle
        if ((ms->l[p] + ms->l[q])%2 != parity) participates =  false;
        if ((ms->tz2[p] + ms->tz2[q]) != 2*Tz) participates =  false;
        if (ms->j2[p] + ms->j2[q] < 2*J)       participates =  false;
        if ( abs(ms->j2[p] - ms->j2[q]) > 2*J)  participates =  false;

        ms->localIndex[ich][iket] = participates ? iket_local : -1; //-1 means it does not participate in this channel
        if(participates)
        {
          ms->globalIndex[ich][iket_local] = iket;
          iket_local += 1;
        } 
    }

    if(iket_local != ms->Nkets_channel[ich]) printf("For channel %d the number of kets is not as predicted %d != %d!\n", ich, iket_local, ms->Nkets_channel[ich]);


}

__global__ void cuMS_construct_cc_local_hh_ph(cuModelSpace* ms)
{
  int ich = threadIdx.x;
  int J = ms->Jchannel_cc[ich];
  int dTz = ms->dTzchannel_cc[ich];
  int parity = ms->Pchannel[ich];

  //Allocate global index map to fill it up at the same time
  cudaMalloc(&ms->globalIndex_cc[ich], sizeof(int)*ms->Nkets_channel_cc[ich]);
  cudaMalloc(&ms->Kets_cc_hh_ph[ich], sizeof(int)*ms->Nkets_channel_cc_hh_ph[ich] );

  int iket_local = 0;
  int iket_local_hh_ph = 0;

  for(int iket = 0; iket < ms->Nkets_modelspace; ++iket)
  {
      int p = ms->Ket_p[iket];
      int q = ms->Ket_q[iket];
      
      //Check if this ket participates in this channel
      bool participates = true;
      
      //if ((p==q) and (J%2 != 0)) participates = false; // Pauli principle not relevant for cc channels
      if ((ms->l[p] + ms->l[q])%2 != parity) participates =  false;
      if ( abs(ms->tz2[p] - ms->tz2[q]) != 2*dTz) participates =  false;
      if (ms->j2[p] + ms->j2[q] < 2*J)       participates =  false;
      if ( abs(ms->j2[p] - ms->j2[q]) > 2*J)  participates =  false;

      ms->localIndex_cc[ich][iket] = participates ? iket_local : -1; //-1 means it does not participate in this channel
      if(participates)
      {
        ms->globalIndex_cc[ich][iket_local] = iket;
        iket_local += 1;
      }
      //In the first pass we only fill hh kets (this is needed to make everything consistent...)
      if(ms->occ[p] > 1e-6 and ms->occ[q]>1e-6 and participates)
      {
        ms->Kets_cc_hh_ph[ich][iket_local_hh_ph] = iket;
        iket_local_hh_ph += 1;
      }
  }

  //This pass will add ph kets
  for(int iket = 0; iket < ms->Nkets_modelspace; ++iket)
  {
      int p = ms->Ket_p[iket];
      int q = ms->Ket_q[iket];
      
      //Check if this ket participates in this channel
      bool participates = true;
      
      //if ((p==q) and (J%2 != 0)) participates = false; // Pauli principle not relevant for cc channels
      if ((ms->l[p] + ms->l[q])%2 != parity) participates =  false;
      if ( abs(ms->tz2[p] - ms->tz2[q]) != 2*dTz) participates =  false;
      if (ms->j2[p] + ms->j2[q] < 2*J)       participates =  false;
      if ( abs(ms->j2[p] - ms->j2[q]) > 2*J)  participates =  false;

      //In the second pass we add ph kets
      if( (ms->occ[p] > 1e-6 xor ms->occ[q]>1e-6) and participates)
      {
        ms->Kets_cc_hh_ph[ich][iket_local_hh_ph] = iket;
        iket_local_hh_ph += 1;
      }
  }

  if(iket_local != ms->Nkets_channel_cc[ich]) printf("For cc channel %d the number of kets is not as predicted %d != %d!\n", ich, iket_local, ms->Nkets_channel_cc[ich]);
  if(iket_local_hh_ph != ms->Nkets_channel_cc_hh_ph[ich]) printf("For cc channel %d the number of hh ph kets is not as predicted %d != %d!\n", ich, iket_local, ms->Nkets_channel_cc_hh_ph[ich]);

}

__device__ int cuModelSpace::Index2(int p, int q)
{
    return p * (2 * Norbits - 1 - p) / 2 + q;
}

__device__ int cuModelSpace::GetGlobalIndex(int ch, int pq_local)
{
    int pq_global = globalIndex[ch][pq_local];
    return pq_global; 
}

__device__ int cuModelSpace::GetLocalIndex(int ch, int p, int q)
{
    int global_Index = Index2(p,q);
    return localIndex[ch][global_Index];
}

__device__ int cuModelSpace::GetLocalIndex_cc(int ch_cc, int p, int q)
{
    int global_Index = Index2(p,q);
    return localIndex_cc[ch_cc][global_Index];
}

__device__ int cuModelSpace::phase(int ch, int p, int q)
{
    int phase_ket = ((j2[p]+j2[q]) / 2 +1) % 2 == 0 ? 1 : -1;
    int phase_ch = Jchannel[ch]%2==0 ? 1 : -1;
    return phase_ket*phase_ch;
}

__device__ uint64_t cuModelSpace::SixJHash(double j1, double j2, double j3, double J1, double J2, double J3)
{
    
  uint64_t twoj1 = 2*j1;
  uint64_t twoj2 = 2*j2;
  uint64_t twoj3 = 2*j3;
  uint64_t twoJ1 = 2*J1;
  uint64_t twoJ2 = 2*J2;
  uint64_t twoJ3 = 2*J3;
  // The 6j can contain 0,3, or 4 half-integer arguments
  // If there are 3, then they can always be permuted so all the half-integers are on the bottom row.
   if ( (twoj1+twoj2+twoj3+twoJ1+twoJ2+twoJ3)%2==1)
   {
    if ( (twoj1%2)==1 )
    { 
      thrust::swap( twoj1, twoJ1);
      thrust::swap( twoj2, twoJ2);
    }
    if ( (twoj2%2)==1 )
    { 
      thrust::swap( twoj2, twoJ2);
      thrust::swap( twoj3, twoJ3);
    }
   }
   else // otherwise, we can permute so the larger entries are on the bottom row
   {
    if ( (twoj1>twoJ1) )
    { 
      thrust::swap( twoj1, twoJ1);
      thrust::swap( twoj2, twoJ2);
    }
    if ( (twoj2>twoJ2) )
    { 
      thrust::swap( twoj2, twoJ2);
      thrust::swap( twoj3, twoJ3);
    }
   }

  // Use the 6J symmetry under permutation of columns. Combine each column into a single integer
  // then sort the column indices so that any of the 6 equivalent permutations will give the same key
  uint64_t jJ1 = twoj1 + (twoJ1 << 10);
  uint64_t jJ2 = twoj2 + (twoJ2 << 10);
  uint64_t jJ3 = twoj3 + (twoJ3 << 10);


  if (jJ3 < jJ2)
    thrust::swap(jJ3, jJ2);
  if (jJ2 < jJ1)
    thrust::swap(jJ2, jJ1);
  if (jJ3 < jJ2)
    thrust::swap(jJ3, jJ2);

  return jJ1 + (jJ2 << 20) + (jJ3 << 40);
}

__device__ uint64_t cuModelSpace::JJSToIndex(int jj1, int jj2, int j_3, int jj4, int jj5, int j_6)
{
    int dim_1 = dim1_sixj;
    int dim_2 = dim2_sixj;
    return dim_1 * dim_2 * dim_1 * dim_1 * dim_2 * JJ1BToIndex(jj1)
      +            dim_2 * dim_1 * dim_1 * dim_2 * JJ1BToIndex(jj2)
      +                    dim_1 * dim_1 * dim_2 * J2BToIndex(j_3)
      +                            dim_1 * dim_2 * JJ1BToIndex(jj4)
      +                                    dim_2 * JJ1BToIndex(jj5)
      +                                            J2BToIndex(j_6);
}

__device__ uint64_t cuModelSpace::JJ1BToIndex(int jj1) {
return (jj1 - 1) / 2;
}

__device__ uint64_t cuModelSpace::J2BToIndex(int j_2) {
return j_2;
}

__device__ double cuModelSpace::SixJ(int jj1, int jj2, int j_3, int jj4, int jj5, int j_6)
{
    uint64_t index = JJSToIndex(jj1, jj2, j_3, jj4, jj5, j_6);
    return SixJCache_112112[index];
}

