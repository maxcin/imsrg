//Class to store modelspace information for use in GPU code
//Only include necessary information i.e. we don't want to generate on the GPU but just store information

//This is supposed to me a minimal working implementation

#ifndef cudaModelSpace 
#define cudaModelSpace 1

#include <stdint.h>

class cuModelSpace
{
    public:

    //One-Body includes number of orbits and a list for qns each of size Norbtis
    int Norbits;
    int* n;
    int* l;
    int* j2;
    int* tz2;
    double* occ;
    int* cvq;

    //Two-Body
    int Nchannels;
    int Nkets_modelspace;
    int* Ket_p; //List mapping Ket |pq> -> p (p<=q)
    int* Ket_q; //List mapping Ket |pq> -> q (p<=q)

    int* Jchannel;
    int* Tzchannel;
    int* Pchannel;
    int* Nkets_channel; //Number of Kets for each channel
    int** localIndex; //For each channel gives a array of length Nkets_modelspace that contains the local index in that channel (-1 if not participating)
    //use: localIndex[ch][global_index]
    int** globalIndex; //For each channel give array of length Nket[channel] containing the global Ket index
    //use: globalIndex[ch][local_index]

    //Two-Body cc (for pandya transformation...)
    int Nchannels_cc;

    int* Jchannel_cc;
    int* dTzchannel_cc; // Tz = tz_1 + tz_2 is no longer good quantum number but rather |dTz| = |tz_1 -tz_2|
    int* Pchannel_cc;

    //Need access to the full cross coupled matrices
    int* Nkets_channel_cc;
    int** localIndex_cc; //For each channel gives a array of length Nkets_modelspace that contains the local index in that channel (-1 if not participating)
    int** globalIndex_cc;  //For each channel give array of length Nket_cc[channel] containing the global Ket index
    //And also just the ph interface (contains hh and ph)
    int* Nkets_channel_cc_hh_ph;
    int** Kets_cc_hh_ph; //For each channel gives array of length Nkets_channel_cc_hh_ph[channel] containing the global Ket index
    
    //SixJs
    int N_SixJ;
    int dim1_sixj;
    int dim2_sixj;
    double* SixJCache_112112;

    __device__ int Index2(int p, int q);
    __device__ int GetTwoBodyChannelIndex(int J, int P, int Tz) {return 6 * J + 2 * (Tz + 1) + P;}
    __device__ int GetLocalIndex(int ch, int p, int q);
    __device__ int GetLocalIndex_cc(int ch_cc, int p, int q);
    __device__ int GetGlobalIndex(int ch, int ab_local);
    __device__ int phase(int ch, int p, int q);

    __device__ uint64_t SixJHash(double j1, double j2, double j3, double J1, double J2, double J3);

    __device__ uint64_t JJSToIndex(int jj1, int jj2, int j_3, int jj4, int jj5, int j_6);

    __device__ uint64_t JJ1BToIndex(int jj1);

    __device__ uint64_t J2BToIndex(int j_2);

    __device__ double SixJ(int jj1, int jj2, int j_3, int jj4, int jj5, int j_6);

};

__global__ void cuMS_allocate(cuModelSpace* ms); //allocate memory on the GPU
__global__ void cuMS_deallocate(cuModelSpace* ms); //deallocate memory

//Copy the MS information
__global__ void cuMS_Memcpy_Orbital(cuModelSpace* ms, int i, int n, int l, int j2, int tz2, double occ, int cvq);
__global__ void cuMS_Memcpy_TBC(cuModelSpace* ms, int ich, int nkets_channel, int J, int Tz, int P);

__global__ void cuMS_Memcpy_TBC_CC(cuModelSpace* ms, int ich_cc, int nkets_channel, int nkets_channel_cc_hh_ph, int J, int Tz, int P);

//Construct the remaining information (why is this not member function?)
__global__ void cuMS_construct_Ketpq(cuModelSpace* ms);
__global__ void cuMS_construct_localIndex_map(cuModelSpace* ms);
__global__ void cuMS_construct_cc_local_hh_ph(cuModelSpace* ms);


#endif