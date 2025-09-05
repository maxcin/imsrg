//Class to run commutator expressions on the GPU
//The class structure is meant to facilitate the creation and managment of
//memory required to perform commutators
//This is one way to do it but probably not the best

//Instance meant to be on the GPU

#ifndef cudaTwoBody 
#define cudaTwoBody 1

#include "cuModelSpace.hh"
//#include "cuOperator.hh"
#include "Matrix.hh"

class cuTwoBodyME
{
    public:
    cuModelSpace* modelspace;
    //Here come temporary storage for two-body commutators
    
    Matrix* MatEl;

    int J = 0;
    int T = 0;
    int P = 0;

    bool hermitian = false;
    bool antihermitian = false;
    //TwoBody functions
    __device__ double GetTBME(int ch, int a, int b, int c, int d) const;
    __device__ double GetTBME_norm(int ch, int a, int b, int c, int d) const;
    __device__ void   SetTBME(int ch, int a, int b, int c, int d, double tbme);
    __device__ void   AddToTBME(int ch, int a, int b, int c, int d, double tbme);

    
    __device__ double GetTBME_J(int j, int a, int b, int c, int d) const;
    __device__ double GetTBME_J_norm(int j, int a, int b, int c, int d) const;
    __device__ void   SetTBME_J(int j, int a, int b, int c, int d, double tbme);
    __device__ void   AddToTBME_J(int j, int a, int b, int c, int d, double tbme);    

    __device__ double GetTBMEmonopole(int a, int b, int c, int d) const;
    __device__ double GetTBMEmonopole_norm(int a, int b, int c, int d) const;

    __device__ void GetTBME_J_norm_twoOps(cuTwoBodyME& OtherTBME, int J, int a, int b, int c, int d, double& tbme_this, double& tbme_other);


    bool allocated = false;
    __device__ void allocate();
    __device__ void deallocate();
};

__global__ void cuTBMEallocate(cuTwoBodyME* TBME);
__global__ void cuTBME_setMatEl(cuTwoBodyME* TBME, int ich, int Nbras, int Nkets, double* mem_ptr);
__global__ void cuTBMEdeallocate(cuTwoBodyME* TBME);

__global__ void cuTBMEassignhermitian(cuTwoBodyME* TBME, bool hermitian, bool antihermitian);


#endif