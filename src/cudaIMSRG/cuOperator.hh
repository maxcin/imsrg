//Class to store Operator
//Always meant to be used with the modelspace

#ifndef cudaOperator 
#define cudaOperator 1

#include "cuModelSpace.hh"
#define ARMA_ALLOW_FAKE_GCC //DO NOT DELETE THIS LINE (EVERYTHING BREAKS)
#include <armadillo>
#include "cuTwoBodyME.hh"

//class cuTwoBodyME;
#include "Matrix.hh"

class Operator;



//Always assume scalar operator 

class cuOperator
{
    public:

    int Jrank = 0;
    int Tzrank = 0;
    int Prank = 0;

    bool hermitian = false;
    bool antihermitian = false;

    bool allocated = false;
    cuModelSpace* modelspace;

    //zeroBody
    double ZeroBody;

    //OneBody
    Matrix OneBody; //Matrix of one-body matrix elements

    //TwoBody
    cuTwoBodyME* TwoBody;


    //Operator functions
    __device__ double GetMP2_Energy();
    
};

__global__ void SetOneBody_mem_ptr(cuOperator* Op ,double* mem_ptr);

__global__ void cuOpassignhermitian(cuOperator* Op, bool hermitian, bool antihermitian);
__global__ void cuOptransferZeroBody(cuOperator* Op, double* dst);

__global__ void cuOpSetZeroBody(cuOperator* Op, double ZeroBody_new);
// __global__ void allocateOperator(cuOperator* Op);
// __global__ void deallocateOperator(cuOperator* Op);

//Because the memory is managed by the Object itself we need to do some ugly stuff to be able to transfer the data from host
// __global__ void device_memcpy_OneBody(cuOperator* Op, double* MatEl);
// __global__ void device_memcpy_TwoBodyMatEl(cuOperator* Op, double* MatEl, int ichannel);

// void memcpy_OneBody(cuOperator* Op,arma::mat& OneBody);
// void memcpy_TwoBody(cuOperator* Op,arma::mat& TwoBody, int ichannel);
// void memcpy_Operator(cuOperator* cudaOp, Operator& Op);





#endif