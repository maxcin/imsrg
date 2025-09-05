#ifndef GPUOp
#define GPUOp
#define ARMA_ALLOW_FAKE_GCC
#include "../Operator.hh"
#include "cuOperator.hh"

#include "GPUModelSpace.hh"
#include "GPUTwoBodyME.hh"

#include <bandicoot>

class GPUOperator
{
    private:
        double* ZeroBody_mem_ptr; //Device pointer to retrieve ZeroBody Term
    public:
        cuOperator* cuOp; //device pointer to data

        GPUModelSpace* gpumodelspace;

        //ZeroBody piece lives only on the GPU in cuOp
        //Reading any single matrix element is in general very slow so we should avoid it
        double GetZeroBody();
        coot::mat OneBody;
        GPUTwoBodyME GPUTwoBody;

        bool hermitian = false;
        bool antihermitian = false;

        int Jrank = 0;
        int Trank = 0;
        int Prank = 0;

        GPUOperator(GPUModelSpace& gpums, Operator& Op);
        GPUOperator(GPUModelSpace& gpums);
        ~GPUOperator();

        void Erase();
        double Norm();
        void SetZeroBody(double ZeroBody);
        // double OneBodyNorm(); //This needs to be implemented using a kernel call

        void SetHermitian();
        void SetAntiHermitian();

        void substractFromCPUOperator(Operator& cpu_Op); //mainly good for testing
        
};

#endif