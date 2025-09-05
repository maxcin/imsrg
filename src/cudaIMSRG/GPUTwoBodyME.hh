#ifndef GPUTBME
#define GPUTBME
#define ARMA_ALLOW_FAKE_GCC
#include "../TwoBodyME.hh"
#include "cuTwoBodyME.hh"

#include "GPUModelSpace.hh"

#define COOT_DONT_USE_OPENCL
#define COOT_USE_CUDA
#define COOT_DEFAULT_BACKEND CUDA_BACKEND
#include <bandicoot>

class GPUTwoBodyME
{
    public:
        cuTwoBodyME* cuTwoBody; //GPU Pointer

        //TODO: Implement more complicated operator
        int J = 0;
        int T = 0;
        int P = 0;

        bool hermitian = false;
        bool antihermitian = false;

        std::vector<coot::mat> MatEl; //MatEl in convenient way to access
        GPUModelSpace* gpumodelspace;

        GPUTwoBodyME(GPUModelSpace* GPUmodelspace, TwoBodyME& TBME);
        GPUTwoBodyME(GPUModelSpace* GPUmodelspace);
        ~GPUTwoBodyME();

        //Careful we have to transfer the information also to the cuda Object
        void SetHermitian();
        void SetAntihermitian();

        void Erase();
        double Norm();
};

#endif