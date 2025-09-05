#ifndef GPUModelSpace_
#define GPUModelSpace_
#define ARMA_ALLOW_FAKE_GCC
// Wrapper class to work with the modelspace

#include "cuModelSpace.hh"
#include "../ModelSpace.hh"

class GPUModelSpace
{
    public:
        ModelSpace* modelspace;

        cuModelSpace* cumodelspace;
        double* sixj_device;

        GPUModelSpace(ModelSpace& modelspace) ;
        ~GPUModelSpace();
};

#endif