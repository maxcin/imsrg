#ifndef GPUComm
#define GPUComm
#define ARMA_ALLOW_FAKE_GCC
//We need some things to do the commutator
// In this case I want to avoid having to think about tranferring things onto the GPU so this is a class

#include "cuCommutator.hh"
#include "GPUModelSpace.hh"
#include "GPUOperator.hh"
#include "GPUTwoBodyME.hh"

#include <bandicoot>

class GPUCommutator
{
    public:
        cuCommutator* cuComm;
        GPUModelSpace* gpumodelspace;
        
        //For pppp hhhh 
        GPUTwoBodyME Mpp;
        GPUTwoBodyME Mhh;

        std::vector<coot::vec> P_pp;
        std::vector<coot::vec> P_hh;

        //For phph
        std::vector<coot::mat> X_cc;
        std::vector<coot::mat> Y_cc;

        std::vector<coot::mat> Z_cc;

        std::vector<coot::mat> Z_phasemat;
        std::vector<coot::mat> Y_phasemat_nohy;

        GPUCommutator(GPUModelSpace& gpums);
        ~GPUCommutator();

        void Reset(); //Call after running commutator. Clear out Mpp, Mhh,  X_cc, Y_cc, Z_cc
        void Commutator(GPUOperator& X, GPUOperator& Y, GPUOperator& Z); // Z = [X,Y]


        //Functions to launch kernels to perform commutator
        void cuComm110ss(GPUOperator& X, GPUOperator& Y, GPUOperator& Z);

        void cuComm220ss(GPUOperator& X, GPUOperator& Y, GPUOperator& Z);

        void cuComm111ss(GPUOperator& X, GPUOperator& Y, GPUOperator& Z);

        void cuComm121ss(GPUOperator& X, GPUOperator& Y, GPUOperator& Z);

        void cuComm122ss(GPUOperator& X, GPUOperator& Y, GPUOperator& Z);

        void cuComm222_pp_hh_221ss(GPUOperator& X, GPUOperator& Y, GPUOperator& Z);
        void cu222_221add_pp_hhss(GPUOperator& Z);
        void cuConstructScalarMpp_Mhh(GPUOperator& X, GPUOperator& Y);

        void cuComm222_phss(GPUOperator& X, GPUOperator& Y, GPUOperator& Z);
        void cuPandyaXY(GPUOperator& X, GPUOperator& Y);
        void cuAddInversePandya(GPUOperator& Z);

};


#endif