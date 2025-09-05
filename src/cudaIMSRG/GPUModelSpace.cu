
#define ARMA_ALLOW_FAKE_GCC
#include "GPUModelSpace.hh"

GPUModelSpace::GPUModelSpace(ModelSpace& modelspace)
{
    this->modelspace = &modelspace;

    cuModelSpace cuda_host_modelspace;

    cuda_host_modelspace.Norbits = modelspace.norbits;
    cuda_host_modelspace.Nchannels = modelspace.GetNumberTwoBodyChannels();  
    cuda_host_modelspace.Nchannels_cc = modelspace.GetNumberTwoBodyChannels_CC();
    cuda_host_modelspace.Nkets_modelspace = modelspace.GetNumberKets();  
    cuda_host_modelspace.dim1_sixj = modelspace.six_j_cache_2b_.dim_1_;
    cuda_host_modelspace.dim2_sixj = modelspace.six_j_cache_2b_.dim_2_;
    cuda_host_modelspace.N_SixJ = modelspace.six_j_cache_2b_.six_js_.size();

    // double* sixj_device;
    // std::cout <<"There are " <<modelspace.six_j_cache_2b_.six_js_.size() <<" sixJ symbols to copy" <<std::endl;
    cudaMalloc(&sixj_device, sizeof(double)*modelspace.six_j_cache_2b_.six_js_.size());
    cudaMemcpy(sixj_device, modelspace.six_j_cache_2b_.six_js_.data(), sizeof(double)*modelspace.six_j_cache_2b_.six_js_.size(), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cuda_host_modelspace.SixJCache_112112 = sixj_device;

    //Allocate Memory for modelspace and copy
    cuModelSpace* cuMS;
    cudaMalloc(&cuMS, sizeof(cuModelSpace));
    cudaMemcpy(cuMS, &cuda_host_modelspace, sizeof(cuModelSpace), cudaMemcpyHostToDevice);
    cuMS_allocate<<<1,1>>>(cuMS);
    cudaDeviceSynchronize();

    //One-Body part
    for(int i : modelspace.all_orbits)
    { 
        Orbit& oi = modelspace.GetOrbit(i);
        cuMS_Memcpy_Orbital<<<1,1>>>(cuMS, i, oi.n, oi.l, oi.j2, oi.tz2, oi.occ, oi.cvq);
    }
    cuMS_construct_Ketpq<<<1,1>>>(cuMS);

    //Two-Body part two-body channel information
    for(int ich = 0; ich < modelspace.GetNumberTwoBodyChannels(); ++ich)
    {
        TwoBodyChannel& tbc = modelspace.GetTwoBodyChannel(ich);
        //tbc.GetLocalIndex(1);
        cuMS_Memcpy_TBC<<<1,1>>>(cuMS, ich, tbc.NumberKets, tbc.J, tbc.Tz, tbc.parity);
    }
    //Two-Body part localIndex information
    cudaDeviceSynchronize();
    cuMS_construct_localIndex_map<<<1,modelspace.GetNumberTwoBodyChannels()>>>(cuMS);

    //Two-Body channel cc
    for(int ich_cc = 0 ; ich_cc < modelspace.GetNumberTwoBodyChannels_CC(); ++ich_cc)
    {
        TwoBodyChannel_CC& tbc_cc = modelspace.GetTwoBodyChannel_CC(ich_cc);
        int nph_kets = tbc_cc.GetKetIndex_hh().size() + tbc_cc.GetKetIndex_ph().size();
        cuMS_Memcpy_TBC_CC<<<1,1>>>(cuMS, ich_cc, tbc_cc.NumberKets, nph_kets, tbc_cc.J, tbc_cc.Tz, tbc_cc.parity);
    }
    cudaDeviceSynchronize();
    cuMS_construct_cc_local_hh_ph<<<1,modelspace.GetNumberTwoBodyChannels_CC()>>>(cuMS);

    cudaDeviceSynchronize();
    
    cumodelspace = cuMS;
}

GPUModelSpace::~GPUModelSpace()
{
    cudaFree(sixj_device);
    //Free memory managed by modelspace
    cuMS_deallocate<<<1,1>>>(cumodelspace);
    cudaDeviceSynchronize();
    //Free remaining modelspace
    cudaFree(cumodelspace);
}  