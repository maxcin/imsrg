
#define ARMA_ALLOW_FAKE_GCC
#include "GPUTwoBodyME.hh"

GPUTwoBodyME::GPUTwoBodyME(GPUModelSpace* GPUmodelspace, TwoBodyME& TBME)
{
    gpumodelspace = GPUmodelspace;

    int Nchannels = GPUmodelspace->modelspace->GetNumberTwoBodyChannels();

    MatEl.resize(Nchannels);

    for(int ich = 0; ich < Nchannels; ++ich)
    {
        MatEl.at(ich) = coot::conv_to<coot::mat>::from(TBME.GetMatrix(ich));
        // MatEl.at(ich) = coot::mat(TBME.GetMatrix(ich));
    }
    coot::coot_synchronise();

    cuTwoBodyME host_cuTBME;
    host_cuTBME.allocated = true;
    host_cuTBME.hermitian = TBME.hermitian;
    host_cuTBME.antihermitian = TBME.antihermitian;
    host_cuTBME.modelspace = GPUmodelspace->cumodelspace;

    //Move to GPU
    cudaMalloc(&cuTwoBody, sizeof(cuTwoBodyME));
    cudaMemcpy(cuTwoBody, &host_cuTBME, sizeof(cuTwoBodyME), cudaMemcpyHostToDevice);
    cuTBMEallocate<<<1,1>>>(cuTwoBody);
    cudaDeviceSynchronize();


    for(int ich = 0; ich < Nchannels; ++ich)
    {
        int Nkets = GPUmodelspace->modelspace->GetTwoBodyChannel(ich).GetNumberKets();
        double* device_mem_ptr = MatEl.at(ich).get_dev_mem(false).cuda_mem_ptr;
        cuTBME_setMatEl<<<1,1>>>(cuTwoBody, ich, Nkets, Nkets, device_mem_ptr);
    }

}

GPUTwoBodyME::GPUTwoBodyME(GPUModelSpace* GPUmodelspace)
{
    gpumodelspace = GPUmodelspace;

    int Nchannels = GPUmodelspace->modelspace->GetNumberTwoBodyChannels();

    MatEl.resize(Nchannels);

    for(int ich = 0; ich < Nchannels; ++ich)
    {
        int Nkets = GPUmodelspace->modelspace->GetTwoBodyChannel(ich).GetNumberKets();
        MatEl.at(ich).zeros(Nkets,Nkets);
    }
    coot::coot_synchronise();

    cuTwoBodyME host_cuTBME;
    host_cuTBME.allocated = true;
    host_cuTBME.hermitian = false;
    host_cuTBME.antihermitian = false;
    host_cuTBME.modelspace = GPUmodelspace->cumodelspace;

    //Move to GPU
    cudaMalloc(&cuTwoBody, sizeof(cuTwoBodyME));
    cudaMemcpy(cuTwoBody, &host_cuTBME, sizeof(cuTwoBodyME), cudaMemcpyHostToDevice);
    cuTBMEallocate<<<1,1>>>(cuTwoBody);
    cudaDeviceSynchronize();


    for(int ich = 0; ich < Nchannels; ++ich)
    {
        int Nkets = GPUmodelspace->modelspace->GetTwoBodyChannel(ich).GetNumberKets();
        double* device_mem_ptr = MatEl.at(ich).get_dev_mem(false).cuda_mem_ptr;
        cuTBME_setMatEl<<<1,1>>>(cuTwoBody, ich, Nkets, Nkets, device_mem_ptr);
    }
}

GPUTwoBodyME::~GPUTwoBodyME()
{
    cuTBMEdeallocate<<<1,1>>>(cuTwoBody);
}

void GPUTwoBodyME::Erase()
{
  for ( auto& matrix : MatEl )
  {
     matrix.zeros();
  }
}

void GPUTwoBodyME::SetHermitian()
{
    hermitian = true;
    antihermitian = false;
    cuTBMEassignhermitian<<<1,1>>>(cuTwoBody, hermitian, antihermitian);
}

void GPUTwoBodyME::SetAntihermitian()
{
    hermitian = false;
    antihermitian = true;
    cuTBMEassignhermitian<<<1,1>>>(cuTwoBody, hermitian, antihermitian);
}

double GPUTwoBodyME::Norm()
{
    double nrm = 0;
    int Nchannels = gpumodelspace->modelspace->GetNumberTwoBodyChannels();
//    if (not allocated) return 0;
    for ( int ich = 0; ich < Nchannels; ++ich)
    {
        coot::mat& matrix = MatEl.at(ich);
        int J = gpumodelspace->modelspace->GetTwoBodyChannel( ich ).J;
        int degeneracy = (2*J+1);
        double n2 = coot::norm(matrix,"fro") * degeneracy;
    }
    return std::sqrt(nrm);
}