

#define ARMA_ALLOW_FAKE_GCC
#include "GPUOperator.hh"
#include "GPUTwoBodyME.hh"
#include "cuOperator.hh"

GPUOperator::GPUOperator(GPUModelSpace& gpums, Operator& Op) : 
GPUTwoBody(gpumodelspace, Op.TwoBody), gpumodelspace(&gpums), hermitian(Op.hermitian), antihermitian(Op.antihermitian), OneBody(Op.OneBody)
{
    //Build up the basic Operator here before handing of to GPU
    cuOperator host_Op;
    host_Op.Jrank = Op.rank_J;
    host_Op.Tzrank = Op.rank_T;
    host_Op.Prank = Op.parity;
    host_Op.hermitian = Op.hermitian;
    host_Op.antihermitian = Op.antihermitian;
    host_Op.modelspace = gpumodelspace->cumodelspace;
    host_Op.ZeroBody = Op.ZeroBody;

    host_Op.TwoBody = GPUTwoBody.cuTwoBody;


    cudaMalloc(&cuOp, sizeof(cuOperator));
    cudaMemcpy(cuOp, &host_Op, sizeof(cuOperator), cudaMemcpyHostToDevice);

    //Here do something with the Operator
    // allocateOperator<<<1,1>>>(cuOp);
    cudaDeviceSynchronize();

    SetOneBody_mem_ptr<<<1,1>>>(cuOp, OneBody.get_dev_mem(false).cuda_mem_ptr);
    // memcpy_Operator(cuOp, Op);

    cudaMalloc(&ZeroBody_mem_ptr, sizeof(double));
    cudaDeviceSynchronize();
    
}

GPUOperator::GPUOperator(GPUModelSpace& gpums) : 
GPUTwoBody(gpumodelspace), gpumodelspace(&gpums), hermitian(false), antihermitian(false), OneBody(gpums.modelspace->norbits, gpums.modelspace->norbits)
{
    //Build up the basic Operator here before handing of to GPU
    cuOperator host_Op;
    host_Op.Jrank = 0;
    host_Op.Tzrank = 0;
    host_Op.Prank = 0;
    host_Op.hermitian = false;
    host_Op.antihermitian = false;
    host_Op.modelspace = gpumodelspace->cumodelspace;
    host_Op.ZeroBody = 0.0;

    host_Op.TwoBody = GPUTwoBody.cuTwoBody;


    cudaMalloc(&cuOp, sizeof(cuOperator));
    cudaMemcpy(cuOp, &host_Op, sizeof(cuOperator), cudaMemcpyHostToDevice);

    //Here do something with the Operator
    // allocateOperator<<<1,1>>>(cuOp);
    cudaDeviceSynchronize();

    SetOneBody_mem_ptr<<<1,1>>>(cuOp, OneBody.get_dev_mem(false).cuda_mem_ptr);
    // memcpy_Operator(cuOp, Op);
    cudaMalloc(&ZeroBody_mem_ptr, sizeof(double));
    cudaDeviceSynchronize();
    
}

GPUOperator::~GPUOperator()
{
    cudaFree(cuOp);
    cudaFree(ZeroBody_mem_ptr);
}

double GPUOperator::GetZeroBody()
{
    cuOptransferZeroBody<<<1,1>>>(cuOp, ZeroBody_mem_ptr);
    double E_0 = 0.0;
    cudaMemcpy(&E_0, ZeroBody_mem_ptr, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return E_0;
}

void GPUOperator::SetHermitian()
{
    hermitian = true;
    antihermitian = false;
    GPUTwoBody.hermitian = hermitian;
    GPUTwoBody.antihermitian = antihermitian;
    cuOpassignhermitian<<<1,1>>>(cuOp, hermitian, antihermitian);
}

void GPUOperator::SetAntiHermitian()
{
    hermitian = false;
    antihermitian = true;
    GPUTwoBody.hermitian = hermitian;
    GPUTwoBody.antihermitian = antihermitian;
    cuOpassignhermitian<<<1,1>>>(cuOp, hermitian, antihermitian);
}

void GPUOperator::substractFromCPUOperator(Operator& cpu_Op)
{
    cpu_Op.ZeroBody -= GetZeroBody();
    cpu_Op.OneBody -= coot::conv_to<arma::mat>::from(OneBody);

    int Nch = gpumodelspace->modelspace->GetNumberTwoBodyChannels();
    for(int ich = 0; ich < Nch; ++ich)
    {
        cpu_Op.TwoBody.GetMatrix(ich) -= coot::conv_to<arma::mat>::from(GPUTwoBody.MatEl.at(ich));
    }
}

void GPUOperator::SetZeroBody(double ZeroBody)
{
    cuOpSetZeroBody<<<1,1>>>(cuOp, ZeroBody);
    cudaDeviceSynchronize();
}

void GPUOperator::Erase()
{
    SetZeroBody(0.0);
    OneBody.zeros();
    GPUTwoBody.Erase();
}