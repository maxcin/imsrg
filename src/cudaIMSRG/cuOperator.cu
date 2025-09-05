#include <stdio.h>
#include "cuOperator.hh"

#include <Operator.hh>


// __global__ void allocateOperator(cuOperator* Op)
// {
//     int norbits = Op->modelspace->Norbits;
//     int nchannels = Op->modelspace->Nchannels;
//     //allocate One-Body part of the Operator
//     Op->OneBody.allocate(norbits, norbits);

//     //TwoBody
//     Op->TwoBody.modelspace = Op->modelspace;
    
//     Op->TwoBody.J = Op->Jrank;
//     Op->TwoBody.P = Op->Prank;
//     Op->TwoBody.T = Op->Tzrank;

//     Op->TwoBody.hermitian = Op->hermitian;
//     Op->TwoBody.antihermitian = Op->antihermitian;

//     Op->TwoBody.allocate();
//     //cudaMalloc(&Op->TwoBody, sizeof(Matrix)*nchannels); //Here is assumed scalar
//     //for(int ich = 0; ich < nchannels; ++ich)
//     //{
//     //   int nkets = Op->modelspace->Nkets_channel[ich];
//     //    Op->TwoBody[ich].allocate(nkets, nkets);
//     //}
//     Op->allocated = true;
// }

// __global__ void deallocateOperator(cuOperator* Op)
// {
//     int nchannels = Op->modelspace->Nchannels;

//     Op->OneBody.deallocate();
//     Op->TwoBody.deallocate();

//     //for(int ich = 0; ich < nchannels; ++ich)
//     //{
//     //    Op->TwoBody[ich].deallocate();
//     //}

//     //cudaFree(Op->TwoBody);


//     Op->allocated = false;
// }

// __global__ void device_memcpy_OneBody(cuOperator* Op, double* MatEl)
// {
//     int N_MatEl = Op->OneBody.ncols*Op->OneBody.nrows;
//     memcpy(Op->OneBody.data, MatEl, sizeof(double)*N_MatEl);
// }

// __global__ void device_memcpy_TwoBodyMatEl(cuOperator* Op, double* MatEl, int ichannel)
// {
//     int N_MatEl = Op->TwoBody.MatEl[ichannel].ncols*Op->TwoBody.MatEl[ichannel].nrows;
//     memcpy(Op->TwoBody.MatEl[ichannel].data, MatEl, sizeof(double)*N_MatEl);
// }

// void memcpy_Operator(cuOperator* cudaOp, Operator& Op)
// {
//     memcpy_OneBody(cudaOp, Op.OneBody);

//     int nchannel = Op.modelspace->GetNumberTwoBodyChannels();
//     for(int ich = 0; ich <nchannel; ++ich)
//     {
//         memcpy_TwoBody(cudaOp, Op.TwoBody.GetMatrix(ich), ich);
//     }
// }

// void memcpy_TwoBody(cuOperator* Op,arma::mat& TwoBodyMat, int ichannel)
// {
//     double* data_pointer_device = 0; //here we want a pointer to gpu memory
    

//     int N_MatEl = TwoBodyMat.n_cols*TwoBodyMat.n_rows;
//     cudaMalloc(&data_pointer_device, sizeof(double)*N_MatEl);

//     cudaMemcpy(data_pointer_device, TwoBodyMat.memptr(), sizeof(double)*N_MatEl, cudaMemcpyHostToDevice);
//     device_memcpy_TwoBodyMatEl<<<1,1>>>(Op, data_pointer_device, ichannel);

//     cudaDeviceSynchronize();
//     cudaFree(data_pointer_device);
// }


// void memcpy_OneBody(cuOperator* Op,arma::mat& OneBody)
// {
//     double* data_pointer_device = 0; //here we want a pointer to gpu memory
    

//     int N_MatEl = OneBody.n_cols*OneBody.n_rows;
//     cudaMalloc(&data_pointer_device, sizeof(double)*N_MatEl);

//     cudaMemcpy(data_pointer_device, OneBody.memptr(), sizeof(double)*N_MatEl, cudaMemcpyHostToDevice);
//     device_memcpy_OneBody<<<1,1>>>(Op, data_pointer_device);

//     cudaDeviceSynchronize();
//     cudaFree(data_pointer_device);
// }

__global__ void cuOpSetZeroBody(cuOperator* Op, double ZeroBody_new)
{
    Op->ZeroBody = ZeroBody_new;
}

__global__ void SetOneBody_mem_ptr(cuOperator* Op ,double* mem_ptr)
{
    int Norbits = Op->modelspace->Norbits;
    Op->OneBody.ncols = Norbits;
    Op->OneBody.nrows = Norbits;
    Op->OneBody.data = mem_ptr;
}

__global__ void cuOpassignhermitian(cuOperator* Op, bool hermitian, bool antihermitian)
{
    Op->hermitian = hermitian;
    Op->antihermitian = antihermitian;
    Op->TwoBody->hermitian = hermitian;
    Op->TwoBody->antihermitian = antihermitian;
}

__global__ void cuOptransferZeroBody(cuOperator* Op, double* dst)
{
    *dst = Op->ZeroBody;
}

//Operator functions

//This is only single threaded for now to have a reference implementation
//Because this is computationally not too expensive single-threaded is fine for now
__device__ double cuOperator::GetMP2_Energy()
{
    double* occ = modelspace->occ;
    int* j2 = modelspace->j2;
    int* l = modelspace->l;
    int* tz2 = modelspace->tz2;

    double Emp2 = 0;
    for (int a = 0; a < modelspace->Norbits; ++a) //particle
    {
        if(occ[a] > 1e-6) continue;
        double ea = OneBody(a, a);
        for (int i = 0; i< modelspace->Norbits; ++i)
        {
            if(occ[i] < 1e-6) continue;
            double ei = OneBody(i, i);
            if (abs(OneBody(i, a)) > 1e-9)
                Emp2 += (modelspace->j2[i] + 1) * modelspace->occ[i] * OneBody(a, i) * OneBody(a, i) / (OneBody(i, i) - OneBody(a, a));
            for (int b = 0; b <modelspace->Norbits; ++b) //particle
            {
                if (b < a)
                    continue;
                if(occ[b]>1e-6) continue;
                double eb = OneBody(b, b);
                for (int j = 0; j<modelspace->Norbits; ++j)
                {
                    if (j < i)continue;
                    if(occ[j] < 1e-6) continue;
                    
                    if ((l[a] + l[b] + l[i] + l[j]) % 2 > 0)
                    continue;
                    if ((tz2[a] + tz2[b]) != (tz2[i] + tz2[j]))
                    continue;
                    double ej = OneBody(j, j);
                    double denom = ei + ej - ea - eb;
                    int Jmin = max(abs(j2[a] - j2[b]), abs(j2[i] - j2[j])) / 2;
                    int Jmax = min(j2[a] + j2[b], j2[i] + j2[j]) / 2;
                    int dJ = 1;
                    if (a == b or i == j)
                    {
                    Jmin += Jmin % 2;
                    dJ = 2;
                    }
                    for (int J = Jmin; J <= Jmax; J += dJ)
                    {
                        double tbme = TwoBody->GetTBME_J_norm(J, i, j, a, b);
                        if (abs(tbme) > 1e-9)
                        {
                            Emp2 += (2 * J + 1) * occ[i] * occ[j] * tbme * tbme / denom; // no factor 1/4 because of the restricted sum
                            //printf("%f\n",Emp2);
                        }
                    }
                }
            }
        }
    }
    return Emp2;
}