
#define ARMA_ALLOW_FAKE_GCC //DO NOT DELETE THIS LINE (EVERYTHING BREAKS)
// #include <cuco/static_map.cuh>
// #ifndef CUCO_BITWISE_DOUBLE
// #define CUCO_BITWISE_DOUBLE
// CUCO_DECLARE_BITWISE_COMPARABLE(double)  
// #endif

#include "cuTest.hh"
#include <ModelSpace.hh>
#include <Commutator.hh>
#include "cuModelSpace.hh"
#include "cuOperator.hh"
#include "cuCommutator.hh"
#include <iostream>

// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>

#include "GPUModelSpace.hh"
#include "GPUOperator.hh"
#include "GPUCommutator.hh"

#define COOT_DONT_USE_OPENCL
#define COOT_USE_CUDA
#define COOT_DEFAULT_BACKEND CUDA_BACKEND
#include <bandicoot>

#include <UnitTest.hh>

//Don't ask...
// using static_map = cuco::static_map<uint64_t, double, std::size_t, cuda::std::__4::thread_scope_device, cuda::std::__4::equal_to<uint64_t>, cuco::linear_probing<1, cuco::default_hash_function<uint64_t>>, cuco::cuda_allocator<cuco::pair<uint64_t, double>>, cuco::storage<1>>::ref_type<cuco::op::find_tag>;

namespace cuTest
{
    __global__ void test_kernel()
    {
        printf("GPU works\n");
    }

    void TestGPU()
    {
        //std::cout <<"No testGPU implementation made" <<std::endl;
        test_kernel<<<1,1>>>();
        cudaDeviceSynchronize();
    }

    void TestModelSpace(ModelSpace& modelspace)
    {
        GPUModelSpace gpuMS(modelspace);
        std::cout <<"SixJ address " <<gpuMS.sixj_device <<std::endl;
    }
    

    __global__ void test_kernel_Operator_OneBody(cuOperator* cuOp)
    {
        cuOp->OneBody.print();
    }


    __global__ void test_kernel_Operator_TwoBody(cuOperator* cuOp)
    {
        cuOp->TwoBody->MatEl[12].print();
    }

    void TestOperator(Operator& Op)
    {
        GPUModelSpace gpuMS(*Op.modelspace);
        GPUOperator gpuOp(gpuMS, Op);

        std::cout <<Op.OneBody <<std::endl;
        std::cout <<gpuOp.OneBody <<std::endl;
        test_kernel_Operator_OneBody<<<1,1>>>(gpuOp.cuOp);
        cudaDeviceSynchronize();

        std::cout <<Op.TwoBody.GetMatrix(12) <<std::endl;
        std::cout <<gpuOp.GPUTwoBody.MatEl.at(12) <<std::endl;
        test_kernel_Operator_TwoBody<<<1,1>>>(gpuOp.cuOp);
        cudaDeviceSynchronize();

        std::cout <<Op.ZeroBody <<std::endl;
        std::cout <<gpuOp.GetZeroBody() <<std::endl;
    }

    __global__ void testSixJ(cuModelSpace* cuMS)
    {
        for(int i = 0; i<20; ++i)
        {
            printf("%f\n", cuMS->SixJCache_112112[i]);
        }
        printf("%p\n", cuMS->SixJCache_112112);
    }

    
    void TimeCommutator(ModelSpace& modelspace)
    {
        UnitTest unittest(modelspace);
        Operator X = unittest.RandomOp(modelspace, 0, 0, 0, 2, 1);
        Operator Y = unittest.RandomOp(modelspace, 0, 0, 0, 2, -1);

        GPUModelSpace gpuMS(*X.modelspace);
        GPUCommutator gpuComm(gpuMS);

        GPUOperator gpuX(gpuMS, X);
        GPUOperator gpuY(gpuMS, Y);

        GPUOperator gpuZ(gpuMS); 

        Operator Z = X;
        Z.Erase();

        coot::wall_clock timer;
        timer.tic();
        for(int i = 0; i<20; ++i)
        {
            Z = Commutator::Commutator(X,Y);
        }

        double t_cpu = timer.toc();
        std::cout <<"20 Commutators take " <<t_cpu <<"s on CPU" <<std::endl;

        timer.tic();
        for(int i = 0; i<20; ++i)
        {
            gpuComm.Commutator(gpuX,gpuY,gpuZ);
        }

        double t_gpu = timer.toc();

        std::cout <<"20 Commutators take " <<t_gpu <<"s on GPU" <<std::endl;

        modelspace.profiler.PrintTimes();
    }

    void TestCommutatorKernels(ModelSpace& modelspace)
    {
        UnitTest unittest(modelspace);
        Operator X = unittest.RandomOp(modelspace, 0, 0, 0, 2, 1);
        Operator Y = unittest.RandomOp(modelspace, 0, 0, 0, 2, -1);

        Operator X2 = unittest.RandomOp(modelspace, 0, 0, 0, 2, -1);
        Operator Y2 = unittest.RandomOp(modelspace, 0, 0, 0, 2, 1);
        // Operator Z(modelspace);

        GPUModelSpace gpuMS(*X.modelspace);
        GPUCommutator gpuComm(gpuMS);

        
        // TestComm110ss(X,Y);
        std::cout <<std::setw(20) <<"Function" <<std::setw(6) <<"hX" <<std::setw(6) <<"hY" <<std::setw(16) << std::setprecision(9) <<"||X||" <<std::setw(16) << std::setprecision(9) 
        <<"||Y||" <<std::setw(16) << std::setprecision(9) <<"||Z||" <<std::setw(16) << std::setprecision(6) <<"||Z - Z_GPU||" 
            <<std::setw(16) <<"Free GPU (MB)" <<std::setw(16) <<"Total GPU (MB)" <<std::setw(10) <<"Passed" <<std::endl;
        
        //Hermitian antihermitian
        Test_against_CPU(X,Y, Commutator::comm110ss, gpuComm, "comm110ss");
        Test_against_CPU(X,Y, Commutator::comm220ss, gpuComm, "comm220ss");
        Test_against_CPU(X,Y, Commutator::comm111ss, gpuComm, "comm111ss");
        Test_against_CPU(X,Y, Commutator::comm121ss, gpuComm, "comm121ss");
        Test_against_CPU(X,Y, Commutator::comm122ss, gpuComm, "comm122ss");
        Test_against_CPU(X,Y, Commutator::comm222_pp_hh_221ss, gpuComm, "comm222_pp_hh_221ss");
        Test_against_CPU(X,Y, Commutator::comm222_phss, gpuComm, "comm222_phss");

        gpuComm.Reset();
        //Antihermitian hermitian
        Test_against_CPU(Y,X, Commutator::comm110ss, gpuComm, "comm110ss");
        Test_against_CPU(Y,X, Commutator::comm220ss, gpuComm, "comm220ss");
        Test_against_CPU(Y,X, Commutator::comm111ss, gpuComm, "comm111ss");
        Test_against_CPU(Y,X, Commutator::comm121ss, gpuComm, "comm121ss");
        Test_against_CPU(Y,X, Commutator::comm122ss, gpuComm, "comm122ss");
        Test_against_CPU(Y,X, Commutator::comm222_pp_hh_221ss, gpuComm, "comm222_pp_hh_221ss");
        Test_against_CPU(Y,X, Commutator::comm222_phss, gpuComm, "comm222_phss");

        gpuComm.Reset();
        //hermitian hermitian
        Test_against_CPU(X,Y2, Commutator::comm110ss, gpuComm, "comm110ss");
        Test_against_CPU(X,Y2, Commutator::comm220ss, gpuComm, "comm220ss");
        Test_against_CPU(X,Y2, Commutator::comm111ss, gpuComm, "comm111ss");
        Test_against_CPU(X,Y2, Commutator::comm121ss, gpuComm, "comm121ss");
        Test_against_CPU(X,Y2, Commutator::comm122ss, gpuComm, "comm122ss");
        Test_against_CPU(X,Y2, Commutator::comm222_pp_hh_221ss, gpuComm, "comm222_pp_hh_221ss");
        Test_against_CPU(X,Y2, Commutator::comm222_phss, gpuComm, "comm222_phss");

        gpuComm.Reset();
        //antihermitian antihermitian
        Test_against_CPU(X2,Y, Commutator::comm110ss, gpuComm, "comm110ss");
        Test_against_CPU(X2,Y, Commutator::comm220ss, gpuComm, "comm220ss");
        Test_against_CPU(X2,Y, Commutator::comm111ss, gpuComm, "comm111ss");
        Test_against_CPU(X2,Y, Commutator::comm121ss, gpuComm, "comm121ss");
        Test_against_CPU(X2,Y, Commutator::comm122ss, gpuComm, "comm122ss");
        Test_against_CPU(X2,Y, Commutator::comm222_pp_hh_221ss, gpuComm, "comm222_pp_hh_221ss");
        Test_against_CPU(X2,Y, Commutator::comm222_phss, gpuComm, "comm222_phss");
    }

    void Test_against_CPU(Operator& X, Operator& Y, cpu_func comm_cpu , GPUCommutator& gpuComm, std::string name)
    {
        GPUModelSpace gpuMS(*X.modelspace);
        GPUOperator gpuX(gpuMS, X);
        GPUOperator gpuY(gpuMS, Y);

        Operator Z(*X.modelspace);
        Z.Erase();
        GPUOperator gpuZ(gpuMS); //creates empty operator Z
        if( (X.IsHermitian() and Y.IsHermitian()) or (X.IsAntiHermitian() and Y.IsAntiHermitian()) )
        {
            gpuZ.SetAntiHermitian();
            Z.SetAntiHermitian();
        } 
        else 
        {
            gpuZ.SetHermitian();
            Z.SetHermitian();
        }

        comm_cpu(X,Y,Z);

        if(name=="comm110ss") gpuComm.cuComm110ss(gpuX, gpuY, gpuZ);
        if(name=="comm220ss") gpuComm.cuComm220ss(gpuX, gpuY, gpuZ);
        if(name=="comm111ss") gpuComm.cuComm111ss(gpuX, gpuY, gpuZ);
        if(name=="comm121ss") gpuComm.cuComm121ss(gpuX, gpuY, gpuZ);
        if(name=="comm122ss") gpuComm.cuComm122ss(gpuX, gpuY, gpuZ);
        if(name=="comm222_pp_hh_221ss") gpuComm.cuComm222_pp_hh_221ss(gpuX, gpuY, gpuZ);
        if(name=="comm222_phss") gpuComm.cuComm222_phss(gpuX, gpuY, gpuZ);
        
        //Now substract from Z the GPU Result and print out the result
        double results_norm = Z.Norm() + std::abs(Z.ZeroBody);
        gpuZ.substractFromCPUOperator(Z);
        double norm = Z.Norm() + std::abs(Z.ZeroBody);

        double normX = X.Norm();
        double normY = Y.Norm();

        std::string passed = norm / std::max(results_norm,1.0)  < 1e-12 ? "Yes" : "No";

        size_t free, total;
        cudaMemGetInfo( &free, &total );
    //     std::cout << "GPU memory: free=" << free*1e-6 << " MB, total=" << total*1e-6 <<"MB" << std::endl;

        int hx = X.IsHermitian() ? 1 : -1;
        int hy = Y.IsHermitian() ? 1 : -1;

        std::cout <<std::setw(20) <<name <<std::setw(6) <<hx <<std::setw(6) <<hy  <<std::setw(16) << std::setprecision(9) <<normX <<std::setw(16) << std::setprecision(9) 
                <<normY <<std::setw(16) << std::setprecision(9) <<results_norm <<std::setw(16) << std::setprecision(6) <<norm 
                <<std::setw(16) <<free*1e-6 <<std::setw(16) <<total*1e-6  <<std::setw(10) <<passed <<std::endl;
    
    }

    // __global__ void test_kernel()
    // {
    //     printf("GPU works\n");
    // }

    // //Supposed to test that transfer of modelspace works correctly
    // __global__ void test_MS(cuModelSpace* cuMS)
    // {
    //     printf("GPU norbits: %d\n", cuMS->Norbits);
    //     printf("GPU Nchannels: %d\n", cuMS->Nchannels);
    //     printf("Orbital information: \n");
    //     for(int i = 0; i<cuMS->Norbits; ++i)
    //     {
    //         printf("%d\t%d\t%d\t%d\t%d\t%f\t%d\n", i, cuMS->n[i],cuMS->l[i],cuMS->j2[i],cuMS->tz2[i],cuMS->occ[i],cuMS->cvq[i]);
    //     }
    // }

    // __global__ void test_TBMS(cuModelSpace* cuMS)
    // {
    //     printf("DEVICE: Ket combining 1, 3 %d\n", cuMS->Index2(1,3));
    //     printf("DEVICE: Local index of ket 14 in channel 13 %d\n" ,cuMS->localIndex[13][31]);
    // }

    // void TestGPU()
    // {
    //     //std::cout <<"No testGPU implementation made" <<std::endl;
    //     test_kernel<<<1,1>>>();
    //     cudaDeviceSynchronize();
    // }

    
    // //cuModelSpace* moveGPUcuModelSpace(cuMemory modelspace_memory)
    // cuModelSpace* moveGPUcuModelSpace(ModelSpace& modelspace)
    // {        
    //     cuModelSpace cuTestMS;

    //     cuTestMS.Norbits = modelspace.norbits;
    //     cuTestMS.Nchannels = modelspace.GetNumberTwoBodyChannels();  
    //     cuTestMS.Nchannels_cc = modelspace.GetNumberTwoBodyChannels_CC();
    //     cuTestMS.Nkets_modelspace = modelspace.GetNumberKets();  
    //     cuTestMS.dim1_sixj = modelspace.six_j_cache_2b_.dim_1_;
    //     cuTestMS.dim2_sixj = modelspace.six_j_cache_2b_.dim_2_;
    //     cuTestMS.N_SixJ = modelspace.six_j_cache_2b_.six_js_.size();

    //     //Allocate Memory for modelspace and copy
    //     cuModelSpace* cuMS;
    //     cudaMalloc(&cuMS, sizeof(cuModelSpace));
    //     cudaMemcpy(cuMS, &cuTestMS, sizeof(cuModelSpace), cudaMemcpyHostToDevice);
    //     cuMS_allocate<<<1,1>>>(cuMS);
    //     cudaDeviceSynchronize();

    //     //One-Body part
    //     for(int i : modelspace.all_orbits)
    //     { 
    //         Orbit& oi = modelspace.GetOrbit(i);
    //         cuMS_Memcpy_Orbital<<<1,1>>>(cuMS, i, oi.n, oi.l, oi.j2, oi.tz2, oi.occ, oi.cvq);
    //     }
    //     cuMS_construct_Ketpq<<<1,1>>>(cuMS);

    //     //Two-Body part two-body channel information
    //     for(int ich = 0; ich < modelspace.GetNumberTwoBodyChannels(); ++ich)
    //     {
    //         TwoBodyChannel& tbc = modelspace.GetTwoBodyChannel(ich);
    //         //tbc.GetLocalIndex(1);
    //         cuMS_Memcpy_TBC<<<1,1>>>(cuMS, ich, tbc.NumberKets, tbc.J, tbc.Tz, tbc.parity);
    //     }
    //     //Two-Body part localIndex information
    //     cudaDeviceSynchronize();
    //     cuMS_construct_localIndex_map<<<1,modelspace.GetNumberTwoBodyChannels()>>>(cuMS);

    //     //Two-Body channel cc
    //     for(int ich_cc = 0 ; ich_cc < modelspace.GetNumberTwoBodyChannels_CC(); ++ich_cc)
    //     {
    //         TwoBodyChannel_CC& tbc_cc = modelspace.GetTwoBodyChannel_CC(ich_cc);
    //         int nph_kets = tbc_cc.GetKetIndex_hh().size() + tbc_cc.GetKetIndex_ph().size();
    //         cuMS_Memcpy_TBC_CC<<<1,1>>>(cuMS, ich_cc, tbc_cc.NumberKets, nph_kets, tbc_cc.J, tbc_cc.Tz, tbc_cc.parity);
    //     }
    //     cudaDeviceSynchronize();
    //     cuMS_construct_cc_local_hh_ph<<<1,modelspace.GetNumberTwoBodyChannels_CC()>>>(cuMS);

    //     double* sixj_device;
    //     cudaMalloc(&sixj_device, sizeof(double)*modelspace.six_j_cache_2b_.six_js_.size());
    //     cudaMemcpy(sixj_device, modelspace.six_j_cache_2b_.six_js_.data(), sizeof(double)*modelspace.six_j_cache_2b_.six_js_.size(), cudaMemcpyHostToDevice);
    //     cudaDeviceSynchronize();
    //     // cuMS_Memcpy_SixJcache<<<1,1>>>(cuMS, sixj_device);
    //     cudaDeviceSynchronize();

    //     cudaFree(sixj_device);

    //     cudaDeviceSynchronize();
    //     return cuMS;
    // }

    // void deallocatecuMS(cuModelSpace* cuMS)
    // {
    //     //Free allocated memory in modelspace
    //     cuMS_deallocate<<<1,1>>>(cuMS);
    //     cudaDeviceSynchronize();
    //     //Free remaining modelspace
    //     cudaFree(cuMS);
    // }

    // cuOperator* moveGPUcuOperator(cuModelSpace* cuMS, Operator& Op)
    // {
    //     //Build up the basic Operator here before handing of to GPU
    //     cuOperator host_Op;
    //     host_Op.Jrank = Op.rank_J;
    //     host_Op.Tzrank = Op.rank_T;
    //     host_Op.Prank = Op.parity;
    //     host_Op.hermitian = Op.hermitian;
    //     host_Op.antihermitian = Op.antihermitian;
    //     host_Op.modelspace = cuMS;
    //     host_Op.ZeroBody = Op.ZeroBody;

    //     cuOperator* cuOp;
    //     cudaMalloc(&cuOp, sizeof(cuOperator));
    //     cudaMemcpy(cuOp, &host_Op, sizeof(cuOperator), cudaMemcpyHostToDevice);

    //     //Here do something with the Operator
    //     allocateOperator<<<1,1>>>(cuOp);
    //     cudaDeviceSynchronize();
    //     memcpy_Operator(cuOp, Op);
    //     cudaDeviceSynchronize();
    //     return cuOp;
    // }

    // void deallocatecuOperator(cuOperator* cuOp)
    // {
    //     deallocateOperator<<<1,1>>>(cuOp);
    //     cudaDeviceSynchronize();
    //     cudaFree(cuOp);
    // }

    // __global__ void test_MS_cc(cuModelSpace* cuMS)
    // {
    //     printf("cc indices channel 5\n");
    //     for(int iket_cc = 0; iket_cc < cuMS->Nkets_channel_cc[5]; iket_cc++)
    //     {
    //         printf("%d ", cuMS->globalIndex_cc[5][iket_cc]);
    //     }
    //     printf("\n");
    //     printf("hh ph indices channel 5\n");
    //     for(int iket_cc = 0; iket_cc < cuMS->Nkets_channel_cc_hh_ph[5]; iket_cc++)
    //     {
    //         printf("%d ", cuMS->Kets_cc_hh_ph[5][iket_cc]);
    //     }
    //     printf("\n");
    // }

    // void testModelSpace(ModelSpace& modelspace)
    // {
    //     std::cout <<"Modelspace:" <<std::endl;
    //     cuModelSpace* cuMS = moveGPUcuModelSpace(modelspace);

    //     TwoBodyChannel_CC& tbc_cc = modelspace.GetTwoBodyChannel_CC(5);
        
    //     arma::uvec kets_global_hh = tbc_cc.GetKetIndexFromList(modelspace.KetIndex_hh);
    //     arma::uvec kets_global_ph = tbc_cc.GetKetIndexFromList(modelspace.KetIndex_ph);
    //     arma::uvec kets_ph = arma::join_cols(kets_global_hh, kets_global_ph);
    //     for(int i = 0; i< kets_ph.size(); ++i)
    //     {
    //         kets_ph(i) = tbc_cc.GetKetIndex(kets_ph(i));
    //     }
    //     std::cout <<"hh ph indices channel 5\n" <<kets_ph.t() <<std::endl;
    //     test_MS_cc<<<1,1>>>(cuMS);

    //     cudaDeviceSynchronize();
        
    //     deallocatecuMS(cuMS);
    // }
    

    // __global__ void kernel_test_Op(cuOperator* Op)
    // {
    //     Op->OneBody.print();
    // }

    // __global__ void kernel_test_Op_TwoBody(cuOperator* Op)
    // {
    //     printf("%f\n", Op->TwoBody.GetTBME_norm(13,2,0,4,8));
    //     printf("%f\n", Op->TwoBody.GetTBME(12,2,2,6,8));
    //     Op->TwoBody.SetTBME(12,2,2,6,8,69.420);
    //     //printf("%f\n", Op->GetTBME(12,2,2,6,8));
    //     Op->TwoBody.MatEl[12].print();
    // }

    // void testOperator(Operator& Op)
    // {
    //     std::cout <<"Modelspace:" <<std::endl;
    //     cuModelSpace* cuMS = moveGPUcuModelSpace(*Op.modelspace);

    //     std::cout <<"Operator:" <<std::endl;
    //     cuOperator* cuOp = moveGPUcuOperator(cuMS, Op);

    //     std::cout <<"Host:" <<std::endl;
    //     std::cout <<Op.OneBody;

    //     std::cout <<"Device:" <<std::endl;
    //     kernel_test_Op<<<1,1>>>(cuOp);
    //     cudaDeviceSynchronize();

    //     std::cout <<"Host:" <<std::endl;
    //     std::cout <<Op.TwoBody.GetMatrix(13);

    //     std::cout <<"Device:" <<std::endl;
    //     kernel_test_Op_TwoBody<<<1,1>>>(cuOp);
    //     cudaDeviceSynchronize();
    //     std::cout <<"Host:" <<std::endl;
    //     std::cout <<Op.TwoBody.GetTBME_norm(13,2,0,4,8) <<std::endl;
    //     std::cout <<Op.TwoBody.GetTBME(12,2,2,6,8) <<std::endl;
    //     Op.TwoBody.SetTBME(12,2,2,6,8,69.420);
    //     std::cout <<Op.TwoBody.GetMatrix(12) <<std::endl;
    //     std::cout <<Op.modelspace->GetTwoBodyChannel(13).J <<" " <<Op.modelspace->GetTwoBodyChannel(13).parity <<" " <<Op.modelspace->GetTwoBodyChannel(13).Tz <<std::endl;

    //     deallocatecuOperator(cuOp);
    //     deallocatecuMS(cuMS);
    //     cudaDeviceSynchronize();
    // }

    // __global__ void kernel_testMP2(cuOperator* Op)
    // {
    //     double EMP2 = Op->GetMP2_Energy();
    //     printf("%f\n", EMP2);
    // }

    // void testMP2(Operator& Op)
    // {
    //     std::cout <<"Modelspace:" <<std::endl;
    //     cuModelSpace* cuMS = moveGPUcuModelSpace(*Op.modelspace);

    //     std::cout <<"Operator:" <<std::endl;
    //     cuOperator* cuOp = moveGPUcuOperator(cuMS, Op);

    //     std::cout <<"Host: " <<Op.GetMP2_Energy() <<std::endl;

    //     std::cout <<"Device: " <<std::endl;
    //     kernel_testMP2<<<1,1>>>(cuOp);
        
    //     cudaDeviceSynchronize();
    //     deallocatecuOperator(cuOp);
    //     deallocatecuMS(cuMS);
    //     cudaDeviceSynchronize();
    // }

    // __global__ void test_110ss(cuOperator* Z)
    // {
    //     printf("%f\n", Z->ZeroBody);
    // }

    // __global__ void test_111ss(cuOperator* Z)
    // {
    //     Z->OneBody.print();
    // }

    // cuCommutator* createGPUcomm(cuModelSpace* cuMS)
    // {
    //     cuCommutator host_comm;
    //     host_comm.modelspace = cuMS;
    //     cuCommutator* cuC;
    //     cudaMalloc(&cuC, sizeof(cuCommutator));
    //     cudaMemcpy(cuC, &host_comm, sizeof(cuCommutator), cudaMemcpyHostToDevice);
    //     cudaDeviceSynchronize();

    //     cuCommSetup(cuC);
    //     cudaDeviceSynchronize();

    //     return cuC;
    // }

    // void deallocatecuComm(cuCommutator* C)
    // {
    //     cuCommClean(C);
    //     cudaDeviceSynchronize();
    //     cudaFree(C);
    // }

    // __global__ void test_MppMhhss(cuCommutator* C)
    // {
    //     C->Mpp.MatEl[13].print();
    //     C->Mhh.MatEl[12].print();
    // }

    // __global__ void test_pp_hhss(cuOperator* Z)
    // {
    //     Z->TwoBody.MatEl[13].print();
    //     printf("\n");
    //     Z->OneBody.print();

    // }

    // __global__ void test_122(cuOperator* Z)
    // {
    //     Z->TwoBody.MatEl[12].print();
    // }

    // __global__ void test_220(cuOperator* Z)
    // {
    //     printf("%f\n",Z->ZeroBody);
    // }

    // __global__ void test_121(cuOperator* Z)
    // {
    //     Z->OneBody.print();
    // }

    // __global__ void test_SixJHash(static_map SixJList, cuModelSpace* modelspace, double j1, double j2, double j3, double J1, double J2, double J3)
    // {
    //     uint64_t hash = modelspace->SixJHash(j1,  j2,  j3,  J1,  J2,  J3);
    //     printf("SixJ example %llu\n", hash);
    //     double sixj = SixJList.find(hash)->second;
    //     printf("%f\n", sixj);
    // }

    // void testComm(Operator& X, Operator& Y, Operator& Z)
    // {
    //     size_t free, total;
    //     cudaMemGetInfo( &free, &total );
    //     std::cout << "GPU memory: free=" << free*1e-6 << " MB, total=" << total*1e-6 <<"MB" << std::endl;

    //     cuModelSpace* cuMS = moveGPUcuModelSpace(*X.modelspace);

    //     cuOperator* cuX = moveGPUcuOperator(cuMS, X);
    //     cuOperator* cuY = moveGPUcuOperator(cuMS, Y);
    //     cuOperator* cuZ = moveGPUcuOperator(cuMS, Z);

    //     cuCommutator* cuC = createGPUcomm(cuMS);


    //     // std::cout <<"Host 110ss" <<std::endl;
    //     // Commutator::comm110ss(X,Y,Z);
    //     // std::cout <<Z.ZeroBody <<std::endl;

    //     // std::cout <<"Device 110ss" <<std::endl;

    //     // cuComm110ss(cuX,cuY,cuZ);
    //     // cudaDeviceSynchronize();
    //     // test_110ss<<<1,1>>>(cuZ);

    //     // cudaDeviceSynchronize();
    //     // std::cout <<"Host 111ss" <<std::endl;
    //     // Commutator::comm111ss(X,Y,Z);
    //     // std::cout <<Z.OneBody <<std::endl;

    //     // std::cout <<"Device 111ss" <<std::endl;

    //     // cuComm111ss(X.modelspace, cuX,cuY,cuZ);
    //     // test_111ss<<<1,1>>>(cuZ);
    //     // cudaDeviceSynchronize();

    //     // TwoBodyME Mpp = Z.TwoBody;
    //     // TwoBodyME Mhh = Z.TwoBody;
    //     // Mpp.Erase();
    //     // Mhh.Erase();
    //     // Commutator::ConstructScalarMpp_Mhh(X, Y, Z, Mpp, Mhh);

    //     // std::cout <<"Host MppMhh" <<std::endl;
    //     // std::cout <<Mpp.GetMatrix(13) <<std::endl;;
    //     // std::cout <<Mhh.GetMatrix(12) <<std::endl;

    //     // std::cout <<"Device MppMhh" <<std::endl;
    //     // cuConstructScalarMpp_Mhh(X.modelspace, cuX, cuY ,cuZ, cuC);
    //     // test_MppMhhss<<<1,1>>>(cuC);
        
    //     // cudaDeviceSynchronize();

    //     // //Z.Erase();
    //     // Commutator::comm222_pp_hh_221ss(X,Y,Z);
    //     // std::cout <<"Host pp_hh_ss " <<std::endl;
    //     // std::cout <<Z.TwoBody.GetMatrix(13) <<std::endl;;
    //     // std::cout <<Z.OneBody <<std::endl;

    //     // std::cout <<"Device pp_hh_ss" <<std::endl;
    //     // cu222_221add_pp_hhss(X.modelspace, cuZ, cuC);
    //     // test_pp_hhss<<<1,1>>>(cuZ);

    //     // cudaDeviceSynchronize();

    //     // Commutator::comm122ss(X,Y,Z);
    //     // std::cout <<"Host122ss" <<std::endl;
    //     // std::cout <<Z.TwoBody.GetMatrix(12) <<std::endl;

    //     // std::cout <<"Device 122ss" <<std::endl;
    //     // cuComm122ss(X.modelspace,cuX,cuY,cuZ);
    //     // test_122<<<1,1>>>(cuZ);

    //     // cudaDeviceSynchronize();

    //     // Commutator::comm220ss(X,Y,Z);
    //     // std::cout <<"Host 220ss" <<std::endl;
    //     // std::cout <<Z.ZeroBody <<std::endl;

    //     // std::cout <<"Device 220ss" <<std::endl;
    //     // cuComm220ss(X.modelspace,cuX,cuY,cuZ);
    //     // test_220<<<1,1>>>(cuZ);

    //     // cudaDeviceSynchronize();

    //     Commutator::comm121ss(X,Y,Z);
    //     std::cout <<"Host 121ss" <<std::endl;
    //     std::cout <<Z.OneBody <<std::endl;

    //     std::cout <<"Device 121ss" <<std::endl;
    //     cuComm121ss(X.modelspace,cuX,cuY,cuZ);
    //     test_121<<<1,1>>>(cuZ);

    //     cudaDeviceSynchronize();


    //     X.modelspace->PreCalculateSixJ();


    //     std::cout <<"SixJ example " <<X.modelspace->SixJHash(0.5,1.5,1.0,0.5,1.5,1.0) <<std::endl;
    //     std::cout <<X.modelspace->GetSixJ(0.5,1.5,1.0,0.5,1.5,1.0) <<std::endl;


    //     //Here run a test to move the sixJ std::unordered_map to be accessible on GPU

    //     // Empty slots are represented by reserved "sentinel" values. These values should be selected such
    //     // that they never occur in your input data.
    //     uint64_t constexpr empty_key_sentinel = -1;
    //     double constexpr empty_value_sentinel = -1e9;

    //     // Number of key/value pairs to be inserted
    //     std::size_t num_keys = X.modelspace->SixJList.size();

    //     // Compute capacity based on a 50% load factor
    //     auto constexpr load_factor = 0.5;
    //     std::size_t const capacity = std::ceil(num_keys / load_factor);

    //     // Extent capacity,
    //     // empty_key<Key> empty_key_sentinel,
    //     // empty_value<T> empty_value_sentinel,
    //     // KeyEqual const& pred                = {},
    //     // ProbingScheme const& probing_scheme = {},
    //     // cuda_thread_scope<Scope> scope      = {},
    //     // Storage storage                     = {},
    //     // Allocator const& alloc              = {},
    //     // cuda::stream_ref stream             = {});
    //     auto map = cuco::static_map{
    //                 capacity, cuco::empty_key{empty_key_sentinel}, cuco::empty_value{empty_value_sentinel},
    //                 cuda::std::equal_to<uint64_t>{}, cuco::linear_probing<1, cuco::default_hash_function<uint64_t>>{}
    //                 };
    //     //

    //     std::cout <<cuco::is_bitwise_comparable_v<int> <<std::endl;
    //     std::cout <<cuco::is_bitwise_comparable_v<double> <<std::endl;
    //     std::cout <<std::has_unique_object_representations_v<int> <<std::endl;
    //     std::cout <<std::has_unique_object_representations_v<double> <<std::endl;

    //     // Create a sequence of keys and values in thrust from the map
    //     thrust::host_vector<uint64_t> host_keys(num_keys);
    //     thrust::host_vector<double> host_values(num_keys);

    //     int count_insertions = 0;
    //     for(auto& [key, value] : X.modelspace->SixJList)
    //     {
    //         host_keys[count_insertions] = key;
    //         host_values[count_insertions] = value;
    //         count_insertions += 1;
    //     }

    //     thrust::device_vector<uint64_t> insert_keys(host_keys);
    //     thrust::device_vector<double> insert_values(host_values);
    //     // Combine keys and values into pairs
    //     auto pairs = thrust::make_transform_iterator(
    //         thrust::counting_iterator<std::size_t>{0},
    //         cuda::proclaim_return_type<cuco::pair<uint64_t, double>>(
    //         [keys = insert_keys.begin(), values = insert_values.begin()] __device__(auto i) {
    //             return cuco::pair<uint64_t, double>{keys[i], values[i]};
    //         }));
        

    //     map.insert(pairs, pairs+num_keys);

    //     static_map Map_ref = map.ref(cuco::find);

    //     test_SixJHash<<<1,1>>>(Map_ref, cuMS, 0.5,1.5,1.0,0.5,1.5,1.0);
    //     cudaDeviceSynchronize();

        
    //     //Start the real dela here
    //     std::cout <<"Pandya 222phss" <<std::endl;
    //     arma::mat Xt_bar_ph;
    //     arma::mat Y_bar_ph;
    //     // Y has dimension (2*nph , nKets_CC)
    //     // X has dimension (nKets_CC, 2*nph )
    //     // This function gives  <ij`| X |ab`> *na(1-nb)  and   <ab`| Y |ij`>,  for both orderings ab` and ba`
    //     Commutator::DoPandyaTransformation_SingleChannel_XandY(X, Y, Xt_bar_ph, Y_bar_ph, 5);
    //     std::cout <<Xt_bar_ph <<std::endl;
    //     std::cout <<Y_bar_ph <<std::endl;

    //     cuComm222_phss(X.modelspace, Map_ref, cuX, cuY, cuZ, X,Y,Z);
        
    //     cudaMemGetInfo( &free, &total );
    //     std::cout << "GPU memory: free=" << free*1e-6 << " MB, total=" << total*1e-6 <<"MB" << std::endl;

    //     deallocatecuOperator(cuX);
    //     deallocatecuOperator(cuY);
    //     deallocatecuOperator(cuZ);
    //     deallocatecuComm(cuC);
    //     deallocatecuMS(cuMS);
    //     cudaDeviceSynchronize();
    // }
}

