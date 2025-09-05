
#include "GPUCommutator.hh"
#include "cuCommutator.hh"

#define ARMA_ALLOW_FAKE_GCC //DO NOT DELETE THIS LINE (EVERYTHING BREAKS)
#include <armadillo>

GPUCommutator::GPUCommutator(GPUModelSpace& gpums) : gpumodelspace(&gpums), Mpp(&gpums), Mhh(&gpums)
{
    //pppp hhhh 
    int Nch = gpumodelspace->modelspace->GetNumberTwoBodyChannels();
    P_pp.resize(Nch);
    P_hh.resize(Nch);
    for(int ich = 0; ich<Nch; ++ich)
    {
        TwoBodyChannel& tbc = gpumodelspace->modelspace->GetTwoBodyChannel(ich);
        int Nkets = tbc.GetNumberKets();
        arma::vec nbarnbar(Nkets, arma::fill::zeros);
        arma::vec nn(Nkets, arma::fill::zeros);
        for(int iket = 0; iket < Nkets; ++iket)
        {
            Ket& ket = tbc.GetKet(iket);
            Orbit& op = gpumodelspace->modelspace->GetOrbit(ket.p);
            Orbit& oq = gpumodelspace->modelspace->GetOrbit(ket.q);

            nbarnbar(iket) = (1-op.occ)*(1-oq.occ);
            nn(iket) = op.occ*oq.occ;
        }

        P_pp.at(ich) = coot::vec(nbarnbar);
        P_hh.at(ich) = coot::vec(nn);

        coot::coot_synchronise();
    }

    //phph precalculating
    //Start by creating the correct matrices (part of this is adapted from the 222ph commutator)

    int Nch_cc = gpumodelspace->modelspace->GetNumberTwoBodyChannels_CC();

    X_cc.resize(Nch_cc);
    Y_cc.resize(Nch_cc);
    Z_cc.resize(Nch_cc);

    Z_phasemat.resize(Nch_cc);
    Y_phasemat_nohy.resize(Nch_cc);

    for (int ch = 0; ch < Nch_cc; ++ch)
    {
        const TwoBodyChannel_CC &tbc_cc = gpumodelspace->modelspace->GetTwoBodyChannel_CC(ch);
        index_t nKets_cc = tbc_cc.GetNumberKets();

        Z_cc.at(ch).zeros(nKets_cc, 2*nKets_cc); //(nKets_cc, 2*nKets_cc)

        size_t nph_kets = tbc_cc.GetKetIndex_hh().size() + tbc_cc.GetKetIndex_ph().size();

        arma::mat Y_bar_ph;
        arma::mat Xt_bar_ph;

        // Y has dimension (2*nph , nKets_CC)
        // X has dimension (nKets_CC, 2*nph )
        Y_bar_ph.zeros(2*nph_kets, nKets_cc);
        Xt_bar_ph.zeros(nKets_cc, 2*nph_kets);

        if (Y_bar_ph.size() < 1 or Xt_bar_ph.size() < 1 )
            continue;

        // get the phases for taking the transpose of a Pandya-transformed operator
        arma::mat PhaseMatZ(nKets_cc, nKets_cc, arma::fill::ones);
        for (index_t iket = 0; iket < nKets_cc; iket++)
        {
            const Ket &ket = tbc_cc.GetKet(iket);
            if (gpumodelspace->modelspace->phase((ket.op->j2 + ket.oq->j2) / 2) < 0)
            {
            PhaseMatZ.col(iket) *= -1;
            PhaseMatZ.row(iket) *= -1;
            }
        }
        arma::uvec phkets = arma::join_cols(tbc_cc.GetKetIndex_hh(), tbc_cc.GetKetIndex_ph());
        arma::mat PhaseMatY = PhaseMatZ.rows(phkets); //This is technically missing a factor of hy that we need to add when taking commutators

        X_cc.at(ch) = coot::conv_to<coot::mat>::from(Xt_bar_ph); //(nKets_CC, 2*nph )
        Y_cc.at(ch) = coot::conv_to<coot::mat>::from(Y_bar_ph); //(2*nph , nKets_CC)
        

        Z_phasemat.at(ch) = coot::conv_to<coot::mat>::from(PhaseMatZ); //(nkets_cc, nKets_cc)
        Y_phasemat_nohy.at(ch) = coot::conv_to<coot::mat>::from(PhaseMatY); //(nph, nKets_cc)

        coot::coot_synchronise(); //Is this needed here?
    
        // std::cout <<ch <<" has address " <<Z_cc.at(ch).get_dev_mem(false).cuda_mem_ptr <<" and " <<tbc_cc.GetNumberKets() <<" kets and size "  <<Z_cc.at(ch).size() <<std::endl;
    }

    cuCommutator host_comm;
    host_comm.modelspace = gpumodelspace->cumodelspace;
    host_comm.Mpp = Mpp.cuTwoBody;
    host_comm.Mhh = Mhh.cuTwoBody;

    cudaMalloc(&cuComm, sizeof(cuCommutator));
    cudaMemcpy(cuComm, &host_comm, sizeof(cuCommutator), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cuCommSetup(cuComm);
    cudaDeviceSynchronize();

    for (int ch = 0; ch < Nch_cc; ++ch)
    {
        kernel_Z_cc_mem_ptr<<<1,1>>>(cuComm, ch, Z_cc.at(ch).get_dev_mem(false).cuda_mem_ptr);
    }
    cudaDeviceSynchronize();

}

GPUCommutator::~GPUCommutator()
{
    cuCommClean(cuComm);
    cudaDeviceSynchronize();
    cudaFree(cuComm);
    cudaDeviceSynchronize();
}

void GPUCommutator::Reset()
{
    Mpp.Erase();
    Mhh.Erase();
    int Nch_cc = gpumodelspace->modelspace->GetNumberTwoBodyChannels_CC();
    //for ( auto& matrix : Z_cc )
    for(int ich_cc = 0; ich_cc < Nch_cc; ++ich_cc)
    {
        Z_cc.at(ich_cc).zeros();
        X_cc.at(ich_cc).zeros();
        Y_cc.at(ich_cc).zeros();
    }

    coot::coot_synchronise();
}

//For now single threaded
void GPUCommutator::cuComm110ss(GPUOperator& X, GPUOperator& Y, GPUOperator& Z)
{
    coot::wall_clock timer;
    timer.tic();
    kernel_cuComm110ss<<<1,1>>>(X.cuOp,Y.cuOp,Z.cuOp);
    cudaDeviceSynchronize();

    gpumodelspace->modelspace->profiler.timer[__func__] += timer.toc();
}

void GPUCommutator::cuComm220ss(GPUOperator& X, GPUOperator& Y, GPUOperator& Z)
{
    coot::wall_clock timer;
    timer.tic();
    ModelSpace* modelspace = X.gpumodelspace->modelspace;
    int nchannels = modelspace->GetNumberTwoBodyChannels();
    for(int ich = 0; ich<nchannels; ++ich)
    {
        TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ich);
        int Nkets = tbc.GetNumberKets();
        if(Nkets==0) continue;
        // kernel_cuComm220ss<<<1,1>>>(ich, X.cuOp, Y.cuOp, Z.cuOp);

        kernel_cuComm220ss_new<<<1,Nkets>>>(ich, X.cuOp, Y.cuOp, Z.cuOp);
    }    
    cudaDeviceSynchronize();

    gpumodelspace->modelspace->profiler.timer[__func__] += timer.toc();
}

void GPUCommutator::cuComm111ss(GPUOperator& X, GPUOperator& Y, GPUOperator& Z)
{
    coot::wall_clock timer;
    timer.tic();
    //thread and block each dimension of one-body matrix elements (not optimal but good enough)
    int Norbits = X.gpumodelspace->modelspace->GetNumberOrbits();
    kernel_cuComm111ss<<<Norbits, Norbits>>>(X.cuOp,Y.cuOp,Z.cuOp);
    cudaDeviceSynchronize();

    gpumodelspace->modelspace->profiler.timer[__func__] += timer.toc();
}

//Launchin 32x32 kernels fails on a 1070 likely due to using too many registers in the actual kernel
//This should be investigated...
void GPUCommutator::cuComm122ss(GPUOperator& X, GPUOperator& Y, GPUOperator& Z)
{
    coot::wall_clock timer;
    timer.tic();
    ModelSpace* modelspace = X.gpumodelspace->modelspace;
    int nchannels = modelspace->GetNumberTwoBodyChannels();
    for(int ich = 0; ich<nchannels; ++ich)
    {
        TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ich);
        int Nkets = tbc.GetNumberKets();
        if(Nkets==0) continue;
        //Block dimension is (16x16) threads
        dim3 blockDim(16,16);
        //Figure out how many blocks we need        
        int Nblocks = (int) ceil( ((double) Nkets) / 16.0);
        dim3 gridDim( Nblocks, Nblocks);

        kernel_cuComm122ss<<<gridDim,blockDim>>>(ich, X.cuOp, Y.cuOp, Z.cuOp);
    }    
    cudaDeviceSynchronize();

    gpumodelspace->modelspace->profiler.timer[__func__] += timer.toc();
}

void GPUCommutator::cuComm222_pp_hh_221ss(GPUOperator& X, GPUOperator& Y, GPUOperator& Z)
{
    coot::wall_clock timer;
    timer.tic();
    cuConstructScalarMpp_Mhh(X,Y);
    cu222_221add_pp_hhss(Z);

    gpumodelspace->modelspace->profiler.timer[__func__] += timer.toc();
}

void GPUCommutator::cuConstructScalarMpp_Mhh(GPUOperator& X, GPUOperator& Y)
{
    coot::wall_clock timer;
    timer.tic();


    ModelSpace* modelspace = X.gpumodelspace->modelspace;
    int nchannels = modelspace->GetNumberTwoBodyChannels();

    //This launches fairly slow kernels
    // for(int ich = 0; ich<nchannels; ++ich)
    // {
    //     TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ich);
    //     int Nkets = tbc.GetNumberKets();
    //     if(Nkets==0) continue;
    //     //Block dimension is (32x32) threads
    //     dim3 blockDim(32,32);
    //     //Figure out how many blocks we need        
    //     int Nblocks = (int) ceil( ((double) Nkets) / 32.0);
    //     dim3 gridDim( Nblocks, Nblocks);

    //     kernel_cuConstructScalarMpp_Mhh<<<gridDim,blockDim>>>(ich, X.cuOp, Y.cuOp, cuComm);
    // }    
    // cudaDeviceSynchronize();

    //This is slightly faster but there is more to gain here!
    for(int ich = 0; ich<nchannels; ++ich)
    {
        TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ich);
        int Nkets = tbc.GetNumberKets();
        if(Nkets==0) continue;
        
        coot::mat& X_ch = X.GPUTwoBody.MatEl.at(ich);
        coot::mat& Y_ch = Y.GPUTwoBody.MatEl.at(ich);

        coot::vec& pp_ch = P_pp.at(ich);
        coot::vec& hh_ch = P_hh.at(ich);

        coot::mat& Mpp_ch = Mpp.MatEl.at(ich);
        coot::mat& Mhh_ch = Mhh.MatEl.at(ich);

        Mpp_ch = X_ch * coot::diagmat(pp_ch) * Y_ch - Y_ch * coot::diagmat(pp_ch) * X_ch;
        Mhh_ch = X_ch * coot::diagmat(hh_ch) * Y_ch - Y_ch * coot::diagmat(hh_ch) * X_ch;

    }    

    coot::coot_synchronise();

    gpumodelspace->modelspace->profiler.timer[__func__] += timer.toc();

}

void GPUCommutator::cu222_221add_pp_hhss(GPUOperator& Z)
{
    coot::wall_clock timer;
    timer.tic();
    ModelSpace* modelspace = Z.gpumodelspace->modelspace;
    int Norbits = modelspace->GetNumberOrbits();
    kernel_cu221add<<<Norbits, Norbits>>>(Z.cuOp, cuComm);

    int nchannels = modelspace->GetNumberTwoBodyChannels();
    for(int ich = 0; ich<nchannels; ++ich)
    {
        TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ich);
        int Nkets = tbc.GetNumberKets();
        if(Nkets==0) continue;
        //Block dimension is (32x32) threads
        dim3 blockDim(32,32);
        //Figure out how many blocks we need        
        int Nblocks = (int) ceil( ((double) Nkets) / 32.0);
        dim3 gridDim( Nblocks, Nblocks);

        kernel_cu222add_pp_hhss<<<gridDim,blockDim>>>(ich, Z.cuOp, cuComm);
    }    
    cudaDeviceSynchronize();
    gpumodelspace->modelspace->profiler.timer[__func__] += timer.toc();
}

void GPUCommutator::cuComm121ss(GPUOperator& X, GPUOperator& Y, GPUOperator& Z)
{
    coot::wall_clock timer;
    timer.tic();
    //thread and block each dimension of one-body matrix elements (not optimal but good enough)
    int Norbits = X.gpumodelspace->modelspace->GetNumberOrbits();
    kernel_cuComm121ss<<<Norbits, Norbits>>>(X.cuOp,Y.cuOp,Z.cuOp);
    cudaDeviceSynchronize();

    gpumodelspace->modelspace->profiler.timer[__func__] += timer.toc();
}

//TODO: rewrite this properly
void GPUCommutator::cuComm222_phss(GPUOperator& X, GPUOperator& Y, GPUOperator& Z)
{
    coot::wall_clock timer;
    timer.tic();
    ModelSpace* modelspace = X.gpumodelspace->modelspace;
    int Nch_cc = modelspace->GetNumberTwoBodyChannels_CC();
    int hy = Y.hermitian ? 1 : -1;
    cuPandyaXY(X,Y);

    // std::cout <<X_cc.at(5) <<std::endl;
    // std::cout <<Y_cc.at(5) <<std::endl;
    // for (int ch = 0; ch < Nch_cc; ++ch)
    // {
    //     // kernel_Z_cc_mem_ptr<<<1,1>>>(cuComm, ch, Z_cc.at(ch).get_dev_mem(false).cuda_mem_ptr);
    //     TwoBodyChannel_CC& tbc = modelspace->GetTwoBodyChannel_CC(ch);
    //     std::cout <<ch <<" has address " <<Z_cc.at(ch).get_dev_mem(false).cuda_mem_ptr <<" and " <<tbc.GetNumberKets() <<" kets" <<std::endl;
    // }

    //The following is just adapted from the CPU implementation using bandicoot instead of armadillo
    for (int ich_cc = 0; ich_cc < Nch_cc; ++ich_cc)
    {
        const TwoBodyChannel_CC &tbc_cc = modelspace->GetTwoBodyChannel_CC(ich_cc);
        index_t nKets_cc = tbc_cc.GetNumberKets();
        size_t nph_kets = tbc_cc.GetKetIndex_hh().size() + tbc_cc.GetKetIndex_ph().size();

        //      auto& Zbar_ch = Z_bar.at(ch);
        coot::mat& Xt_bar_ph = X_cc.at(ich_cc);
        coot::mat& Y_bar_ph = Y_cc.at(ich_cc);

        coot::mat& Zbar = Z_cc.at(ich_cc);

        if (Y_bar_ph.size() < 1 or Xt_bar_ph.size() < 1)
            continue;

        coot::mat& PhaseMatZ = Z_phasemat.at(ich_cc);
        coot::mat& PhaseMatY_nohy = Y_phasemat_nohy.at(ich_cc);

        //                                           [      |     ]
        //     create full Y matrix from the half:   [  Yhp | Y'ph]   where the prime indicates multiplication by (-1)^(i+j+k+l) h_y
        //                                           [      |     ]   Flipping hp <-> ph and multiplying by the phase is equivalent to
        //                                           [  Yph | Y'hp]   having kets |kj> with k>j.
        //
        //
        //      so <il|Zbar|kj> =  <il|Xbar|hp><hp|Ybar|kj> + <il|Xbar|ph><ph|Ybar|kj>
        //
        // arma::mat Y_bar_ph_flip = arma::join_vert(Y_bar_ph.tail_rows(nph_kets) % PhaseMatY, Y_bar_ph.head_rows(nph_kets) % PhaseMatY);
        Zbar = Xt_bar_ph * coot::join_horiz(Y_bar_ph, hy * coot::join_vert(Y_bar_ph.tail_rows(nph_kets) % PhaseMatY_nohy, Y_bar_ph.head_rows(nph_kets) % PhaseMatY_nohy));

        // If Z is hermitian, then XY is anti-hermitian, and so XY - YX = XY + (XY)^T
        if (Z.hermitian and X.hermitian != Y.hermitian)
        {
            Zbar.head_cols(nKets_cc) += Zbar.head_cols(nKets_cc).t();
        }
        else // Z is antihermitian, so XY is hermitian, XY - YX = XY - XY^T
        {
            Zbar.head_cols(nKets_cc) -= Zbar.head_cols(nKets_cc).t();
        }

        // By taking the transpose, we get <il|Zbar|kj> with i>l and k<j, and we want the opposite
        // By the symmetries of the Pandya-transformed matrix element, that means we pick
        // up a factor hZ * phase(i+j+k+l). The hZ cancels the hXhY we have for the "head" part of the matrix
        // so we end up adding in either case.
        Zbar.tail_cols(nKets_cc) += Zbar.tail_cols(nKets_cc).t() % PhaseMatZ;
    }

    coot::coot_synchronise();

    //std::cout <<Z_cc.at(5) <<std::endl;
    // for (int ch = 0; ch < Nch_cc; ++ch)
    // {
    //     // kernel_Z_cc_mem_ptr<<<1,1>>>(cuComm, ch, Z_cc.at(ch).get_dev_mem(false).cuda_mem_ptr);
    //     TwoBodyChannel_CC& tbc = modelspace->GetTwoBodyChannel_CC(ch);
    //     std::cout <<ch <<" has address " <<Z_phasemat.at(ch).get_dev_mem(false).cuda_mem_ptr <<" and " <<tbc.GetNumberKets() <<" kets" <<std::endl;
    // }

    cuAddInversePandya(Z);

    gpumodelspace->modelspace->profiler.timer[__func__] += timer.toc();

}

void GPUCommutator::cuPandyaXY(GPUOperator& X, GPUOperator& Y)
{
    coot::wall_clock timer;
    timer.tic();
    ModelSpace* modelspace = X.gpumodelspace->modelspace;
    int Nch_cc = modelspace->GetNumberTwoBodyChannels_CC();

    for(int ich_cc = 0; ich_cc < Nch_cc; ++ich_cc)
    {
        // Y_bar_cc has dimension (2*nph , nKets_CC)
        // X_bar_t_cc has dimension (nKets_CC, 2*nph )
        TwoBodyChannel_CC& tbc_cc = modelspace->GetTwoBodyChannel_CC(ich_cc);
        int nph_kets = tbc_cc.GetKetIndex_hh().size() + tbc_cc.GetKetIndex_ph().size();
        int Nkets_cc = tbc_cc.GetNumberKets();
        // coot::mat Y_bar_cc(2*nph_kets, Nkets_cc);
        // coot::mat Xt_bar_cc(Nkets_cc, 2*nph_kets);

        // double* Y_memory = Y_bar_cc.get_dev_mem().cuda_mem_ptr;
        // double* X_memory = Xt_bar_cc.get_dev_mem().cuda_mem_ptr;
        double* X_memory = X_cc.at(ich_cc).get_dev_mem(false).cuda_mem_ptr;
        double* Y_memory = Y_cc.at(ich_cc).get_dev_mem(false).cuda_mem_ptr;

        //Block dimension is (16x16) threads
        dim3 blockDim(16,16);
        //Figure out how many blocks we need        
        int Nblocks_x = (int) ceil( ((double) 2*nph_kets) / 16.0);
        int Nblocks_y = (int) ceil( ((double) Nkets_cc) / 16.0);
        dim3 gridDim( Nblocks_x, Nblocks_y);

        kernel_cuPandyaTransformSingleChannel<<<gridDim,blockDim>>>(ich_cc, X.cuOp,Y.cuOp, X_memory, Y_memory);
    }

    cudaDeviceSynchronize();

    gpumodelspace->modelspace->profiler.timer[__func__] += timer.toc();

}

void GPUCommutator::cuAddInversePandya(GPUOperator& Z)
{
    coot::wall_clock timer;
    timer.tic();
    ModelSpace* modelspace = Z.gpumodelspace->modelspace;
    int Nch = modelspace->GetNumberTwoBodyChannels();

    //(nKets_cc, 2*nKets_cc) <-- Dimensions of Z in each channel
    for(int ich = 0; ich < Nch; ++ich)
    {
        TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ich);
        int Nkets = tbc.GetNumberKets();

        // double* X_memory = X_cc.at(ich_cc).get_dev_mem().cuda_mem_ptr;

        //Block dimension is (16x16) threads
        dim3 blockDim(16,16);
        //Figure out how many blocks we need        
        int Nblocks_x = (int) ceil( ((double) Nkets) / 16.0);
        int Nblocks_y = (int) ceil( ((double) Nkets) / 16.0);
        dim3 gridDim( Nblocks_x, Nblocks_y);

        // kernel_cuPandyaTransformSingleChannel<<<gridDim,blockDim>>>(ich_cc, X.cuOp,Y.cuOp, X_memory, Y_memory);
        kernel_cuAddInversePandya<<<gridDim,blockDim>>>(ich, Z.cuOp, cuComm);
    }

    cudaDeviceSynchronize();

    gpumodelspace->modelspace->profiler.timer[__func__] += timer.toc();
}

void GPUCommutator::Commutator(GPUOperator& X, GPUOperator& Y, GPUOperator& Z)
{
    coot::wall_clock timer;
    timer.tic();

    if( (X.hermitian and Y.hermitian) or (X.antihermitian and Y.antihermitian) )
    {
        Z.SetAntiHermitian();
    } 
    else 
    {
        Z.SetHermitian();
    }

    Z.Erase();

    cuComm110ss( X,  Y,  Z);

    cuComm220ss( X,  Y,  Z);

    cuComm111ss( X,  Y,  Z);

    cuComm121ss( X,  Y,  Z);
    cuComm122ss( X,  Y,  Z);

    cuComm222_pp_hh_221ss( X,  Y,  Z);

    cuComm222_phss(X,  Y,  Z);

    Reset();

    gpumodelspace->modelspace->profiler.timer[__func__] += timer.toc();
}