#include "FSCPT.hh"
#include "Commutator.hh"
#include "Generator.hh"

FSCPT::FSCPT(Operator& H) :         
        H0(arma::diagmat(H.OneBody)),
        V(*H.modelspace),
        Vod(*H.modelspace),
        Vd(*H.modelspace),
        G1(*H.modelspace),
        G2(*H.modelspace),
        Heff2(*H.modelspace),
        Heff3(*H.modelspace)
{
    // std::cout <<"Started Fock space canonical perturbation theory" <<std::endl;
    //V is by definition without H0
    V.OneBody = H.OneBody - H0;
    V.TwoBody = H.TwoBody;
    Vod = GetOpOd(V);
    Vd = V-Vod;
    
    G1.SetAntiHermitian();
    G2.SetAntiHermitian();
    //Here we can start implementing actual expressions

    // generator.SetType("white");
    // generator.SetDenominatorPartitioning("MP");
    // generator.AddToEta(V,G1);
    G1 = Delta(GetOpOd(V));

    std::cout <<"Norm of G1 " <<G1.Norm() <<std::endl;

    //second order
    FSCPT2();
    if(off_diagonal == "single_reference")
    {
        std::cout <<"Second order = " <<Heff2.ZeroBody <<std::endl;
    }
    // std::cout <<"Second order = " <<Heff2.ZeroBody <<std::endl;

    //Third order
    FSCPT3();
    if(off_diagonal == "single_reference")
    {
        std::cout <<"Third order = " <<Heff3.ZeroBody <<std::endl;
    }
    // std::cout <<"Third order = " <<Heff3.ZeroBody <<std::endl;


    
}

//Needs 1 commutator
void FSCPT::FSCPT2()
{
    Operator vd2vod = 2*Vd + Vod;
    Operator second_order = 0.5*Commutator::Commutator(G1, vd2vod);
    Operator second_order_od = GetOpOd(second_order);

    Heff2 = second_order-second_order_od;

    G2 = Delta(GetOpOd(second_order_od));
}


//Needs 4 commutators
void FSCPT::FSCPT3()
{
    Operator v_vd = V+Vd;
    Operator vv_vd = 2*V+Vd;
    Operator vd_vd_vod = 2*Vd + Vod;

    Operator g1_vd_vd_vod_3 = 3*GetOpOd(Commutator::Commutator(G1,vd_vd_vod));

    Operator combine = 2*Commutator::Commutator(G1,vv_vd) - g1_vd_vd_vod_3;
    
    Operator third_order = 0.5 * Commutator::Commutator(G2,v_vd) + (1/12.0)* Commutator::Commutator(G1,combine);
    
    Heff3 = third_order - GetOpOd(third_order);
    //G3 = Delta(third_order - Heff3);
}

Operator FSCPT::GetOpOd(const Operator& Op)
{
    Operator OpOd(Op);

    if(off_diagonal == "valence") OpOd = generator.GetHod_ShellModel(OpOd);
    else if(off_diagonal == "single_reference") OpOd = generator.GetHod_SingleRef(OpOd);
    else 
    {
        std::cout <<"Unknown definition of off-diagonal" <<std::endl;
        exit(0);
    }

    return OpOd;
}


//The new operator is antihermitian (if Op is hermitian). To make this most general we just add the delta everywhere
Operator FSCPT::Delta(const Operator& Op)
{
    Operator Opdelta(*Op.modelspace);
    Opdelta.SetAntiHermitian();
    
    for ( auto& a : Op.modelspace->all_orbits)
    {
        Orbit& oa = Op.modelspace->GetOrbit(a);
        for ( auto& i : Op.modelspace->OneBodyChannels.at({oa.l,oa.j2,oa.tz2}) )
        {
            if(Opdelta.OneBody(a,i)==0) continue;
            Opdelta.OneBody(a,i) = Op.OneBody(a,i) / (H0(a,a) - H0(i,i)) ;
            Opdelta.OneBody(i,a) = -Opdelta.OneBody(a,i) ;
        }
    }

    // #pragma omp parallel for schedule(dynamic, 1)
    for ( auto& iter : Opdelta.TwoBody.MatEl )
    {
        size_t ch_bra = iter.first[0];
        size_t ch_ket = iter.first[1];
        TwoBodyChannel& tbc_bra = Op.modelspace->GetTwoBodyChannel(ch_bra);
        TwoBodyChannel& tbc_ket = Op.modelspace->GetTwoBodyChannel(ch_ket);
        arma::mat& OpMat =  iter.second;
        for (int iket = 0; iket< tbc_ket.GetNumberKets(); ++iket )
        // for ( auto& iket : tbc_ket.GetKetIndex_cc() )
        {
            Ket& ket = Opdelta.modelspace->GetKet(tbc_ket.GetKetIndex(iket));
            double e_ket = H0(ket.p,ket.p) + H0(ket.q,ket.q);
            for (int ibra = 0; ibra< tbc_bra.GetNumberKets(); ++ibra )
            // for ( auto& ibra : VectorUnion(tbc_bra.GetKetIndex_qq(), tbc_bra.GetKetIndex_vv(), tbc_bra.GetKetIndex_qv() ) )
            {
                double matel = Op.TwoBody.MatEl.at(iter.first)(ibra,iket);
                if(matel==0) continue;
                Ket& bra = Opdelta.modelspace->GetKet(tbc_ket.GetKetIndex(ibra));
                double e_bra = H0(bra.p,bra.p) + H0(bra.q,bra.q);
                OpMat(ibra,iket) = matel / (e_bra - e_ket);
                OpMat(iket,ibra) = -OpMat(ibra,iket);
            }
        }
    }
    return Opdelta;
}