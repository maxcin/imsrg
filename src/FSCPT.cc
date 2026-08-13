#include "FSCPT.hh"
#include "Commutator.hh"
#include "Generator.hh"

#include <omp.h>

FSCPT::FSCPT(ModelSpace& ms_imsrg, std::string od):
        modelspace_imsrg(&ms_imsrg),
        off_diagonal(od) 
{
    std::cout <<"Created FSCPT instance for " <<od <<" type corrections" <<std::endl;
}

//Needs 7 Operators (+3 +6 for 2nd and 3rd order = 13 worst case at once)
Operator FSCPT::Calculate(Operator& H)
{
    //If we do not want any correction we just truncate
    if(order < 2) return H.Truncate(*modelspace_imsrg);

    //Allocate all the operators we will need
    H0 = arma::diagmat(H.OneBody);
    V = Operator(*H.modelspace);
    Vod = Operator(*H.modelspace);
    Vd = Operator(*H.modelspace);
    G1 = Operator(*H.modelspace);
    G2 = Operator(*H.modelspace);
    
    Heff2 = Operator(*H.modelspace);
    Heff3 = Operator(*H.modelspace);

    //initialize
    V.OneBody = H.OneBody - H0;
    V.TwoBody = H.TwoBody;
    Vod = GetOpOd(V);
    Vd = V-Vod;

    G1.SetAntiHermitian();
    G2.SetAntiHermitian();

    //Do the calculation
    G1 = Delta(GetOpOd(V));
    FSCPT2();
    std::cout <<"Second order energy= " <<Heff2.ZeroBody <<std::endl;
    if(order > 2)
    {
        FSCPT3();
        std::cout <<"Third order energy= " <<Heff3.ZeroBody <<std::endl;
    }    

    //For valence space the off-diagonal part lives only in the to be truncated space
    //we can return immediately the answer
    if(off_diagonal == "valence")
        return H.Truncate(*modelspace_imsrg)+Heff2.Truncate(*modelspace_imsrg)+Heff3.Truncate(*modelspace_imsrg);

    //Here we calculate the expansion in the full space P+Q and then subtract the results if only expanded in the P space
    else if(off_diagonal == "valence_diff")
    {
        Operator H_eff = H.Truncate(*modelspace_imsrg)+Heff2.Truncate(*modelspace_imsrg)+Heff3.Truncate(*modelspace_imsrg);
        Operator H_small = H.Truncate(*modelspace_imsrg);

        H0 = arma::diagmat(H_small.OneBody);
        V = Operator(*H_small.modelspace);
        Vod = Operator(*H_small.modelspace);
        Vd = Operator(*H_small.modelspace);
        G1 = Operator(*H_small.modelspace);
        G2 = Operator(*H_small.modelspace);
        
        Heff2 = Operator(*H_small.modelspace);
        Heff3 = Operator(*H_small.modelspace);

        //initialize
        V.OneBody = H_small.OneBody - H0;
        V.TwoBody = H_small.TwoBody;
        Vod = GetOpOd(V);
        Vd = V-Vod;

        G1.SetAntiHermitian();
        G2.SetAntiHermitian();
        //Do the calculation
        G1 = Delta(GetOpOd(V));
        FSCPT2();
        std::cout <<"Second order energy small space= " <<Heff2.ZeroBody <<std::endl;
        if(order > 2)
        {
            FSCPT3();
            std::cout <<"Third order energy small space= " <<Heff3.ZeroBody <<std::endl;
        }    

        H_small += Heff2 + Heff3;
        
        //This essentially says that we add the effective operator from P and Q space but with all parts that originate purely from the P space subtracted
        //Strictly speaking when doing now a decoupling on this H we include also (partly) higher order components.
        return H.Truncate(*modelspace_imsrg) + H_eff - H_small; 
    }
    //For single reference we just return the truncated operator along with the corrections
    //But we need to substract the inner corrections so we do it all again
    else if(off_diagonal == "single_reference")
    {
        double dE_2 = Heff2.ZeroBody;
        double dE_3 = Heff3.ZeroBody;
        Operator H_small = H.Truncate(*modelspace_imsrg);

        H0 = arma::diagmat(H_small.OneBody);
        V = Operator(*H_small.modelspace);
        Vod = Operator(*H_small.modelspace);
        Vd = Operator(*H_small.modelspace);
        G1 = Operator(*H_small.modelspace);
        G2 = Operator(*H_small.modelspace);
        
        Heff2 = Operator(*H_small.modelspace);
        Heff3 = Operator(*H_small.modelspace);

        //initialize
        V.OneBody = H_small.OneBody - H0;
        V.TwoBody = H_small.TwoBody;
        Vod = GetOpOd(V);
        Vd = V-Vod;

        G1.SetAntiHermitian();
        G2.SetAntiHermitian();
        //Do the calculation
        G1 = Delta(GetOpOd(V));
        FSCPT2();
        std::cout <<"Second order energy small space= " <<Heff2.ZeroBody <<std::endl;
        if(order > 2)
        {
            FSCPT3();
            std::cout <<"Third order energy small space= " <<Heff3.ZeroBody <<std::endl;
        }    
        double E_corr = dE_2 + dE_3 - Heff2.ZeroBody - Heff3.ZeroBody;
        std::cout <<"Perturbative correction second order: " <<dE_2 - Heff2.ZeroBody <<std::endl;
        std::cout <<"Perturbative correction third order: " <<dE_3 - Heff3.ZeroBody <<std::endl;
        std::cout <<"Perturbative correction total: " <<E_corr <<std::endl;
        H_small.ZeroBody += E_corr; //Add just the energy corrections
        return H_small;
    }
    else
    {
        std::cout <<"Unknown off diagonal definition in FSCPT..." <<std::endl;
        exit(1);
    }

}

//Creates the dHeff2 and dHeff3 to be used in later applications
//This function can be used to treat the truncated part "consistently" in perturbation theory
//We always calculate up to third order here
void FSCPT::CalculateDHeff(Operator& H)
{

    //Allocate all the operators we will need
    H0 = arma::diagmat(H.OneBody);
    V = Operator(*H.modelspace);
    Vod = Operator(*H.modelspace);
    Vd = Operator(*H.modelspace);
    G1 = Operator(*H.modelspace);
    G2 = Operator(*H.modelspace);
    
    Heff2 = Operator(*H.modelspace);
    Heff3 = Operator(*H.modelspace);

    //initialize
    V.OneBody = H.OneBody - H0;
    V.TwoBody = H.TwoBody;
    Vod = GetOpOd(V);
    Vd = V-Vod;

    G1.SetAntiHermitian();
    G2.SetAntiHermitian();

    //Do the calculation
    G1 = Delta(GetOpOd(V));
    FSCPT2();
    std::cout <<"Second order energy= " <<Heff2.ZeroBody <<std::endl;
    FSCPT3();
    std::cout <<"Third order energy= " <<Heff3.ZeroBody <<std::endl;

    //We need to save Heff2 and Heff3 (we only need the P space component (actually only the ZeroBody + Valence but P space is easer)  )
    Operator Heff2_full = Heff2.Truncate(*modelspace_imsrg);
    Operator Heff3_full = Heff3.Truncate(*modelspace_imsrg);

    Operator H_small = H.Truncate(*modelspace_imsrg);

    H0 = arma::diagmat(H_small.OneBody);
    V = Operator(*H_small.modelspace);
    Vod = Operator(*H_small.modelspace);
    Vd = Operator(*H_small.modelspace);
    G1 = Operator(*H_small.modelspace);
    G2 = Operator(*H_small.modelspace);
    
    Heff2 = Operator(*H_small.modelspace);
    Heff3 = Operator(*H_small.modelspace);

    //initialize
    V.OneBody = H_small.OneBody - H0;
    V.TwoBody = H_small.TwoBody;
    Vod = GetOpOd(V);
    Vd = V-Vod;

    G1.SetAntiHermitian();
    G2.SetAntiHermitian();
    //Do the calculation
    G1 = Delta(GetOpOd(V));
    FSCPT2();
    std::cout <<"Second order energy small space= " <<Heff2.ZeroBody <<std::endl;
    FSCPT3();
    std::cout <<"Third order energy small space= " <<Heff3.ZeroBody <<std::endl;

    //Now Heff2 and Heff3 are the small space results
    //We can now calculate the relevant Operators
    Delta_Heff2 = Heff2_full - Heff2; //pure second order contributions from truncation
    Delta_Heff3 = Heff3_full - Heff3; //pure third order contributions from truncation

    std::cout <<"Second order energy from truncated space: " <<Delta_Heff2.ZeroBody <<std::endl;
    std::cout <<"Third order energy from truncated space: " <<Delta_Heff3.ZeroBody <<std::endl;
        
}

//The definition of Heff changes here
//Above Heff = Heff1 + Heff2 + Heff3 + ...
//Here Heff2 will also contain Heff1
//i.e. Heff = Heff1 //for first order
//     Heff = Heff2 //for second order
//     Heff = Heff3 //for third order
void FSCPT::CalculateAtanDHeff(Operator& H)
{
    //Full space first

    //Setup
    H0 = arma::diagmat(H.OneBody);
    Operator H0Op = Operator(*H.modelspace);
    V = Operator(*H.modelspace);
    Vod = Operator(*H.modelspace);

    G1 = Operator(*H.modelspace);
    G2 = Operator(*H.modelspace);
    G3 = Operator(*H.modelspace);
    G1.SetAntiHermitian();
    G2.SetAntiHermitian();
    G3.SetAntiHermitian();

    
    Heff1 = Operator(*H.modelspace);
    Heff2 = Operator(*H.modelspace);
    Heff3 = Operator(*H.modelspace);

    H0Op.OneBody = H0;
    V.OneBody = H.OneBody - H0;
    V.TwoBody = H.TwoBody;

    //**insert here expressions for calculating Heff1 to Heff3**
    //1
    G1 = atanDelta(GetOpOd(V));
    Heff1 = H0Op + V + Commutator::Commutator(G1,H0Op);
    //2
    //Calculate first the transformed H then the Generator by the offdiagonal part
    //The use add the relevant commutator
    Operator Hn2 = Commutator::Commutator(G1, V) + (1/2.0)*Commutator::Commutator(G1, Commutator::Commutator(G1,H0Op)); 
    G2 = atanDelta(GetOpOd(Heff1 + Hn2));
    Heff2 = Heff1 + Hn2 + Commutator::Commutator(G2, H0Op); 
    //3
    Operator Hn3 = (1/6.0)*Commutator::Commutator(G1, Commutator::Commutator(G1,Commutator::Commutator(G1,H0Op)))
                 + (1/2.0)*Commutator::Commutator(G1, Commutator::Commutator(G2, H0Op))
                 + (1/2.0)*Commutator::Commutator(G2, Commutator::Commutator(G1, H0Op))
                 + (1/2.0)*Commutator::Commutator(G1, Commutator::Commutator(G1, V))
                 + Commutator::Commutator(G2, V);

    G3 = atanDelta(GetOpOd(Heff2 + Hn3));
    Heff3 = Heff2 + Hn3 + Commutator::Commutator(G3, H0Op);
    
    Operator Heff1_full = Heff1.Truncate(*modelspace_imsrg);
    Operator Heff2_full = Heff2.Truncate(*modelspace_imsrg);
    Operator Heff3_full = Heff3.Truncate(*modelspace_imsrg);

    // Now the small space
    //Setup
    Operator H_small = H.Truncate(*modelspace_imsrg);
    H0 = arma::diagmat(H_small.OneBody);
    H0Op = Operator(*H_small.modelspace);
    V = Operator(*H_small.modelspace);
    Vod = Operator(*H_small.modelspace);

    G1 = Operator(*H_small.modelspace);
    G2 = Operator(*H_small.modelspace);
    G3 = Operator(*H_small.modelspace);
    G1.SetAntiHermitian();
    G2.SetAntiHermitian();
    G3.SetAntiHermitian();

    
    Heff1 = Operator(*H_small.modelspace);
    Heff2 = Operator(*H_small.modelspace);
    Heff3 = Operator(*H_small.modelspace);

    H0Op.OneBody = H0;
    V.OneBody = H_small.OneBody - H0;
    V.TwoBody = H_small.TwoBody;

    //**insert here expressions for calculating Heff1 to Heff3**
    //1
    G1 = atanDelta(GetOpOd(V));
    Heff1 = H0Op + V + Commutator::Commutator(G1,H0Op);
    //2
    //Calculate first the transformed H then the Generator by the offdiagonal part
    //The use add the relevant commutator
    Hn2 = Commutator::Commutator(G1, V) + (1/2.0)*Commutator::Commutator(G1, Commutator::Commutator(G1,H0Op)); 
    G2 = atanDelta(GetOpOd(Heff1 + Hn2));
    Heff2 = Heff1 + Hn2 + Commutator::Commutator(G2, H0Op); 
    //3
    Hn3 = (1/6.0)*Commutator::Commutator(G1, Commutator::Commutator(G1,Commutator::Commutator(G1,H0Op)))
                 + (1/2.0)*Commutator::Commutator(G1, Commutator::Commutator(G2, H0Op))
                 + (1/2.0)*Commutator::Commutator(G2, Commutator::Commutator(G1, H0Op))
                 + (1/2.0)*Commutator::Commutator(G1, Commutator::Commutator(G1, V))
                 + Commutator::Commutator(G2, V);

    G3 = atanDelta(GetOpOd(Heff2 + Hn3));
    Heff3 = Heff2 + Hn3 + Commutator::Commutator(G3, H0Op);

    

    Delta_Heff1 = Heff1_full - Heff1;
    Delta_Heff2 = Heff2_full - Heff2;
    Delta_Heff3 = Heff3_full - Heff3;

    //We want to keep only the diagonal part as the remainder is of higher order then indicated
    //This makes a difference when undoing the normal ordering but because it is a higher order
    //effect it should be fine
    //This does not affect the convergence of the expansion (if it converges)
    Delta_Heff1 -= GetOpOd(Delta_Heff1);
    Delta_Heff2 -= GetOpOd(Delta_Heff2);
    Delta_Heff3 -= GetOpOd(Delta_Heff3);

    std::cout <<"Second order energy from truncated space: " <<Delta_Heff2.ZeroBody <<std::endl;
    std::cout <<"Third order energy from truncated space: " <<Delta_Heff3.ZeroBody <<std::endl;
}

//As above but refactored to use less memory in the large space
void FSCPT::CalculateAtanDHeff_memory(Operator& H)
{
    //Full space first

    //Setup

    // I count 8 Operators here
    // At emax=16 this  is ~80 GB
    // Taking the Commutator will also take ~ 100 GB

    // H0 can be a 1B operator

    H0 = arma::diagmat(H.OneBody);
    Operator H0Op = Operator(*H.modelspace, 0 ,0 ,0 ,2);
    V = Operator(*H.modelspace);

    G1 = Operator(*H.modelspace);
    G1.SetAntiHermitian();

    G2 = Operator(*H.modelspace);
    G2.SetAntiHermitian();

    
    // Heff1 = Operator(*H.modelspace);
    // Heff2 = Operator(*H.modelspace);
    // Heff3 = Operator(*H.modelspace);
    Operator Heff = Operator(*H.modelspace);

    H0Op.OneBody = H0;
    V.OneBody = H.OneBody - H0;
    V.TwoBody = H.TwoBody;

    //**insert here expressions for calculating Heff1 to Heff3**
    //1

    G1 = atanDelta(GetOpOd(V));

    Heff = H0Op + V + Commutator::Commutator(G1,H0Op);
    Operator Heff1_full = Heff.Truncate(*modelspace_imsrg);
    //2
    //Calculate first the transformed H then the Generator by the offdiagonal part
    //The use add the relevant commutator
    Operator Hn2 = Commutator::Commutator(G1, V) + (1/2.0)*Commutator::Commutator(G1, Commutator::Commutator(G1,H0Op)); 
    G2 = atanDelta(GetOpOd(Heff + Hn2));
    Heff += Hn2 + Commutator::Commutator(G2, H0Op); 

    //We dont need Hn2 at this point 
    Hn2 = Operator();
    Operator Heff2_full = Heff.Truncate(*modelspace_imsrg);
    //3
    Operator Hn3 = (1/6.0)*Commutator::Commutator(G1, Commutator::Commutator(G1,Commutator::Commutator(G1,H0Op)))
                 + (1/2.0)*Commutator::Commutator(G1, Commutator::Commutator(G2, H0Op))
                 + (1/2.0)*Commutator::Commutator(G2, Commutator::Commutator(G1, H0Op))
                 + (1/2.0)*Commutator::Commutator(G1, Commutator::Commutator(G1, V))
                 + Commutator::Commutator(G2, V);

    //At this point neither G1, G2 or V are needed so lets free them
    G1 = Operator();
    G2 = Operator();
    V = Operator();

    //G3 is only needed at this point
    G3 = Operator(*H.modelspace);
    G3.SetAntiHermitian();
    G3 = atanDelta(GetOpOd(Heff + Hn3));
    Heff += Hn3 + Commutator::Commutator(G3, H0Op);
    Operator Heff3_full = Heff.Truncate(*modelspace_imsrg);
    Heff = Operator();

    //After this everything is in the small space so we dont need to worry about memory

    // Now the small space
    //Setup
    Operator H_small = H.Truncate(*modelspace_imsrg);
    H0 = arma::diagmat(H_small.OneBody);
    H0Op = Operator(*H_small.modelspace);
    V = Operator(*H_small.modelspace);

    G1 = Operator(*H_small.modelspace);
    G1.SetAntiHermitian();

    G2 = Operator(*H_small.modelspace);
    G2.SetAntiHermitian();

    G3 = Operator(*H_small.modelspace);
    G3.SetAntiHermitian();

    
    Heff1 = Operator(*H_small.modelspace);
    Heff2 = Operator(*H_small.modelspace);
    Heff3 = Operator(*H_small.modelspace);

    H0Op.OneBody = H0;
    V.OneBody = H_small.OneBody - H0;
    V.TwoBody = H_small.TwoBody;

    //**insert here expressions for calculating Heff1 to Heff3**
    //1
    G1 = atanDelta(GetOpOd(V));
    Heff1 = H0Op + V + Commutator::Commutator(G1,H0Op);
    //2
    //Calculate first the transformed H then the Generator by the offdiagonal part
    //The use add the relevant commutator
    Hn2 = Commutator::Commutator(G1, V) + (1/2.0)*Commutator::Commutator(G1, Commutator::Commutator(G1,H0Op)); 
    G2 = atanDelta(GetOpOd(Heff1 + Hn2));
    Heff2 = Heff1 + Hn2 + Commutator::Commutator(G2, H0Op); 
    //3
    Hn3 = (1/6.0)*Commutator::Commutator(G1, Commutator::Commutator(G1,Commutator::Commutator(G1,H0Op)))
                 + (1/2.0)*Commutator::Commutator(G1, Commutator::Commutator(G2, H0Op))
                 + (1/2.0)*Commutator::Commutator(G2, Commutator::Commutator(G1, H0Op))
                 + (1/2.0)*Commutator::Commutator(G1, Commutator::Commutator(G1, V))
                 + Commutator::Commutator(G2, V);

    G3 = atanDelta(GetOpOd(Heff2 + Hn3));
    Heff3 = Heff2 + Hn3 + Commutator::Commutator(G3, H0Op);

    

    Delta_Heff1 = Heff1_full - Heff1;
    Delta_Heff2 = Heff2_full - Heff2;
    Delta_Heff3 = Heff3_full - Heff3;

    //We want to keep only the diagonal part as the remainder is of higher order then indicated
    //This makes a difference when undoing the normal ordering but because it is a higher order
    //effect it should be fine
    //This does not affect the convergence of the expansion (if it converges)
    Delta_Heff1 -= GetOpOd(Delta_Heff1);
    Delta_Heff2 -= GetOpOd(Delta_Heff2);
    Delta_Heff3 -= GetOpOd(Delta_Heff3);

    std::cout <<"Second order energy from truncated space: " <<Delta_Heff2.ZeroBody <<std::endl;
    std::cout <<"Third order energy from truncated space: " <<Delta_Heff3.ZeroBody <<std::endl;
}

//Needs 1 commutator and creates 3 Operators
void FSCPT::FSCPT2()
{
    // Operator vd2vod = 2*Vd + Vod;
    Operator second_order = 0.5*Commutator::Commutator(G1, 2*Vd + Vod);
    Operator second_order_od = GetOpOd(second_order);

    Heff2 = second_order-second_order_od;

    G2 = Delta(second_order_od);
}


//Needs 4 (5) commutators and creates 6 Operators
void FSCPT::FSCPT3()
{
    // Operator v_vd = V+Vd;
    // Operator vv_vd = 2*V+Vd;
    // Operator vd_vd_vod = 2*Vd + Vod;

    // Operator g1_vd_vd_vod_3 = 3*GetOpOd(Commutator::Commutator(G1,vd_vd_vod));

    // Operator combine = 2*Commutator::Commutator(G1,vv_vd) - g1_vd_vd_vod_3;
    
    // Operator third_order = 0.5 * Commutator::Commutator(G2,v_vd) + (1/12.0)* Commutator::Commutator(G1,combine);

    Operator third_order = 0.5* Commutator::Commutator(G2, V+Vd)
                        + 1/6. * Commutator::Commutator(G1, Commutator::Commutator(G1, 2*V+Vd))
                        - 1/4. * Commutator::Commutator(G1, GetOpOd(Commutator::Commutator(G1,2*Vd+Vod)) );
    
    Heff3 = third_order - GetOpOd(third_order);

    //G3 is not needed for this type of calculation so we dont store it it
    //G3 = Delta(third_order - Heff3); 
}

Operator FSCPT::GetOpOd(const Operator& Op)
{
    Operator OpOd(Op);

    if(off_diagonal == "valence")
    {
        OpOd = generator.GetHod_ShellModel(OpOd);
        //Off diagonal is only in the large space
        //These lines replace the P space components with 0
        Operator small_space(*modelspace_imsrg);
        OpOd.replaceSubOperator(small_space);        
    } 
    else if(off_diagonal == "valence_diff") OpOd = generator.GetHod_ShellModel(OpOd);
    else if(off_diagonal == "single_reference") OpOd = generator.GetHod_SingleRef_ph(OpOd);
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

    
    int nmatel = Opdelta.TwoBody.MatEl.size();
    //for ( auto& iter : Opdelta.TwoBody.MatEl )
    #pragma omp parallel for schedule(dynamic, 1)
    for(int index = 0 ; index < nmatel; ++index)
    {
        auto iter = Opdelta.TwoBody.MatEl.begin();
        std::advance(iter , index);
        size_t ch_bra = iter->first[0];
        size_t ch_ket = iter->first[1];
        TwoBodyChannel& tbc_bra = Op.modelspace->GetTwoBodyChannel(ch_bra);
        TwoBodyChannel& tbc_ket = Op.modelspace->GetTwoBodyChannel(ch_ket);
        arma::mat& OpMat =  iter->second;
        for (int iket = 0; iket< tbc_ket.GetNumberKets(); ++iket )
        // for ( auto& iket : tbc_ket.GetKetIndex_cc() )
        {
            Ket& ket = Opdelta.modelspace->GetKet(tbc_ket.GetKetIndex(iket));
            double e_ket = H0(ket.p,ket.p) + H0(ket.q,ket.q);
            for (int ibra = 0; ibra< tbc_bra.GetNumberKets(); ++ibra )
            // for ( auto& ibra : VectorUnion(tbc_bra.GetKetIndex_qq(), tbc_bra.GetKetIndex_vv(), tbc_bra.GetKetIndex_qv() ) )
            {
                double matel = Op.TwoBody.MatEl.at(iter->first)(ibra,iket);
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

//The new operator is antihermitian (if Op is hermitian). To make this most general we just add the delta everywhere
Operator FSCPT::atanDelta(const Operator& Op)
{
    Operator Opdelta(*Op.modelspace);
    Opdelta.SetAntiHermitian();
    
    for ( auto& a : Op.modelspace->all_orbits)
    {
        Orbit& oa = Op.modelspace->GetOrbit(a);
        for ( auto& i : Op.modelspace->OneBodyChannels.at({oa.l,oa.j2,oa.tz2}) )
        {
            if(Opdelta.OneBody(a,i)==0) continue;
            Opdelta.OneBody(a,i) = 0.5 * atan(2* Op.OneBody(a,i) / (H0(a,a) - H0(i,i)) );
            Opdelta.OneBody(i,a) = -Opdelta.OneBody(a,i) ;
        }
    }

    
    int nmatel = Opdelta.TwoBody.MatEl.size();
    //for ( auto& iter : Opdelta.TwoBody.MatEl )
    #pragma omp parallel for schedule(dynamic, 1)
    for(int index = 0 ; index < nmatel; ++index)
    {
        auto iter = Opdelta.TwoBody.MatEl.begin();
        std::advance(iter , index);
        size_t ch_bra = iter->first[0];
        size_t ch_ket = iter->first[1];
        TwoBodyChannel& tbc_bra = Op.modelspace->GetTwoBodyChannel(ch_bra);
        TwoBodyChannel& tbc_ket = Op.modelspace->GetTwoBodyChannel(ch_ket);
        arma::mat& OpMat =  iter->second;
        for (int iket = 0; iket< tbc_ket.GetNumberKets(); ++iket )
        // for ( auto& iket : tbc_ket.GetKetIndex_cc() )
        {
            Ket& ket = Opdelta.modelspace->GetKet(tbc_ket.GetKetIndex(iket));
            double e_ket = H0(ket.p,ket.p) + H0(ket.q,ket.q);
            for (int ibra = 0; ibra< tbc_bra.GetNumberKets(); ++ibra )
            // for ( auto& ibra : VectorUnion(tbc_bra.GetKetIndex_qq(), tbc_bra.GetKetIndex_vv(), tbc_bra.GetKetIndex_qv() ) )
            {
                double matel = Op.TwoBody.MatEl.at(iter->first)(ibra,iket);
                if(matel==0) continue;
                Ket& bra = Opdelta.modelspace->GetKet(tbc_ket.GetKetIndex(ibra));
                double e_bra = H0(bra.p,bra.p) + H0(bra.q,bra.q);
                OpMat(ibra,iket) = 0.5 * atan(2*matel / (e_bra - e_ket));
                OpMat(iket,ibra) = -OpMat(ibra,iket);
            }
        }
    }
    return Opdelta;
}
