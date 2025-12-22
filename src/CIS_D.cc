
#include "CIS_D.hh"
#include "PhysicalConstants.hh"

CISD::CISD(Operator& Op, int J, int P, int Tz) : RPA(Op), J(J), P(P), Tz(Tz), Jhat(sqrt(2*J+1))
{
    RunTDA();
}

CISD::CISD(Operator& Op, int J) : RPA(Op), J(J), Jhat(sqrt(2*J+1))
{
    RunTDA();
}

void CISD::RunTDA()
{
    ConstructAMatrix(J, P, Tz, false);
    SolveTDA();
    BuildTDAIndex();
}

void CISD::BuildTDAIndex()
{
    TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(channel);

    TDAindex.ones(modelspace->GetNumberKets());
    TDAindex *= -1;

    size_t I_ph = 0;
    for (auto iket_ai : tbc_CC.GetKetIndex_ph() )
    {
        Ket& ket_ai = tbc_CC.GetKet(iket_ai);
        index_t a = ket_ai.p;
        index_t i = ket_ai.q;
        size_t KetIndex = modelspace->GetKetIndex(std::min(a,i), std::max(a,i));
        //A(I_ph,II_ph) = H1b  + V_aibj * phase_ai *phase_bj;
        TDAindex(KetIndex) = I_ph;
        I_ph ++;
    }

}

size_t CISD::GetTDAIndex(int a, int i)
{
    size_t GlobalIndex = modelspace->GetKetIndex(std::min(a,i), std::max(a,i));
    return TDAindex(GlobalIndex);
}

//Each column of X is normalized to 1
//Our equations use normalization $ \sum_{ai} b^{J}_{ai}b^{J}_{ai} = \hat{J}^2$
//Conversion is to multiply entries by \hat{J}=\sqrt{2J+1}
double CISD::bSingles(int nstate, int a, int i)
{
    //Jhat for normalization
    arma::vec x =Jhat* X.col(nstate);    

    size_t index = GetTDAIndex(a,i);

    if(index == -1) return 0.0;
    else return x(index);
}

void CISD::uPrecalculateDoubles(int nstate)
{

    std::cout <<"Precalculating u_abij J=" <<J <<" P=" <<P <<" Tz=" <<Tz <<" ..." <<std::endl;

    uDoublesCache.insert({nstate, TwoBodyME(modelspace, J, Tz, P)});

    TwoBodyME& uCache = uDoublesCache.at(nstate);
    uCache.SetHermitian();

    //We now loop over the matrix elements and fill them with our amplitudes
    //Because TBME stores only bra <= ket we need to fill pphh AND hhpp if bra < ket
    //#pragma omp parallel for schedule(dynamic, 1)
    for(auto& it : uCache.MatEl)
    {
        int ichannel_bra = it.first[0];
        int ichannel_ket = it.first[1];

        TwoBodyChannel& channel_bra = modelspace->GetTwoBodyChannel(ichannel_bra);
        TwoBodyChannel& channel_ket = modelspace->GetTwoBodyChannel(ichannel_ket);

        int J1 = channel_bra.J;
        int J2 = channel_ket.J;

        arma::mat& Mat = it.second;

        //pp hh
        for(auto& bra_pp : channel_bra.GetKetIndex_pp())
        {
            Ket bra = channel_bra.GetKet(bra_pp);
            int a = bra.p;
            int b = bra.q;

            for(auto& ket_hh : channel_ket.GetKetIndex_hh())
            {
                Ket ket = channel_ket.GetKet(ket_hh);
                int i = ket.p;
                int j = ket.q;

                double u_abij = uDoubles(nstate, J1, J2, a,b,i,j);

                Mat(bra_pp, ket_hh) = u_abij;

            }
        }

        //hh pp, Need to swap J1 and J2
        std::swap(J1,J2);
        for(auto& bra_hh : channel_bra.GetKetIndex_hh())
        {
            Ket bra = channel_bra.GetKet(bra_hh);
            int i = bra.p;
            int j = bra.q;

            for(auto& ket_pp : channel_ket.GetKetIndex_pp())
            {
                Ket ket = channel_ket.GetKet(ket_pp);
                int a = ket.p;
                int b = ket.q;

                double u_abij = uDoubles(nstate, J1, J2, a,b,i,j);

                //We now have <ab J1|u|ij J2> but we need <ij J2|u|ab J1> which introduces an additional phase 
                Mat(bra_hh, ket_pp) = modelspace->phase(J1-J2)*u_abij;

            }
        }
    }//MatEl
}

double CISD::bDoubles(int nstate, int J1, int J2, int a, int b, int i, int j)
{
    double wTDA = Energies(nstate);

    double ea = H.OneBody(a,a);
    double eb = H.OneBody(b,b);
    double ei = H.OneBody(i,i);
    double ej = H.OneBody(j,j);

    double denom = ea + eb - ei - ej - wTDA;

    // double u_abij = uDoubles(nstate, J1, J2, a, b, i, j);
    double u_abij = uDoublesCached(nstate, J1, J2, a, b, i, j);
    return -u_abij / denom;
}

double CISD::uDoublesCached(int nstate, int J1, int J2, int a, int b , int i, int j)
{
    TwoBodyME& u = uDoublesCache.at(nstate);
    
    return u.GetTBME_J_norm(J1,J2, a,b,i,j);
}

double CISD::uDoubles(int nstate, int J1, int J2, int a, int b, int i, int j)
{
    Orbit& oa = modelspace->GetOrbit(a);
    Orbit& ob = modelspace->GetOrbit(b);
    Orbit& oi = modelspace->GetOrbit(i);
    Orbit& oj = modelspace->GetOrbit(j);

    double ja = ( (double) oa.j2 )/ 2.0;
    double jb = ( (double) ob.j2 ) / 2.0;
    double ji = ( (double) oi.j2 ) / 2.0;
    double jj = ( (double) oj.j2 ) / 2.0;

    double u_p = 0.0;

    for(int c : modelspace->particles)
    {
        Orbit& oc = modelspace->GetOrbit(c);
        double jc = ( (double) oc.j2 )/ 2.0;
        double b_ci = bSingles(nstate, c, i);
        double b_cj = bSingles(nstate, c, j);

        if(std::abs(b_ci) > 1e-10)
        {
            double sixj = modelspace->GetSixJ(J1, J2, J, ji,jc,jj);
            double H_abcj = H.TwoBody.GetTBME_J_norm(J1,a,b,c,j);
            int phase = modelspace->phase(J+J2+(oc.j2+oj.j2)/2);

            u_p += phase*H_abcj*b_ci*sixj;
        }

        if(std::abs(b_cj) > 1e-10)
        {
            double sixj = modelspace->GetSixJ(J1, J2, J, jj,jc,ji);
            double H_abci = H.TwoBody.GetTBME_J_norm(J1,a,b,c,i);
            int phase = modelspace->phase(J+(oc.j2+oj.j2)/2);

            u_p += phase*H_abci*b_cj*sixj;
        }

    }

    double u_h = 0.0;

    for(int k : modelspace->holes)
    {
        Orbit& ok = modelspace->GetOrbit(k);
        double jk = ( (double) ok.j2 )/ 2.0;
        double b_bk = bSingles(nstate, b, k);
        double b_ak = bSingles(nstate, a, k);

        if(std::abs(b_bk) > 1e-10)
        {
            double sixj = modelspace->GetSixJ(J1, J2, J, jk,jb,ja);
            double H_kaij = H.TwoBody.GetTBME_J_norm(J2,k,a,i,j);
            int phase = modelspace->phase(J+J1+J2);

            u_h += phase*H_kaij*b_bk*sixj;
        }

        if(std::abs(b_ak) > 1e-10)
        {
            double sixj = modelspace->GetSixJ(J1, J2, J, jk,ja,jb);
            double H_kbij = H.TwoBody.GetTBME_J_norm(J2,k,b,i,j);
            int phase = modelspace->phase(J+J2+(oa.j2+ob.j2)/2);

            u_h -= phase*H_kbij*b_ak*sixj;
        }
    }

    double J1hat = sqrt(2*J1+1);
    double J2hat = sqrt(2*J2+1);

    double u_abij = J1hat*J2hat*(u_p+u_h);

    return u_abij;
}


double CISD::vSingles(int nstate, int a, int i)
{
    double vai2 = 0.0;

    Orbit& oa = modelspace->GetOrbit(a);
    Orbit& oi = modelspace->GetOrbit(i);

    double ja = ( (double) oa.j2 )/ 2.0;
    double ji = ( (double) oi.j2 ) / 2.0;

    double ea = H.OneBody(a,a);
    double ei = H.OneBody(i,i);

    for(int j : modelspace->holes)
    {
        Orbit& oj = modelspace->GetOrbit(j);
        double jj = ( (double) oj.j2 )/ 2.0;
        double ej = H.OneBody(j,j);
        for(int k: modelspace->holes)
        {
            Orbit& ok = modelspace->GetOrbit(k);
            double jk = ( (double) ok.j2 )/ 2.0;
            double ek = H.OneBody(k,k);
            for(int b : modelspace->particles)
            {
                Orbit& ob = modelspace->GetOrbit(b);
                double jb = ( (double) ob.j2 )/ 2.0;
                double eb = H.OneBody(b,b);
                for(int c: modelspace->particles)
                {
                    Orbit& oc = modelspace->GetOrbit(c);
                    double jc = ( (double) oc.j2 )/ 2.0;
                    double ec = H.OneBody(c,c);

                    int J1min = std::max(std::abs(oj.j2-ok.j2),std::abs(ob.j2-oc.j2))/2;
                    int J1max = std::min(oj.j2+ok.j2, ob.j2+oc.j2)/2;

                    
                    //Term 1
                    double b_bi = bSingles(nstate,b,i);
                    if(oa.j2 == ob.j2 and std::abs(b_bi) > 1e-10)
                    {
                        for(int J1 = J1min; J1 <= J1max; ++J1)
                        {
                            double denom = ec + ea - ej - ek;
                            double H_jkbc = H.TwoBody.GetTBME_J_norm(J1,j,k,b,c);
                            double a_cajk = -H.TwoBody.GetTBME_J_norm(J1,c,a,j,k)/ (denom);

                            int phase = modelspace->phase(J1+(oc.j2+oa.j2)/2);

                            double v = phase*(2*J1+1)*H_jkbc*b_bi*a_cajk ;
                            vai2 += v / (2.0*ja+1.0);
                        }
                    }
                    
                    //Term 2
                    double b_aj = bSingles(nstate,a,j);
                    if(oi.j2 == oj.j2 and std::abs(b_aj) > 1e-10)
                    {
                        for(int J1 = J1min; J1 <= J1max; ++J1)
                        {
                            double denom = ec + eb - ei - ek;
                            double H_jkbc = H.TwoBody.GetTBME_J_norm(J1,j,k,b,c);
                            double a_cbik = -H.TwoBody.GetTBME_J_norm(J1,c,b,i,k)/ (denom);

                            int phase = modelspace->phase(J1+(oc.j2+ob.j2)/2);

                            double v = phase*(2*J1+1)*H_jkbc*b_aj*a_cbik ;
                            vai2 += v / (2.0*ji+1.0);
                        }
                    }

                    //Term 3
                    double b_bj = bSingles(nstate, b,j);
                    if (std::abs(b_bj) > 1e-10) 
                    {
                        int J2min = std::max(std::abs(oi.j2-ok.j2),std::abs(oa.j2-oc.j2))/2;
                        int J2max = std::min(oi.j2+ok.j2, oa.j2+oc.j2)/2;
                        double denom = ea + ec - ei - ek;
                        for(int J1 = J1min; J1 <= J1max; ++J1)
                        {
                            double H_jkbc = H.TwoBody.GetTBME_J_norm(J1,j,k,b,c);
                            double sixJ1 = modelspace->GetSixJ(jb,jj,J,jk,jc,J1);
                            for(int J2 = J2min; J2 <= J2max; ++J2)
                            {
                                double sixJ2 = modelspace->GetSixJ(ja,ji,J,jk,jc,J2);
                                double a_acik = -H.TwoBody.GetTBME_J_norm(J2,a,c,i,k) / denom;
                                int phase = modelspace->phase(J1+J2+(oi.j2+oj.j2)/2);
                                double v = phase * (2*J1+1) * (2*J2+1) * H_jkbc * b_bj * a_acik * sixJ1 * sixJ2 ;
                                vai2 -= 2*v;
                            }
                        }
                    }
                    
                }//c
            }//b
        }//k
    }//j

    return vai2 / 2.0;
}

//Energy correction
double CISD::E_CISD(int nstate)
{
    double wTDA = Energies(nstate);
    double wCISD = 0.0;
    for(int a: modelspace->particles)
    {
        Orbit& oa = modelspace->GetOrbit(a);
        double ea = H.OneBody(a,a);
        for(int b: modelspace->particles)
        {
            Orbit& ob = modelspace->GetOrbit(b);
            double eb = H.OneBody(b,b);
            int J1min = std::abs(oa.j2-ob.j2)/2;
            int J1max = (oa.j2+ob.j2)/2;
            for(int i : modelspace->holes)
            {
                Orbit& oi = modelspace->GetOrbit(i);
                double ei = H.OneBody(i,i);
                for(int j : modelspace->holes)
                {
                    Orbit& oj = modelspace->GetOrbit(j);
                    double ej = H.OneBody(j,j);
                    int J2min = std::abs(oi.j2-oj.j2)/2;
                    int J2max = (oi.j2+oj.j2)/2;

                    double denom = ea + eb - ei - ej - wTDA;

                    double wJsum = 0.0;

                    for(int J1 = J1min; J1 <= J1max; ++J1)
                    {
                        for(int J2 = J2min; J2 <= J2max; ++J2)
                        {
                            //if (J > J1+J2 or J <std::abs(J1-J2)) continue;
                            // double u_abij = uDoubles(nstate, J1, J2, a, b, i, j);
                            double u_abij = uDoublesCached(nstate, J1, J2, a, b, i, j);

                            wJsum += u_abij*u_abij;
                        }
                    }


                    wCISD -= 0.25 * wJsum / denom;
                }//j
            }//i
        }//b
    }//a


    for(int a : modelspace->particles)
    {
        for(int i: modelspace->holes)
        {
            double b_ai = bSingles(nstate,a,i);
            if(std::abs(b_ai) < 1e-10) continue;
            double v_ai = vSingles(nstate, a,i);

            wCISD += b_ai*v_ai;
        }
    }

    return wCISD / (Jhat*Jhat);
}

void CISD::Energy_test(int nstate)
{
    std::cout <<"State " <<nstate <<std::endl;

    // double E_mat = arma::as_scalar(X.col(nstate).t() * A * X.col(nstate));

    double wCISD = E_CISD(nstate);
    // std::cout <<"TDA with matrix " <<E_mat <<std::endl;
    std::cout <<"TDA " <<Energies(nstate) <<std::endl;
    std::cout <<"CISD " <<Energies(nstate)+wCISD<<std::endl;
    std::cout <<"CISD correction " <<wCISD<<std::endl;
        
}

//Correction of TDA to the scalar density
arma::mat CISD::TDAScalarDensityPP(int nstate)
{
    int Norbits = modelspace->norbits;
    arma::mat rho_TDA = arma::zeros(Norbits, Norbits);

    for(int a : modelspace->particles)
    {
        Orbit& oa = modelspace->GetOrbit(a);

        for(int b : modelspace->OneBodyChannels.at({oa.l,oa.j2,oa.tz2}))
        {
            Orbit& ob = modelspace->GetOrbit(b);
            if(ob.occ > modelspace->OCC_CUT) continue;
            if(b > a) continue;

            double r_ab = 0.0;

            for(int i : modelspace->holes)
            {
                Orbit& oi = modelspace->GetOrbit(i);

                r_ab += bSingles(nstate, a,i)*bSingles(nstate, b,i);
            }

            rho_TDA(a,b) = r_ab / ((2*J+1)*(oa.j2+1));
            rho_TDA(b,a) = r_ab / ((2*J+1)*(oa.j2+1));

        }
    }

    return rho_TDA;
}

arma::mat CISD::TDAScalarDensityHH(int nstate)
{
    int Norbits = modelspace->norbits;
    arma::mat rho_TDA = arma::zeros(Norbits, Norbits);

    for(int i : modelspace->holes)
    {
        Orbit& oi = modelspace->GetOrbit(i);

        for(int j : modelspace->OneBodyChannels.at({oi.l,oi.j2,oi.tz2}))
        {
            Orbit& oj = modelspace->GetOrbit(j);
            if(oj.occ < modelspace->OCC_CUT) continue;
            if(j > i) continue;

            double r_ij = 0.0;

            for(int a : modelspace->particles)
            {
                Orbit& oa = modelspace->GetOrbit(a);

                r_ij -= bSingles(nstate, a,i)*bSingles(nstate, a,j);
            }

            rho_TDA(i,j) = r_ij / ((2*J+1)*(oi.j2+1));
            rho_TDA(j,i) = r_ij / ((2*J+1)*(oi.j2+1));

        }
    }

    return rho_TDA;
}

//Correction of CIS(D) to the scalar density
arma::mat CISD::CISDScalarDensityPP(int nstate)
{
    int Norbits = modelspace->norbits;
    arma::mat rho_CISD = arma::zeros(Norbits, Norbits);

    for(int a : modelspace->particles)
    {
        Orbit& oa = modelspace->GetOrbit(a);

        for(int b : modelspace->OneBodyChannels.at({oa.l,oa.j2,oa.tz2}))
        {
            Orbit& ob = modelspace->GetOrbit(b);
            if(ob.occ > modelspace->OCC_CUT) continue;
            if(b > a) continue;

            double r_ab = 0.0;
            for(int c : modelspace->particles)
            {
                Orbit& oc = modelspace->GetOrbit(c);
                for(int i : modelspace->holes)
                {
                    Orbit& oi = modelspace->GetOrbit(i);
                    for(int j : modelspace->holes)
                    {
                        Orbit& oj = modelspace->GetOrbit(j);

                        int J1min = std::max(std::abs(oa.j2-oc.j2), std::abs(ob.j2-oc.j2))/2;
                        int J1max = std::min(oa.j2+oc.j2, ob.j2+oc.j2)/2;

                        int J2min = std::abs(oi.j2-oj.j2)/2;
                        int J2max = (oi.j2+oj.j2)/2;

                        for(int J1 = J1min; J1<=J1max; ++J1)
                        {
                            for(int J2 = J2min; J2<=J2max; ++J2)
                            {
                                // if( J < std::abs(J1-J2)  or J > J1+J2) continue;
                                r_ab += bDoubles(nstate, J1, J2, c, a, i ,j)*bDoubles(nstate, J1, J2, c, b, i ,j);
                            }
                        }
                    }//j
                }//i
            }//c
            rho_CISD(a,b) = 0.5*r_ab / ((2*J+1)*(oa.j2+1));
            rho_CISD(b,a) = 0.5*r_ab / ((2*J+1)*(oa.j2+1));
        }//b
    }//a

    return rho_CISD;
}


arma::mat CISD::CISDScalarDensityHH(int nstate)
{
    int Norbits = modelspace->norbits;
    arma::mat rho_CISD = arma::zeros(Norbits, Norbits);

    for(int i : modelspace->holes)
    {
        Orbit& oi = modelspace->GetOrbit(i);

        for(int j : modelspace->OneBodyChannels.at({oi.l,oi.j2,oi.tz2}))
        {
            Orbit& oj = modelspace->GetOrbit(j);
            if(oj.occ < modelspace->OCC_CUT) continue;
            if(j > i) continue;

            double r_ij = 0.0;
            for(int k : modelspace->holes)
            {
                Orbit& ok = modelspace->GetOrbit(k);
                for(int a : modelspace->particles)
                {
                    Orbit& oa = modelspace->GetOrbit(a);
                    for(int b : modelspace->particles)
                    {
                        Orbit& ob = modelspace->GetOrbit(b);

                        int J1min = std::abs(oa.j2-ob.j2)/2;
                        int J1max = (oa.j2+ob.j2)/2;

                        int J2min = std::max(std::abs(oi.j2-ok.j2), std::abs(oj.j2-ok.j2))/2;
                        int J2max = std::min(oi.j2+ok.j2, oj.j2+ok.j2)/2;

                        for(int J1 = J1min; J1<=J1max; ++J1)
                        {
                            for(int J2 = J2min; J2<=J2max; ++J2)
                            {
                                // if( J < std::abs(J1-J2)  or J > J1+J2) continue;
                                r_ij -= bDoubles(nstate, J1, J2, a, b, i ,k)*bDoubles(nstate, J1, J2, a, b, j ,k);
                            }
                        }
                    }//b
                }//a
            }//k
            rho_CISD(i,j) = 0.5*r_ij / ((2*J+1)*(oi.j2+1));
            rho_CISD(j,i) = 0.5*r_ij / ((2*J+1)*(oi.j2+1));
        }//j
    }//i
    return rho_CISD;
}

arma::mat CISD::GetScalarDensityTDA(int nstate)
{
    int Norbits = modelspace->norbits;
    arma::mat rho = arma::zeros(Norbits,Norbits);
    for (auto& i : modelspace->holes)  rho(i,i) = modelspace->GetOrbit(i).occ;

    //TDA corrections
    rho += TDAScalarDensityHH(nstate);
    rho += TDAScalarDensityPP(nstate);

    return rho;
}

arma::mat CISD::GetScalarDensity(int nstate)
{
    int Norbits = modelspace->norbits;
    arma::mat rho = arma::zeros(Norbits,Norbits);
    for (auto& i : modelspace->holes)  rho(i,i) = modelspace->GetOrbit(i).occ;

    //TDA corrections
    rho += TDAScalarDensityHH(nstate);
    rho += TDAScalarDensityPP(nstate);

    //CISD corrections
    rho += CISDScalarDensityHH(nstate);
    rho += CISDScalarDensityPP(nstate);

    return rho;
}

//Run some tests on the density 
void CISD::DensityTest(int nstate)
{
    std::cout <<"Calculating TDA density" <<std::endl;
    arma::mat rho_TDA = GetScalarDensityTDA(nstate);

    std::cout <<"Calculating CIS(D) density" <<std::endl;
    arma::mat rho_CISD = GetScalarDensity(nstate);


    double Z_TDA = 0.0;
    double N_TDA = 0.0;

    double Z_CISD = 0.0;
    double N_CISD = 0.0;

    int Norbits = modelspace->norbits;

    for(int i=0; i< Norbits; ++i)
    {
        Orbit& oi = modelspace->GetOrbit(i);
        if(oi.tz2 == -1)
        {
            Z_TDA += rho_TDA(i,i) * (oi.j2+1);
            Z_CISD += rho_CISD(i,i) * (oi.j2+1);
        }
        
        if(oi.tz2 == 1)
        {
            N_TDA += rho_TDA(i,i) * (oi.j2+1);
            N_CISD += rho_CISD(i,i) * (oi.j2+1);
        }
        
    }

    std::cout <<"TDA  " <<"Z=" <<Z_TDA <<" N=" <<N_TDA <<std::endl; 
    std::cout <<"CISD  " <<"Z=" <<Z_CISD <<" N=" <<N_CISD <<std::endl;

    std::cout <<"TDA:" <<std::endl;
    printDensity(rho_TDA);

    std::cout <<"CISD:" <<std::endl;
    printDensity(rho_CISD);
}

void CISD::printDensity(arma::mat& rho)
{
  for (auto& it : modelspace->OneBodyChannels)
  {
    arma::uvec orbvec(std::vector<index_t>(it.second.begin(),it.second.end()));
    arma::mat rho_ch = rho.submat(orbvec, orbvec);
    std::cout <<"l="<<it.first.at(0) <<"  j2=" <<it.first.at(1)  <<"  tz2=" <<it.first.at(2) <<std::endl;
    std::cout <<rho_ch <<std::endl;
  }
}

