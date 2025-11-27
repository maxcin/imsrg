
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

double CISD::bDoubles(int nstate, int J1, int J2, int a, int b, int i, int j)
{
    double wTDA = Energies(nstate);

    double ea = H.OneBody(a,a);
    double eb = H.OneBody(b,b);
    double ei = H.OneBody(i,i);
    double ej = H.OneBody(j,j);

    double denom = ea + eb - ei - ej - wTDA;

    double u_abij = uDoubles(nstate, J1, J2, a, b, i, j);
    return -u_abij / denom;
}

double CISD::uDoubles(int nstate, int J1, int J2, int a, int b, int i, int j)
{
    double u_p = 0.0;

    Orbit& oa = modelspace->GetOrbit(a);
    Orbit& ob = modelspace->GetOrbit(b);
    Orbit& oi = modelspace->GetOrbit(i);
    Orbit& oj = modelspace->GetOrbit(j);

    double ja = ( (double) oa.j2 )/ 2.0;
    double jb = ( (double) ob.j2 ) / 2.0;
    double ji = ( (double) oi.j2 ) / 2.0;
    double jj = ( (double) oj.j2 ) / 2.0;

    for(int c : modelspace->particles)
    {
        Orbit& oc = modelspace->GetOrbit(c);
        double jc = ( (double) oc.j2 )/ 2.0;
        double b_ci = bSingles(nstate, c, i);
        double b_cj = bSingles(nstate, c, j);

        if(std::abs(b_ci) > 1e-9)
        {
            double sixj = modelspace->GetSixJ(J1, J2, J, ji,jc,jj);
            double H_abcj = H.TwoBody.GetTBME_J(J1,a,b,c,j);
            int phase = modelspace->phase(J+J2+(oc.j2+oj.j2)/2);

            u_p += phase*H_abcj*b_ci*sixj;
        }

        if(std::abs(b_cj) > 1e-9)
        {
            double sixj = modelspace->GetSixJ(J1, J2, J, jj,jc,ji);
            double H_abci = H.TwoBody.GetTBME_J(J1,a,b,c,i);
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

        if(std::abs(b_bk) > 1e-9)
        {
            double sixj = modelspace->GetSixJ(J1, J2, J, jk,jb,ja);
            double H_kaij = H.TwoBody.GetTBME_J(J2,k,a,i,j);
            int phase = modelspace->phase(J+J1+J2);

            u_h += phase*H_kaij*b_bk*sixj;
        }

        if(std::abs(b_ak) > 1e-9)
        {
            double sixj = modelspace->GetSixJ(J1, J2, J, jk,ja,jb);
            double H_kbij = H.TwoBody.GetTBME_J(J2,k,b,i,j);
            int phase = modelspace->phase(J+J2+(oa.j2+ob.j2)/2);

            u_h -= phase*H_kbij*b_ak*sixj;
        }
        
    }

    double J1hat = sqrt(2*J1+1);
    double J2hat = sqrt(2*J2+1);

    double u_abij = Jhat*Jhat*J1hat*J2hat*(u_p+u_h);

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
                    if(oa.j2 == ob.j2)
                    {
                        double b_bi = bSingles(nstate,b,i);
                        if(std::abs(b_bi) < 1e-10) continue;

                        for(int J1 = J1min; J1 <= J1max; ++J1)
                        {
                            double denom = ec + ea - ej - ek;
                            double H_jkbc = H.TwoBody.GetTBME(J1,j,k,b,c);
                            double a_cajk = -H.TwoBody.GetTBME(J1,c,a,j,k)/ (denom);

                            int phase = modelspace->phase(J1+(oc.j2+oa.j2)/2);

                            double v = (2*J1)*phase*H_jkbc*b_bi*a_cajk ;
                            vai2 += v / (2.0*ja+1.0);
                        }
                    }
                    
                    //Term 2
                    if(oi.j2 == oj.j2)
                    {
                        double b_aj = bSingles(nstate,a,j);
                        if(std::abs(b_aj) < 1e-10) continue;

                        for(int J1 = J1min; J1 <= J1max; ++J1)
                        {
                            double denom = ec + eb - ei - ek;
                            double H_jkbc = H.TwoBody.GetTBME(J1,j,k,b,c);
                            double a_cbik = -H.TwoBody.GetTBME(J1,c,b,i,k)/ (denom);

                            int phase = modelspace->phase(J1+(oc.j2+oa.j2)/2);

                            double v = (2*J1)*phase*H_jkbc*b_aj*a_cbik ;
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
                            double H_jkbc = H.TwoBody.GetTBME(J1,j,k,b,c);
                            double sixJ1 = modelspace->GetSixJ(jb,jj,J,jk,jc,J1);
                            for(int J2 = J2min; J2 <= J2max; ++J2)
                            {
                                double sixJ2 = modelspace->GetSixJ(ja,ji,J,jk,jc,J2);
                                double a_acik = -H.TwoBody.GetTBME(J2,a,c,i,k) / denom;
                                int phase = modelspace->phase(J1+J2+(oi.j2+oj.j2)/2);
                                double v = (2*J1) * (2*J2) * phase * H_jkbc * b_bj * a_acik * sixJ1 * sixJ2 ;
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
            if(b>a) continue; //induces factor 2
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
                    if(j>i) continue; //induces factor 2
                    Orbit& oj = modelspace->GetOrbit(j);
                    double ej = H.OneBody(j,j);
                    int J2min = std::abs(oi.j2-oj.j2)/2;
                    int J2max = (oi.j2+oj.j2)/2;

                    double denom = ea+ eb- ei- ej - wTDA;

                    double wJsum = 0.0;

                    for(int J1 = J1min; J1 < J1max; ++J1)
                    {
                        for(int J2 = J2min; J2 < J2max; ++J2)
                        {
                            //if (J > J1+J2 or J <std::abs(J1-J2)) continue;
                            double u_J1J2_abij = uDoubles(nstate, J1, J2, a, b, i, j);
                            wJsum += u_J1J2_abij*u_J1J2_abij;

                            // if(u_J1J2_abij*u_J1J2_abij / denom > 1.0)
                            // {
                            //     std::cout <<"u(" <<J1 <<" " <<J2 <<"," <<a <<" " <<b <<" "  <<i <<" "  <<j <<")=" <<u_J1J2_abij <<"  denom = " << denom<<std::endl;
                            // }
                        }
                    }


                    wCISD -= wJsum / denom;
                }//j
            }//i
        }//b
    }//a

    std::cout <<"u correction " <<wCISD / ((double) (2*J+1) ) <<std::endl;

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

    return wCISD / ((double) (2*J+1) );
}

void CISD::Energy_test(int nstate)
{
    std::cout <<"State " <<nstate <<std::endl;

    double wCISD = E_CISD(nstate);
    std::cout <<"TDA " <<Energies(nstate) <<std::endl;
    std::cout <<"CISD " <<Energies(nstate)+wCISD<<std::endl;
    std::cout <<"CISD correction " <<wCISD<<std::endl;
        
}

arma::mat CISD::GetScalarDensity(int nstate)
{

    arma::mat a = arma::zeros(5);
    return a;
}