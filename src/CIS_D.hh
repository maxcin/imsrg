

#ifndef CISD_h
#define CISD_h

#include "RPA.hh"
#include <armadillo>

class CISD : public RPA
{
    public:
        int J;
        int P = 0;
        int Tz = 0;

        double Jhat;

        //Constructor
        CISD(Operator& Op, int J, int P, int Tz);
        CISD(Operator& Op, int J);

        void RunTDA();

        arma::uvec TDAindex;// Global Index -> Local index to access TDA amplitudes
        void BuildTDAIndex(); //Helper function for building index map
        size_t GetTDAIndex(int a, int i);
        
        //Singles and doubles amplitude of perturbatively improve TDA state
        double bSingles(int nstate, int a, int i); 
        double bDoubles(int nstate, int J1, int J2, int a, int b, int i, int j); //(33) in notes
        
        //Storing in a TBME has lots of zeros because only pphh amplitudes are required but for now this is fine
        std::map<int, TwoBodyME> uDoublesCache; // nstate -> TBME containing amplitudes for that state
        void uPrecalculateDoubles(int nstate); // Precalculate amplitudes for nstate

        //Misc functions for energy correction 
        double uDoubles(int nstate, int J1, int J2, int a, int b, int i, int j);//(16) and (33) in notes .
        double uDoublesCached(int nstate, int J1, int J2, int a, int b , int i, int j);//Access cached amplitudes
        double vSingles(int nstate, int a, int i);//(39) in notes

        //What we are interested in
        double E_CISD(int nstate);

        void Energy_test(int nstate);
        arma::mat TDAScalarDensityPP(int nstate);
        arma::mat TDAScalarDensityHH(int nstate);
        
        arma::mat CISDScalarDensityPP(int nstate);
        arma::mat CISDScalarDensityHH(int nstate);

        arma::mat GetScalarDensityTDA(int nstate);
        arma::mat GetScalarDensity(int nstate);

        void DensityTest(int nstate);
        void printDensity(arma::mat& rho); //Helper function to print density by channel
        void printStateNAT(arma::mat& rho); //Helper function to print state specific NAT basis
        void printOccInBasis(arma::mat& rho, arma::mat& C_HF2NAT); //Helper function to print occupations in (arbitrary) NAT basis

};

#endif