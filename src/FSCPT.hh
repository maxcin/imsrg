#ifndef FSCPT_h
#define FSCPT_h

#include "ModelSpace.hh"
#include "Operator.hh"
#include "Generator.hh"

//Implementation of Fock space canonical perturbation theory expressions up to third order

//Usage: Create instance by providing the full modelspace and the target modelspace
// Then run Calculate with the normal ordered Hamiltonian
// Definition of off_diagonal determines kind of correction (single reference / ensemble or valence based)

//Note that the single-reference works also via commutators for now
//In principle this is the same as the regular expressions but third order
//also includes fractional occupation numbers and works for non-HF basis

class FSCPT
{
    public:
        //Perturbation theory is defined for "single-reference", "valence"
        std::string off_diagonal = "";
        int order = 3; //order to which the correction should be done

        ModelSpace* modelspace_imsrg;

        arma::mat H0;
        Operator V;
        Operator Vod;
        Operator Vd;

        Operator G1;
        Operator G2;

        Operator G3; //Technically only needed in the atan case 

        Generator generator;

        //second and third order corrections
        //Heff1 = Vod
        Operator Heff1; //only used in atan case
        Operator Heff2;
        Operator Heff3;

        //the delta operators that encode order-by-order differences
        Operator Delta_Heff1; //This one can appear in atan expansion
        Operator Delta_Heff2;
        Operator Delta_Heff3;


        FSCPT(ModelSpace& ms_imsrg, std::string od); // ms_imsrg = target model space
        void FSCPT2();
        void FSCPT3();

        Operator Calculate(Operator& H); //this one applies a truncation and will return the changed operator
        void CalculateDHeff(Operator& H); //This one constructs just the corrections dHeff2 and dHeff3 to be used at a later time
        void CalculateAtanDHeff(Operator& H); //Using atan expansion

        Operator GetOpOd(const Operator& Op); //Gets the off diagonal part of Op
        // void sr_diagonal(Operator& Op); //Single-reference diagonal
        // void vs_diagonal(Operator& Op); //valence-space diagonal

        Operator Delta(const Operator& Op); // apply denominator to the Delta

        Operator atanDelta(const Operator& Op); //apply denominator plus atan
};

#endif
