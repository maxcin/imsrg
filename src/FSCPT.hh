#ifndef FSCPT_h
#define FSCPT_h

#include "ModelSpace.hh"
#include "Operator.hh"
#include "Generator.hh"

//Implementation of Fock space canonical perturbation theory expressions up to third order

class FSCPT
{
    public:
        //Perturbation theory is defined for "single-reference" and "valence"
        std::string off_diagonal = "valence";

        arma::mat H0;
        Operator V;
        Operator Vod;
        Operator Vd;

        Operator G1;
        Operator G2;

        Generator generator;

        //second and third order corrections
        //Heff1 = Vod
        Operator Heff2;
        Operator Heff3;

        FSCPT(Operator& H); //H = normal ordered hamiltonian
        void FSCPT2();
        void FSCPT3();

        Operator GetOpOd(const Operator& Op); //Gets the off diagonal part of Op
        // void sr_diagonal(Operator& Op); //Single-reference diagonal
        // void vs_diagonal(Operator& Op); //valence-space diagonal

        Operator Delta(const Operator& Op); // apply denominator to the Delta
};

#endif