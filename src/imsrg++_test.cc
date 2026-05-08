#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <omp.h>
#include "IMSRG.hh"
#include "version.hh"
#include "ReferenceImplementations.hh"


int main(int argc, char** argv)
{
    std::cout << "######  imsrg++ test timings: " << version::BuildVersion() << std::endl;

    int emax = 6;
    std::string reference = "Ca48";
    ModelSpace modelspace_imsrg(emax,reference);

    UnitTest tester(modelspace_imsrg);

    std::cout <<"Creating Operators..." <<std::endl;
    Operator H = tester.RandomOp(modelspace_imsrg, 0,0,0,2,1);
    Operator Omega = tester.RandomOp(modelspace_imsrg, 0,0,0,2,-1);
    Operator E2 = tester.RandomOp(modelspace_imsrg, 2,0,0,2,1);

    Operator Out_H = H;
    Out_H.Erase();

    Operator Out_E2 = E2;
    Out_E2.Erase();

    Commutator::verbose = true;

    int Ncomm = 10;
    std::cout <<"[Scalar , Scalar]" <<std::endl;
    for(int i = 0; i<Ncomm; ++i)
    {
        std::cout <<"\n\n\n=======================  " <<i <<"  =======================" <<std::endl;
        Out_H = Commutator::Commutator(Omega, H);
        H.PrintTimes();
    }

    std::cout <<"[Scalar , Tensor]" <<std::endl;
    for(int i = 0; i<Ncomm; ++i)
    {
        std::cout <<"\n\n\n=======================  " <<i <<"  =======================" <<std::endl;
        Out_E2 = Commutator::Commutator(Omega, E2);
        E2.PrintTimes();
    }


    return 0;
}

