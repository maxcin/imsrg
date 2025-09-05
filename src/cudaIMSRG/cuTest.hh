#ifndef cuTest_ 
#define cuTest_ 1


#include "../ModelSpace.hh"
#include "../Operator.hh"

// #include "GPUCommutator.hh"

class cuModelSpace;
class cuOperator;
class cuCommutator;
class GPUCommutator;


//These are functions to test the functionality of the cuda implementation
//Using compute sanitizer for memory leaks is difficult as there are problems with bandicoot :( (surely I made no mistakes in the implementation...)

//Most important tests are the individual commutator expressions. Those test generate random Operators and compare
//to CPU implementation. Differences are expected to be of order 1e-15 (i.e. they should be purely due to machine precision)


namespace cuTest
{
    void TestGPU();
    void TestModelSpace(ModelSpace& modelspace);
    void TestOperator(Operator& Op);

    typedef void cpu_func(const Operator& X, const Operator& Y, Operator& Z);

    void Test_against_CPU(Operator& X, Operator& Y, cpu_func comm_cpu , GPUCommutator& gpuComm, std::string name);

    void TestCommutatorKernels(ModelSpace& modelspace);

    void TimeCommutator(ModelSpace& modelspace); //Do 20 Commutators and time them on GPU and CPU
    // void TestGPU(); //Tries to run simple kernel on the GPU to check everything work
    // cuModelSpace* moveGPUcuModelSpace(ModelSpace& modelspace); //allocates all memory on the gpu and finally returns the GPU pointer to the modelspace
    // cuOperator* moveGPUcuOperator(cuModelSpace* cuMS, Operator& Op); //allocates all memory on the gpu and finally returns the GPU pointer to the GPU
    // cuCommutator* createGPUcomm(cuModelSpace* cuMS);
    // void deallocatecuMS(cuModelSpace* cuMS); //deallocates memory on the gpu
    // void deallocatecuComm(cuCommutator* C);

    // void testModelSpace(ModelSpace& modelspace);

    // void testOperator(Operator& Op);
    // void testMP2(Operator& Op);
    // void testComm(Operator& X, Operator& Y, Operator& Z);

};

#endif