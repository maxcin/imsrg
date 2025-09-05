//Class to run commutator expressions on the GPU

//Assume always most simple case of scalar, parity/isospin conserving operators

#ifndef cudaCommutator 
#define cudaCommutator 1

#include "cuModelSpace.hh"
#include "cuOperator.hh"
#include "cuTwoBodyME.hh"
// #include <ModelSpace.hh>

//Don't ask...
//#include <cuco/static_map.cuh>
//#ifndef CUCO_BITWISE_DOUBLE
//#define CUCO_BITWISE_DOUBLE
//CUCO_DECLARE_BITWISE_COMPARABLE(double)  
//#endif
//using static_map = cuco::static_map<uint64_t, double, std::size_t, cuda::std::__4::thread_scope_device, cuda::std::__4::equal_to<uint64_t>, cuco::linear_probing<1, cuco::default_hash_function<uint64_t>>, cuco::cuda_allocator<cuco::pair<uint64_t, double>>, cuco::storage<1>>::ref_type<cuco::op::find_tag>;


#define COOT_DONT_USE_OPENCL
#define COOT_USE_CUDA
#define COOT_DEFAULT_BACKEND CUDA_BACKEND
#include <bandicoot>

class cuCommutator
{
    public:
    cuModelSpace* modelspace;
    //Here come temporary storage for two-body commutators
    cuTwoBodyME* Mpp;
    cuTwoBodyME* Mhh;

    Matrix* Z_cc; // Nch_cc matrices for pandy transformed operator (needed for inverse Pandya)

    //Storage management
    // __device__ void initialize_MppMhh();
    // __device__ void destroy_MppMhh();

    //This allocates all memory required for taking commutators
    __device__ void setupComm();
    __device__ void cleanComm();
};

//Kernels
__global__ void kernel_cuComm110ss(cuOperator* X, cuOperator* Y, cuOperator* Z);

__global__ void kernel_cuComm220ss(int ich, cuOperator* X, cuOperator* Y, cuOperator* Z);
__global__ void kernel_cuComm220ss_new(int ich, cuOperator* X, cuOperator* Y, cuOperator* Z);

__global__ void kernel_cuComm111ss(cuOperator* X, cuOperator* Y, cuOperator* Z);

__global__ void kernel_cuComm121ss(cuOperator* X, cuOperator* Y, cuOperator* Z);

__global__ void kernel_cuComm122ss(int ich, cuOperator* X, cuOperator* Y, cuOperator* Z);

__global__ void kernel_cu222add_pp_hhss(int ich, cuOperator* Z, cuCommutator* Comm);
__global__ void kernel_cu221add(cuOperator* Z, cuCommutator* Comm);
__global__ void kernel_cuConstructScalarMpp_Mhh(int ich, cuOperator* X, cuOperator* Y, cuCommutator* Comm);

//
__global__ void kernel_cuPandyaTransformSingleChannel(int ich_cc, cuOperator* X, cuOperator* Y, double* X_bar_t_cc, double* Y_bar_cc);

__global__ void kernel_cuAddInversePandya(int ich, cuOperator* Z, cuCommutator* Comm);


//misc functions
__global__ void kernel_Z_cc_mem_ptr(cuCommutator* Comm, int ich_cc, double* mem_ptr);
__global__ void kernel_setupComm(cuCommutator* C);
__global__ void kernel_cleanComm(cuCommutator* C);
void cuCommSetup(cuCommutator* C);
void cuCommClean(cuCommutator* C);




#endif