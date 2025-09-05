#ifndef cuMatrix 
#define cuMatrix 1

#include <stdio.h>

struct Matrix
{
    int ncols;
    int nrows;

    double* data;

    __device__ void allocate(int number_rows, int number_cols)
    { 
        ncols = number_cols;
        nrows =number_rows;
        cudaMalloc(&data, sizeof(double)*number_rows*number_cols);
    }

    __device__ void deallocate(){ cudaFree(data); }

    __device__ double& operator()(int i, int j){ return data[nrows*j+i];}

    __device__ void print()
    {
        for(int i = 0; i<nrows; ++i)
        {
            for(int j = 0; j<ncols; ++j)
            {
                printf("%f ", data[nrows*j+i]);
            }
            printf("\n");
        }
        printf("\n");
    }
};

#endif
