#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

#define MASK_WIDTH 7
#define MAX_CHANNELS 4
#define TILEWIDTH 16
#define MAX_FEATURE_MAPS 16
#define SM TILEWIDTH+MASK_WIDTH-1

__constant__ float Weights[4096];

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - MASK_WIDTH + 1;
    const int W_out = W - MASK_WIDTH + 1;
    

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) Weights[(i3) * (C * MASK_WIDTH * MASK_WIDTH) + (i2) * (MASK_WIDTH * MASK_WIDTH) + (i1) * (MASK_WIDTH) + i0]


    // make the input shared memory only the size of input tile (TILEWIDTH+MASK_WIDTH-1)^2) * number of channels...
    __shared__ float sharedInput[MAX_CHANNELS*(SM)*(SM)];

    //__shared__ float sharedWeights[(MASK_WIDTH*MASK_WIDTH)];

    int w = threadIdx.x;
    int h = threadIdx.y;

    int batch = blockIdx.z;
    int input_x = w + blockIdx.x*TILEWIDTH;
    int input_y = h + blockIdx.y*TILEWIDTH;
    int input_z = threadIdx.z + batch;

    int c;
    for (c = 0; c < C; c++) {
        int index = (c*(SM)*(SM)) + (h*(SM)) + (w);
        if (input_z < B && input_y < H && input_x < W) {
            sharedInput[index] = x4d(input_z, c, input_y, input_x);
        } else {
            sharedInput[index] = 0.0;
        }
    }
    __syncthreads();
    if ((h < TILEWIDTH) && (w < TILEWIDTH)){ 
        for (int m = 0; m < M; m++){    
            float Pvalue = 0.0f;
            for (int c = 0; c < C; c++){ 
                // if (w < K && h < K) {
                //     int w_index = (h*(K)) + (w);
                //     sharedWeights[w_index] = k4d(m, c, h, w);
                // }   
                __syncthreads();
                // Loop through 2D convolution kernel/filter
                for (int ky = 0; ky < MASK_WIDTH; ky++){
                    for (int kx = 0; kx < MASK_WIDTH; kx++){
                        int index = (c*(SM)*(SM)) + ((h+ky)*(SM)) + (w+kx);
                        int w_index = (ky*MASK_WIDTH) + (kx);
                        Pvalue += k4d(m, c, ky, kx) * sharedInput[index];   // do convolution
                        //Pvalue += sharedWeights[w_index] * sharedInput[index];
                    }
                }
            }
            if ((input_x < W_out) && (input_y < H_out) && (input_z < B)){  // if thread is within output bounds
                y4d(input_z, m, input_y, input_x) = Pvalue;  // set convolution result
            }
        }
    }

    #undef y4d
    #undef x4d
    #undef k4d

}

__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Declare relevant device pointers
    float *deviceInput;
    float *deviceOutput;
    float *deviceKernel;    

    const int H_out = H - MASK_WIDTH + 1;
    const int W_out = W - MASK_WIDTH + 1;

    // Allocate memory and copy over the relevant data structures to the GPU
    std::cout<<"Allocating Memory.. "<<std::endl;
    cudaMalloc((void**)&deviceInput,(H*W*C*B)*sizeof(float));
    cudaMalloc((void**)&deviceOutput,(H_out*W_out*M*B)*sizeof(float));
    cudaMalloc((void**)&deviceKernel,(MASK_WIDTH*MASK_WIDTH*M*C)*sizeof(float));
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error [1]: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    // std::cout<<"Input Height:  "<< H <<std::endl;
    // std::cout<<"Input Width:  "<< W <<std::endl;
    std::cout<<"Copying memory from host to device.. "<<std::endl;
    cudaMemcpy(deviceInput, &host_x[0], (H*W*C*B)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceKernel, &host_k[0], (MASK_WIDTH*MASK_WIDTH*M*C)*sizeof(float), cudaMemcpyHostToDevice);
    // error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error [2]: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    // Set the kernel dimensions and call the kernel
    std::cout<<"Settting Kernel Dimensions "<<std::endl;
    int W_grid = ceil(W/(TILEWIDTH*1.0));
    int H_grid = ceil(H/(TILEWIDTH*1.0));
    int Z = W_grid*H_grid;
    //  launching as many threads as we need to load an input tile 
    // (as opposed to computing an output tile)
    //dim3 dimBlock(TILEWIDTH + K-1, TILEWIDTH+K-1, 1);

    cudaMemcpyToSymbol(Weights, host_k, (MASK_WIDTH * MASK_WIDTH * M * C) * sizeof(float));
    
    dim3 dimBlock(SM, SM, 1);
    //dim3 dimBlock(TILEWIDTH, TILEWIDTH, 1);
    dim3 dimGrid(ceil((1.0*W_out)/(1.0*TILEWIDTH)), ceil((1.0*H_out)/(1.0*TILEWIDTH)), B);
    //dim3 dimGrid(B,M,Z);

    //@@ Launch the GPU kernel here
    std::cout<<"Launching Kernel "<<std::endl;
    std::cout<<"B:      "<<B<<std::endl;
    std::cout<<"M:      "<<M<<std::endl;
    std::cout<<"C:      "<<C<<std::endl;
    std::cout<<"H:      "<<H<<std::endl;
    std::cout<<"W:      "<<W<<std::endl;
    std::cout<<"K:      "<<K<<std::endl;
    std::cout<<"Launching Kernel with block dim: "<< SM <<std::endl;
    // transitioning to useing output tile size..
   // size_t shmem_size = sizeof(float) * ( (TILEWIDTH + K-1)*(TILEWIDTH + K-1)+ K*K );
   //size_t shmem_size = sizeof(float) * (MASK_WIDTH*MASK_WIDTH);
    conv_forward_kernel<<<dimGrid, dimBlock, 0>>>(deviceOutput, deviceInput, deviceKernel, B, M, C, H, W, K);
    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error [3]: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    // cudaDeviceSynchronize();

    // Copy the output back to host
    std::cout<<"Copying Data back to Host"<<std::endl;
    cudaMemcpy(&host_y[0], deviceOutput, (H_out*W_out*M*B)*sizeof(float),cudaMemcpyDeviceToHost);

    // Free device memory
    std::cout<<"Freeing Memory "<<std::endl;
    cudaFree(deviceInput);
    cudaFree(deviceOutput);  
    cudaFree(deviceKernel);  
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}