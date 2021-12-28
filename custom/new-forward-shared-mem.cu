#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

#define TILEWIDTH 16

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

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int X_tile_width = TILEWIDTH + K-1;
    int W_grid = ceil(W/(TILEWIDTH*1.0));
    // __shared__ float X_shared[X_tile_width][X_tile_width];
    // __shared__ float K_shared[K][K];
    extern __shared__ float shmem[];
    // x_shared here is size of [X_tile_width][X_tile_width] and so pointer points from 0 - (X_tile_width*X_tile_width)
    float* X_shared = &shmem[0];
    // K-sharerd here points from (X_tile_width*X_tile_width) to the end of the sharerd mem region 
    float* K_shared = &shmem[X_tile_width * X_tile_width];

    int batchId = blockIdx.x;
    int m = blockIdx.y;
    int h0 = threadIdx.x;
    int w0 = threadIdx.y;
    int h_base = (blockIdx.z / W_grid) * TILEWIDTH; // vertical base out data index for the block
    int w_base = (blockIdx.z % W_grid) * TILEWIDTH; // horizontal base out data index for the block

    int h = h_base + h0;
    int w = w_base + w0;

    float Pvalue = 0.0f;
    int c, p, q, i, j;
    for (c = 0; c < C; c++) { //iterate through channels
        if (( h0 < K) && ( w0 < K)) {
            K_shared[h0*K + w0]= k4d(m, c, h0, w0); 
        }
        __syncthreads();
        //printf("All Threads have loaded in the kernel to sharerd mem \n");
        // load tile from globl mem X[n, c,...] into shared memory
        for (i = h; i < h_base + X_tile_width; i+=TILEWIDTH) {
            for (j = w; j < w_base + X_tile_width; j+=TILEWIDTH) {
                if (i < W_out + X_tile_width && j < H_out + X_tile_width) {
                    X_shared[(i-h_base)*X_tile_width + (j-w_base)] = x4d(batchId, c, i, j);
                }
            }
        }
        __syncthreads();
        for (p = 0; p < K; p++) {
            for (q = 0; q < K; q++) {
                Pvalue = Pvalue + (X_shared[(h+p) * X_tile_width + (w+q)]*K_shared[p * K + q]);
            }
        }
        __syncthreads();
    }
    if (h < H_out && w < W_out) {
        y4d(batchId, m, h, w) = Pvalue;
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

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // Allocate memory and copy over the relevant data structures to the GPU
    std::cout<<"Allocating Memory.. "<<std::endl;
    cudaMalloc((void**)&deviceInput,(H*W*C*B)*sizeof(float));
    cudaMalloc((void**)&deviceOutput,(H_out*W_out*M*B)*sizeof(float));
    cudaMalloc((void**)&deviceKernel,(K*K*M*C)*sizeof(float));
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
    cudaMemcpy(deviceKernel, &host_k[0], (K*K*M*C)*sizeof(float), cudaMemcpyHostToDevice);
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
    dim3 dimBlock(TILEWIDTH, TILEWIDTH, 1);
    dim3 dimGrid(B,M,Z);

    //@@ Launch the GPU kernel here
    std::cout<<"Launching Kernel "<<std::endl;
    size_t shmem_size = sizeof(float) * ((TILEWIDTH + K-1)*(TILEWIDTH + K-1) + K*K );
    conv_forward_kernel<<<dimGrid, dimBlock, shmem_size>>>(deviceOutput, deviceInput, deviceKernel, B, M, C, H, W, K);
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