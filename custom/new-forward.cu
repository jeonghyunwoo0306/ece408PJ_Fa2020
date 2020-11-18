#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 32

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

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil(W_out/(1.0*TILE_WIDTH));

    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;

    float sum = 0.0;

    for (int c = 0; c < C ; c++) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q ++) {
                sum += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
            }
        }
    }
   if((h < H_out) && (w < W_out))
        y4d(b, m, h, w) = sum;
    

#undef y4d
#undef x4d
#undef k4d
}

__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // TODO: Erase this after check Input / Kernel Output size
    // printf("Input H: %d, W: %d, Kernel Width: %d, # of Output %d, # of Input: %d\n", H, W, K, M, C);

    // get_device_properties();

    // Declare Related variables
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int W_grid = ceil(W_out/(1.0*TILE_WIDTH));
    const int H_grid = ceil(H_out/(1.0*TILE_WIDTH));
    const int Z_grid = W_grid * H_grid;

    // Declare relevant device pointers
    float *device_x;
    float *device_y;
    float *device_k;

    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void **) &device_x, (H * W * C * B) * sizeof(float));
    cudaMalloc((void **) &device_y, (H_out * W_out * M * B) * sizeof(float));
    cudaMalloc((void **) &device_k, (K * K * M * C) * sizeof(float));

    cudaMemcpy(device_x, host_x, H * W * C * B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_k, host_k, K * K * M * C * sizeof(float), cudaMemcpyHostToDevice);

    // Set the kernel dimensions and call the kernel
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z_grid);
    conv_forward_kernel<<<gridDim, blockDim>>>(device_y, device_x, device_k, B, M, C, H, W, K);

    // Copy the output back to host
    cudaMemcpy(host_y, device_y, (H_out * W_out * M * B) * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_k);
    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
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
