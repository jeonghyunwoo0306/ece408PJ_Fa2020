#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

#define KERNEL_WIDTH 7 

__constant__ float Weights[4096];

// TODO: Remove every unnecessary calcaution for parameters (e.g., ceiling, modular opetioan, etc)
// TODO: Change namings & delete unnecessary variables in code
// Kernel fusion: unroll + Simple shared Matrix multiply 
__global__ void matrixMultiplyShared(float *B, float *Output,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns, const int W_out, const int W, const int H, const int C) {

    // JH: Comment out this since we are using const memory for weights now
    // __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    // TODO: Change shared memory to 1D array and optimize the access pattern
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

    int bx=blockIdx.x; 
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0;

    // TODO: Potential region using restrict & unrolling
    for (int q=0; q < (numAColumns-1)/TILE_WIDTH +1 ; q++){
    // JH: Comment out this since we are using const memory for weights now 
    // if((Row < numARows) && ((q*TILE_WIDTH+tx) < numAColumns))
    //     subTileA[ty][tx] = A[Row*numAColumns+(q*TILE_WIDTH+tx)];
    // else
    //     subTileA[ty][tx] = 0.0;

    // JH: Add required index for kernel fusion
    // JH: TODO: Substitue this calculation for macro (Currently bottleneck)
    int temp_row = q*TILE_WIDTH + ty;
    int input_c = temp_row / (KERNEL_WIDTH * KERNEL_WIDTH);
    int input_p = (temp_row % (KERNEL_WIDTH * KERNEL_WIDTH)) / KERNEL_WIDTH;
    int input_q = (temp_row % (KERNEL_WIDTH * KERNEL_WIDTH)) % KERNEL_WIDTH;
    int input_h = Col / W_out;
    int input_w = Col % W_out;

    // TODO: Currently uncoalesced global accesses (How we can reduce this)
    if((temp_row < numBRows) && (Col < numBColumns))
        subTileB[ty][tx] = B[blockIdx.z * (W * H * C) + input_c * (W * H) + (input_h + input_p) * W + (input_w + input_q)];
    else
        subTileB[ty][tx] = 0.0;

    __syncthreads();

    // TODO: Potential region using restrict & unrolling (Currently bottleneck)
    for (int k = 0; k < TILE_WIDTH; k++)
        Pvalue += Weights[Row*numAColumns + q*TILE_WIDTH+k] * subTileB[k][tx];
    __syncthreads();

    // TODO: Currently uncoalesced global accesses (How we can reduce this)
    if((Row < numCRows) && (Col < numCColumns))
        Output[blockIdx.z * numCColumns * numCRows + Row*numCColumns + (Col/W_out)*W_out + (Col%W_out)] = Pvalue;
    }
}

__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Declare Related variables
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // Declare relevant device pointers
    float *device_x;
    float *device_y;
    // float *device_k;

    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void **) &device_x, (H * W * C * B) * sizeof(float));
    cudaMalloc((void **) &device_y, (H_out * W_out * M * B) * sizeof(float));
    // JH: Temporary comment out to check benefit from Constant memory
    // cudaMalloc((void **) &device_k, (K * K * M * C) * sizeof(float));

    // Copy input & weights to Device
    cudaMemcpy(device_x, host_x, H * W * C * B * sizeof(float), cudaMemcpyHostToDevice);
    // JH: Temporary comment out to check benefit from Constant memory
    // cudaMemcpy(device_k, host_k, K * K * M * C * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(Weights, host_k, (K * K * M * C) * sizeof(float));

    // Set the kernel dimensions
    dim3 DimBlockMatmul(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 DimGridMatmul(ceil((W_out * H_out)/(1.0*TILE_WIDTH)), ceil(M/(1.0*TILE_WIDTH)), B);

    // Launch fused kernel here
    matrixMultiplyShared<<<DimGridMatmul, DimBlockMatmul>>>(device_x, device_y, M, (K * K * C), (K * K * C), (W_out * H_out), (M), (W_out*H_out), W_out, W, H, C);
    
    // Copy the output back to host
    cudaMemcpy(host_y, device_y, (H_out * W_out * M * B) * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Complete an copy result to host\n");    

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
    // cudaFree(device_k);
    
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
