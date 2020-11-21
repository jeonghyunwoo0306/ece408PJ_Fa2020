#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

// #define KERNEL_WIDTH 7 

// __constant__ float Weights[4096];

__global__ void unroll_input(float *x_unroll, const float *x, const int C, const int H, const int W, const int K, int base){
    


    const int W_out = W - K + 1;
    const int H_out = H - K + 1;
    
    int t = 1024 * blockIdx.x + threadIdx.x;

    const int W_unroll = H_out * W_out;

    int c, s, h_out, w_out, h_unroll, w_base, w_unroll;

    if(t < C * W_unroll){
        c = t / W_unroll;
        s = t % W_unroll;
        h_out = s / W_out;
        w_out = s % W_out;
        h_unroll = h_out * W_out + w_out;
        w_base = c * K * K;
        for ( int p =0; p < K; p++){
            for ( int q =0; q < K; q++){
                w_unroll = w_base + p*K + q;
                x_unroll[w_unroll * W_unroll + s] = x[base + c * (W * H) + (h_out + p) * W + w_out + q];
            }
        }
    }
}

__global__ void unroll_weights(float *k_unroll, float *k, const int M, const int C, const int K){
    int t = blockIdx.x * 1024 + threadIdx.x;

    if(t < (M*C)){
        int m = t / C;
        int c = t % C;
        int h_base = c * K * K;
        int unroll_width = C * K * K;

        for(int p=0; p < K; p++){
            for (int q=0; q < K; q++){
                int h_unroll = h_base + p * K + q;
                k_unroll[m * unroll_width + h_unroll] = k[m * K * K * C + c * K * K + p * K + q];
            }
        }
    }
}

// Simple shared Matmul kernel 
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns) {

    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

    int bx=blockIdx.x; 
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0;

    for (int q=0; q < (numAColumns-1)/TILE_WIDTH +1 ; q++){
    if((Row < numARows) && ((q*TILE_WIDTH+tx) < numAColumns))
        subTileA[ty][tx] = A[Row*numAColumns+(q*TILE_WIDTH+tx)];
    else
        subTileA[ty][tx] = 0.0;
    if(((q*TILE_WIDTH+ty) < numBRows) && (Col < numBColumns))
        subTileB[ty][tx] = B[(q*TILE_WIDTH+ty)*numBColumns+Col];
    else
        subTileB[ty][tx] = 0.0;

    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++)
        Pvalue += subTileA[ty][k] * subTileB[k][tx];
    __syncthreads();

    if((Row < numCRows) && (Col < numCColumns))
        C[Row*numCColumns+Col] = Pvalue;
    }
}

__global__ void reshpae_outputs(float *y, float *y_matmul, const int M, const int H, const int W, const int K, int base){
    int t = blockIdx.x * 1024 + threadIdx.x;

    const int W_out = W - K + 1;
    const int H_out = H - K + 1;

    if(t< W_out * H_out ){
        int w_in  = t % (W_out * H_out);
        int w_out = t % W_out;
        int h_out = t / W_out;
        
        for(int i=0; i <M; i++) {
            y[base + i * W_out * H_out + h_out * W_out + w_out] = y_matmul[i * W_out * H_out + w_in];
        }
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
    float *device_k;

    float *device_unroll_x;
    float *device_unroll_k;
    float *device_unroll_y;
	
    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void **) &device_x, (H * W * C * B) * sizeof(float));
    cudaMalloc((void **) &device_y, (H_out * W_out * M * B) * sizeof(float));
    cudaMalloc((void **) &device_k, (K * K * M * C) * sizeof(float));

    // Allocate memory for unrolled input & Weights
    cudaMalloc((void **) &device_unroll_x, (H_out * W_out * K * K * C) * sizeof(float));
    cudaMalloc((void **) &device_unroll_k, (K * K * C * M) * sizeof(float));
    cudaMalloc((void **) &device_unroll_y, (H_out * W_out * M) * sizeof(float));


    // Copy input & weights to Device
    cudaMemcpy(device_x, host_x, H * W * C * B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_k, host_k, K * K * M * C * sizeof(float), cudaMemcpyHostToDevice);

    // cudaMemcpyToSymbol(Weights, host_k, (K * K * M * C) * sizeof(float));


    // Set the kernel dimensions and call the kernel TODO: Change this
    dim3 DimBlockUnrollX(1024, 1, 1);
    dim3 DimGridUnrollX(ceil((C * H_out * W_out)/(1024.0)), 1, 1); // 1024: MAX_NUM_THREADS in per block

    dim3 DimBlockUnrollK(1024, 1, 1);
    dim3 DimGridUnrollK(1, 1, 1);

    dim3 DimBlockMatmul(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 DimGridMatmul(ceil((W_out * H_out)/(1.0*TILE_WIDTH)), ceil(M/(1.0*TILE_WIDTH)), 1);

    dim3 DimBlockReshape(1024, 1, 1);
    dim3 DimGridReshape(ceil((W_out * H_out)/(1024.0)), 1, 1);
    //Launch weigths unroll kernel here
    unroll_weights<<<DimGridUnrollK, DimBlockUnrollK>>> (device_unroll_k, device_k, M, C, K);
    cudaDeviceSynchronize(); 
    for(int i = 0; i < B; i++) {
        // Declare base index for input & Output
        int baseX = i * W * H * C;

        int baseY = i * W_out * H_out * M;

        // Launch input unroll kernel here
        unroll_input<<<DimGridUnrollX, DimBlockUnrollX>>>(device_unroll_x, device_x, C, H, W, K, baseX);
        // Do tiled matrix multiplication here
        matrixMultiplyShared<<<DimGridMatmul, DimBlockMatmul>>>(device_unroll_k, device_unroll_x, device_unroll_y, M, (K * K * C), (K * K * C), (W_out * H_out), (M), (W_out*H_out));
        // Launch kernel which reshape output from Matmul to orignal shape
        reshpae_outputs<<<DimGridReshape, DimBlockReshape>>>(device_y, device_unroll_y, M, H, W, K, baseY);
    }

    // Copy the output back to host
    cudaMemcpy(host_y, device_y, (H_out * W_out * M * B) * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Complete an copy result to host\n");    

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_k);
    cudaFree(device_unroll_x);
    cudaFree(device_unroll_k);
    cudaFree(device_unroll_y);

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
