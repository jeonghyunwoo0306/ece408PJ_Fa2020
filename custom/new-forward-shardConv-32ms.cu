#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

#define MASK_WIDTH 7

// For single Kernel
// #define TILEWIDTH 17

// #define X_tile_width 23
// For multiple Kernels
#define TILEWIDTH1 32 //TODO: Choose best width for layer1
#define TILEWIDTH2 18 //TODO: Choose best width for layer2
#define X_tile_width1 38 //TODO: Fill this based on TILEWIDTH1
#define X_tile_width2 24 //TODO: Fill this based on TILEWIDTH2

__constant__ float Weights[4096];

__global__ void conv_forward_kernel1(float *y, const float* __restrict__ x)
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
    #define H 86
    #define W 86
    #define M 4
    #define C 1
    #define H_out 80
    #define W_out 80
    #define W_grid 3  //TODO: Need to add this: ceil(W_out/TILEWIDTH1)
    #define h_base ((blockIdx.z / W_grid) * TILEWIDTH1)
    #define w_base ((blockIdx.z % W_grid) * TILEWIDTH1) 
    #define h (h_base+threadIdx.y)  
    #define w (w_base+threadIdx.x)      
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) Weights[(i3) * (C * MASK_WIDTH * MASK_WIDTH) + (i2) * (MASK_WIDTH * MASK_WIDTH) + (i1) * (MASK_WIDTH) + i0]


    // const int H_out = H - MASK_WIDTH + 1;
    // const int W_out = W - MASK_WIDTH + 1;

    

    // int X_tile_width = TILEWIDTH + MASK_WIDTH-1;
    // int W_grid = ceil(W/(TILEWIDTH*1.0));
    __shared__ float X_shared[X_tile_width1*X_tile_width1];

    int batchId = blockIdx.x;
    // int m = blockIdx.y;
    int h0 = threadIdx.y;
    int w0 = threadIdx.x;
    // int h_base = (blockIdx.z / W_grid) * TILEWIDTH; // vertical base out data index for the block
    // int w_base = (blockIdx.z % W_grid) * TILEWIDTH; // horizontal base out data index for the block

    // int h = h_base + h0;
    // int w = w_base + w0;

    int p, i, j;

    // JH: We don't need for loop for layer1 kernel since number of channel is one

    // load tile from globl mem X[n, c,...] into shared memory
    // JH: Original loading from global to shared memory
    // for (i = h; i < h_base + X_tile_width; i+=TILEWIDTH) {
    //     for (j = w; j < w_base + X_tile_width; j+=TILEWIDTH) {
    //         if (i < H && j < W) {
    //             X_shared[(i-h_base)*X_tile_width + (j-w_base)] = x4d(batchId, c, i, j);
    //         }
    //     }
    // }

    // Added by JH: To unroll for loops for shared memory loading
    if(h < H && w < W){
        X_shared[h0*X_tile_width1 + w0] = x4d(batchId, 0, h, w);
    }
    
    if((w+TILEWIDTH1) < (w_base + X_tile_width1)){
        if(h < H && ((w+TILEWIDTH1) < W))
            X_shared[h0*X_tile_width1 + (w0 + TILEWIDTH1)] = x4d(batchId, 0, h, w+TILEWIDTH1);
    }

    if((h+TILEWIDTH1) < (h_base + X_tile_width1)){
        if((h+TILEWIDTH1) < H && w < W)
            X_shared[(h0+TILEWIDTH1)*X_tile_width1 + w0] = x4d(batchId, 0, h+TILEWIDTH1, w);
    }

    if(((w+TILEWIDTH1) < (w_base + X_tile_width1)) && ((h+TILEWIDTH1) < (h_base + X_tile_width1))){
        if((h+TILEWIDTH1) < H && (w+TILEWIDTH1) < W)
            X_shared[(h0+TILEWIDTH1)*X_tile_width1 + w0 + TILEWIDTH1] = x4d(batchId, 0, h+TILEWIDTH1, w+TILEWIDTH1);
    }

    __syncthreads();
    
    for(int m=0; m<M; m++){
        float Pvalue = 0.0f;

        #pragma unroll
        for (p = 0; p < MASK_WIDTH; p++) {
            Pvalue += X_shared[(h0+p) * X_tile_width1 + (w0+0)] * Weights[m * 49 + p * 7 + 0];
            Pvalue += X_shared[(h0+p) * X_tile_width1 + (w0+1)] * Weights[m * 49 + p * 7 + 1];
            Pvalue += X_shared[(h0+p) * X_tile_width1 + (w0+2)] * Weights[m * 49 + p * 7 + 2];
            Pvalue += X_shared[(h0+p) * X_tile_width1 + (w0+3)] * Weights[m * 49 + p * 7 + 3];
            Pvalue += X_shared[(h0+p) * X_tile_width1 + (w0+4)] * Weights[m * 49 + p * 7 + 4];
            Pvalue += X_shared[(h0+p) * X_tile_width1 + (w0+5)] * Weights[m * 49 + p * 7 + 5];
            Pvalue += X_shared[(h0+p) * X_tile_width1 + (w0+6)] * Weights[m * 49 + p * 7 + 6];            
        }
        // Pvalue += X_shared[(h0+0) * X_tile_width1 + (w0+0)] * Weights[m * C * 49 + c * 49 + 0 * 7 + 0];
        // Pvalue += X_shared[(h0+0) * X_tile_width1 + (w0+1)] * Weights[m * C * 49 + c * 49 + 0 * 7 + 1];
        // Pvalue += X_shared[(h0+0) * X_tile_width1 + (w0+2)] * Weights[m * C * 49 + c * 49 + 0 * 7 + 2];
        // Pvalue += X_shared[(h0+0) * X_tile_width1 + (w0+3)] * Weights[m * C * 49 + c * 49 + 0 * 7 + 3];
        // Pvalue += X_shared[(h0+0) * X_tile_width1 + (w0+4)] * Weights[m * C * 49 + c * 49 + 0 * 7 + 4];
        // Pvalue += X_shared[(h0+0) * X_tile_width1 + (w0+5)] * Weights[m * C * 49 + c * 49 + 0 * 7 + 5];
        // Pvalue += X_shared[(h0+0) * X_tile_width1 + (w0+6)] * Weights[m * C * 49 + c * 49 + 0 * 7 + 6];

        // Pvalue += X_shared[(h0+1) * X_tile_width1 + (w0+0)] * Weights[m * C * 49 + c * 49 + 1 * 7 + 0];
        // Pvalue += X_shared[(h0+1) * X_tile_width1 + (w0+1)] * Weights[m * C * 49 + c * 49 + 1 * 7 + 1];
        // Pvalue += X_shared[(h0+1) * X_tile_width1 + (w0+2)] * Weights[m * C * 49 + c * 49 + 1 * 7 + 2];
        // Pvalue += X_shared[(h0+1) * X_tile_width1 + (w0+3)] * Weights[m * C * 49 + c * 49 + 1 * 7 + 3];
        // Pvalue += X_shared[(h0+1) * X_tile_width1 + (w0+4)] * Weights[m * C * 49 + c * 49 + 1 * 7 + 4];
        // Pvalue += X_shared[(h0+1) * X_tile_width1 + (w0+5)] * Weights[m * C * 49 + c * 49 + 1 * 7 + 5];
        // Pvalue += X_shared[(h0+1) * X_tile_width1 + (w0+6)] * Weights[m * C * 49 + c * 49 + 1 * 7 + 6];

        // Pvalue += X_shared[(h0+2) * X_tile_width1 + (w0+0)] * Weights[m * C * 49 + c * 49 + 2 * 7 + 0];
        // Pvalue += X_shared[(h0+2) * X_tile_width1 + (w0+1)] * Weights[m * C * 49 + c * 49 + 2 * 7 + 1];
        // Pvalue += X_shared[(h0+2) * X_tile_width1 + (w0+2)] * Weights[m * C * 49 + c * 49 + 2 * 7 + 2];
        // Pvalue += X_shared[(h0+2) * X_tile_width1 + (w0+3)] * Weights[m * C * 49 + c * 49 + 2 * 7 + 3];
        // Pvalue += X_shared[(h0+2) * X_tile_width1 + (w0+4)] * Weights[m * C * 49 + c * 49 + 2 * 7 + 4];
        // Pvalue += X_shared[(h0+2) * X_tile_width1 + (w0+5)] * Weights[m * C * 49 + c * 49 + 2 * 7 + 5];
        // Pvalue += X_shared[(h0+2) * X_tile_width1 + (w0+6)] * Weights[m * C * 49 + c * 49 + 2 * 7 + 6];

        // Pvalue += X_shared[(h0+3) * X_tile_width1 + (w0+0)] * Weights[m * C * 49 + c * 49 + 3 * 7 + 0];
        // Pvalue += X_shared[(h0+3) * X_tile_width1 + (w0+1)] * Weights[m * C * 49 + c * 49 + 3 * 7 + 1];
        // Pvalue += X_shared[(h0+3) * X_tile_width1 + (w0+2)] * Weights[m * C * 49 + c * 49 + 3 * 7 + 2];
        // Pvalue += X_shared[(h0+3) * X_tile_width1 + (w0+3)] * Weights[m * C * 49 + c * 49 + 3 * 7 + 3];
        // Pvalue += X_shared[(h0+3) * X_tile_width1 + (w0+4)] * Weights[m * C * 49 + c * 49 + 3 * 7 + 4];
        // Pvalue += X_shared[(h0+3) * X_tile_width1 + (w0+5)] * Weights[m * C * 49 + c * 49 + 3 * 7 + 5];
        // Pvalue += X_shared[(h0+3) * X_tile_width1 + (w0+6)] * Weights[m * C * 49 + c * 49 + 3 * 7 + 6];

        // Pvalue += X_shared[(h0+4) * X_tile_width1 + (w0+0)] * Weights[m * C * 49 + c * 49 + 4 * 7 + 0];
        // Pvalue += X_shared[(h0+4) * X_tile_width1 + (w0+1)] * Weights[m * C * 49 + c * 49 + 4 * 7 + 1];
        // Pvalue += X_shared[(h0+4) * X_tile_width1 + (w0+2)] * Weights[m * C * 49 + c * 49 + 4 * 7 + 2];
        // Pvalue += X_shared[(h0+4) * X_tile_width1 + (w0+3)] * Weights[m * C * 49 + c * 49 + 4 * 7 + 3];
        // Pvalue += X_shared[(h0+4) * X_tile_width1 + (w0+4)] * Weights[m * C * 49 + c * 49 + 4 * 7 + 4];
        // Pvalue += X_shared[(h0+4) * X_tile_width1 + (w0+5)] * Weights[m * C * 49 + c * 49 + 4 * 7 + 5];
        // Pvalue += X_shared[(h0+4) * X_tile_width1 + (w0+6)] * Weights[m * C * 49 + c * 49 + 4 * 7 + 6];

        // Pvalue += X_shared[(h0+5) * X_tile_width1 + (w0+0)] * Weights[m * C * 49 + c * 49 + 5 * 7 + 0];
        // Pvalue += X_shared[(h0+5) * X_tile_width1 + (w0+1)] * Weights[m * C * 49 + c * 49 + 5 * 7 + 1];
        // Pvalue += X_shared[(h0+5) * X_tile_width1 + (w0+2)] * Weights[m * C * 49 + c * 49 + 5 * 7 + 2];
        // Pvalue += X_shared[(h0+5) * X_tile_width1 + (w0+3)] * Weights[m * C * 49 + c * 49 + 5 * 7 + 3];
        // Pvalue += X_shared[(h0+5) * X_tile_width1 + (w0+4)] * Weights[m * C * 49 + c * 49 + 5 * 7 + 4];
        // Pvalue += X_shared[(h0+5) * X_tile_width1 + (w0+5)] * Weights[m * C * 49 + c * 49 + 5 * 7 + 5];
        // Pvalue += X_shared[(h0+5) * X_tile_width1 + (w0+6)] * Weights[m * C * 49 + c * 49 + 5 * 7 + 6];

        // Pvalue += X_shared[(h0+6) * X_tile_width1 + (w0+0)] * Weights[m * C * 49 + c * 49 + 6 * 7 + 0];
        // Pvalue += X_shared[(h0+6) * X_tile_width1 + (w0+1)] * Weights[m * C * 49 + c * 49 + 6 * 7 + 1];
        // Pvalue += X_shared[(h0+6) * X_tile_width1 + (w0+2)] * Weights[m * C * 49 + c * 49 + 6 * 7 + 2];
        // Pvalue += X_shared[(h0+6) * X_tile_width1 + (w0+3)] * Weights[m * C * 49 + c * 49 + 6 * 7 + 3];
        // Pvalue += X_shared[(h0+6) * X_tile_width1 + (w0+4)] * Weights[m * C * 49 + c * 49 + 6 * 7 + 4];
        // Pvalue += X_shared[(h0+6) * X_tile_width1 + (w0+5)] * Weights[m * C * 49 + c * 49 + 6 * 7 + 5];
        // Pvalue += X_shared[(h0+6) * X_tile_width1 + (w0+6)] * Weights[m * C * 49 + c * 49 + 6 * 7 + 6];
        

        __syncthreads();
        
        if (h < H_out && w < W_out) {
            y4d(batchId, m, h, w) = Pvalue;
        }
    }
    #undef H 
    #undef W 
    #undef M 
    #undef C 
    #undef H_out 
    #undef W_out 
    #undef W_grid 
    #undef h_base
    #undef w_base
    #undef h   
    #undef w  
    #undef y4d
    #undef x4d
    #undef k4d

}

// Kernel which unroll Channels with #pragma keyword (Rely on Compiler)
__global__ void conv_forward_kernel2(float *y, const float* __restrict__ x)
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
    #define H 40
    #define W 40
    #define M 16
    #define C 4
    #define H_out 34
    #define W_out 34
    #define W_grid 2  //TODO: Need to add this: ceil(W_out/TILEWIDTH2)
    #define h_base ((blockIdx.z / W_grid) * TILEWIDTH2)
    #define w_base ((blockIdx.z % W_grid) * TILEWIDTH2) 
    #define h (h_base+threadIdx.y)  
    #define w (w_base+threadIdx.x)      
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) Weights[(i3) * (C * MASK_WIDTH * MASK_WIDTH) + (i2) * (MASK_WIDTH * MASK_WIDTH) + (i1) * (MASK_WIDTH) + i0]

    // const int H_out = H - MASK_WIDTH + 1;
    // const int W_out = W - MASK_WIDTH + 1;

    // int X_tile_width = TILEWIDTH + MASK_WIDTH-1;
    // int W_grid = ceil(W_out/(TILEWIDTH*1.0));
    __shared__ float X_shared[X_tile_width2*X_tile_width2];

    int batchId = blockIdx.x;
    // int m = blockIdx.y;
    int h0 = threadIdx.y;
    int w0 = threadIdx.x;
    // int h_base = (blockIdx.z / W_grid) * TILEWIDTH; // vertical base out data index for the block
    // int w_base = (blockIdx.z % W_grid) * TILEWIDTH; // horizontal base out data index for the block

    // int h = h_base + h0;
    // int w = w_base + w0;

    int c, p, q, i, j;
    #pragma unroll
    for(int m=0; m<M; m++){
        float Pvalue = 0.0f;   
        #pragma unroll
        for (c = 0; c < C; c++) { //iterate through channels
            // load tile from globl mem X[n, c,...] into shared memory
            // Original loading from global to shared memory
            // for (i = h; i < h_base + X_tile_width2; i+=TILEWIDTH2) {
            //     for (j = w; j < w_base + X_tile_width2; j+=TILEWIDTH2) {
            //         if (i < H && j < W) {
            //             X_shared[(i-h_base)*X_tile_width2 + (j-w_base)] = x4d(batchId, c, i, j);
            //         }
            //     }
            // }

            // Added by JH: To unroll for loops for shared memory loading
            if(h < H && w < W){
                X_shared[h0*X_tile_width2 + w0] = x4d(batchId, c, h, w);
            }
            
            if((w+TILEWIDTH2) < (w_base + X_tile_width2)){
                if(h < H && ((w+TILEWIDTH2) < W))
                    X_shared[h0*X_tile_width2 + (w0 + TILEWIDTH2)] = x4d(batchId, c, h, w+TILEWIDTH2);
            }

            if((h+TILEWIDTH2) < (h_base + X_tile_width2)){
                if((h+TILEWIDTH2) < H && w < W)
                    X_shared[(h0+TILEWIDTH2)*X_tile_width2 + w0] = x4d(batchId, c, h+TILEWIDTH2, w);
            }

            if(((w+TILEWIDTH2) < (w_base + X_tile_width2)) && ((h+TILEWIDTH2) < (h_base + X_tile_width2))){
                if((h+TILEWIDTH2) < H && (w+TILEWIDTH2) < W)
                    X_shared[(h0+TILEWIDTH2)*X_tile_width2 + w0 + TILEWIDTH2] = x4d(batchId, c, h+TILEWIDTH2, w+TILEWIDTH2);
            }

            __syncthreads();

        
            // #pragma unroll
            for (p = 0; p < MASK_WIDTH; p++) {
                // #pragma unroll
                for (q = 0; q < MASK_WIDTH; q++) {
                    Pvalue += (X_shared[(h0+p) * X_tile_width2 + (w0+q)]*k4d(m,c,p,q));
                }
            }

            // Added by JH: Forced unrolling for Pvalue calculation
            // Pvalue += X_shared[(h0+0) * X_tile_width2 + (w0+0)] * Weights[m * C * 49 + c * 49 + 0 * 7 + 0];
            // Pvalue += X_shared[(h0+0) * X_tile_width2 + (w0+1)] * Weights[m * C * 49 + c * 49 + 0 * 7 + 1];
            // Pvalue += X_shared[(h0+0) * X_tile_width2 + (w0+2)] * Weights[m * C * 49 + c * 49 + 0 * 7 + 2];
            // Pvalue += X_shared[(h0+0) * X_tile_width2 + (w0+3)] * Weights[m * C * 49 + c * 49 + 0 * 7 + 3];
            // Pvalue += X_shared[(h0+0) * X_tile_width2 + (w0+4)] * Weights[m * C * 49 + c * 49 + 0 * 7 + 4];
            // Pvalue += X_shared[(h0+0) * X_tile_width2 + (w0+5)] * Weights[m * C * 49 + c * 49 + 0 * 7 + 5];
            // Pvalue += X_shared[(h0+0) * X_tile_width2 + (w0+6)] * Weights[m * C * 49 + c * 49 + 0 * 7 + 6];

            // Pvalue += X_shared[(h0+1) * X_tile_width2 + (w0+0)] * Weights[m * C * 49 + c * 49 + 1 * 7 + 0];
            // Pvalue += X_shared[(h0+1) * X_tile_width2 + (w0+1)] * Weights[m * C * 49 + c * 49 + 1 * 7 + 1];
            // Pvalue += X_shared[(h0+1) * X_tile_width2 + (w0+2)] * Weights[m * C * 49 + c * 49 + 1 * 7 + 2];
            // Pvalue += X_shared[(h0+1) * X_tile_width2 + (w0+3)] * Weights[m * C * 49 + c * 49 + 1 * 7 + 3];
            // Pvalue += X_shared[(h0+1) * X_tile_width2 + (w0+4)] * Weights[m * C * 49 + c * 49 + 1 * 7 + 4];
            // Pvalue += X_shared[(h0+1) * X_tile_width2 + (w0+5)] * Weights[m * C * 49 + c * 49 + 1 * 7 + 5];
            // Pvalue += X_shared[(h0+1) * X_tile_width2 + (w0+6)] * Weights[m * C * 49 + c * 49 + 1 * 7 + 6];

            // Pvalue += X_shared[(h0+2) * X_tile_width2 + (w0+0)] * Weights[m * C * 49 + c * 49 + 2 * 7 + 0];
            // Pvalue += X_shared[(h0+2) * X_tile_width2 + (w0+1)] * Weights[m * C * 49 + c * 49 + 2 * 7 + 1];
            // Pvalue += X_shared[(h0+2) * X_tile_width2 + (w0+2)] * Weights[m * C * 49 + c * 49 + 2 * 7 + 2];
            // Pvalue += X_shared[(h0+2) * X_tile_width2 + (w0+3)] * Weights[m * C * 49 + c * 49 + 2 * 7 + 3];
            // Pvalue += X_shared[(h0+2) * X_tile_width2 + (w0+4)] * Weights[m * C * 49 + c * 49 + 2 * 7 + 4];
            // Pvalue += X_shared[(h0+2) * X_tile_width2 + (w0+5)] * Weights[m * C * 49 + c * 49 + 2 * 7 + 5];
            // Pvalue += X_shared[(h0+2) * X_tile_width2 + (w0+6)] * Weights[m * C * 49 + c * 49 + 2 * 7 + 6];

            // Pvalue += X_shared[(h0+3) * X_tile_width2 + (w0+0)] * Weights[m * C * 49 + c * 49 + 3 * 7 + 0];
            // Pvalue += X_shared[(h0+3) * X_tile_width2 + (w0+1)] * Weights[m * C * 49 + c * 49 + 3 * 7 + 1];
            // Pvalue += X_shared[(h0+3) * X_tile_width2 + (w0+2)] * Weights[m * C * 49 + c * 49 + 3 * 7 + 2];
            // Pvalue += X_shared[(h0+3) * X_tile_width2 + (w0+3)] * Weights[m * C * 49 + c * 49 + 3 * 7 + 3];
            // Pvalue += X_shared[(h0+3) * X_tile_width2 + (w0+4)] * Weights[m * C * 49 + c * 49 + 3 * 7 + 4];
            // Pvalue += X_shared[(h0+3) * X_tile_width2 + (w0+5)] * Weights[m * C * 49 + c * 49 + 3 * 7 + 5];
            // Pvalue += X_shared[(h0+3) * X_tile_width2 + (w0+6)] * Weights[m * C * 49 + c * 49 + 3 * 7 + 6];

            // Pvalue += X_shared[(h0+4) * X_tile_width2 + (w0+0)] * Weights[m * C * 49 + c * 49 + 4 * 7 + 0];
            // Pvalue += X_shared[(h0+4) * X_tile_width2 + (w0+1)] * Weights[m * C * 49 + c * 49 + 4 * 7 + 1];
            // Pvalue += X_shared[(h0+4) * X_tile_width2 + (w0+2)] * Weights[m * C * 49 + c * 49 + 4 * 7 + 2];
            // Pvalue += X_shared[(h0+4) * X_tile_width2 + (w0+3)] * Weights[m * C * 49 + c * 49 + 4 * 7 + 3];
            // Pvalue += X_shared[(h0+4) * X_tile_width2 + (w0+4)] * Weights[m * C * 49 + c * 49 + 4 * 7 + 4];
            // Pvalue += X_shared[(h0+4) * X_tile_width2 + (w0+5)] * Weights[m * C * 49 + c * 49 + 4 * 7 + 5];
            // Pvalue += X_shared[(h0+4) * X_tile_width2 + (w0+6)] * Weights[m * C * 49 + c * 49 + 4 * 7 + 6];

            // Pvalue += X_shared[(h0+5) * X_tile_width2 + (w0+0)] * Weights[m * C * 49 + c * 49 + 5 * 7 + 0];
            // Pvalue += X_shared[(h0+5) * X_tile_width2 + (w0+1)] * Weights[m * C * 49 + c * 49 + 5 * 7 + 1];
            // Pvalue += X_shared[(h0+5) * X_tile_width2 + (w0+2)] * Weights[m * C * 49 + c * 49 + 5 * 7 + 2];
            // Pvalue += X_shared[(h0+5) * X_tile_width2 + (w0+3)] * Weights[m * C * 49 + c * 49 + 5 * 7 + 3];
            // Pvalue += X_shared[(h0+5) * X_tile_width2 + (w0+4)] * Weights[m * C * 49 + c * 49 + 5 * 7 + 4];
            // Pvalue += X_shared[(h0+5) * X_tile_width2 + (w0+5)] * Weights[m * C * 49 + c * 49 + 5 * 7 + 5];
            // Pvalue += X_shared[(h0+5) * X_tile_width2 + (w0+6)] * Weights[m * C * 49 + c * 49 + 5 * 7 + 6];

            // Pvalue += X_shared[(h0+6) * X_tile_width2 + (w0+0)] * Weights[m * C * 49 + c * 49 + 6 * 7 + 0];
            // Pvalue += X_shared[(h0+6) * X_tile_width2 + (w0+1)] * Weights[m * C * 49 + c * 49 + 6 * 7 + 1];
            // Pvalue += X_shared[(h0+6) * X_tile_width2 + (w0+2)] * Weights[m * C * 49 + c * 49 + 6 * 7 + 2];
            // Pvalue += X_shared[(h0+6) * X_tile_width2 + (w0+3)] * Weights[m * C * 49 + c * 49 + 6 * 7 + 3];
            // Pvalue += X_shared[(h0+6) * X_tile_width2 + (w0+4)] * Weights[m * C * 49 + c * 49 + 6 * 7 + 4];
            // Pvalue += X_shared[(h0+6) * X_tile_width2 + (w0+5)] * Weights[m * C * 49 + c * 49 + 6 * 7 + 5];
            // Pvalue += X_shared[(h0+6) * X_tile_width2 + (w0+6)] * Weights[m * C * 49 + c * 49 + 6 * 7 + 6];
            

            __syncthreads();
            if (h < H_out && w < W_out) {
                y4d(batchId, m, h, w) = Pvalue;
            }            
        }
    }
    #undef H 
    #undef W 
    #undef M 
    #undef C 
    #undef H_out 
    #undef W_out 
    #undef W_grid 
    #undef h_base
    #undef w_base
    #undef h   
    #undef w  
    #undef y4d
    #undef x4d
    #undef k4d
}


// Kernel which unrolls channels Manually
// __global__ void conv_forward_kernel2(float *y, const float* __restrict__ x)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     y - output
//     x - input
//     k - kernel
//     B - batch_size (number of images in x)
//     M - number of output feature maps
//     C - number of input feature maps
//     H - input height dimension
//     W - input width dimension
//     K - kernel height and width (K x K)
//     */
//     #define H 40
//     #define W 40
//     #define M 16
//     #define C 4
//     #define H_out 34
//     #define W_out 34
//     #define W_grid 2  //TODO: Need to add this: ceil(W_out/TILEWIDTH2)
//     #define h_base ((blockIdx.z / W_grid) * TILEWIDTH2)
//     #define w_base ((blockIdx.z % W_grid) * TILEWIDTH2) 
//     #define h (h_base+threadIdx.y)  
//     #define w (w_base+threadIdx.x)      
//     #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//     #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//     #define k4d(i3, i2, i1, i0) Weights[(i3) * (C * MASK_WIDTH * MASK_WIDTH) + (i2) * (MASK_WIDTH * MASK_WIDTH) + (i1) * (MASK_WIDTH) + i0]

//     // const int H_out = H - MASK_WIDTH + 1;
//     // const int W_out = W - MASK_WIDTH + 1;

//     // int X_tile_width = TILEWIDTH + MASK_WIDTH-1;
//     // int W_grid = ceil(W_out/(TILEWIDTH*1.0));
//     __shared__ float X_shared[X_tile_width2*X_tile_width2];

//     int batchId = blockIdx.x;
//     int m = blockIdx.y;
//     int h0 = threadIdx.y;
//     int w0 = threadIdx.x;
//     // int h_base = (blockIdx.z / W_grid) * TILEWIDTH; // vertical base out data index for the block
//     // int w_base = (blockIdx.z % W_grid) * TILEWIDTH; // horizontal base out data index for the block

//     // int h = h_base + h0;
//     // int w = w_base + w0;

//     float Pvalue = 0.0f;
//     int c, p, q, i, j;
//     // Unroll Loop manually
//     // Channel 0
//     // Added by JH: To unroll for loops for shared memory loading
//     if(h < H && w < W){
//         X_shared[h0*X_tile_width2 + w0] = x4d(batchId, 0, h, w);
//     }
    
//     if((w+TILEWIDTH2) < (w_base + X_tile_width2)){
//         if(h < H && ((w+TILEWIDTH2) < W))
//             X_shared[h0*X_tile_width2 + (w0 + TILEWIDTH2)] = x4d(batchId, 0, h, w+TILEWIDTH2);
//     }

//     if((h+TILEWIDTH2) < (h_base + X_tile_width2)){
//         if((h+TILEWIDTH2) < H && w < W)
//             X_shared[(h0+TILEWIDTH2)*X_tile_width2 + w0] = x4d(batchId, 0, h+TILEWIDTH2, w);
//     }

//     if(((w+TILEWIDTH2) < (w_base + X_tile_width2)) && ((h+TILEWIDTH2) < (h_base + X_tile_width2))){
//         if((h+TILEWIDTH2) < H && (w+TILEWIDTH2) < W)
//             X_shared[(h0+TILEWIDTH2)*X_tile_width2 + w0 + TILEWIDTH2] = x4d(batchId, 0, h+TILEWIDTH2, w+TILEWIDTH2);
//     }

//     __syncthreads();
    
//     #pragma unroll
//     for (p = 0; p < MASK_WIDTH; p++) {
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+0)] * Weights[m * 196 + 0 *49 + p * 7 + 0];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+1)] * Weights[m * 196 + 0 *49 + p * 7 + 1];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+2)] * Weights[m * 196 + 0 *49 + p * 7 + 2];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+3)] * Weights[m * 196 + 0 *49 + p * 7 + 3];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+4)] * Weights[m * 196 + 0 *49 + p * 7 + 4];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+5)] * Weights[m * 196 + 0 *49 + p * 7 + 5];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+6)] * Weights[m * 196 + 0 *49 + p * 7 + 6]; 
//     }

//     __syncthreads();
//     // Channel 1
//     // Added by JH: To unroll for loops for shared memory loading
//     if(h < H && w < W){
//         X_shared[h0*X_tile_width2 + w0] = x4d(batchId, 1, h, w);
//     }
    
//     if((w+TILEWIDTH2) < (w_base + X_tile_width2)){
//         if(h < H && ((w+TILEWIDTH2) < W))
//             X_shared[h0*X_tile_width2 + (w0 + TILEWIDTH2)] = x4d(batchId, 1, h, w+TILEWIDTH2);
//     }

//     if((h+TILEWIDTH2) < (h_base + X_tile_width2)){
//         if((h+TILEWIDTH2) < H && w < W)
//             X_shared[(h0+TILEWIDTH2)*X_tile_width2 + w0] = x4d(batchId, 1, h+TILEWIDTH2, w);
//     }

//     if(((w+TILEWIDTH2) < (w_base + X_tile_width2)) && ((h+TILEWIDTH2) < (h_base + X_tile_width2))){
//         if((h+TILEWIDTH2) < H && (w+TILEWIDTH2) < W)
//             X_shared[(h0+TILEWIDTH2)*X_tile_width2 + w0 + TILEWIDTH2] = x4d(batchId, 1, h+TILEWIDTH2, w+TILEWIDTH2);
//     }

//     __syncthreads();
    
//     #pragma unroll
//     for (p = 0; p < MASK_WIDTH; p++) {
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+0)] * Weights[m * 196 + 1 *49 + p * 7 + 0];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+1)] * Weights[m * 196 + 1 *49 + p * 7 + 1];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+2)] * Weights[m * 196 + 1 *49 + p * 7 + 2];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+3)] * Weights[m * 196 + 1 *49 + p * 7 + 3];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+4)] * Weights[m * 196 + 1 *49 + p * 7 + 4];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+5)] * Weights[m * 196 + 1 *49 + p * 7 + 5];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+6)] * Weights[m * 196 + 1 *49 + p * 7 + 6]; 
//     }
    
//     __syncthreads();

//     // Channel 2
//     // Added by JH: To unroll for loops for shared memory loading
//     if(h < H && w < W){
//         X_shared[h0*X_tile_width2 + w0] = x4d(batchId, 2, h, w);
//     }
    
//     if((w+TILEWIDTH2) < (w_base + X_tile_width2)){
//         if(h < H && ((w+TILEWIDTH2) < W))
//             X_shared[h0*X_tile_width2 + (w0 + TILEWIDTH2)] = x4d(batchId, 2, h, w+TILEWIDTH2);
//     }

//     if((h+TILEWIDTH2) < (h_base + X_tile_width2)){
//         if((h+TILEWIDTH2) < H && w < W)
//             X_shared[(h0+TILEWIDTH2)*X_tile_width2 + w0] = x4d(batchId, 2, h+TILEWIDTH2, w);
//     }

//     if(((w+TILEWIDTH2) < (w_base + X_tile_width2)) && ((h+TILEWIDTH2) < (h_base + X_tile_width2))){
//         if((h+TILEWIDTH2) < H && (w+TILEWIDTH2) < W)
//             X_shared[(h0+TILEWIDTH2)*X_tile_width2 + w0 + TILEWIDTH2] = x4d(batchId, 2, h+TILEWIDTH2, w+TILEWIDTH2);
//     }

//     __syncthreads();
    
//     #pragma unroll
//     for (p = 0; p < MASK_WIDTH; p++) {
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+0)] * Weights[m * 196 + 2 *49 + p * 7 + 0];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+1)] * Weights[m * 196 + 2 *49 + p * 7 + 1];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+2)] * Weights[m * 196 + 2 *49 + p * 7 + 2];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+3)] * Weights[m * 196 + 2 *49 + p * 7 + 3];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+4)] * Weights[m * 196 + 2 *49 + p * 7 + 4];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+5)] * Weights[m * 196 + 2 *49 + p * 7 + 5];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+6)] * Weights[m * 196 + 2 *49 + p * 7 + 6]; 
//     }
    
//     __syncthreads();

//     // Channel 3
//     // Added by JH: To unroll for loops for shared memory loading
//     if(h < H && w < W){
//         X_shared[h0*X_tile_width2 + w0] = x4d(batchId, 3, h, w);
//     }
    
//     if((w+TILEWIDTH2) < (w_base + X_tile_width2)){
//         if(h < H && ((w+TILEWIDTH2) < W))
//             X_shared[h0*X_tile_width2 + (w0 + TILEWIDTH2)] = x4d(batchId, 3, h, w+TILEWIDTH2);
//     }

//     if((h+TILEWIDTH2) < (h_base + X_tile_width2)){
//         if((h+TILEWIDTH2) < H && w < W)
//             X_shared[(h0+TILEWIDTH2)*X_tile_width2 + w0] = x4d(batchId, 3, h+TILEWIDTH2, w);
//     }

//     if(((w+TILEWIDTH2) < (w_base + X_tile_width2)) && ((h+TILEWIDTH2) < (h_base + X_tile_width2))){
//         if((h+TILEWIDTH2) < H && (w+TILEWIDTH2) < W)
//             X_shared[(h0+TILEWIDTH2)*X_tile_width2 + w0 + TILEWIDTH2] = x4d(batchId, 3, h+TILEWIDTH2, w+TILEWIDTH2);
//     }

//     __syncthreads();
    
//     #pragma unroll
//     for (p = 0; p < MASK_WIDTH; p++) {
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+0)] * Weights[m * 196 + 3 *49 + p * 7 + 0];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+1)] * Weights[m * 196 + 3 *49 + p * 7 + 1];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+2)] * Weights[m * 196 + 3 *49 + p * 7 + 2];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+3)] * Weights[m * 196 + 3 *49 + p * 7 + 3];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+4)] * Weights[m * 196 + 3 *49 + p * 7 + 4];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+5)] * Weights[m * 196 + 3 *49 + p * 7 + 5];
//         Pvalue += X_shared[(h0+p) * X_tile_width2 + (w0+6)] * Weights[m * 196 + 3 *49 + p * 7 + 6]; 
//     }
    
//     __syncthreads();
    
//     if (h < H_out && w < W_out) {
//         y4d(batchId, m, h, w) = Pvalue;
//     }

//     #undef H 
//     #undef W 
//     #undef M 
//     #undef C 
//     #undef H_out 
//     #undef W_out 
//     #undef W_grid 
//     #undef h_base
//     #undef w_base
//     #undef h   
//     #undef w  
//     #undef y4d
//     #undef x4d
//     #undef k4d
// }

__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Declare relevant device pointers
    float *deviceInput;
    float *deviceOutput;
    // float *deviceKernel;    

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // Allocate memory and copy over the relevant data structures to the GPU
    std::cout<<"Allocating Memory.. "<<std::endl;
    cudaMalloc((void**)&deviceInput,(H*W*C*B)*sizeof(float));
    cudaMalloc((void**)&deviceOutput,(H_out*W_out*M*B)*sizeof(float));
    // cudaMalloc((void**)&deviceKernel,(K*K*M*C)*sizeof(float));
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
    cudaMemcpyToSymbol(Weights, host_k, (MASK_WIDTH * MASK_WIDTH * M * C) * sizeof(float));

    // cudaMemcpy(deviceKernel, &host_k[0], (K*K*M*C)*sizeof(float), cudaMemcpyHostToDevice);
    // error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error [2]: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    // Set the kernel dimensions and call the kernel
    std::cout<<"Settting Kernel Dimensions "<<std::endl;
    // Kernel Dimensions for Layer1
    int W_grid1 = ceil(W_out/(TILEWIDTH1*1.0));
    int H_grid1 = ceil(H_out/(TILEWIDTH1*1.0));
    int Z1 = W_grid1*H_grid1;

    dim3 dimBlock1(TILEWIDTH1, TILEWIDTH1, 1);
    dim3 dimGrid1(B,1,Z1);
    // Kernel Dimensions for Layer2
    int W_grid2 = ceil(W_out/(TILEWIDTH2*1.0));
    int H_grid2 = ceil(H_out/(TILEWIDTH2*1.0));
    int Z2 = W_grid2*H_grid2;

    dim3 dimBlock2(TILEWIDTH2, TILEWIDTH2, 1);
    dim3 dimGrid2(B,1,Z2);


    //@@ Launch the GPU kernel here
    std::cout<<"Launching Kernel "<<std::endl;
    // size_t shmem_size = sizeof(float) * ((TILEWIDTH + K-1)*(TILEWIDTH + K-1) + K*K );
    if(M==4)
        conv_forward_kernel1<<<dimGrid1, dimBlock1>>>(deviceOutput, deviceInput);
    else
        conv_forward_kernel2<<<dimGrid2, dimBlock2>>>(deviceOutput, deviceInput);
        
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
    // cudaFree(deviceKernel);  
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
