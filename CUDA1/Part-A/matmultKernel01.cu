///
/// matmultKernel00.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-23 DVN
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Copy this file and modify the MatMultKernel device function for
/// each of your experiments. 
///

#include "matmultKernel.h"

#define FOOTPRINT_SIZE BLOCK_SIZE

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  Csub = &C.elements[C.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * block_col];

  // Each thread computes one element of Csub in its copy of CValue
  float Cvalue[4] = {0, 0, 0, 0};

  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int m = 0;  m < (A.width / BLOCK_SIZE); ++m){
    // Get Asub and Bsub descriptors
    Asub = &A.elements[A.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * m];
    Bsub = &B.elements[B.stride * BLOCK_SIZE * m + BLOCK_SIZE * block_col];

    // Copy ELEMENTS OF  ASub and Bsub into shared memory
    // EACH THREAD loads ONE ELEMENT of ASub and ONE of Bsub
    // Notice: it does not need to be the element it requires to
    //         compute its Cvalue, as long as all elements are 
    //         collaboratively read. 

    // Notice: every thread declares shared_A and shared_B in shared memory
    //         even though a thread block has only one shared_A and one shared_B
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread copies just one element of shared_A and one element of shared_B
   // shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
   // shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];

    shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
    shared_A[thread_row][thread_col+1] = Asub[thread_row * A.stride + thread_col + 1];
    shared_A[thread_row+1][thread_col] = Asub[(thread_row+1) * A.stride + thread_col];
    shared_A[thread_row+1][thread_col+1] = Asub[(thread_row+1) * A.stride + thread_col + 1];

    shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];
    shared_B[thread_row][thread_col+1] = Bsub[thread_row * B.stride + thread_col + 1];
    shared_B[thread_row+1][thread_col] = Bsub[(thread_row+1) * B.stride + thread_col];
    shared_B[thread_row+1][thread_col+1] = Bsub[(thread_row+1) * B.stride + thread_col + 1];

    // Synchronize to ensure all elements are read
    __syncthreads();

    // Do an inproduct of one row of shared_A and one col of shared_B
    // computing one Cvalue by accumulation
#pragma unroll
    int e;
    for(e=0; e<BLOCK_SIZE; ++e)
      // Cvalue[0] += shared_A[thread_row][e] * shared_B[e][thread_col];
    
    // Perform the multiplication and add to the accumulator (Cvalue)
    
    Cvalue[0] += shared_A[thread_row][e] * shared_B[e][thread_col];  // Top-left element of the 2x2 block
    Cvalue[1] += shared_A[thread_row][e] * shared_B[e][thread_col + 1];  // Top-right element of the 2x2 block
    Cvalue[2] += shared_A[thread_row + 1][e] * shared_B[e][thread_col];  // Bottom-left element of the 2x2 block
    Cvalue[3] += shared_A[thread_row + 1][e] * shared_B[e][thread_col + 1];  // Bottom-right element of the 2x2 block
    
    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
  }
  Csub[thread_row * C.stride + thread_col] = Cvalue[0];
  Csub[thread_row * C.stride + thread_col + 1] = Cvalue[1];
  Csub[(thread_row + 1) * C.stride + thread_col] = Cvalue[2];
  Csub[(thread_row + 1) * C.stride + thread_col + 1] = Cvalue[3];
  // Write Csub to GLOBAL memory.
  // Each thread writes its own cell value.
 // Csub[thread_row * C.stride + thread_col] = Cvalue;
}

