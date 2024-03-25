#include "matmultKernel.h"

#define FOOTPRINT_SIZE BLOCK_SIZE

// Adjusted kernel definition
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
  float *Asub, *Bsub, *Csub;
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x * 4; // Each thread now handles 4 columns
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Adjust Csub calculation for the 4 values per thread
  Csub = &C.elements[C.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * block_col];

  // Declare an array to hold the computed values for C
  float Cvalue[4] = {0, 0, 0, 0};

  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
    Asub = &A.elements[A.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * m];
    Bsub = &B.elements[B.stride * BLOCK_SIZE * m + BLOCK_SIZE * block_col];

    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    // Load elements into shared memory
    shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
    shared_A[thread_row][thread_col + 1] = Asub[thread_row * A.stride + thread_col + 1];
    shared_A[thread_row][thread_col + 2] = Asub[thread_row * A.stride + thread_col + 2];
    shared_A[thread_row][thread_col + 3] = Asub[thread_row * A.stride + thread_col + 3];

    shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];
    shared_B[thread_row][thread_col + 1] = Bsub[thread_row * B.stride + thread_col + 1];
    shared_B[thread_row][thread_col + 2] = Bsub[thread_row * B.stride + thread_col + 2];
    shared_B[thread_row][thread_col + 3] = Bsub[thread_row * B.stride + thread_col + 3];

    __syncthreads();

    // Unroll this loop to compute the four Cvalues
    for (int e = 0; e < BLOCK_SIZE; ++e) {
      Cvalue[0] += shared_A[thread_row][e] * shared_B[e][thread_col];
      Cvalue[1] += shared_A[thread_row][e] * shared_B[e][thread_col + 1];
      Cvalue[2] += shared_A[thread_row][e] * shared_B[e][thread_col + 2];
      Cvalue[3] += shared_A[thread_row][e] * shared_B[e][thread_col + 3];
    }

    __syncthreads();
  }

  // Write the computed Cvalues to global memory
  Csub[thread_row * C.stride + thread_col] = Cvalue[0];
  Csub[thread_row * C.stride + thread_col + 1] = Cvalue[1];
  Csub[thread_row * C.stride + thread_col + 2] = Cvalue[2];
  Csub[thread_row * C.stride + thread_col + 3] = Cvalue[3];
}

