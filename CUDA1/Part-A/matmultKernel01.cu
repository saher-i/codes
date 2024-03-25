#include "matmultKernel.h"

#define FOOTPRINT_SIZE BLOCK_SIZE

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
  float *Asub, *Bsub, *Csub;
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x; // Operate directly for shared memory indexing
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Compute the starting point for each thread in global memory
  // Each thread handles four elements in a row, but accesses should be coalesced
  Csub = &C.elements[C.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * block_col];

  __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

  // Initialize Cvalue for accumulation
  float Cvalue[4] = {0, 0, 0, 0};

  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
    Asub = &A.elements[A.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * m];
    Bsub = &B.elements[B.stride * BLOCK_SIZE * m + BLOCK_SIZE * block_col];

    // Load elements into shared memory - Each thread loads one element for shared_A and shared_B
    shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
    shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];

    __syncthreads();

    // Perform the multiplication for the four elements this thread is responsible for
    // This loop is for dot-product computation; keep it for shared memory utilization
    for (int e = 0; e < BLOCK_SIZE; ++e) {
      for (int i = 0; i < 4; ++i) { // Loop unrolled for computing four elements
        // Adjust index for global Csub write back
        int col = thread_col * 4 + i;
        // Ensure within bounds for the case when the matrix width is not a multiple of BLOCK_SIZE * 4
        if (col < C.width) {
          Cvalue[i] += shared_A[thread_row][e] * shared_B[e][col];
        }
      }
    }

    __syncthreads();
  }

  // Write the computed Cvalues back to global memory, ensuring coalesced writes
  for (int i = 0; i < 4; ++i) {
    int col = thread_col * 4 + i;
    if (col < C.width) { // Check bounds
      Csub[thread_row * C.stride + col] = Cvalue[i];
    }
  }
}

