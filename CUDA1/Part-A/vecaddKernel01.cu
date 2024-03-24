///
/// vecAddKernel00.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// By David Newman
/// Created: 2011-02-16
/// Last Modified: 2011-02-16 DVN
///
/// This Kernel adds two Vectors A and B in C on GPU
/// without using coalesced memory access.
///

__global__ void AddVectors(const float *A, const float *B, float *C, int N) {
  
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < N; i+blockDim.x)) {
        C[index] = A[index] + B[index];
    }
/*
    for(int i = index; i < N; i = i+blockDim.x) {
        C[i] = A[i] + B[i];
    }

    for(int i = index; i < blockDim.x*gridDim.x*N; i = i+N) {
        C[i] = A[i] + B[i];
    }
*/
}
