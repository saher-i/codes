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
  
    // Global thread index calculation
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computing N elements
    //int startindex = index * N;
    //int endindex = startindex + N;

    // Addition for 'N' elements assigned to a thread
    for(int i = index; i < N; i = i+blockDim.x) {
        C[i] = A[i] + B[i];
    }
}
