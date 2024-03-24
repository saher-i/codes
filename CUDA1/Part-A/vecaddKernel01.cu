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
  
        // Calculate the global thread index
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes 'N' elements
    int start = index * N;
    int end = start + N;

    // Perform the addition for 'N' elements assigned to this thread
    for(int i = start; i < end; i++) {
        C[i] = A[i] + B[i];
    }
    /*
    int blockStartIndex  = blockIdx.x * blockDim.x * N;
    int threadStartIndex = (blockStartIndex + (threadIdx.x * N))%N;
    int threadEndIndex   = threadStartIndex + N;
    int i;

    for( i=threadStartIndex; i<threadEndIndex; ++i ){
        C[i] = A[i] + B[i];
    }
*/
    /*
    int idx = (blockDim.x * blockIdx.x + threadIdx.x);
    if (idx < N) {
            C[idx] = A[idx] + B[idx];
    }*/
}
