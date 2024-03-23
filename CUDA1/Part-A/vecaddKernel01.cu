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

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Unique grid index of a thread
    //int stride = blockDim.x * gridDim.x; // Total number of threads in the grid
    
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
/*
    for (int i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
        __syncthreads();
    }
*/    
}

