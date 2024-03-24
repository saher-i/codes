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
    int startidx = ((blockIdx.x * blockDim.x) + threadIdx.x)*32; // Unique grid index of a thread
    if(startidx < N) {
        for (int j = 0; j < 32 && (startIdx + j) < N; ++j){
            int idx = startIdx + j;
            C[idx] = A[idx] + B[idx];
        }
    }
}

