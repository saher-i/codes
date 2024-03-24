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

__global__ void AddVectors(const float *A, const float *B, float *C, int numElements) {
    int startIdx = (blockDim.x * blockIdx.x + threadIdx.x) * 32; // Each thread starts at a different base index
    if (startIdx < numElements) {
        for (int j = 0; j < 32 && (startIdx + j) < numElements; ++j) { // Ensure we don't go out of bounds
            int idx = startIdx + j;
            C[idx] = A[idx] + B[idx];
        }
    }
}
