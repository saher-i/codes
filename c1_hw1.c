// dp1.c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define REPS_SMALL 1000
#define REPS_LARGE 20

float dp(long N, float *pA, float *pB) 
{
  float R = 0.0;
  int j;
  for (j = 0; j < N; j++)
    R += pA[j] * pB[j];
  return R;
}

// The main() function can receive arguments from the command line, the number of arguments is referred by argument_count
//

int main(int argument_count, char *argv[]) 
{
  if (argument_count != 3)                             
  {
    printf("For the results, use the following format: %s <vector_size> <num_reps> \n", argv[0]);
    return 1;
  }

  long N = atol(argv[1]);
  int num_reps = atoi(argv[2]);
  struct timespec start, end;

  // Allocate memory for vectors
  float *vectorA = (float *)malloc(N * sizeof(float));
  float *vectorB = (float *)malloc(N * sizeof(float));

  // Initialize vectors to 1.0
  for (long i = 0; i < N; i++) {
    vectorA[i] = 1.0;
    vectorB[i] = 1.0;
  }

  // Measure execution time for N=1000000
  printf("Vector Size: %ld, Number of Repetitions: %d\n", N, num_reps);
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int rep = 0; rep < num_reps; rep++) {
    dp(N, vectorA, vectorB);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);

  double elapsed_time_small = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  double mean_time_small = elapsed_time_small / REPS_SMALL / 2;

  // Measure execution time for N=300000000
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int rep = 0; rep < num_reps; rep++) {
    dp(N, vectorA, vectorB);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);

  double elapsed_time_large = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  double mean_time_large = elapsed_time_large / REPS_LARGE / 2;

  // Calculate bandwidth and throughput
  double bandwidth_small = (2 * N * sizeof(float) * REPS_SMALL) / (mean_time_small * 1e9);
  double bandwidth_large = (2 * N * sizeof(float) * REPS_LARGE) / (mean_time_large * 1e9);
  double throughput_small = (2 * N * REPS_SMALL) / (mean_time_small * 1e9);
  double throughput_large = (2 * N * REPS_LARGE) / (mean_time_large * 1e9);

  // Print results
  printf("Mean Execution Time (N=1000000): %e seconds\n", mean_time_small);
  printf("Bandwidth (N=1000000): %e GB/sec\n", bandwidth_small);
  printf("Throughput (N=1000000): %e FLOP/sec\n", throughput_small);

  printf("\nMean Execution Time (N=300000000): %e seconds\n", mean_time_large);
  printf("Bandwidth (N=300000000): %e GB/sec\n", bandwidth_large);
  printf("Throughput (N=300000000): %e FLOP/sec\n", throughput_large);

  // Free allocated memory
  free(vectorA);
  free(vectorB);

  return 0;
}

