#include <stdio.h>
#include <stdlib.h>
#include <time.h>


float dpunroll(long N, float *pA, float *pB) 
{
  float R = 0.0;
  int j;
  for (j=0;j<N;j+=4)
  {
    R += pA[j]*pB[j] + pA[j+1]*pB[j+1] + pA[j+2]*pB[j+2] + pA[j+3] * pB[j+3];
  }  
    return R; 
}

// The main() function can receive arguments from the command line, the number of arguments is referred by argument_count
// The argument_vector in the main function below is a char array of pointers and argument_vector[0] represents the name
//of the executable

int main(int argument_count, char *argument_vector[]) 
{
  if (argument_count != 3)                             
  {
    printf("For the results, use the following format: %s <vector_size> <num_of_rep> \n", argument_vector[0]);
    return 1;
  }

//argument_vector[1] represents the <vector_size>
//argument_vector[2] represents the <num_of_rep>

  long N = atol(argument_vector[1]);
  int num_of_rep = atoi(argument_vector[2]);
  struct timespec start, end;

  // Allocate memory for vectors
  float *vectorA = (float *)malloc(N * sizeof(float));
  float *vectorB = (float *)malloc(N * sizeof(float));

  // Initialize vectors to 1.0
  for (long i = 0; i < N; i++) 
  {
    vectorA[i] = 1.0;
    vectorB[i] = 1.0;
  }

  // Execution time measurement
  float a = 0.0;

  for (int rep = 0; rep < num_of_rep/2; rep++)
  {
    a = dpunroll(N, vectorA, vectorB);
  }

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int rep = num_of_rep/2; rep < num_of_rep; rep++)
  {
    a = dpunroll(N, vectorA, vectorB);
  }

  clock_gettime(CLOCK_MONOTONIC, &end);

  printf("dot-product for reference (ignore) : %f\n", a);

  double elapsed_time = (end.tv_sec - start.tv_sec) + ((end.tv_nsec - start.tv_nsec) * 1e-9);
  double mean_time = elapsed_time / num_of_rep / 2;

  // Bandwidth and throughput calculation
  double bandwidth = (2 * N * sizeof(float) * num_of_rep) / (mean_time * 1e9);
  double throughput = (2 * N * num_of_rep) / (mean_time);

  // Results
  printf("N: %ld ", N);
  printf("<T>: %e seconds ", mean_time);
  printf("B: %e GB/sec ", bandwidth);
  printf("F: %e FLOP/sec ", throughput);
  printf("\n");

  // Free allocated memory
  free(vectorA);
  free(vectorB);

  return 0;
}

