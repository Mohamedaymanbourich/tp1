#include <omp.h>
#include <stdio.h>
static long num_steps = 100000;
static long num_threads = 100;
double step;

 int main () {
double pi, sum = 0.0;
double start_time = omp_get_wtime();
#pragma omp parallel num_threads(num_threads) reduction(+:sum)
  {
    int thread_id = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    
    int i;
    double x; 
    for (i = (int) thread_id*num_steps/num_threads; i < (int)(thread_id+1)*num_steps/num_threads ; i++) {
        step = 1.0 / (double) num_steps;
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
 }
  }

  printf("Pi is approximately: %f\n", step * sum);
  double end_time = omp_get_wtime();
  printf("Time taken: %f seconds\n", end_time - start_time );

return 0;
}