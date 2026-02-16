#include <stdio.h>
#include <omp.h>

static long num_steps = 100000;
long num_threads = 1000;
double step;

int main () 
{
int i; double x, pi, sum = 0.0;
step = 1.0 / (double) num_steps;

#pragma omp parallel for num_threads(num_threads) reduction(+:sum) 
for (i = 0; i < num_steps; i++) {
x = (i + 0.5) * step;
sum = sum + 4.0 / (1.0 + x * x);
}

pi = step * sum;
printf("Pi is approximately: %f\n", pi);

 return 0 ; 
}