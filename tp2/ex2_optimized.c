#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000000

int main() {
    double a = 1.1, b = 1.2;
    double x = 0.0, y = 0.0;
    double ab;  // Pre-compute a*b
    clock_t start, end;

    // Manual optimization: compute a*b once before the loop
    ab = a * b;
    
    start = clock();
    // Manually optimized version - reduce redundant multiplications
    // and improve instruction scheduling
    for (int i = 0; i < N; i++) {
        x = ab + x; // stream 1
        y = ab + y; // independent stream 2
    }
    end = clock();

    printf("x = %f, y = %f, time = %f s\n",
           x, y, (double)(end - start) / CLOCKS_PER_SEC);
    return 0;
}
