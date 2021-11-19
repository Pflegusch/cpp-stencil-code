# Dependencies
To make the code work the Agner Fog vector class library is neccessary, which one can find here: https://github.com/vectorclass/version2.
# Build and Run
Simply type ``make`` to build the code.
To run the code, simply run the binary with ``./stencil 1024 8``
````
Reference:  36ms
Vanilla:    50ms
Blocked:    40ms
Vectorized: 38ms
Multithreaded: 14ms

Running Jacobi method with N = 1024 and 50 Iterations
Jacobi 1 Thread OpenMP: 387ms
Jacobi 2 Threads OpenMP: 227ms
Jacobi 3 Threads OpenMP: 168ms
Jacobi 4 Threads OpenMP: 176ms
````

Note: g++ version 10.3.0 (Ubuntu 10.3.0-1ubuntu1~20.04) was used when building the project.
# Example Results 
Example output of the program for my local machine with an Intel Core i5-750:
````
N: 4096, K: 15, B: 32
Reference:  4896ms
Vanilla:    2203ms
Blocked:    1225ms
Vectorized: 856ms

Running Jacobi method with N = 4096 and 100 Iterations
Jacobi 1 Thread OpenMP: 11961ms
Jacobi 2 Threads OpenMP: 6150ms
Jacobi 3 Threads OpenMP: 4572ms
Jacobi 4 Threads OpenMP: 4103ms
````
