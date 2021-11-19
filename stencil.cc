#include <iostream>
#include <chrono>
#include <numeric>
#include <assert.h>
#include <math.h>
#include <vector>
#include <memory>
#include <iomanip>
#include "vcl/vectorclass.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

#ifndef __GNUC__
#define __restrict__
#endif

struct GlobalContext {
    int n;
    int iterations;
    double* u0;
    double* u1;

    GlobalContext (int n_)
        : n(n_)
    {}
};

// Print the grid for debugging purposes
template <typename T>
void print_grid(int N, T vec) {
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            cout << setprecision(2) << vec[x*N + y] << " ";
        }
        cout << endl;
    }
}

// Test if one grid equals another with a small margin of error epsilon
bool test_grids(int N, float* __restrict__ grid_1, float* __restrict__ grid_2) {
    const double epsilon = 0.0002;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (!(fabs(grid_1[i*N + j] - grid_2[i*N + j]) < epsilon)) {
                cout << "ERROR: " << grid_1[i*N + j] << " != " << grid_2[i*N + j] << endl;
                return false;
            }
        }
    }
    return true;
}

// Return a random variable between fMin and fMax
float fRand(float fMin, float fMax) {
    float f = (float)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

// Initialize a N * N grid with random variables coming from a seed
float* initialize_grid(int N, float fMin, float fMax) {
    float* vec = new(std::align_val_t(64)) float[N * N];
    const unsigned int seed = 42;
    srand(seed);
    for (unsigned int x = 0; x < N; ++x) {
        for (unsigned int y = 0; y < N; ++y) {
            // For better debugging
            // vec[x*N + y] = x*N + y + 100;
            vec[x*N + y] = fRand(fMin, fMax);
        }
    }
    return vec;
}

// Transpose a given grid and return that transposed one
float* transpose(int N, float* src) {
    float* tar = new(std::align_val_t(64)) float[N * N];
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            tar[i*N + k] = src[k*N + i];
        }
    }
    return tar;
}

// Reference solution for the local mean kernel, allocate a tmp array of size 
// N * N for temporary write and later read backs
// Used to ensure that other implementations work correctly
void reference_solution(int N, int K, float* vec) {
    // Allocate tmp grid
    float* tmp = new(std::align_val_t(64)) float[N*N];

    // Find neighbors
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            int left_boundary = max(y - K, 0), right_boundary = min(y + K, N - 1);
            int bottom_boundary = max(x - K, 0), upper_boundary = min(x + K, N - 1);
            int neighbors_count = (upper_boundary - bottom_boundary) + (right_boundary - left_boundary);
            float sum = 0.0;

            for (int i = left_boundary; i <= right_boundary; i++) {
                if (i != y) {
                    sum += vec[x*N + i];
                }
            }

            for (int i = bottom_boundary; i <= upper_boundary; i++) {
                if (i != x) {
                    sum += vec[i*N + y];
                }
            }

            // Replace current value at u(x, y) with local mean
            tmp[x*N + y] = sum / neighbors_count;
        }
    }

    // Write back
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            vec[x*N + y] = tmp[x*N + y];
        }
    }
}

// Vanilla version, no blocking or vectorization applied. Use the original grid and the transposed
// version to benefit from consecutive reads from neighbors below or above grud point u(i, j)
void vanilla_local_mean(int N, int K, float* __restrict__ vec, float* __restrict__ trans) {
    // Allocate tmp grid
    float* tmp = new(std::align_val_t(64)) float[N*N];

    // Find neighbors
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            int left_boundary = max(y - K, 0), right_boundary = min(y + K, N - 1);
            int bottom_boundary = max(x - K, 0), upper_boundary = min(x + K, N - 1);
            int neighbors_count = (right_boundary - left_boundary) + (upper_boundary - bottom_boundary);
            float sum = 0.0;

            for (int i = left_boundary; i <= right_boundary; i++) {
                if (i != y) {
                    sum += vec[x*N + i];
                }
            }

            for (int i = bottom_boundary; i <= upper_boundary; i++) {
                if (i != x) {
                    sum += trans[y*N+i]; // Makes it a lot faster due to consecutive reads
                }
            }

            // Replace current value at u(x, y) with local mean
            tmp[x*N + y] = sum / neighbors_count;
        }
    }

    // Write back
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            vec[x*N + y] = tmp[x*N + y];
        }
    }
}

// Blocked version of the vanilla version implemented above
// Block size is compile parameter B
template<int B>
void blocked_local_mean(int N, int K, float* __restrict__ vec, float* __restrict__ trans) {
    // Divisibility contraints
    if (N % B != 0) {
        cout << "N must be divisible through B" << endl;
        exit(-1);
    }
    // Allocate tmp grid
    float* tmp = new(std::align_val_t(64)) float[N*N];

    // Find neighbors
    for (int I = 0; I < N; I+=B) {
        for (int J = 0; J < N; J+=B) {
            for (int x = I; x < I + B; ++x) {
                for (int y = J; y < J + B; ++y) {
                    int left_boundary = max(y - K, 0), right_boundary = min(y + K, N - 1);
                    int bottom_boundary = max(x - K, 0), upper_boundary = min(x + K, N - 1);
                    int neighbors_count = (right_boundary - left_boundary) + (upper_boundary - bottom_boundary);
                    float sum = 0.0;

                    for (int i = left_boundary; i <= right_boundary; i++) {
                        if (i != y) {
                            sum += vec[x*N + i];
                        }
                    }

                    for (int i = bottom_boundary; i <= upper_boundary; i++) {
                        if (i != x) {
                            sum += trans[y*N+i]; // Makes it a lot faster due to consecutive reads
                        }
                    }

                    // Replace current value at u(x, y) with local mean
                    tmp[x*N + y] = sum / neighbors_count;
                }
            }
        }
    }

    // Write back
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            vec[x*N + y] = tmp[x*N + y];
        }
    }
}

// Vectorized version of the blocked implementation from above
template<int B>
void blocked_vectorized_local_mean(int N, int K, float* __restrict__ vec, float* __restrict__ trans) {
    // Divisibility contraints
    if (N % B != 0) {
        cout << "N must be divisible through B" << endl;
        exit(-1);
    }
    // Allocate tmp grid
    float* tmp = new(std::align_val_t(64)) float[N*N];

    // Find neighbors
    for (int I = 0; I < N; I+=B) {
        for (int J = 0; J < N; J+=B) {
            for (int x = I; x < I + B; ++x) {
                for (int y = J; y < J + B; ++y) {
                    int left_boundary = max(y - K, 0), right_boundary = min(y + K, N - 1);
                    int bottom_boundary = max(x - K, 0), upper_boundary = min(x + K, N - 1);
                    int neighbors_count = (right_boundary - left_boundary) + (upper_boundary - bottom_boundary);
                    Vec4f result_vec(0.0);

                    for (int i = left_boundary; i < y; i+=4) {
                        Vec4f tmp;
                        if (i + 4 < y)
                            tmp.load(&vec[x*N + i]);
                        else
                            tmp.load_partial(y - i, &vec[x*N + i]);

                        result_vec += tmp;
                    }

                    for (int i = y; i < right_boundary; i+=4) {
                        Vec4f tmp;
                        if (i + 4 < right_boundary)
                            tmp.load(&vec[x*N + i + 1]);
                        else
                            tmp.load_partial(right_boundary - i, &vec[x*N + i + 1]);

                        result_vec += tmp;
                    }

                    for (int i = bottom_boundary; i < x; i+=4) {
                        Vec4f tmp;
                        if (i + 4 < x)
                            tmp.load(&trans[y*N + i]);
                        else
                            tmp.load_partial(x - i, &trans[y*N + i]);

                        result_vec += tmp;
                    }

                    for (int i = x; i < upper_boundary; i+=4) {
                        Vec4f tmp;
                        if (i + 4 < upper_boundary)
                            tmp.load(&trans[y*N + i + 1]);
                        else
                            tmp.load_partial(upper_boundary - i, &trans[y*N + i + 1]);

                        result_vec += tmp;
                    }

                    float sum = horizontal_add(result_vec); // Add up the numbers from a single vector - medium efficiency

                    // Replace current value at u(x, y) with local mean
                    tmp[x*N + y] = sum / neighbors_count;
                }
            }
        }
    }

    // Write back
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            vec[x*N + y] = tmp[x*N + y];
        }
    }
}

// Vectorized version of the blocked implementation from above
template<int B>
void blocked_vectorized_local_mean_multithreaded(int N, int K, float* __restrict__ vec, float* __restrict__ trans) {
    // Divisibility contraints
    if (N % B != 0) {
        cout << "N must be divisible through B" << endl;
        exit(-1);
    }
    // Allocate tmp grid
    float* tmp = new(std::align_val_t(64)) float[N*N];

    // Find neighbors
#pragma omp parallel
#pragma omp for
    for (int I = 0; I < N; I+=B) {
        for (int J = 0; J < N; J+=B) {
            for (int x = I; x < I + B; ++x) {
                for (int y = J; y < J + B; ++y) {
                    int left_boundary = max(y - K, 0), right_boundary = min(y + K, N - 1);
                    int bottom_boundary = max(x - K, 0), upper_boundary = min(x + K, N - 1);
                    int neighbors_count = (right_boundary - left_boundary) + (upper_boundary - bottom_boundary);
                    Vec4f result_vec(0.0);

                    for (int i = left_boundary; i < y; i+=4) {
                        Vec4f tmp;
                        if (i + 4 < y)
                            tmp.load(&vec[x*N + i]);
                        else
                            tmp.load_partial(y - i, &vec[x*N + i]);

                        result_vec += tmp;
                    }

                    for (int i = y; i < right_boundary; i+=4) {
                        Vec4f tmp;
                        if (i + 4 < right_boundary)
                            tmp.load(&vec[x*N + i + 1]);
                        else
                            tmp.load_partial(right_boundary - i, &vec[x*N + i + 1]);

                        result_vec += tmp;
                    }

                    for (int i = bottom_boundary; i < x; i+=4) {
                        Vec4f tmp;
                        if (i + 4 < x)
                            tmp.load(&trans[y*N + i]);
                        else
                            tmp.load_partial(x - i, &trans[y*N + i]);

                        result_vec += tmp;
                    }

                    for (int i = x; i < upper_boundary; i+=4) {
                        Vec4f tmp;
                        if (i + 4 < upper_boundary)
                            tmp.load(&trans[y*N + i + 1]);
                        else
                            tmp.load_partial(upper_boundary - i, &trans[y*N + i + 1]);

                        result_vec += tmp;
                    }

                    float sum = horizontal_add(result_vec); // Add up the numbers from a single vector - medium efficiency

                    // Replace current value at u(x, y) with local mean
                    tmp[x*N + y] = sum / neighbors_count;
                }
            }
        }
    }

    // Write back
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            vec[x*N + y] = tmp[x*N + y];
        }
    }
}

// Implemented jacobi method parallized with OpenMP
// make it possible to specify a number of threads that will run in parallel
void jacobi_parallel_kernel(int n, int iterations, int threads, double *__restrict uold, double *__restrict unew)
{
    for (int i = 0; i < iterations; i++) {
        // parallellize inner loops (parallel reads on uold)
        #pragma omp parallel for num_threads(threads) collapse(2)
        for (int i1 = 1; i1 < n - 1; i1++) {
            for (int i0 = 1; i0 < n - 1; i0++) {
                unew[i1*n + i0] = pow(max(abs(i0), abs(i1)), -2) * (uold[i1*n + i0-n] + uold[i1*n + i0-1] +
                                    uold[i1*n + i0+1] + uold[i1*n + i0+n]);
            }
        }
        swap(uold, unew);
    }
}

int main(int argc, char* argv[]) {
    // Block size for blocked implementations
    const int B = 32;

    // Grid size N and amount of neighbors K
    int N = 1024;
    int K = 8;
    
    // Make it possible to initialize the variables while the program is running
    if (argc == 3) {
        N = atoi(argv[1]);
        K = atoi(argv[2]);
    }

    if (argc != 1 && argc != 3) {
        cout << "Usage:" << endl;
        cout << "./program" << endl;
        cout << "./program <N> <K>" << endl;
        exit(-1);
    }

    cout << "N: " << N << ", K: " << K << ", B: " << B << endl;

    // Run the reference solution
    float* reference = initialize_grid(N, -100, 100);
    auto begin = chrono::high_resolution_clock::now();
    reference_solution(N, K, reference);
    auto end = chrono::high_resolution_clock::now();
    cout << "Reference:  " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;

    // Run the vanilla version
    float* vec_vanilla = initialize_grid(N, -100, 100);
    float* vec_vanilla_transposed = transpose(N, vec_vanilla);
    begin = chrono::high_resolution_clock::now();
    vanilla_local_mean(N, K, vec_vanilla, vec_vanilla_transposed);
    end = chrono::high_resolution_clock::now();
    cout << "Vanilla:    " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;
    assert(test_grids(N, reference, vec_vanilla));
    delete[] vec_vanilla_transposed;
    delete[] vec_vanilla;

    // Run the blocked version
    float* vec_blocked = initialize_grid(N, -100, 100);    
    float* vec_blocked_transposed = transpose(N, vec_blocked);
    begin = chrono::high_resolution_clock::now();
    blocked_local_mean<B>(N, K, vec_blocked, vec_blocked_transposed);
    end = chrono::high_resolution_clock::now();
    cout << "Blocked:    " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;
    assert(test_grids(N, reference, vec_blocked));
    delete[] vec_blocked_transposed;
    delete[] vec_blocked;

    // Run the vectorized version
    float* vec_blocked_vectorized = initialize_grid(N, -100, 100);
    float* vec_blocked_vectorized_transposed = transpose(N, vec_blocked_vectorized);
    begin = chrono::high_resolution_clock::now();
    blocked_vectorized_local_mean<B>(N, K, vec_blocked_vectorized, vec_blocked_vectorized_transposed);
    end = chrono::high_resolution_clock::now();
    cout << "Vectorized: " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;
    assert(test_grids(N, reference, vec_blocked_vectorized));
    delete[] vec_blocked_vectorized_transposed; 
    delete[] vec_blocked_vectorized;

    // Run multithreaded version
    float* vec_blocked_vectorized_multithreaded = initialize_grid(N, -100, 100);
    float* vec_blocked_vectorized_transposed_multithreaded = transpose(N, vec_blocked_vectorized_multithreaded);
    begin = chrono::high_resolution_clock::now();
    blocked_vectorized_local_mean_multithreaded<B>(N, K, vec_blocked_vectorized_multithreaded, vec_blocked_vectorized_transposed_multithreaded);
    end = chrono::high_resolution_clock::now();
    cout << "Multithreaded: " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;
    assert(test_grids(N, reference, vec_blocked_vectorized_multithreaded));
    delete[] vec_blocked_vectorized_transposed_multithreaded; 
    delete[] vec_blocked_vectorized_multithreaded;
    
    // Free memory
    delete[] reference;

    // New grid size for Jacobi and iterations
    N = 1024;
    const int iterations = 50;
    
    auto context = make_shared<GlobalContext>(N);     
    context->iterations = iterations;
    context->u0 = new(std::align_val_t(64)) double [N*N];
    context->u1 = new(std::align_val_t(64)) double [N*N];

    // fill boundary values and initial values
    auto g = [&](int i0, int i1) { return  (i0>0 && i0<N-1 && i1>0 && i1<N-1) ? 0.0 : 1.0; };

    for (int i1 = 0; i1 < context->n; ++i1) {
        for (int i0 = 0; i0 < context->n; ++i0) {
            context->u0[i1*context->n+i0] = context->u1[i1*context->n+i0] = g(i0, i1);
        }
    }

    cout << "---------------------" << endl;
    cout << "Running Jacobi method with N = "  << N << " and " << iterations << " Iterations" << endl;

    context->u0[0] = 50;

    // Jacobi sequential
    begin = chrono::high_resolution_clock::now();
    jacobi_parallel_kernel(context->n, context->iterations, 1, context->u0, context->u1);
    end = chrono::high_resolution_clock::now();
    cout << "Jacobi 1 Thread OpenMP: " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;

    // Reset grid
    for (int i1 = 0; i1 < context->n; ++i1) {
        for (int i0 = 0; i0 < context->n; ++i0) {
            context->u0[i1*context->n+i0] = context->u1[i1*context->n+i0] = g(i0, i1);
        }
    }

    // Jacobi 2 Threads
    begin = chrono::high_resolution_clock::now();
    jacobi_parallel_kernel(context->n, context->iterations, 2, context->u0, context->u1);
    end = chrono::high_resolution_clock::now();
    cout << "Jacobi 2 Threads OpenMP: " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;

    // Reset grid
    for (int i1 = 0; i1 < context->n; ++i1) {
        for (int i0 = 0; i0 < context->n; ++i0) {
            context->u0[i1*context->n+i0] = context->u1[i1*context->n+i0] = g(i0, i1);
        }
    }

    // Jacobi 3 Threads
    begin = chrono::high_resolution_clock::now();
    jacobi_parallel_kernel(context->n, context->iterations, 3, context->u0, context->u1);
    end = chrono::high_resolution_clock::now();
    cout << "Jacobi 3 Threads OpenMP: " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;

    // Reset grid
    for (int i1 = 0; i1 < context->n; ++i1) {
        for (int i0 = 0; i0 < context->n; ++i0) {
            context->u0[i1*context->n+i0] = context->u1[i1*context->n+i0] = g(i0, i1);
        }
    }

    // Jacobi 4 Threads
    begin = chrono::high_resolution_clock::now();
    jacobi_parallel_kernel(context->n, context->iterations, 4, context->u0, context->u1);
    end = chrono::high_resolution_clock::now();
    cout << "Jacobi 4 Threads OpenMP: " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;

    // Free memory
    delete[] context->u0;
    delete[] context->u1;
 
    return 0;
}
