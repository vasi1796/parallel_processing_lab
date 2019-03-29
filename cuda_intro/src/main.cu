#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>

constexpr auto PI = 3.14f;

__global__ void fill_array2D(float *a, float *b, int N, int M) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < M)
    {
        a[row*N + col] = powf(sinf(2*PI*row/N), 2) + powf(cosf(2*PI*col/M), 2);
        b[row*N + col] = powf(cosf(2 * PI*row / N), 2) + powf(sinf(2 * PI*col / M), 2);
    }
}

__global__ void fill_array1D(float *a,float*b, int N, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N;
    int col = idx % N;
    if (row < N && col < M)
    {
        a[row*N + col] = powf(sinf(2 * PI*row/N), 2) + powf(cosf(2 * PI*col/N), 2);
        b[row*N + col] = powf(cosf(2 * PI*row/M), 2) + powf(sinf(2 * PI*col/M), 2);
    }
}

__global__ void sum_vectors1D(float *a, float *b, float *c, int N, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx/N;
    int col = idx%N;
    if (row < N && col < M)
    {
        c[row*N + col] = a[row*N + col] + b[row*N + col];
    }
}

__global__ void sum_vectors2D(float *a, float *b, float *c, int N,int M) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < M)
    {
        c[row*N + col] = a[row*N + col]+ b[row*N + col];
    }
}

// main routine that executes on the host
int main()
{

    float *a_h, *a_d, *b_h, *b_d, *c_h, *c_d;
    const int N = 512;
    const int M = 512;
    size_t size = N * M * sizeof(float);
    
    //alocare host
    a_h = (float*)malloc(size);
    b_h = (float*)malloc(size);
    c_h = (float*)malloc(size);
    
    //alocare device
    cudaMalloc((void**)&a_d, size);
    cudaMalloc((void**)&b_d, size);
    cudaMalloc((void**)&c_d, size);

    //dimensiuni grid si threads
    dim3 grid2D(16,16,1);
    dim3 threads2D(32,32,1);
    dim3 grid1D(512, 1, 1);
    dim3 threads1D(512, 1, 1);

    //fill arrays
    //fill_array2D <<< grid2D, threads2D >>> (a_d, b_d,N, M);
    //sum_vectors2D <<< grid2D, threads2D >>> (a_d, b_d, c_d, N, M);
    fill_array1D <<< grid1D, threads1D >>> (a_d, b_d, N, M);
    sum_vectors1D <<< grid1D, threads1D >>> (a_d, b_d, c_d, N, M);
    
    //copy device data to host
    cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_h, b_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; ++i) 
    {
        for (int j = 0; j < M; ++j) 
        {
            std::cout << c_h[i*N + j]<<" ";
        }
        std::cout << std::endl;
    }

    //cuda cleanup
    free(a_h);
    free(b_h);
    free(c_h);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return 0;
}