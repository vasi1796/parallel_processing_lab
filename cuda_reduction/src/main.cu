#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <utilities.hpp>

__global__ void cuda_max(float* a_d, float* out)
{
    __shared__ float a_sh[128];
    const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    a_sh[threadIdx.x] = a_d[index];
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            a_sh[threadIdx.x] = a_sh[threadIdx.x] > a_sh[threadIdx.x + s] ? a_sh[threadIdx.x] : a_sh[threadIdx.x + s];
        }
    }
    if (threadIdx.x == 0)
    {
        out[blockIdx.x] = a_sh[0];
    }

}

int main()
{
    float *a_h, *a_d, *b_h, *b_d;
    b_d = NULL;
    int N = 1024*1024;
    size_t size_of_vector = N * sizeof(float);

    a_h = (float*)malloc(size_of_vector);
    cudaMalloc((void**)&a_d, size_of_vector);

    const int grid = N / 128;

    dim3 grids(grid, 1, 1);
    dim3 threads(128, 1, 1);

    int rand_end = 1;
    for (int i = 0; i < N; ++i)
    {
        if (rand_end == 255) 
        {
            rand_end = 0;
        }
        rand_end++;
        a_h[i] = 1 + (rand() % rand_end);
    }
    float max_h = -1;

    for (int i = 0; i < N; ++i)
    {
        if (a_h[i] > max_h)
        {
            max_h = a_h[i];
        }
    }

    cudaMemcpy(a_d, a_h, size_of_vector, cudaMemcpyHostToDevice);
    while(N > threads.x)
    {
        cudaMalloc((void**)&b_d, N / 128 * sizeof(float));
        dim3 grids(N / 128);
        cuda_max << <grids, threads >> > (a_d, b_d);
        cudaFree(a_d);
        a_d = b_d;
        N /= 128;
    }
    b_h = (float*)malloc(N*sizeof(float));
    cudaMemcpy(b_h, a_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    float max_d = -1;
    for (int i = 0; i < N; ++i) 
    {
        if (b_h[i] > max_d)
        {
            max_d = b_h[i];
        }
    }
    std::cout << "cpu max " << max_h << std::endl;
    std::cout << "gpu max " << max_d << std::endl;
    
    cudaFree(a_d);
    free(a_h);
    free(b_h);
    return 0;
}