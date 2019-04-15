#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <utilities.hpp>

__global__ void cuda_max(float* a_d, float* out, int size)
{
    __shared__ float a_sh[128];
    const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= size)
    {
        return;
    }
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
    size_t size = N * sizeof(float);

    a_h = (float*)malloc(size);
    b_h = (float*)malloc(size);
    cudaMalloc((void**)&a_d, size);
    cudaMalloc((void**)&b_d, size);

    //dimensiuni grid si threads
    const int thread = 128;

    const int grid = N / thread;

    dim3 grids(grid, 1, 1);
    dim3 threads(thread, 1, 1);

    for (int i = 0; i < N; ++i)
    {
        a_h[i] = 1 + (rand() % (i + 1));
    }
    float max = -1;

    for (int i = 0; i < N; ++i)
    {
        if (a_h[i] > max)
        {
            max = a_h[i];
        }
    }

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    while(N>grids.x)
    {
        cudaMalloc((void**)&b_d, N / 128 * sizeof(float));
        dim3 grids(N / 128);
        cuda_max << <grids, threads >> > (a_d, b_d, N);
        cudaFree(a_d);
        a_d = b_d;
        //cudaMemcpy(a_d, b_d, N*sizeof(float), cudaMemcpyDeviceToDevice);
        N /= 128;
    }
    float *c_h;
    c_h = (float*)malloc(N*sizeof(float));
    cudaMemcpy(c_h, a_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    float maxim = -1;
    for (int i = 0; i < N; ++i) 
    {
        if (c_h[i] > maxim)
        {
            maxim = c_h[i];
        }
    }
    /*cuda_max << <grids, threads >> > (a_d, b_d, N);
    cudaMemcpy(b_h, b_d, size, cudaMemcpyDeviceToHost);*/
    std::cout << "cpu max " << max << std::endl;
    std::cout << "gpu max " << maxim << std::endl;
    
    cudaFree(a_d);
    //cudaFree(b_d);
    free(a_h);
    free(b_h);
    free(c_h);
    return 0;
}