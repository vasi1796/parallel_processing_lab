﻿#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <utilities.hpp>

__global__ void median_filter(float *a,float *b, int N, int M, int win_size) 
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int factor = win_size / 2;
    const float elems = win_size * win_size;
    if (row < (N - factor) && col < (M - factor) && row >= factor && col >= factor)
    {
        float sum = 0.f;
        for (int i = row - factor; i <= row + factor; ++i)
        {
            for (int j = col - factor; j <= col + factor; ++j)
            {
                sum += a[i*N + j];
            }
        }
        b[row * N + col] = sum/ elems;
    }
}

__global__ void shared_median_filter(float *a, float *b, int N, int M, int win_size)
{
    __shared__ float img_copy[32][32];
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int grid_i = threadIdx.x;
    const int grid_j = threadIdx.y;
    const int factor = win_size / 2;
    const float elems = win_size * win_size;

    if (row < N && col < M)
    {
        img_copy[grid_i][grid_j] = a[row*N + col];
        __syncthreads();
        if (grid_i < (32 - factor) && grid_j < (32 - factor) && grid_i >= factor && grid_j >= factor)
        {
            float sum = 0.f;
            for (int i = grid_i - factor; i <= grid_i + factor; ++i)
            {
                for (int j = grid_j - factor; j <= grid_j + factor; ++j)
                {
                    sum += img_copy[i][j];
                }
            }
            b[row * N + col] = sum / elems;
        }
    }
}

// main routine that executes on the host
int main(int argc, char* argv[])
{
    int filter_size;
    cv::Mat input_img, filtered_img;

    if (argc > 2) 
    {
        int size = atoi(argv[1]);
        if (size % 2) 
        {
            filter_size = size;
        }
        input_img = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    }
    else
    {
        input_img = cv::imread("../../res/ex_sp.png", cv::IMREAD_GRAYSCALE);
        filter_size = 3;
    }
    
    float *a_h, *a_d, *b_d;
    cv::imshow("image unfiltered", input_img);
    input_img.convertTo(input_img, CV_32FC1);
    filtered_img.create(cv::Size(input_img.rows, input_img.cols), CV_32FC1);
    
    size_t size = input_img.rows * input_img.cols * sizeof(float);
    
    //alocare host
    a_h = (float*)malloc(size);
    for (int i = 0; i < input_img.rows; ++i) 
    {
        for (int j = 0; j < input_img.cols; ++j) 
        {
            a_h[i*input_img.rows + j] = input_img.at<float>(i,j);
        }
    }
    
    //alocare device
    cudaMalloc((void**)&a_d, size);
    cudaMalloc((void**)&b_d, size);

    //copiere date pe device
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

    //dimensiuni grid si threads
    const int thread_x = 32;
    const int thread_y = 32;
    const int grid_x = input_img.rows / thread_x + input_img.rows % thread_x;
    const int grid_y = input_img.cols / thread_y + input_img.cols % thread_y;
    dim3 grid(grid_x,grid_y,1);
    dim3 thread(thread_y, thread_y,1);

    //utilities::timeit([&] {median_filter <<< grid, thread >>> (a_d, b_d, input_img.rows, input_img.cols, filter_size); });
    utilities::timeit([&] {shared_median_filter << < grid, thread >> > (a_d, b_d, input_img.rows, input_img.cols, filter_size); });

    //copiere data pe host
    cudaMemcpy(a_h, b_d, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < input_img.rows; ++i)
    {
        for (int j = 0; j < input_img.cols; ++j)
        {
            filtered_img.at<float>(i, j) = a_h[i*input_img.rows + j];
        }
    }

    filtered_img.convertTo(filtered_img, CV_8UC1);
    cv::imshow("image filtered", filtered_img);
    std::stringstream sstream;
    sstream << "filter_" << filter_size << ".jpg";
    cv::imwrite(sstream.str(), filtered_img);
    cv::waitKey(0);

    //cuda cleanup
    free(a_h);
    cudaFree(a_d);
    cudaFree(b_d);

    return 0;
}