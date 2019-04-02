#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

__global__ void median_filter(float *a,float *b, int N, int M) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < (N - 1) && col < (M - 1) && row > 0 && col > 0) 
    {
        float sum = 0.f;
        for (int i = row - 1; i < row + 1; ++i) 
        {
            for (int j = col - 1; j < col + 1; ++j)
            {
                sum += a[i*N + j];
            }
        }
        b[row * N + col] = sum/9.f;
    }
}

// main routine that executes on the host
int main()
{
    float *a_h, *a_d, *b_d;
    cv::Mat input_img,filtered_img;
    input_img = cv::imread("D:/dev/Programming/School/PPD/res/ex_sp.png", cv::IMREAD_GRAYSCALE);
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
    dim3 grid(32,32,1);
    dim3 thread(32,32,1);

    median_filter <<<grid, thread>>> (a_d,b_d,1000,1000);

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
    cv::waitKey(0);

    //cuda cleanup
    free(a_h);
    cudaFree(a_d);
    cudaFree(b_d);

    return 0;
}