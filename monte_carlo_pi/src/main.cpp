#include <iostream>
#include <omp.h>
#include <chrono>

float get_random_number(float a, float b);
float classic_monte_carlo(unsigned int numbers, float a, float b, float radius);
float parallel_monte_carlo(unsigned int numbers, float a, float b, float radius);
float man_parallel_monte_carlo(unsigned int numbers, float a, float b, float radius);

int main() 
{
    const unsigned int numbers = 10'000'000;

    //find number of threads fit for rows, should be improved
    const short no_threads = 8;
    omp_set_num_threads(no_threads);
    std::cout << "working on " << no_threads << " threads."<< std::endl;
    
    const float a = -1;
    const float b = 1;
    const float cx = 0.0f;
    const float cy = 0.0f;
    const float radius = 1;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    float pi = classic_monte_carlo(numbers, a, b, radius);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Sequential time = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;
    std::cout << "classic pi = " << pi << std::endl;

    begin = std::chrono::steady_clock::now();
    pi = parallel_monte_carlo(numbers, a, b, radius);
    end = std::chrono::steady_clock::now();
    std::cout << "Parallel time = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;
    std::cout << "parallel pi = " << pi << std::endl;

    return 0;
}

float get_random_number(float a, float b) 
{
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

float classic_monte_carlo(unsigned int numbers,float a,float b,float radius) 
{
    float counter = 0;

    for (int i = 0; i < numbers; i++)
    {
        float rand_x = get_random_number(a, b);
        float rand_y = get_random_number(a, b);
        float distance = sqrt((double)(rand_x - 0.0)*(rand_x - 0.0) + (rand_y - 0.0)*(rand_y - 0.0));
        if (distance < radius)
        {
            counter+=1;
        }
    }
    return 4 * (counter / (float)(numbers));
}

float parallel_monte_carlo(unsigned int numbers, float a, float b, float radius) 
{
    float counter = 0;

#pragma omp parallel for reduction(+:counter)
    for (int i = 0; i < numbers; i++)
    {
        float rand_x = get_random_number(a, b);
        float rand_y = get_random_number(a, b);
        float distance = sqrt((double)(rand_x - 0.0)*(rand_x - 0.0) + (rand_y - 0.0)*(rand_y - 0.0));
        if (distance < radius)
        {
            counter+=1;
        }
    }
    return 4 * (counter / (float)(numbers));
}