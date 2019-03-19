#include <iostream>
#include <omp.h>
#include <utilities.hpp>

void classic_histogram(int imgSize, int *hist,unsigned char *img);
void parallel_histogram(int imgSize, int *hist, unsigned char *img, int*hist_thread,short no_threads);
void print_histogram(int *hist);

int main()
{
    // Alocarea de memorie pentru imagine
    int imgSize = 10000;
    unsigned char *img = new unsigned char[imgSize*imgSize];
    // Alocare de memorie pentru histogramă
    int *hist = new int[256];
    // Initializarea imaginii
    for (int i = 0; i < imgSize; ++i)
        for (int j = 0; j < imgSize; ++j)
            img[i*imgSize + j] = i % 256;

    // Inițializarea histogramei
    for (int i = 0; i < 256; ++i)
        hist[i] = 0;

    utilities::timeit([&]()
    {
        classic_histogram(imgSize, hist, img);
    });
    
    print_histogram(hist);

    // Inițializarea histogramei
    for (int i = 0; i < 256; ++i)
        hist[i] = 0;

    const short no_threads = 8;
    omp_set_num_threads(no_threads);
    std::cout << "working on " << no_threads << " threads." << std::endl;
    int *hist_thread = new int[256 * no_threads];
    for (int i = 0; i < 256 * no_threads; ++i)
    {
        hist_thread[i] = 0;
    }

    utilities::timeit([&]()
    {
        parallel_histogram(imgSize, hist, img, hist_thread, no_threads);
    });

    print_histogram(hist);
    
    // Dealocarea memoriei
    delete img;
    delete hist;

    return 0;
}

void classic_histogram(int imgSize, int *hist, unsigned char *img)
{
    // Calcului histogramei
    for (int i = 0; i < imgSize; ++i)
        for (int j = 0; j < imgSize; ++j)
            hist[img[i*imgSize + j]]++;
}

void parallel_histogram(int imgSize, int *hist, unsigned char *img, int*hist_thread, short no_threads)
{
    // Calcului histogramei
#pragma omp parallel for
    for (int i = 0; i < imgSize; ++i)
    {
        int current_thread = omp_get_thread_num();
        for (int j = 0; j < imgSize; ++j) 
        {
            hist_thread[current_thread * 256 + img[i * imgSize + j]]++;
        }
    }
    
    for (int i = 0; i < 256; ++i) 
    {
        for (int j = 0; j < no_threads; ++j) 
        {
            hist[i] += hist_thread[j * 256 + i];
        }
    }
}

void print_histogram(int *hist) 
{
    // Afișarea histogramei
    for (int i = 0; i < 256; i++)
        std::cout << hist[i] << " ";
    std::cout << std::endl;
}