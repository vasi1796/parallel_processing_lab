#include <iostream>
#include <omp.h>
#include <utilities.hpp>

void assign_matrix(int** mat, unsigned int rows,unsigned int cols);
void assign_vector(int* vec, unsigned int rows,bool is_result_vec);
void classic_mul(int* vec, int **mat, int* res_vec, unsigned int rows, unsigned int cols);
void parallel_mul(int* vec, int **mat, int* res_vec, unsigned int rows, unsigned int cols);
void print_mat(int** mat, unsigned int rows, unsigned int cols);
void print_vec(int* vec, unsigned int rows);

int main() 
{
    const unsigned int rows = 10'000;
    const unsigned int cols = rows;

    //find number of threads fit for rows, should be improved
    const short no_threads = 8;
    omp_set_num_threads(no_threads);
    std::cout << "working on " << no_threads << " threads."<< std::endl;
    
    //create matrix
    int** matrix = new int*[rows];
    for (int i = 0; i < rows; ++i) 
    {
        matrix[i] = new int[cols];
    }

    //create vector and result
    int* vector = new int[rows];
    int* result = new int[rows];

    //assign values to created matrix/vectors
    assign_matrix(matrix,rows,cols);
    assign_vector(vector, rows, false);
    assign_vector(result, rows, true);

    //begin classic mul
    utilities::timeit([&]()
    {
        classic_mul(vector, matrix, result, rows, cols);
    });

    //reset result vector
    assign_vector(result, rows, true);

    //begin parallel mul
    utilities::timeit([&]()
    {
        parallel_mul(vector, matrix, result, rows, cols);
    });

    //print only for small sizes
    if (rows < 10) 
    {
        print_mat(matrix, rows, cols);
        print_vec(vector, rows);
        print_vec(result, rows);
    }

    //deallocate memory
    for (int i = 0; i < rows; ++i) 
    {
        delete[] matrix[i];
    } 
    delete[] matrix;
    delete[] vector; 
    delete[] result;

    return 0;
}

void assign_matrix(int** mat, unsigned int rows, unsigned int cols)
{
#pragma omp parallel for
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                mat[i][j] = rand() % 100 + i*j;
            }
        }
    //// assign values to allocated matrix, sequential
    //for (int i = 0; i < rows; i++)
    //{
    //    for (int j = 0; j < cols; j++)
    //    {
    //        mat[i][j] = rand() % 100;
    //    }

    //}
}

void assign_vector(int* vec, unsigned int rows, bool is_result_vec)
{
#pragma omp parallel for
        for (int i = 0; i < rows; ++i)
        {
            if (is_result_vec)
            {
                vec[i] = 0;
            }
            else
            {
                vec[i] = rand() % 100 + i;
            }

        }
    ////assign values to allocated vector, sequential
    //for (int i = 0; i < rows; i++)
    //{
    //    if (is_result_vec) 
    //    {
    //        vec[i] = 0;
    //    }
    //    else 
    //    {
    //        vec[i] = rand() % 100;
    //    }
    //    
    //}
}

void classic_mul(int* vec, int **mat, int* res_vec, unsigned int rows, unsigned int cols)
{
    //perform classic multiplication
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            res_vec[i] += mat[i][j] * vec[j];
        }
    }
}

void parallel_mul(int* vec, int **mat, int* res_vec, unsigned int rows, unsigned int cols)
{
    //multiplication on threads
#pragma omp parallel for
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                res_vec[i] += mat[i][j] * vec[j];
            }
        }
}

void print_mat(int** mat, unsigned int rows, unsigned int cols)
{
    std::cout << "matrix: " << std::endl;
    // print the matrix
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << mat[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void print_vec(int* vec, unsigned int rows)
{
    std::cout << "vector: " << std::endl;
    // print the vector
    for (int i = 0; i < rows; ++i)
    {
        std::cout << vec[i] << std::endl;
    }
}