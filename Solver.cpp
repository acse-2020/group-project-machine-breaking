#include <iostream>
#include <math.h>
#include "Solver.h"

template <class T>
Solver<T>::Solver(Matrix<T> *A, Matrix<T> *b) : A(A), b(b)
{
}

// destructor

template <class T>
Solver<T>::~Solver()
{
}

template <class T>
void Solver<T>::stationaryIterative(Matrix<T> &x, double &tol, int &it_max, bool isGaussSeidel)
{
    // TODO: check diagonal dominance
    // TODO: add more comments

    // Initialise residual, matrix for row-matrix multiplication
    // and matrix for storing previous iteration
    double residual;
    Matrix<T> new_array(x.rows, x.cols, true);
    Matrix<T> x_old(x.rows, x.cols, true);

    // Check our dimensions match
    if (this->A->cols != this->b->rows)
    {
        std::cerr << "Input dimensions for matrices don't match" << std::endl;
        return;
    }

    // Check if our output matrix has had space allocated to it
    if (x.values != nullptr)
    {
        // Check our dimensions match
        if (this->A->rows != x.rows || this->b->cols != x.cols)
        {
            std::cerr << "Input dimensions for matrices don't match" << std::endl;
            return;
        }
    }

    // The output hasn't been preallocated, so we are going to do that
    else
    {
        x.values = new T[this->A->rows * this->b->cols];
        x.preallocated = true;
    }

    // Set values to zero before hand
    for (int i = 0; i < x.size_of_values; i++)
    {
        x.values[i] = 0;
    }
    int k;
    for (k = 0; k < it_max; k++)
    {
        for (int i = 0; i < this->A->rows; i++)
        {
            // Initialise sums of aij * xj
            double sum = 0;
            double sum2 = 0;
            for (int j = 0; j < this->A->rows; j++)
            {
                if (j < i)
                {
                    sum += this->A->values[i * this->A->cols + j] * x.values[j];
                }
                else if (j > i && isGaussSeidel)
                {
                    sum2 += this->A->values[i * this->A->cols + j] * x.values[j];
                }
            }
            x.values[i] = (1.0 / this->A->values[i + i * this->A->rows]) * (this->b->values[i] - sum - sum2);
        }
        // A x = b(estimate)
        this->A->matMatMult(x, new_array);

        // Find the norm between old value and new guess
        residual = 0;
        for (int i = 0; i < this->A->rows; i++)
        {
            residual += pow(new_array.values[i] - this->b->values[i], 2.0);
        }
        residual = sqrt(residual);

        // End iterations if tolerance convergence is reached
        if (residual < tol)
        {
            break;
        }
    }
    std::cout << "k is :" << k << std::endl;
    std::cout << "residual is :" << residual << std::endl;
    for (int i = 0; i < this->A->rows; i++)
    {
        std::cout << "new_array value :" << new_array.values[i];
    }
    std::cout << std::endl;
}

// LU decomposition method to solve linear system of equations (Ax=b)
// Based on algorithm provided in Lecture 3 of ACSE3
template <class T>
void Solver<T>::lu_solve(Matrix<T> &x)
{
    // Initialise lower and upper matrices
    int N = A->rows;
    double sum;
    Matrix<T> L(N, A->cols, true);
    Matrix<T> U(N, A->cols, true);
    Matrix<T> y(N, 1, true);

    // Check if square matrix
    if (A->cols != A->rows)
    {
        std::cerr << "Only implemented for square matrix" << std::endl;
        return;
    }

    // Check our dimensions match
    if (A->cols != b->rows)
    {
        std::cerr << "Input dimensions for matrices don't match" << std::endl;
        return;
    }

    // Check if our output matrix has had space allocated to it
    if (x.values != nullptr)
    {
        // Check our dimensions match
        if (A->rows != x.rows || b->cols != x.cols)
        {
            std::cerr << "Input dimensions for matrices don't match" << std::endl;
            return;
        }
    }

    // The output hasn't been preallocated, so we are going to do that
    else
    {
        x.values = new T[A->rows * b->cols];
        x.preallocated = true;
    }

    // Initialise L and U matrix.
    // Set L_ii = 1 (Crout's method for LU decomp)
    // Look into avoiding to copy U to save memory
    for (int i = 0; i < A->size_of_values; i++)
    {
        if (i % (L.cols + 1) == 0)
        {
            L.values[i] = 1;
        }
        else
        {
            L.values[i] = 0;
        }
        U.values[i] = A->values[i];
    }

    // Perform LU decomposition
    for (int k = 0; k < N - 1; k++)
    {
        for (int i = k + 1; i < N; i++)
        {
            T s = U.values[i * U.cols + k] / U.values[k * U.cols + k];
            for (int j = k; j < N; j++)
            {
                U.values[i * U.cols + j] = U.values[i * U.cols + j] - s * U.values[k * U.cols + j];
            }
            L.values[i * L.cols + k] = s;
        }
    }

    // Now we solve the equations L*y = b and U*x = y to find x x
    // We don't need to initialise x and y as we only use values
    // already set in the substitution, step-wise

    // Perform forward substitution to solve L*y = b
    for (int k = 0; k < N; k++)
    {
        sum = 0;
        for (int j = 0; j < k; j++)
        {
            sum += L.values[k * L.cols + j] * y.values[j];
        }
        y.values[k] = (b->values[k] - sum) / L.values[k * L.cols + k];
    }

    // Perform backward substitution to solve U*x = y
    for (int k = N - 1; k > -1; k--)
    {
        sum = 0;
        for (int j = k + 1; j < N; j++)
        {
            sum += U.values[k * U.cols + j] * x.values[j];
        }
        x.values[k] = (y.values[k] - sum) / U.values[k * U.cols + k];
    }
}
