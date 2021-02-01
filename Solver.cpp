#include <iostream>
#include <math.h>
#include "Solver.h"
#include <stdexcept>
#include <vector>

template <class T>
Solver<T>::Solver(Matrix<T> &A, std::vector<T> &b) : A(A), b(b)
{
    // Check our dimensions match
    if (A.cols != b.size())
    {
        std::cerr << "Input dimensions for matrices don't match" << std::endl;
        return;
    }
}

// destructor

template <class T>
Solver<T>::~Solver()
{
}

template <class T>
void Solver<T>::checkDimensions(Matrix<T> &M1, std::vector<T> &vec)
{
    if (A.cols != vec.size())
    {
        throw std::invalid_argument("Dimensions don't match");
    }
}

template <class T>
double Solver<T>::residualCalc(std::vector<T> &x, std::vector<T> &b_estimate)
{
    double residual = 0;
    // A x = b(estimate)
    A.matVecMult(x, b_estimate);

    // Find the norm between old value and new guess
    for (int i = 0; i < A.rows; i++)
    {
        residual += pow(b_estimate[i] - b[i], 2.0);
    }
    return sqrt(residual);
}

template <class T>
void Solver<T>::stationaryIterative(std::vector<T> &x, double &tol, int &it_max, bool isGaussSeidel)
{
    // TODO: check diagonal dominance
    // TODO: add more comments

    // Initialise matrix for row-matrix multiplication
    // and matrix for storing previous iteration
    double residual;
    std::vector<T> b_estimate(x.size(), 0);
    std::vector<T> x_old(x.size(), 0);

    // Check our dimensions match
    checkDimensions(A, b);
    checkDimensions(A, x);

    // Set values to zero before hand
    for (int i = 0; i < x.size(); i++)
    {
        x[i] = 0;
    }
    int k;
    for (k = 0; k < it_max; k++)
    {
        for (int i = 0; i < A.rows; i++)
        {
            // Initialise sums of aij * xj
            double sum = 0;
            double sum2 = 0;
            for (int j = 0; j < A.rows; j++)
            {
                if (j != i && isGaussSeidel == false)
                {
                    sum += A.values[i * A.cols + j] * x_old[j];
                }
                else if (j < i && isGaussSeidel)
                {
                    sum += A.values[i * A.cols + j] * x[j];
                }
                else if (j > i && isGaussSeidel)
                {
                    sum2 += A.values[i * A.cols + j] * x[j];
                }
            }
            x[i] = (1.0 / A.values[i + i * A.rows]) * (b[i] - sum - sum2);
        }

        // Call residual calculation method
        residual = residualCalc(x, b_estimate);
        // // A x = b(estimate)
        // A.matMatMult(x, b_estimate);

        // // Find the norm between old value and new guess
        // residual = 0;
        // for (int i = 0; i < A.rows; i++)
        // {
        //     residual += pow(b_estimate[i] - b[i], 2.0);
        // }
        // residual = sqrt(residual);

        // End iterations if tolerance convergence is reached
        if (residual < tol)
        {
            break;
        }
        if (isGaussSeidel == false)
        {
            // Update the solution from previous iteration with
            // new estimate for Jacobi
            for (int i = 0; i < x.size(); i++)
            {
                x_old[i] = x[i];
            }
        }
    }
    std::cout << "k is :" << k << std::endl;
    std::cout << "residual is :" << residual << std::endl;
}

// LU decomposition method to solve linear system of equations (Ax=b)
// Based on algorithm provided in Lecture 3 of ACSE3
template <class T>
void Solver<T>::lu_solve(std::vector<T> &x)
{
    // Initialise lower and upper matrices
    int N = A.rows;
    double sum;
    Matrix<T> L(N, A.cols, true);
    Matrix<T> U(N, A.cols, true);
    std::vector<T> y(N, 0);

    // Check if square matrix
    if (A.cols != A.rows)
    {
        std::cerr << "Only implemented for square matrix" << std::endl;
        return;
    }

    // Check our dimensions match
    checkDimensions(A, b);
    checkDimensions(A, x);

    // Initialise L and U matrix.
    // Set L_ii = 1 (Crout's method for LU decomp)
    // Look into avoiding to copy U to save memory
    for (int i = 0; i < A.size_of_values; i++)
    {
        if (i % (L.cols + 1) == 0)
        {
            L.values[i] = 1;
        }
        else
        {
            L.values[i] = 0;
        }
        U.values[i] = A.values[i];
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
            sum += L.values[k * L.cols + j] * y[j];
        }
        y[k] = (b[k] - sum) / L.values[k * L.cols + k];
    }

    // Perform backward substitution to solve U*x = y
    for (int k = N - 1; k > -1; k--)
    {
        sum = 0;
        for (int j = k + 1; j < N; j++)
        {
            sum += U.values[k * U.cols + j] * x[j];
        }
        x[k] = (y[k] - sum) / U.values[k * U.cols + k];
    }
}
