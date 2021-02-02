#include <iostream>
#include <math.h>
#include "Solver.h"
#include <stdexcept>
#include <vector>
#include "utilities.h"

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
template <class T>
void Solver<T>::lu_solve(std::vector<T> &x)
/*
LU decomposition
The input matrix A is copied to LU, which is modified 'in place'.
Uses Crout's method by setting L_ii = 1.
Partial pivoting is implemented to ensure the stability of the method.
Implicit pivoting used to make it independent of scaling of equations.
*/
{
    // Initialise lower and upper matrices
    int n, max_ind, i, j, k;
    n = A.rows;
    double max, temp;
    // Use copy constructor to copy A, LU will be modified 'in place'
    Matrix<T> LU(A);
    std::vector<T> y(N, 0);  // For forward substitution
    std::vector<T> perm_indx;  // Store index of permutation
    std::vector<T> scaling(N);  // Store implicit scaling of each row
    
    // Check our dimensions match
    checkDimensions(A, b);
    checkDimensions(A, x);

    // Implicit scaling, find max in each row and store scaling factor
    for (i = 0; i < n; i++)
    {
        max = 0.0;
        for (j = 0; i < n; j++)
        {
            temp = abs(LU[i * A.cols + j]);
            if (temp > max)
                max = temp;
        }
        if (max == 0)
            throw ("Matrix is singular");
        scaling[i] = 1.0/max;
    }

    // Perform LU decomposition
    // Inner LU loop resembles inner loop of matrix multiplication.
    // Uses kij permutation to loop over elements as fastest for 
    // row major storage and easiest to implement pivoting for.




    /*
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
    */

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
