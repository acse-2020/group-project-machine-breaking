#include <iostream>
#include <math.h>
#include "SparseSolver.h"
#include <stdexcept>
#include <vector>

template <class T>
SparseSolver<T>::SparseSolver(CSRMatrix<T> &A, std::vector<T> &b) : A(A), b(b)
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
SparseSolver<T>::~SparseSolver()
{
}

// TODO: move this to utilities?
template <class T>
void SparseSolver<T>::checkDimensions(CSRMatrix<T> &M1, std::vector<T> &vec)
{
    if (A.cols != vec.size())
    {
        throw std::invalid_argument("Dimensions don't match");
    }
}

// TODO: move this to utilities?
template <class T>
double SparseSolver<T>::residualCalc(std::vector<T> &x, std::vector<T> &b_estimate)
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

// NOTE: this is currently only implemented for isGaussSeidel = false
template <class T>
void SparseSolver<T>::stationaryIterative(std::vector<T> &x, double &tol, int &it_max, bool isGaussSeidel)
{
    // TODO: check diagonal dominance
    // TODO: add more comments

    if (isGaussSeidel)
    {
        throw "Sparse Gauss-Seidel is not yet implemented.";
    }

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

            // loop over rows
            for (int r = 0; r < A.rows; r++)
            {
                T diagonal = 0;
                T sum = 0;
                // T sum2 = 0;

                // number of iterations == number of non-zero items in that row
                for (int item_index = A.row_position[r]; item_index < A.row_position[r + 1]; item_index++)
                {
                    int col_ind = A.col_index[item_index];
                    if (col_ind == r)
                    {
                        // this is a diagonal element
                        diagonal = A.values[item_index];
                    }
                    else
                    {
                        sum += A.values[item_index] * x_old[col_ind];
                    }
                }

                x[r] = (1.0 / diagonal) * (b[r] - sum);
            }
        }

        // Call residual calculation method
        residual = residualCalc(x, b_estimate);

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