#include <iostream>
#include <math.h>
#include "SparseSolver.h"
#include <stdexcept>
#include <vector>
#include "utilities.h"

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

template <class T>
T SparseSolver<T>::residualCalc(std::vector<T> &x, std::vector<T> &b_estimate)
{
    T residual = 0;
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
    double residual;
    std::vector<T> b_estimate(x.size(), 0);

    // vector for storing previous iteration if necessary
    std::vector<T> x_old;
    if (isGaussSeidel == false)
    {
        x_old = std::vector<T>(x.size(), 0);
    }

    // Check our dimensions match
    checkDimensions(A, b);
    checkDimensions(A, x);

    // Set values to zero before hand
    for (int i = 0; i < x.size(); i++)
    {
        x[i] = 0;
    }
    // loop up to a max number of iterations in case the solution doesn't converge
    int k;
    for (k = 0; k < it_max; k++)
    {
        // loop over rows
        for (int r = 0; r < A.rows; r++)
        {
            // Initialise sums of aij * xj
            T diagonal = 0;
            T sum = 0;
            // T sum2 = 0;

            // loop over non-zero values in row r
            for (int item_index = A.row_position[r]; item_index < A.row_position[r + 1]; item_index++)
            {
                int col_ind = A.col_index[item_index];
                if (r == col_ind)
                {
                    // this is a diagonal element
                    diagonal = A.values[item_index];
                }
                else if (!isGaussSeidel)
                {
                    sum += A.values[item_index] * x_old[col_ind];
                }
                else
                {
                    sum += A.values[item_index] * x[col_ind];
                }
            }

            x[r] = (1.0 / diagonal) * (b[r] - sum);
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

template <class T>
void SparseSolver<T>::conjugateGradient(std::vector<T> &x, double &tol, int &it_max)
{
    // TODO: check diagonal dominance
    // TODO: add more comments
    double residual;
    double alpha;
    double beta;
    std::vector<T> b_estimate(x.size(), 0);
    std::vector<T> residue_vec(x.size(), 0);
    std::vector<T> p(x.size(), 0);
    std::vector<T> Ap_product(x.size(), 0);
    std::vector<T> r_old(x.size(), 0);

    // Check our dimensions match
    checkDimensions(A, b);
    checkDimensions(A, x);

    // Set values to zero before hand
    for (int i = 0; i < x.size(); i++)
    {
        x[i] = 0;
    }

    // Calculate estimate of b
    // A.matVecMult(x, b_estimate);

    // Find the norm between old value and new guess
    for (int i = 0; i < x.size(); i++)
    {
        r_old[i] = b[i];
        p[i] = r_old[i];
    }

    int k;
    for (k = 0; k < it_max; k++)
    {
        A.matVecMult(p, Ap_product);

        // Calculate alpha gradient
        alpha = vecDotProduct(r_old, r_old) / vecDotProduct(p, Ap_product);

        residual = 0.0;
        for (int i = 0; i < x.size(); i++)
        {
            x[i] += alpha * p[i];
            residue_vec[i] = r_old[i] - alpha * Ap_product[i];
            residual += pow(residue_vec[i], 2.0);
        }

        residual = sqrt(residual);

        if (residual < tol)
        {
            break;
        }

        // Calculate beta gradient
        beta = vecDotProduct(residue_vec, residue_vec) / vecDotProduct(r_old, r_old);
        for (int i = 0; i < x.size(); i++)
        {
            p[i] = residue_vec[i] + beta * p[i];

            // Update "old"(k) residue vector with new residue (k+1)
            // for next iteration
            r_old[i] = residue_vec[i];
        }
    }
    std::cout << "k is :" << k << std::endl;
    std::cout << "residual is :" << residual << std::endl;
}