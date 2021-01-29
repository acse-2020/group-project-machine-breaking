#include <iostream>
#include <math.h>
#include "Solver.h"

template <class T>
Solver<T>::Solver(Matrix<T> *LHS, Matrix<T> *RHS) : LHS(LHS), RHS(RHS)
{
}

// destructor

template <class T>
Solver<T>::~Solver()
{
}

// Jacobi method to solve linear system of equations (Ax=b)
// Based on algorithm provided in Lecture 3 of ACSE3
template <class T>
void Solver<T>::jacobi(Matrix<T> &unknowns, double &tol, int &it_max)
{
    // Initialise residual, matrix for row-matrix multiplication
    // and matrix for storing previous iteration
    double residual;
    Matrix<T> new_array(unknowns.rows, unknowns.cols, true);
    Matrix<T> x_old(unknowns.rows, unknowns.cols, true);

    // Check our dimensions match
    if (this->LHS->cols != this->RHS->rows)
    {
        std::cerr << "Input dimensions for matrices don't match" << std::endl;
        return;
    }

    // Check if our output matrix has had space allocated to it
    if (unknowns.values != nullptr)
    {
        // Check our dimensions match
        if (this->LHS->rows != unknowns.rows || this->RHS->cols != unknowns.cols)
        {
            std::cerr << "Input dimensions for matrices don't match" << std::endl;
            return;
        }
    }

    // The output hasn't been preallocated, so we are going to do that
    else
    {
        unknowns.values = new T[this->LHS->rows * this->RHS->cols];
        unknowns.preallocated = true;
    }

    // Set values to zero before hand
    for (int i = 0; i < unknowns.size_of_values; i++)
    {
        unknowns.values[i] = 0;
        x_old.values[i] = unknowns.values[i];
    }

    int k;
    for (k = 0; k < it_max; k++)
    {
        for (int i = 0; i < this->LHS->rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < this->LHS->rows; j++)
            {
                if (j != i)
                {
                    sum += this->LHS->values[i * this->LHS->cols + j] * x_old.values[j];
                }
            }
            unknowns.values[i] = (1.0 / this->LHS->values[i + i * this->LHS->rows]) * (this->RHS->values[i] - sum);
            //(1. / A[i, i]) * (b[i] - (A [i, :i] @x[:i]) - (A [i, i + 1:] @x [i + 1:]))
        }

        // COULD JUST HAVE new_array AS VECTOR. REVISIT
        // A x = b(estimate)
        this->LHS->matMatMult(unknowns, new_array);

        // Find the norm between old value and new guess
        residual = 0;
        for (int i = 0; i < this->LHS->rows; i++)
        {
            residual += abs(pow(new_array.values[i] - this->RHS->values[i], 2));
        }
        residual = sqrt(residual);

        // End iterations if tolerance convergence is reached
        if (residual < tol)
        {
            break;
        }

        // Update the solution from previous iteration with new estimate
        for (int i = 0; i < unknowns.size_of_values; i++)
        {
            x_old.values[i] = unknowns.values[i];
        }
    }
    std::cout << "Final value of k in Jacobi:" << k << std::endl;
    std::cout << "Residual is " << residual << std::endl;
}

template <class T>
void Solver<T>::gaussSeidel(Matrix<T> &unknowns, double &tol, int &it_max)
{
    // TODO: check diagonal dominance
    // TODO: add more comments

    // Initialise residual, matrix for row-matrix multiplication
    // and matrix for storing previous iteration
    double residual;
    Matrix<T> new_array(unknowns.rows, unknowns.cols, true);
    Matrix<T> x_old(unknowns.rows, unknowns.cols, true);

    // Check our dimensions match
    if (this->LHS->cols != this->RHS->rows)
    {
        std::cerr << "Input dimensions for matrices don't match" << std::endl;
        return;
    }

    // Check if our output matrix has had space allocated to it
    if (unknowns.values != nullptr)
    {
        // Check our dimensions match
        if (this->LHS->rows != unknowns.rows || this->RHS->cols != unknowns.cols)
        {
            std::cerr << "Input dimensions for matrices don't match" << std::endl;
            return;
        }
    }

    // The output hasn't been preallocated, so we are going to do that
    else
    {
        unknowns.values = new T[this->LHS->rows * this->RHS->cols];
        unknowns.preallocated = true;
    }

    // Set values to zero before hand
    for (int i = 0; i < unknowns.size_of_values; i++)
    {
        unknowns.values[i] = 0;
    }
    int k;
    for (k = 0; k < it_max; k++)
    {
        for (int i = 0; i < this->LHS->rows; i++)
        {
            // Initialise sums of aij * xj
            double sum = 0;
            double sum2 = 0;
            for (int j = 0; j < this->LHS->rows; j++)
            {
                if (j < i)
                {
                    sum += this->LHS->values[i * this->LHS->cols + j] * unknowns.values[j];
                }
                else if (j > i)
                {
                    sum2 += this->LHS->values[i * this->LHS->cols + j] * unknowns.values[j];
                }
            }
            unknowns.values[i] = (1.0 / this->LHS->values[i + i * this->LHS->rows]) * (this->RHS->values[i] - sum - sum2);
        }
        // A x = b(estimate)
        this->LHS->matMatMult(unknowns, new_array);

        // Find the norm between old value and new guess
        residual = 0;
        for (int i = 0; i < this->LHS->rows; i++)
        {
            residual += pow(new_array.values[i] - this->RHS->values[i], 2.0);
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
    for (int i = 0; i < this->LHS->rows; i++)
    {
        std::cout << "new_array value :" << new_array.values[i];
    }
    std::cout << std::endl;
}

// LU decomposition method to solve linear system of equations (Ax=b)
// Based on algorithm provided in Lecture 3 of ACSE3
template <class T>
void Solver<T>::lu_solve(Matrix<T> &unknowns, double &tol, int &it_max)
{
    // Initialise lower and upper matrices
    int N = LHS->rows;
    double sum;
    Matrix<T> L(N, LHS->cols, true);
    Matrix<T> U(N, LHS->cols, true);
    Matrix<T> y(N, 1, true);

    // Check if square matrix
    if (LHS->cols != LHS->rows)
    {
        std::cerr << "Only implemented for square matrix" << std::endl;
        return;
    }

    // Check our dimensions match
    if (LHS->cols != RHS->rows)
    {
        std::cerr << "Input dimensions for matrices don't match" << std::endl;
        return;
    }

    // Check if our output matrix has had space allocated to it
    if (unknowns.values != nullptr)
    {
        // Check our dimensions match
        if (LHS->rows != unknowns.rows || RHS->cols != unknowns.cols)
        {
            std::cerr << "Input dimensions for matrices don't match" << std::endl;
            return;
        }
    }

    // The output hasn't been preallocated, so we are going to do that
    else
    {
        unknowns.values = new T[LHS->rows * RHS->cols];
        unknowns.preallocated = true;
    }

    // Initialise L and U matrix.
    // Set L_ii = 1 (Crout's method for LU decomp)
    // Look into avoiding to copy U to save memory
    for (int i = 0; i < LHS->size_of_values; i++)
    {
        if (i % (L.cols + 1) == 0) 
        {
            L.values[i] = 1;
        }
        else 
        {
            L.values[i] = 0;
        }
        U.values[i] = LHS->values[i];
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

    // Now we solve the equations L*y = b and U*x = y to find unknowns x
    // We don't need to initialise unknowns and y as we only use values
    // already set in the substitution, step-wise

    // Perform forward substitution to solve L*y = b
    for (int k = 0; k < N; k++)
    {
        sum = 0;
        for (int j = 0; j < k; j++)
        {
            sum += L.values[k * L.cols + j] * y.values[j];
        }
        y.values[k] = (RHS->values[k] - sum) / L.values[k * L.cols + k];
    }

    // Perform backward substitution to solve U*x = y
    for (int k = N-1; k > -1; k--)
    {
        sum = 0;
        for (int j = k + 1; j < N; j++)
        {
            sum += U.values[k * U.cols + j] * unknowns.values[j];
        }
        unknowns.values[k] = (y.values[k] - sum) / U.values[k * U.cols + k];
    }
}
