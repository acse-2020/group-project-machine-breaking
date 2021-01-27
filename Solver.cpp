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

    for (int k = 0; k < it_max; k++)
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
            residual += pow(abs(new_array.values[i] - this->RHS->values[i]), 2);
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

    for (int k = 0; k < it_max; k++)
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

            // A x = b(estimate)
            this->LHS->matMatMult(unknowns, new_array);

            // Find the norm between old value and new guess
            residual = 0;
            for (int i = 0; i < this->LHS->rows; i++)
            {
                residual += pow(abs(new_array.values[i] - this->RHS->values[i]), 2);
            }
            residual = sqrt(residual);

            // End iterations if tolerance convergence is reached
            if (residual < tol)
            {
                break;
            }
        }
    }
}
