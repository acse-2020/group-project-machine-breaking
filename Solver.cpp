#include <iostream>
#include <math.h>
#include "Solver.h"
#include "Matrix.h"
#include <stdexcept>
#include <vector>
#include <memory>
#include <random>
#include "utilities.h"

// Constructor
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

// Constructor - creates a random matrix
template <class T>
Solver<T>::Solver(int size) : size(size)
{
    // retuns a pointer that needs to be deleted
    // create random diagonally dominant matrices
    A = Matrix<T>(size, size, true);
    b.reserve(size);

    for (int i = 0; i < size; i++)
    {
        b.push_back(T(rand() % 10 + 1));
        for (int j = 0; j < size; j++)
        {
            if (i == j)
            {
                A.values[i * size + j] = T(rand() % 10 + 10);
            }
            else
            {
                A.values[i * size + j] = T(rand() % 10);
            }
        }
    }
}

// Copy constructor
template <class T>
Solver<T>::Solver(const Solver<T> &S2)
{
    A = S2.A;                    // Assignment operator overloaded for Matrix to deepcopy
    std::vector<T> btemp = S2.b; // vector comes with copy constructor
    b = btemp;
}

// destructor
template <class T>
Solver<T>::~Solver()
{
}

template <class T>
T Solver<T>::residualCalc(std::vector<T> &x, std::vector<T> &output_b)
{
    T residual = 0;

    A.matVecMult(x, output_b);

    // Find the norm between old value and new guess
    for (int i = 0; i < A.rows; i++)
    {
        residual += pow(output_b[i] - b[i], 2.0);
    }
    return sqrt(residual);
}

// Jacobi and Gauss-Seidel iterative solvers
template <class T>
void Solver<T>::stationaryIterative(std::vector<T> &x, double &tol, int &it_max, bool isGaussSeidel)
{
    T residual;
    T sum;

    // Initialise vector for row-matrix multiplication
    std::vector<T> output_b(x.size(), 0);

    // vector for storing previous iteration if necessary
    std::vector<T> x_old;
    if (isGaussSeidel == false)
    {
        x_old = std::vector<T>(x.size(), 0);
    }

    // Check our dimensions match
    checkDimensions(A, b);
    checkDimensions(A, x);

    // Set values to zero beforehand
    for (int i = 0; i < x.size(); i++)
    {
        x[i] = 0;
    }

    // declare k beforehand so it is available outside of the for loop scope
    int k;
    for (k = 0; k < it_max; k++)
    {
        for (int i = 0; i < A.rows; i++)
        {
            // sums of aij * xj
            sum = 0;

            for (int j = 0; j < A.cols; j++)
            {
                if (j != i && isGaussSeidel == false)
                {
                    sum += A.values[i * A.cols + j] * x_old[j];
                }
                else if (j != i)
                {
                    sum += A.values[i * A.cols + j] * x[j];
                }
            }
            x[i] = (1.0 / A.values[i + i * A.rows]) * (b[i] - sum);
        }

        // Call residual calculation method
        residual = residualCalc(x, output_b);

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

// LU decomposition
template <class T>
std::vector<int> Solver<T>::lu_decomp(Matrix<T> &LU)
/*
LU decomposition
Algorithm based on similar method as in 'Numerical recipes C++'.
The input matrix A is copied to LU, which is modified 'in place'.
Uses Crout's method by setting U_ii = 1.
Partial pivoting is implemented to ensure the stability of the method.
Implicit pivoting used to make it independent of scaling of equations.
*/
{
    int n, max_ind, i, j, k;
    n = A.rows;
    T max, temp;

    checkDimensions(A, b);

    std::vector<int> perm_indx(n); // Store index of permutation
    std::vector<T> scaling(n);     // Store implicit scaling of each row

    // Copy values into LU, want to do this with a copy constryctor later
    for (int i = 0; i < A.size_of_values; i++)
    {
        LU.values[i] = A.values[i];
    }

    // Implicit scaling, find max in each row and store scaling factor
    for (i = 0; i < n; i++)
    {
        max = 0.0;
        for (j = 0; j < n; j++)
        {
            temp = abs(LU.values[i * LU.cols + j]);
            if (temp > max)
                max = temp;
        }
        if (max == 0)
            throw std::invalid_argument("Matrix is singular");
        scaling[i] = 1.0 / max;
    }

    // Perform LU decomposition
    // Inner LU loop resembles inner loop of matrix multiplication.
    // Uses kij permutation to loop over elements as fastest for
    // row major storage and easiest to implement pivoting for.
    for (k = 0; k < n; k++)
    {
        max = 0.0;
        for (i = k; i < n; i++)
        {
            temp = scaling[i] * abs(LU.values[i * LU.cols + k]);
            // Store best pivot row so far
            if (temp > max)
            {
                max = temp;
                max_ind = i;
            }
        }
        // If k not best pivot row, swap rows
        if (k != max_ind)
        {
            for (j = 0; j < n; j++)
            {
                temp = LU.values[max_ind * LU.cols + j];
                LU.values[max_ind * LU.cols + j] = LU.values[k * LU.cols + j];
                LU.values[k * LU.cols + j] = temp;
            }
            scaling[max_ind] = scaling[k];
        }
        perm_indx[k] = max_ind;

        // Inner loop of LU decomposition
        for (i = k + 1; i < n; i++)
        {
            // Divide by pivot element
            temp = LU.values[i * LU.cols + k] /= LU.values[k * LU.cols + k];

            for (j = k + 1; j < n; j++)
            {
                LU.values[i * LU.cols + j] -= temp * LU.values[k * LU.cols + j];
            }
        }
    }
    return perm_indx;
}

// Linear solver that uses LU decomposition
template <class T>
void Solver<T>::lu_solve(Matrix<T> &LU, std::vector<int> &perm_indx, std::vector<T> &x)
// Solve the equations L*y = b and U*x = y to find x.
{
    int n, kp, i, j, k;
    n = LU.rows;
    T sum;

    checkDimensions(A, x);

    // The unknown x will be used as temporary storage for y.
    // The equations for forward and backward substitution have
    // been simplified by combining (b and sum) and (y and sum).
    for (i = 0; i < n; i++)
    {
        x[i] = b[i];
    }

    // Perform forward substitution to solve L*y = b.
    // Need to keep track of permutation of RHS as well
    for (k = 0; k < n; k++)
    {
        kp = perm_indx[k];
        sum = x[kp];
        x[kp] = x[k];
        for (j = 0; j < k; j++)
        {
            sum -= LU.values[k * LU.cols + j] * x[j];
        }
        x[k] = sum;
    }

    // Perform backward substitution to solve U*x = y
    // Here x = y before being updated.
    for (k = n - 1; k >= 0; k--)
    {
        sum = x[k];
        for (j = k + 1; j < n; j++)
        {
            sum -= LU.values[k * LU.cols + j] * x[j];
        }
        x[k] = sum / LU.values[k * LU.cols + k];
    }
}
