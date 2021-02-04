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


// LU decomposition
template <class T>
std::vector<int> SparseSolver<T>::lu_decomp(CSRMatrix<T> &LU)
/*
LU decomposition
Algorithm based on similar method as in 'Numerical recipes C++'.
The input matrix A is copied to LU, which is modified 'in place'.
Uses Crout's method by setting U_ii = 1.
Partial pivoting is implemented to ensure the stability of the method.
Implicit pivoting used to make it independent of scaling of equations.
An extensions would be to create a pivoting scheme that minimised the 
number of non-zero elements.
*/
{
    int n, max_ind, i, j, k, row_start, row_len;
    n = A.rows;
    T max, temp, piv_el;

    checkDimensions(A, b);

    std::vector<int> perm_indx(n); // Store index of permutation
    std::vector<T> scaling(n);     // Store implicit scaling of each row
    std::vector<T> pivot_elemets(n);     // Store implicit scaling of each row

    // Copy values into LU, want to do this with a copy constryctor later
    for (int i = 0; i <= A.nnzs; i++)
    {
        LU.values[i] = A.values[i];
        std::cout << " i " << i << " LU val " << LU.values[i] << std::endl;
        LU.col_index[i] = A.col_index[i];
    }
        for (int i = 0; i < A.rows + 1; i++)
    {
        LU.row_position[i] = A.row_position[i];
    }
    LU.print2DMatrix();
    // Implicit scaling, find max in each row and store scaling factor
    // This part is working
    for (i = 0; i < n; i++)
    {
        max = 0.0;
        row_start = LU.row_position[i];
        row_len = LU.row_position[i + 1] - row_start;
        for (j = 0; j < row_len; j++)
        {
            temp = abs(LU.values[row_start + j]);
            if (temp > max)
                max = temp;
        }
        if (max == 0)
            throw std::invalid_argument("Matrix is singular");
        scaling[i] = 1.0 / max;
    }
    printVector(scaling);
    
    // Perform LU decomposition
    // Inner LU loop resembles inner loop of matrix multiplication.
    // Uses kij permutation to loop over elements as fastest for
    // row major storage and easiest to implement pivoting for.

    // Loop over Upper matrix to find largest B to pivot with
    // Pivot by swapping row k by row with max pivot element
    // Perform the inner loop of LU decomp, reduce remaining submatrix
    
    for (k = 0; k < n; k++)
    {
        // The search of finding the max element and performing the row swapping is not implemented yet
        /*
        max = 0.0;
        for (i = k; i < n; i++)
        {
            temp = scaling[i] *  abs(LU.values[i * LU.cols + k]);
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
        */

       // Find and store pivot element
        row_start = LU.row_position[k];
        row_len = LU.row_position[k + 1] - row_start;
        for (j = 0; j < row_len; j++)
            {
                col_indx = LU.col_index[row_start + j];
                if (col_indx == k) 
                {
                    piv_elements[k] = LU.values[row_start+ j];
                    break;
                }
            }

        // Inner loop of LU decomposition
        for (i = k + 1; i < n; i++)
        {
            row_start = LU.row_position[i];
            row_len = LU.row_position[i + 1] - row_start;
            
            // Divide by pivot element
            // Check if LU[i, k] is non-zero
            temp = 0;
            for (j = 0; j < row_len; j++)
            {
                col_indx = LU.col_index[row_start + j];
                if (col_index == k)
                    temp = LU.values[row_start + j] /= piv_elements[k];
                    break;
            }

            if (temp != 0)
            { 
                for (j = 0; j < row_len; j++)
                {
                    col_indx = LU.col_index[row_start + j];
                    if (col_indx >= k + 1)  // check from original loop
                    {
                        if (col_indx ==
                        row_start2 = LU.row_position[k];
                        row_len2 = LU.row_position[k + 1] - row_start2;
                        for (int r = 0; r < row_len2; r++)
                        {
                            // need to check if things in this expression are not zero
                            // need to insert new elements
                            LU.values[i * LU.cols + j] -= temp * LU.values[k * LU.cols + j];
                        }
                        break;
                    }
                }
            }
        }
    }
    return perm_indx;
}

// Linear solver that uses LU decomposition
template <class T>
void SparseSolver<T>::lu_solve(CSRMatrix<T> &LU, std::vector<int> &perm_indx, std::vector<T> &x)
// Solve the equations L*y = b and U*x = y to find x.
{
    int n, ip, i, j, row_start, row_len, col_start, col_indx;
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
    for (i = 0; i < n; i++)
    {
        ip = perm_indx[i];
        sum = x[ip];
        x[ip] = x[i];
        row_start = LU.row_position[i];
        row_len = LU.row_position[i + 1] - row_start;
        for (j = 0; j < row_len; j++)
        {
            col_indx = LU.col_index[row_start + j];
            // check if valid and exits loop to avoid uneccesary checks
            if (col_indx >= i)
                break;
            sum -= LU.values[row_start + j] * x[col_indx];
        }
        x[i] = sum;
    } 
    
    // Perform backward substitution to solve U*x = y
    // Here x = y before being updated.
    for (i = n - 1; i >= 0; i--)
    {
        sum = x[i];
        row_start = LU.row_position[i];
        row_len = LU.row_position[i + 1] - row_start;
        for (j = 0; j < row_len; j++)
        {
            col_indx = LU.col_index[row_start + j];
            if (col_indx >= i + 1) 
                sum -= LU.values[row_start + j] * x[col_indx];
        }
        // Find diagonal element
        for (j = 0; j < row_len; j++)
        {
            col_indx = LU.col_index[row_start + j];
            if (col_indx == i) {
                x[i] = sum / LU.values[row_start + j];
                break;}
        }
    }
}
