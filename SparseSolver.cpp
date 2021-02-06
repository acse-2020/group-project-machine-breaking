#include <iostream>
#include <math.h>
#include "SparseSolver.h"
#include <stdexcept>
#include <vector>
#include "utilities.h"
#include <memory>

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
std::shared_ptr<CSRMatrix<T>> SparseSolver<T>::lu_decomp()
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
    int n, max_ind, i, j, k, row_start, row_start2, row_len, col_indx;
    n = A.rows;
    T max, temp, piv_el;

    checkDimensions(A, b);

    std::vector<int> perm_indx(n);    // Store index of permutation
    std::vector<T> scaling(n);        // Store implicit scaling of each row
    std::vector<T> pivot_elements(n); // Store implicit scaling of each row

    // Implicit scaling, find max in each row and store scaling factor
    for (i = 0; i < n; i++)
    {
        max = 0.0;
        row_start = A.row_position[i];
        row_len = A.row_position[i + 1] - row_start;
        for (j = 0; j < row_len; j++)
        {
            temp = abs(A.values[row_start + j]);
            if (temp > max)
                max = temp;
        }
        if (max == 0)
            throw std::invalid_argument("Matrix is singular");
        scaling[i] = 1.0 / max;
    }
    printVector(scaling);

    bool matching = false;

    CSRMatrix<T> matrix_before = A;
    std::shared_ptr<CSRMatrix<T>> LU;
    while (!matching)
    {
        LU = matrix_before.matMatMultSymbolic(matrix_before);

        if (matrix_before.nnzs != LU->nnzs || matrix_before.rows != LU->rows)
        {
            // If they are not the same size
            matrix_before = *LU;
            continue;
        }

        bool allSame = true;

        for (int i = 0; i < LU->rows; i++)
        {
            if (matrix_before.row_position[i] != LU->row_position[i])
            {
                allSame = false;
                break;
            }
        }

        if (!allSame)
        {
            matrix_before = *LU;
            continue;
        }

        for (int i = 0; i < LU->nnzs; i++)
        {
            if (matrix_before.col_index[i] != LU->col_index[i])
            {
                allSame = false;
                break;
            }
        }

        if (!allSame)
        {
            matrix_before = *LU;
            continue;
        }

        matching = true;
    }

    std::cout << "BEFORE " << std::endl;
    for (int i = 0; i < 7; i++)
    {
        std::cout << LU->row_position[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < LU->nnzs; i++)
    {
        std::cout << LU->col_index[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < LU->nnzs; i++)
    {
        std::cout << LU->values[i] << " ";
    }
    std::cout << std::endl;

    // the lower and upper triangular matrix values are stored in the LU matrix
    // loop over rows of LU
    for (int row = 0; row < LU->rows; row++)
    {
        // loop over non-zero columns in that row
        for (int cii = LU->row_position[row]; cii < LU->row_position[row + 1]; cii++)
        {
            // col index of a non-zero
            int col = LU->col_index[cii];

            // first element is simply equal to A_00
            if (col == 0 && row == 0)
            {
                LU->values[0] = A.values[0];
                continue;
            }

            T a_ij = 0.0;
            // search in our original matrix for a_ij
            for (int a_row_pos = A.row_position[row]; a_row_pos < A.row_position[row + 1]; a_row_pos++)
            {
                if (A.col_index[a_row_pos] == col)
                {
                    a_ij = A.values[a_row_pos];
                    break;
                }
            }

            // number of products to sum over (k-limit)
            int cutoff = std::min(row, col) - 1;

            // sum over alpha_ik * beta_kj
            // look for values in same row first, if they exist check for col equivalents
            T valsum = 0.0;
            for (int col_vec_ind = LU->row_position[row]; col_vec_ind < LU->row_position[row + 1]; col_vec_ind++)
            {
                // check on this row, preceding the current value
                int k = LU->col_index[col_vec_ind];

                // if k is above cutoff, break the loop
                if (k > cutoff)
                {
                    break;
                }

                // if alpha_ik is non-zero
                if (LU->values[col_vec_ind])
                {
                    // check whether corresponding beta_kj also exists -> add to valsum
                    for (int tempcols = LU->row_position[k]; tempcols < LU->row_position[k + 1]; tempcols++)
                    {
                        if (LU->col_index[tempcols] == col)
                        {
                            valsum += (LU->values[col_vec_ind] * LU->values[tempcols]);
                        }
                    }
                }
            }

            if (col >= row)
            {
                // this means that we are in the upper triangle or the diagonal (U)
                LU->values[row * LU->cols + col] = a_ij - valsum;
            }
            else if (col < row)
            {
                // we are in the lower triangle (L)
                // we need the beta value from the LU_jj above
                T b_jj = 0.0;

                for (int LU_vals_index = LU->row_position[col]; LU_vals_index < LU->row_position[col + 1]; LU_vals_index++)
                {
                    int temp_col_ind = LU->col_index[LU_vals_index];
                    if (temp_col_ind == col)
                    {
                        b_jj = LU->values[LU_vals_index];
                        break;
                    }
                }

                if (b_jj == 0)
                {
                    std::cout << "Error: b_jj is zero" << std::endl;
                }

                LU->values[row * LU->cols + col] = (1.0 / b_jj) * (a_ij - valsum);
            }
        }
    }

    std::cout << "END " << std::endl;
    for (int i = 0; i < 7; i++)
    {
        std::cout << LU->row_position[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < LU->nnzs; i++)
    {
        std::cout << LU->col_index[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < LU->nnzs; i++)
    {
        std::cout << LU->values[i] << " ";
    }
    std::cout << std::endl;

    return LU;
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
        // row_start = LU.row_position[i];
        // row_len = LU.row_position[i + 1] - row_start;
        for (j = LU.row_position[i]; j < LU.row_position[i + 1]; j++)
        {
            col_indx = LU.col_index[j];
            // check if valid and exits loop to avoid uneccesary checks
            if (col_indx >= i)
                break;
            sum -= LU.values[j] * x[col_indx];
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
            if (col_indx == i)
            {
                x[i] = sum / LU.values[row_start + j];
                break;
            }
        }
    }
}
