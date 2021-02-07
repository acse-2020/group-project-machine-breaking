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

template <class T>
std::shared_ptr<CSRMatrix<T>> SparseSolver<T>::cholesky_decomp()
{
    // for now, we assume output has been preallocated
    std::vector<T> R_values{};
    std::vector<int> R_cols{};
    std::vector<int> R_row_position(A.rows + 1, 0);

    // loop over rows of A
    for (int i = 0; i < A.rows; i++)
    {
        // rows indices of left matrix
        int r_start = A.row_position[i];
        int r_end = A.row_position[i + 1];

        // loop over column indices for this row in A - equivalent to rows in B
        std::vector<int> cols_nnzs{};
        // cii - index of col_index of R array
        int ci = 0;
        std::vector<int> infills_left{};
        for (int cii = r_start; (cii < r_end && ci < i); cii++)
        {
            ci = A.col_index[cii];

            for (int k = 0; k < ci; k++)
            {
                if (cii != r_start && k == A.col_index[cii - 1])
                {
                    infills_left.push_back(A.col_index[cii - 1]);
                }
                for (int r = 0; r < k && k > 0; r++)
                {
                    int r_start_above = A.row_position[r];
                    int r_end_above = A.row_position[r + 1];
                    for (int a = r_start_above; a < r_end_above; a++)
                    {
                        if (infills_left.size() > 0 && A.col_index[a] == k)
                        {
                            cols_nnzs.push_back(k);
                        }
                    }
                }
            }
            if (ci < i)
            {
                cols_nnzs.push_back(ci);
            }
        }
        cols_nnzs.push_back(i);
        // sort non_zeros & remove duplicates
        sort(cols_nnzs.begin(), cols_nnzs.end());
        cols_nnzs.erase(unique(cols_nnzs.begin(), cols_nnzs.end()), cols_nnzs.end());

        // Concatenate
        for (int j = 0; j < cols_nnzs.size(); j++)
        {
            R_cols.push_back(cols_nnzs[j]);
        }
        R_row_position[i + 1] = R_row_position[i] + cols_nnzs.size();

        if (i == 0)
        {
            R_values.push_back(sqrt(A.values[0]));
        }

        // Entries in R on row i
        int R_r_start = R_row_position[i];
        int R_r_end = R_row_position[i + 1];
        if (i != 0)
        {
            for (int cii = R_r_start; cii < R_r_end; cii++)
            {
                ci = R_cols[cii];

                if (ci == 0)
                {

                    T A_i0;
                    for (int k = r_start; k < r_end; k++)
                    {
                        if (A.col_index[k] == 0)
                        {
                            A_i0 = A.values[k];
                            R_values.push_back(A_i0 / R_values[0]);
                        }
                    }
                }

                T sum_ij = 0;
                if (ci != 0 && ci != i)
                {

                    int r_start_above = R_row_position[ci];
                    int r_end_above = R_row_position[ci + 1];

                    for (int n = R_r_start; n < R_r_end; n++)
                    {
                        // Loop through columns[:j] in row ci above current entry row[i, ci]
                        for (int a = r_start_above; a < r_end_above; a++)
                        {
                            // Match cols in current and above rows
                            if (R_cols[n] == R_cols[a] && R_cols[n] != ci)
                            {
                                sum_ij += R_values[n] * R_values[a];
                            }
                        }
                    }

                    int diag_r_start = R_row_position[ci];
                    int diag_r_end = R_row_position[ci + 1];

                    // Loop through row that contains diagonal L entry on current column
                    for (int diag = diag_r_start; diag < diag_r_end; diag++)
                    {
                        if (R_cols[diag] == ci)
                        {
                            T A_ij = 0;
                            // Determine whether ij entry in L also appears in A, else we use a 0
                            for (int k = r_start; k < r_end; k++)
                            {
                                if (ci == A.col_index[k])
                                {
                                    A_ij = A.values[k];
                                }
                            }
                            // std::cout << "A_ij " << A_ij << std::endl;
                            // std::cout << "sum_ij " << sum_ij << std::endl;

                            // std::cout << "R_ij " << R_values[diag] << std::endl;

                            // Lij = (Lij - sum(Lik[:j]*Rjk[:j])/Ljj; for i>j
                            R_values.push_back((A_ij - sum_ij) / R_values[diag]);
                        }
                    }
                }

                T sum_jj = 0;
                if (ci != 0 && ci == i)
                {
                    // Loop over Ljk entries in row j
                    for (int k = R_r_start; (k < R_r_end && R_cols[k] < i); k++)
                    {
                        sum_jj += pow(R_values[k], 2);
                    }

                    // Access jj value in A
                    T A_jj;
                    int diag;
                    for (diag = r_start; diag < r_end; diag++)
                    {
                        if (A.col_index[diag] == i)
                        {
                            A_jj = A.values[diag];
                        }
                    }

                    // Ljj = sqrt(Ajj - sum(Ljk[:j]^2)
                    R_values.push_back(sqrt(A_jj - sum_jj));
                }
            }
        }
    }

    std::shared_ptr<CSRMatrix<T>> sparse_mat_ptr(new CSRMatrix<T>(A.rows, A.cols, R_values.size(), true));

    for (int i = 0; i < R_values.size(); i++)
    {
        sparse_mat_ptr->col_index[i] = R_cols[i];
        sparse_mat_ptr->values[i] = R_values[i];
    }
    for (int i = 0; i <= A.rows; i++)
    {
        sparse_mat_ptr->row_position[i] = R_row_position[i];
    }

    return sparse_mat_ptr;
}

// Linear solver that uses LU decomposition
template <class T>
void SparseSolver<T>::cholesky_solve(CSRMatrix<T> &R, std::vector<T> &x)
// Solve the equations L*y = b and U*x = y to find x.
{
    int n, ip, i, j, row_start, row_len, col_start, col_indx;
    n = R.rows;
    T sum;

    checkDimensions(A, x);

    std::shared_ptr<CSRMatrix<T>> R_T = R.transpose();

    R_T->print2DMatrix();

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
        // ip = perm_indx[i];
        sum = x[i];
        // x[ip] = x[i];
        // row_start = R.row_position[i];
        // row_len = R.row_position[i + 1] - row_start;
        for (j = R.row_position[i]; j < R.row_position[i + 1]; j++)
        {
            col_indx = R.col_index[j];
            // check if valid and exits loop to avoid uneccesary checks
            if (col_indx >= i)
                break;
            sum -= R.values[j] * x[col_indx];
        }
        x[i] = sum;
        // std::cout << "sum: " << x[i] << std::endl;
    }

    // Perform backward substitution to solve U*x = y
    // Here x = y before being updated.
    for (i = n - 1; i >= 0; i--)
    {
        sum = x[i];
        row_start = R_T->row_position[i];
        row_len = R_T->row_position[i + 1] - row_start;
        for (j = 0; j < row_len; j++)
        {
            col_indx = R_T->col_index[row_start + j];
            if (col_indx >= i + 1)
                sum -= R_T->values[row_start + j] * x[col_indx];
        }
        // Find diagonal element
        for (j = 0; j < row_len; j++)
        {
            col_indx = R_T->col_index[row_start + j];
            if (col_indx == i)
            {
                x[i] = sum / R_T->values[row_start + j];
                break;
            }
        }
    }
}