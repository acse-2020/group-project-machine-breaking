#include <iostream>
#include "CSRMatrix.h"

template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, bool preallocate) : Matrix<T>(rows, cols, false), nnzs(nnzs)
{
    this->preallocated = preallocate;
    if (this->preallocated)
    {
        // Values and col index should be same length, while rows should be no.rows + 1
        this->values = new T[this->nnzs];
        this->row_position = new int[this->rows + 1];
        this->col_index = new int[this->nnzs];
    }
}

template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, T *values_ptr, int *row_pos, int *col_ind) : Matrix<T>(rows, cols, values_ptr), nnzs(nnzs), row_position(row_pos), col_index(col_ind)
{
}

template <class T>
CSRMatrix<T>::~CSRMatrix()
{
    // values pointer is deleted in Matrix destructor
    if (this->preallocated)
    {
        delete[] this->row_position;
        delete[] this->col_index;
    }
}

template <class T>
void CSRMatrix<T>::printMatrix()
{
    std::cout << "Printing matrix" << std::endl;
    std::cout << "Values: ";
    for (int j = 0; j < this->nnzs; j++)
    {
        std::cout << this->values[j] << " ";
    }
    std::cout << std::endl;
    std::cout << "row_position: ";
    for (int j = 0; j < this->rows + 1; j++)
    {
        std::cout << this->row_position[j] << " ";
    }
    std::cout << std::endl;
    std::cout << "col_index: ";
    for (int j = 0; j < this->nnzs; j++)
    {
        std::cout << this->col_index[j] << " ";
    }
    std::cout << std::endl;
}

template <class T>
void CSRMatrix<T>::print2DMatrix()
{
    // Initialise dense matrix format
    std::vector<T> vals(this->rows * this->cols, 0);
    std::cout << "Printing 2D Matrix" << std::endl;
    for (int i = 0; i < this->rows; i++)
    {
        // rows indices of matrix
        int r_start = row_position[i];
        int r_end = row_position[i + 1];

        // cii - index of col_index of array
        for (int cii = r_start; cii < r_end; cii++)
        {
            int ci = col_index[cii];
            // Store non-zeros in row-major order
            vals[ci + i * this->cols] = this->values[cii];
        }
        for (int j = 0; j < this->cols; j++)
        {
            std::cout << " " << vals[j + i * this->cols] << " ";
        }

        std::cout << std::endl;
    }
}

template <class T>
void CSRMatrix<T>::matVecMult(std::vector<T> &input, std::vector<T> &output)
{
    // TODO: check the sizes

    for (int i = 0; i < this->rows; i++)
    {
        output[i] = 0.0;
    }

    for (int i = 0; i < this->rows; i++)
    {
        for (int val_index = this->row_position[i]; val_index < this->row_position[i + 1]; val_index++)
        {
            output[i] += this->values[val_index] * input[this->col_index[val_index]];
        }
    }
}

template <class T>
CSRMatrix<T> CSRMatrix<T>::matMatMult(CSRMatrix<T> &mat_right)
{
    // for now, we assume output has been preallocated
    std::vector<T> output_all_values{};
    std::vector<int> output_cols{};
    std::vector<int> output_row_position(this->rows + 1, 0);

    // loop over rows of A
    for (int i = 0; i < this->rows; i++)
    {
        // rows indices of left matrix
        int r_start = row_position[i];
        int r_end = row_position[i + 1];

        // loop over column indices for this row in A - equivalent to rows in B
        std::vector<int> cols_nnzs{};

        // cii - index of col_index of left array
        for (int cii = r_start; cii < r_end; cii++)
        {
            int ci = col_index[cii];
            // left col index corresponds to right row index
            int row_right_start = mat_right.row_position[ci];
            int row_right_end = mat_right.row_position[ci + 1];
            for (int rr = row_right_start; rr < row_right_end; rr++)
            {
                // this is the column index of output that is nnz
                // the corresponding row index is i
                cols_nnzs.push_back(mat_right.col_index[rr]);
            }
        }
        // sort non_zeros & remove duplicates
        sort(cols_nnzs.begin(), cols_nnzs.end());
        cols_nnzs.erase(unique(cols_nnzs.begin(), cols_nnzs.end()), cols_nnzs.end());

        std::vector<T> output_vals_per_row(cols_nnzs.size(), 0);

        for (int cii = r_start; cii < r_end; cii++)
        {
            int ci = col_index[cii];
            // left col index corresponds to right row index
            int row_right_start = mat_right.row_position[ci];
            int row_right_end = mat_right.row_position[ci + 1];
            for (int rr = row_right_start; rr < row_right_end; rr++)
            {
                // multiply
                for (int n = 0; n < cols_nnzs.size(); n++)
                {
                    if (cols_nnzs[n] == mat_right.col_index[rr])
                    {
                        output_vals_per_row[n] += this->values[cii] * mat_right.values[rr];
                    }
                }
            }
        }
        output_row_position[i + 1] = output_row_position[i] + cols_nnzs.size();

        // Concatenate
        for (int j = 0; j < cols_nnzs.size(); j++)
        {
            output_cols.push_back(cols_nnzs[j]);
            output_all_values.push_back(output_vals_per_row[j]);
        }
    }

    //(int rows, int cols, int nnzs, T *values_ptr, int *row_pos, int *col_ind) : Matrix<T>(rows, cols, values_ptr), nnzs(nnzs), row_position(row_pos), col_index(col_ind)

    int init_row_position[] = {0, 1, 2, 3, 4};
    int init_col_index[] = {0, 1, 2, 3};
    double init_sparse_values[] = {2, 1, 3, 7};

    CSRMatrix<T> result = CSRMatrix<T>(this->rows, mat_right.cols, output_cols.size(), true);

    for (int i = 0; i < output_cols.size(); i++)
    {
        result.col_index[i] = output_cols[i];
    }
    for (int i = 0; i <= this->rows; i++)
    {
        result.row_position[i] = output_row_position[i];
    }
    for (int i = 0; i < output_all_values.size(); i++)
    {
        result.values[i] = output_all_values[i];
    }

    return result;
}
