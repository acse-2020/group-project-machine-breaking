#include <iostream>
#include "CSRMatrix.h"
#include <algorithm>

// Default constructor - creates emmpty matrix
template <class T>
CSRMatrix<T>::CSRMatrix()
{
}

// Constructor
template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, bool preallocate) : Matrix<T>(rows, cols, false), nnzs(nnzs)
{

    this->preallocated = preallocate;
    if (this->preallocated)
    {
        // Values and col index should be same length, while rows should be no.rows + 1
        std::shared_ptr<T[]> vals(new T[this->nnzs]);
        std::shared_ptr<int[]> rows(new int[this->rows + 1]);
        std::shared_ptr<int[]> cols(new int[this->nnzs]);
        this->values = vals;
        this->row_position = rows;
        this->col_index = cols;
    }
}

// Constructor
template <class T>
//CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, T *values_ptr, int *row_pos, int *col_ind) : Matrix<T>(rows, cols, values_ptr), nnzs(nnzs), row_position(row_pos), col_index(col_ind)
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, std::shared_ptr<T[]> values_ptr, std::shared_ptr<int[]> row_pos, std::shared_ptr<int[]> col_ind)
    : Matrix<T>(rows, cols, values_ptr), nnzs(nnzs), row_position(row_pos), col_index(col_ind)
{
}

// Copy constructor
template <class T>
CSRMatrix<T>::CSRMatrix(const CSRMatrix<T> &M2)
{
    this->rows = M2.rows;
    this->cols = M2.cols;
    this->nnzs = M2.nnzs;
    this->values = std::shared_ptr<T[]>(new T[this->nnzs]);
    this->row_position = std::shared_ptr<int[]>(new int[this->rows + 1]);
    this->col_index = std::shared_ptr<int[]>(new int[this->nnzs]);
    for (int i = 0; i < this->nnzs; i++)
    {
        this->values[i] = M2.values[i];
        this->col_index[i] = M2.col_index[i];
    }
    for (int i = 0; i < this->rows + 1; i++)
    {
        this->row_position[i] = M2.row_position[i];
    }
    this->preallocated = true;
}

// Copy constructor - overloading the assignement operator
template <class T>
CSRMatrix<T> &CSRMatrix<T>::operator=(const CSRMatrix<T> &M2)
{
    // self-assignment check
    if (this == &M2)
        return *this;
    this->rows = M2.rows;
    this->cols = M2.cols;
    this->nnzs = M2.nnzs;
    this->values = std::shared_ptr<T[]>(new T[this->nnzs]);
    this->row_position = std::shared_ptr<int[]>(new int[this->rows + 1]);
    this->col_index = std::shared_ptr<int[]>(new int[this->nnzs]);
    for (int i = 0; i < this->nnzs; i++)
    {
        this->values[i] = M2.values[i];
        this->col_index[i] = M2.col_index[i];
    }
    for (int i = 0; i < this->rows + 1; i++)
    {
        this->row_position[i] = M2.row_position[i];
    }
    this->preallocated = true;
    return *this;
}

template <class T>
CSRMatrix<T>::~CSRMatrix()
{
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

// populate the row_pos and col_ind vectors
template <class T>
std::shared_ptr<CSRMatrix<T>> CSRMatrix<T>::matMatMultSymbolic(CSRMatrix<T> &mat_right)
{
    std::vector<int> col_ind;
    std::vector<int> row_pos(this->rows + 1, 0);
    //std::shared_ptr<int[]> row_pos(new int[this->rows + 1]);
    //row_pos.reserve(this->rows + 1);
    //row_pos.push_back(0);
    row_pos[0] = 0;

    // loop over rows of A
    for (int i = 0; i < this->rows; i++)
    {
        // rows indices of left matrix
        int r_start = this->row_position[i];
        int r_end = this->row_position[i + 1];

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

        int nnzs_per_row = cols_nnzs.size();

        row_pos[i + 1] = row_pos[i] + nnzs_per_row;

        // Concatenate
        for (int j = 0; j < nnzs_per_row; j++)
        {
            col_ind.push_back(cols_nnzs[j]);
        }
    }

    int new_nnzs = col_ind.size();
    /*
    std::shared_ptr<int[]> col_ind_ptr(new int[new_nnzs]);
    std::shared_ptr<T[]> values_ptr(new T[new_nnzs]);

    for (int i = 0; i < new_nnzs; i++)
    {
        col_ind_ptr[i] = col_ind[i];
        values_ptr[i] = 0;
    }

    std::shared_ptr<CSRMatrix<T>> result(new CSRMatrix<T>(this->rows, mat_right.cols, new_nnzs, values_ptr, row_pos, col_ind_ptr));
    */
    std::shared_ptr<CSRMatrix<T>> result(new CSRMatrix<T>(this->rows, mat_right.cols, new_nnzs, true));

    for (int i = 0; i < new_nnzs; i++)
    {
        result->col_index[i] = col_ind[i];
        result->values[i] = 0;
    }
    for (int i = 0; i < this->rows + 1; i++)
    {
        result->row_position[i] = row_pos[i];
    }

    return result;
}

template <class T>
std::shared_ptr<CSRMatrix<T>> CSRMatrix<T>::matMatMult(CSRMatrix<T> &mat_right)
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

    std::shared_ptr<CSRMatrix<T>> result(new CSRMatrix<T>(this->rows, mat_right.cols, output_cols.size(), true));

    for (int i = 0; i < output_cols.size(); i++)
    {
        result->col_index[i] = output_cols[i];
    }
    for (int i = 0; i <= this->rows; i++)
    {
        result->row_position[i] = output_row_position[i];
    }
    for (int i = 0; i < output_all_values.size(); i++)
    {
        result->values[i] = output_all_values[i];
    }

    return result;
}
