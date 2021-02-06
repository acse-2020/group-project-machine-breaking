#include <iostream>
#include "CSRMatrix.h"
#include <algorithm>

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

template <class T>
//CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, T *values_ptr, int *row_pos, int *col_ind) : Matrix<T>(rows, cols, values_ptr), nnzs(nnzs), row_position(row_pos), col_index(col_ind)
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, std::shared_ptr<T[]> values_ptr, std::shared_ptr<int[]> row_pos, std::shared_ptr<int[]> col_ind)
    : Matrix<T>(rows, cols, values_ptr), nnzs(nnzs), row_position(row_pos), col_index(col_ind)
{
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
CSRMatrix<T> CSRMatrix<T>::transpose()
{
    // TO DO: comment
    std::vector<int> t_values;
    std::vector<int> t_cols;
    std::vector<int> t_rows;

    // First element in row vector will always be 0
    t_rows.push_back(0);

    // Note that outer loop is cols as there will be as many rows
    // in transposed matrix as columns in the original one
    for (int i = 0; i < this->cols; i++)
    {
        std::vector<int> cols_nnzs{};
        int k = 0;
        for (int c = 0; c < nnzs; c++)
        {
            // Store values and column indices on column i,
            // which corresponds to row i in transposed matrix
            if (i == col_index[c])
            {
                cols_nnzs.push_back(i);
                t_values.push_back(this->values[c]);
                for (int k = 0; k < this->rows; k++)
                    if (c >= row_position[k] && c < row_position[k + 1])
                    {
                        t_cols.push_back(k);
                    }
            }
        }
        // Store row position incrementally based on number of nnzs in row
        t_rows.push_back(cols_nnzs.size() + t_rows.back());
    }

    // Construct transposed matrix
    CSRMatrix<T> t_Matrix = CSRMatrix<T>(t_rows.size() - 1, this->rows, nnzs, true);

    for (int i = 0; i < t_cols.size(); i++)
    {
        t_Matrix.col_index[i] = t_cols[i];
    }
    for (int i = 0; i < t_rows.size(); i++)
    {
        t_Matrix.row_position[i] = t_rows[i];
    }
    for (int i = 0; i < t_values.size(); i++)
    {
        t_Matrix.values[i] = t_values[i];
    }

    return t_Matrix;
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

template <class T>
CSRMatrix<T> CSRMatrix<T>::cholesky()
{
    // for now, we assume output has been preallocated
    std::vector<T> R_values{};
    std::vector<int> R_cols{};
    std::vector<int> R_row_position(this->rows + 1, 0);

    /*
    Rjj = sqrt(Ajj - sum(Rjk[:j]^2)

    Rij = sqrt(Aij - sum(Rik[:j]*Rjk[:j]^2); for i>j
    */

    // loop over rows of A
    for (int i = 0; i < this->rows; i++)
    {
        // rows indices of left matrix
        int r_start = row_position[i];
        int r_end = row_position[i + 1];

        // loop over column indices for this row in A - equivalent to rows in B
        std::vector<int> cols_nnzs{};
        // cii - index of col_index of R array
        int ci = 0;
        std::vector<int> infills_left{};
        for (int cii = r_start; (cii < r_end && ci < i); cii++)
        {
            ci = col_index[cii];

            // std::cout << "No. of items in row: " << cols_nnzs.size() << std::endl;
            // std::cout << "row: " << i << std::endl;
            // std::cout << "column: " << ci << std::endl;
            for (int k = 0; k < ci; k++)
            {
                // std::cout << "columns left: " << k << std::endl;
                if (cii != r_start && k == col_index[cii - 1])
                {
                    infills_left.push_back(col_index[cii - 1]);
                }
                for (int r = 0; r < k && k > 0; r++)
                {
                    int r_start_above = row_position[r];
                    int r_end_above = row_position[r + 1];
                    // std::cout << "rows above: " << r << std::endl;
                    for (int a = r_start_above; a < r_end_above; a++)
                    {
                        if (infills_left.size() > 0 && col_index[a] == k)
                        {

                            cols_nnzs.push_back(k);
                            // std::cout << "No. of items in row: " << cols_nnzs.size() << std::endl;
                        }
                    }
                }
            }
            if (ci < i)
            {
                cols_nnzs.push_back(ci);
            }

            // std::cout << "No. of items in row: " << cols_nnzs.size() << std::endl;
        }
        cols_nnzs.push_back(i);

        R_row_position[i + 1] = R_row_position[i] + cols_nnzs.size();

        // Concatenate
        for (int j = 0; j < cols_nnzs.size(); j++)
        {
            R_cols.push_back(cols_nnzs[j]);
            R_values.push_back(1);
        }
    }

    CSRMatrix<T> R = CSRMatrix<T>(this->rows, this->cols, R_values.size(), true);

    for (int i = 0; i < R_values.size(); i++)
    {
        R.col_index[i] = R_cols[i];
        R.values[i] = R_values[i];
    }
    for (int i = 0; i <= this->rows; i++)
    {
        R.row_position[i] = R_row_position[i];
    }

    return R;
}
