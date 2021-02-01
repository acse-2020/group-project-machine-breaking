#include <iostream>
#include "CSRMatrix.h"

template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, bool preallocate) : Matrix<T>(rows, cols, false), nnzs(nnzs)
{
    this->preallocated = preallocate;
    if (this->preallocated)
    {
        this->values = new T[this->nnzs];
        this->row_position = new int[this->nnzs + 1];
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

// template <class T>
// void CSRMatrix<T>::matMatMult(CSRMatrix<T> &mat_right, CSRMatrix<T> &output)
// {
// }