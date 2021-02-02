#pragma once
#include "Matrix.h"
#include <vector>
template <class T>
class CSRMatrix : public Matrix<T>
{
public:
    CSRMatrix(int rows, int cols, int nnzs, bool preallocate);
    CSRMatrix(int rows, int cols, int nnzs, T *values_ptr, int *row_pos, int *col_ind);

    ~CSRMatrix();

    virtual void printMatrix();

    void matVecMult(std::vector<T> &input, std::vector<T> &output);

    // TODO: implement sparse matMatMult
    // void matMatMult(CSRMatrix<T> &mat_right, CSRMatrix<T> &output);

    int *row_position = nullptr;
    int *col_index = nullptr;

    // number of non-zeros
    int nnzs = -1;

    // we're inheriting the values pointer so we don't have to include it here
};