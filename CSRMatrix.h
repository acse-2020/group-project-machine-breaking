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

    virtual void print2DMatrix();

    void matVecMult(std::vector<T> &input, std::vector<T> &output);

    // TODO: implement sparse matMatMult
    CSRMatrix<T> matMatMult(CSRMatrix<T> &mat_right);

    int *row_position = nullptr;
    int *col_index = nullptr;

    // number of non-zeros
    int nnzs = -1;

    // we're inheriting the values pointer so we don't have to include it here
};