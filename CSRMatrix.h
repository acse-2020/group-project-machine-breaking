#pragma once
#include "Matrix.h"
#include <vector>
#include <memory>

template <class T>
class CSRMatrix : public Matrix<T>
{
public:
    CSRMatrix(int rows, int cols, int nnzs, bool preallocate);

    CSRMatrix(int rows, int cols, int nnzs, std::shared_ptr<T[]> values_ptr, std::shared_ptr<int[]> row_pos, std::shared_ptr<int[]> col_ind);

    ~CSRMatrix();

    virtual void printMatrix();

    virtual void print2DMatrix();

    void matVecMult(std::vector<T> &input, std::vector<T> &output);

    std::shared_ptr<CSRMatrix<T>>  matMatMult(CSRMatrix<T> &mat_right);

    std::shared_ptr<int[]> row_position;  //create nullpointer
    std::shared_ptr<int[]> col_index; // create nullpointer

    // number of non-zeros
    int nnzs = -1;

    // we're inheriting the values pointer so we don't have to include it here
};