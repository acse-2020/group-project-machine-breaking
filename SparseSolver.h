
#pragma once
#include "CSRMatrix.h"
#include <vector>

template <class T>
class SparseSolver
{
public:
    CSRMatrix<T> A;

    std::vector<T> b{};

    SparseSolver(CSRMatrix<T> &A, std::vector<T> &b);

    ~SparseSolver();

    void checkDimensions(CSRMatrix<T> &M1, std::vector<T> &vec);

    double residualCalc(std::vector<T> &x, std::vector<T> &b_estimate);
};