
#pragma once
#include <iostream>
#include "Matrix.h"
#include <vector>

template <class T>
class Solver
{
public:
    Matrix<T> A;
    std::vector<T> b{};

    // constructor
    Solver(Matrix<T> &A, std::vector<T> &b);

    // destructor
    virtual ~Solver();

    // factory function
    static Solver<T> *makeSolver(int size);

    double residualCalc(std::vector<T> &x, std::vector<T> &b_estimate);

    // Jacobi or Gauss-Seidel
    void stationaryIterative(std::vector<T> &x, double &tol, int &it_max, bool isGaussSeidel);

    std::vector<int> lu_decomp(Matrix<T> &LU);
    void lu_solve(Matrix<T> &LU, std::vector<int> &piv, std::vector<T> &x);
};