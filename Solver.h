
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

    void checkDimensions(Matrix<T> &M1, std::vector<T> &vec);

    double residualCalc(std::vector<T> &x, std::vector<T> &b_estimate);

    // Jacobi or Gauss-Seidel
    void stationaryIterative(std::vector<T> &x, double &tol, int &it_max, bool isGaussSeidel);

    void lu_solve(std::vector<T> &x);
};