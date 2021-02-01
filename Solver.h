
#pragma once
#include <iostream>
#include "Matrix.h"
//#include "Matrix.cpp"

template <class T>
class Solver
{
public:
    Matrix<T> *A = nullptr;
    Matrix<T> *b = nullptr;
    // constructor where we want to preallocate ourselves
    Solver(Matrix<T> *A, Matrix<T> *b);
    // constructor where we already have allocated memory outside
    //Solver(Matrix LHS, int cols, T *values_ptr);

    // destructor
    virtual ~Solver();

    void jacobi(Matrix<T> &x, double &tol, int &it_max);
    void gaussSeidel(Matrix<T> &x, double &tol, int &it_max);
    void lu_solve(Matrix<T> &x, double &tol, int &it_max);
};