
#pragma once
#include <iostream>
#include "Matrix.h"
//#include "Matrix.cpp"

template <class T>
class Solver
{
public:
    Matrix<T> *LHS = nullptr;
    Matrix<T> *RHS = nullptr;
    // constructor where we want to preallocate ourselves
    Solver(Matrix<T> *LHS, Matrix<T> *RHS);
    // constructor where we already have allocated memory outside
    //Solver(Matrix LHS, int cols, T *values_ptr);

    // destructor
    virtual ~Solver();

    void jacobi(Matrix<T> &unknowns, double &tol, int &it_max);
    void gaussSeidel(Matrix<T> &unknowns, double &tol, int &it_max);
    void lu_solve(Matrix<T> &unknowns, double &tol, int &it_max);
};