#pragma once
#include <iostream>
#include <vector>
#include <memory>

template <class T>
class Matrix
{
public:
    // default constructor
    Matrix(void);

    // constructor where we want to preallocate ourselves
    Matrix(int rows, int cols, bool preallocate);

    // constructor where we already have allocated memory outside
    Matrix(int rows, int cols, std::shared_ptr<T[]> &values_ptr);

    // Copy constructor
    Matrix(const Matrix<T> &M2);

    Matrix<T> &operator=(const Matrix<T> &M2);

    // destructor
    virtual ~Matrix();

    // Print out the values in our matrix
    void printValues();
    virtual void printMatrix();

    /* matMatMul function should not be virtual, since in the derived class
   CSRMatrix we want to specify the input type as CSRMatrix */
    void matMatMult(Matrix<T> &mat_right, Matrix<T> &output);

    void matVecMult(std::vector<T> &vec, std::vector<T> &output);

    // Solve simple Ax=b system with Jacobi method
    //virtual void Jacobi(Matrix &RHS, Matrix &unknowns, double &tol, int &it_max);

    // Explicitly using the C++11 nullptr here
    //T *values = nullptr;
    std::shared_ptr<T[]> values;
    int rows = -1;
    int cols = -1;

    int size_of_values = -1;
    bool preallocated = false;
};