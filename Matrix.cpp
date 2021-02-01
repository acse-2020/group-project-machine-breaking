// #pragma once
#include <iostream>
#include "Matrix.h"
#include <math.h>

// Constructor - using an initialisation list here
template <class T>
Matrix<T>::Matrix(int rows, int cols, bool preallocate) : rows(rows), cols(cols), size_of_values(rows * cols), preallocated(preallocate)
{
    // If we want to handle memory ourselves
    if (this->preallocated)
    {
        // Must remember to delete this in the destructor
        this->values = new T[size_of_values];
    }
}

// Constructor - now just setting the value of our pointer

template <class T>
Matrix<T>::Matrix(int rows, int cols, T *values_ptr) : rows(rows), cols(cols), size_of_values(rows * cols), values(values_ptr)
{
}

// destructor

template <class T>
Matrix<T>::~Matrix()
{
    // Delete the values array
    if (this->preallocated)
    {
        delete[] this->values;
    }
}

// Just print out the values in our values array
template <class T>
void Matrix<T>::printValues()
{
    std::cout << "Printing values" << std::endl;
    for (int i = 0; i < this->size_of_values; i++)
    {
        std::cout << this->values[i] << " ";
    }
    std::cout << std::endl;
}

// Explicitly print out the values in values array as if they are a matrix
template <class T>
void Matrix<T>::printMatrix()
{
    std::cout << "Printing matrix" << std::endl;
    for (int j = 0; j < this->cols; j++)
    {
        std::cout << std::endl;
        for (int i = 0; i < this->rows; i++)
        {
            // We have explicitly used a row-major ordering here
            std::cout << this->values[i + j * this->rows] << " ";
        }
    }
    std::cout << std::endl;
}

// Do matrix vector multiplication for rowmajor
// output =  this * vec
template <class T>
void Matrix<T>::matVecMult(std::vector<T> &vec, std::vector<T> &output)
{
    double sum1;

    for (int i = 0; i < this->rows; i++)
    {
        // This is a dot product and can have been done with BLAS dot
        sum1 = 0;
        for (int j = 0; j < this->cols; j++)
        {
            sum1 += this->values[i * this->cols + j] * vec[j];
        }
        output[i] = sum1;
    }
}

// Do matrix matrix multiplication
template <class T> // output = this * mat_right
void Matrix<T>::matMatMult(Matrix &mat_right, Matrix &output)
{

    // Check our dimensions match
    if (this->cols != mat_right.rows)
    {
        std::cerr << "Input dimensions for matrices don't match" << std::endl;
        return;
    }

    // Check if our output matrix has had space allocated to it
    if (output.values != nullptr)
    {
        // Check our dimensions match
        if (this->rows != output.rows || mat_right.cols != output.cols)
        {
            std::cerr << "Input dimensions for matrices don't match" << std::endl;
            return;
        }
    }
    // The output hasn't been preallocated, so we are going to do that
    else
    {
        output.values = new T[this->rows * mat_right.cols];
        output.preallocated = true;
    }

    // Set values to zero before hand
    for (int i = 0; i < output.size_of_values; i++)
    {
        output.values[i] = 0;
    }

    // Now we can do our matrix-matrix multiplication
    // CHANGE THIS FOR LOOP ORDERING AROUND
    // AND CHECK THE TIME SPENT
    // Does the ordering matter for performance. Why??
    for (int i = 0; i < this->rows; i++)
    {
        for (int k = 0; k < this->cols; k++)
        {
            for (int j = 0; j < mat_right.cols; j++)
            {
                output.values[i * output.cols + j] += this->values[i * this->cols + k] * mat_right.values[k * mat_right.cols + j];
            }
        }
    }
}