#pragma once
#include <iostream>

class Matrix
{
public:
   // constructor where we want to preallocate ourselves
   Matrix(int rows, int cols, bool preallocate);
   // constructor where we already have allocated memory outside
   Matrix(int rows, int cols, double *values_ptr);
   // destructor
   virtual ~Matrix();

   // Print out the values in our matrix
   void printValues();
   virtual void printMatrix();

   // Perform some operations with our matrix
   virtual void matMatMult(Matrix &mat_right, Matrix &output);

   // Solve simple Ax=b system with Jacobi method
   virtual void Jacobi(Matrix &RHS, Matrix &unknowns, double &tol, int &it_max);

   // Explicitly using the C++11 nullptr here
   double *values = nullptr;
   int rows = -1;
   int cols = -1;

   // Private variables - there is no need for other classes
   // to know about these variables
private:
   int size_of_values = -1;
   bool preallocated = false;
};