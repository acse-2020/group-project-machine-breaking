#include <iostream>
#include "Matrix.h"

// Constructor - using an initialisation list here
Matrix::Matrix(int rows, int cols, bool preallocate) : rows(rows), cols(cols), size_of_values(rows * cols), preallocated(preallocate)
{
   // If we want to handle memory ourselves
   if (this->preallocated)
   {
      // Must remember to delete this in the destructor
      this->values = new double[size_of_values];
   }
}

// Constructor - now just setting the value of our double pointer
Matrix::Matrix(int rows, int cols, double *values_ptr) : rows(rows), cols(cols), size_of_values(rows * cols), values(values_ptr)
{
}

// destructor
Matrix::~Matrix()
{
   // Delete the values array
   if (this->preallocated)
   {
      delete[] this->values;
   }
}

// Just print out the values in our values array
void Matrix::printValues()
{
   std::cout << "Printing values" << std::endl;
   for (int i = 0; i < this->size_of_values; i++)
   {
      std::cout << this->values[i] << " ";
   }
   std::cout << std::endl;
}

// Explicitly print out the values in values array as if they are a matrix
void Matrix::printMatrix()
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

// Do matrix matrix multiplication
// output = this * mat_right
void Matrix::matMatMult(Matrix &mat_right, Matrix &output)
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
      output.values = new double[this->rows * mat_right.cols];
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
   for (int i = 0; i < this->cols; i++)
   {
      for (int k = 0; k < this->rows; k++)
      {
         for (int j = 0; j < mat_right.rows; j++)
         {
            output.values[i * output.rows + j] += this->values[i * this->rows + k] * mat_right.values[k * mat_right.rows + j];
         }
      }
   }
}