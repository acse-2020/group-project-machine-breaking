#include <iostream>
#include "Matrix.h"
#include <math.h>

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

// Jacobi method to solve linear system of equations (Ax=b)
// Based on algorithm provided in Lecture 3 of ACSE3
void Matrix::Jacobi(Matrix &RHS, Matrix &unknowns, double &tol, int &it_max)
{
   // Initialise residual, matrix for row-matrix multiplication
   // and matrix for storing previous iteration
   double residual;
   Matrix new_array(unknowns.rows, unknowns.cols, true);
   Matrix x_old(unknowns.rows, unknowns.cols, true);

   // Check our dimensions match
   if (this->cols != RHS.rows)
   {
      std::cerr << "Input dimensions for matrices don't match" << std::endl;
      return;
   }

   // Check if our output matrix has had space allocated to it
   if (unknowns.values != nullptr)
   {
      // Check our dimensions match
      if (this->rows != unknowns.rows || RHS.cols != unknowns.cols)
      {
         std::cerr << "Input dimensions for matrices don't match" << std::endl;
         return;
      }
   }
   // The output hasn't been preallocated, so we are going to do that
   else
   {
      unknowns.values = new double[this->rows * RHS.cols];
      unknowns.preallocated = true;
   }

   // Set values to zero before hand
   for (int i = 0; i < unknowns.size_of_values; i++)
   {
      unknowns.values[i] = 0;
      x_old.values[i] = unknowns.values[i];
   }

   for (int k = 0; k < it_max; k++)
   {
      for (int i = 0; i < this->rows; i++)
      {
         double sum = 0;
         for (int j = 0; j < this->rows; j++)
         {
            if (j != i)
            {
               sum += this->values[i * this->cols + j] * x_old.values[j];
            }
         }
         unknowns.values[i] = (1.0 / this->values[i + i * this->rows]) * (RHS.values[i] - sum);
         //(1. / A[i, i]) * (b[i] - (A [i, :i] @x[:i]) - (A [i, i + 1:] @x [i + 1:]))
      }

      // COULD JUST HAVE new_array AS VECTOR. REVISIT
      // A x = b(estimate)
      matMatMult(unknowns, new_array);

      // Find the norm between old value and new guess
      residual = 0;
      for (int i = 0; i < this->rows; i++)
      {
         residual += pow(abs(new_array.values[i] - RHS.values[i]), 2);
      }
      residual = sqrt(residual);

      // End iterations if tolerance convergence is reached
      if (residual < tol)
      {
         break;
      }

      // Update the solution from previous iteration with new estimate
      for (int i = 0; i < unknowns.size_of_values; i++)
      {
         x_old.values[i] = unknowns.values[i];
      }
   }
}