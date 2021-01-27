#include <iostream>
#include <math.h>
#include <ctime>
#include "Matrix.h"
#include "Matrix.cpp"

using namespace std;

int main()
{
   int rows = 4;
   int cols = 4;
   double tol = 1e-6;
   int it_max = 1000;

   double init_dense_values[] = {10., 2., 3., 5., 1., 14., 6., 2., -1., 4., 16., -4, 5., 4., 3., 11.};

   double init_RHS_values[] = {1., 2., 3., 4.};

   // Testing our matrix class
   auto *dense_mat = new Matrix<double>(rows, cols, true);
   auto *RHS = new Matrix<double>(rows, 1, true);
   auto *unknowns = new Matrix<double>(rows, 1, true);

   // Now we need to go and fill our matrices
   for (int i = 0; i < rows * cols; i++)
   {
      dense_mat->values[i] = init_dense_values[i];
   }

   for (int i = 0; i < rows; i++)
   {
      RHS->values[i] = init_RHS_values[i];
   }

   for (int i = 0; i < rows; i++)
   {
      unknowns->values[i] = 0;
   }
   dense_mat->printMatrix();
   dense_mat->Jacobi(*RHS, *unknowns, tol, it_max);
   unknowns->printMatrix();

   delete dense_mat;
   delete RHS;
   delete unknowns;
}