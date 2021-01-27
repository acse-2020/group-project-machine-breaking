#include <iostream>
#include <math.h>
#include <ctime>
#include "Matrix.h"
#include "Matrix.cpp"
#include "Solver.h"
#include "Solver.cpp"

using namespace std;

int main()
{
    int rows = 4;
    int cols = 4;
    double tol = 1e-6;
    int it_max = 1000;

    double init_dense_values[] = { 10., 2., 3., 5., 1., 14., 6., 2., -1., 4., 16., -4, 5., 4., 3., 11. };

    double init_RHS_values[] = { 1., 2., 3., 4. };

    // Testing our matrix class
    auto* dense_mat = new Matrix<double>(rows, cols, true);
    auto* RHS = new Matrix<double>(rows, 1, true);
    auto* unknowns_j = new Matrix<double>(rows, 1, true);
    auto* unknowns_gs = new Matrix<double>(rows, 1, true);
    auto* unknowns_lu = new Matrix<double>(rows, 1, true);

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
        unknowns_j->values[i] = 0;
        unknowns_gs->values[i] = 0;
        unknowns_lu->values[i] = 0;
    }

    // testing our solver
    auto* solver_example = new Solver<double>(dense_mat, RHS);

    dense_mat->printMatrix();
    clock_t t = clock();
    solver_example->jacobi(*unknowns_j, tol, it_max);
    t = clock() - t;
    std::cout << "time: " << ((float)t) / CLOCKS_PER_SEC << std::endl;

    t = clock();
    solver_example->gaussSeidel(*unknowns_gs, tol, it_max);
    t = clock() - t;
    std::cout << "time: " << ((float)t) / CLOCKS_PER_SEC << std::endl;
    unknowns_j->printMatrix();
    unknowns_gs->printMatrix();


    dense_mat->printMatrix();
    clock_t t = clock();
    solver_example->lu_solve(*unknowns_lu, tol, it_max);
    t = clock() - t;
    std::cout << "time: " << ((float)t) / CLOCKS_PER_SEC << std::endl;

    delete dense_mat;
    delete RHS;
    delete unknowns_j;
    delete unknowns_gs;
    delete unknowns_lu;
    delete solver_example;
}