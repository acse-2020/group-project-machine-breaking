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

    double init_dense_values[] = {10., 2., 3., 5., 1., 14., 6., 2., -1., 4., 16., -4, 5., 4., 3., 11.};

    double init_b_values[] = {1., 2., 3., 4.};

    // Testing our matrix class
    auto *dense_mat = new Matrix<double>(rows, cols, true);
    auto *b = new Matrix<double>(rows, 1, true);
    auto *x_j = new Matrix<double>(rows, 1, true);
    auto *x_gs = new Matrix<double>(rows, 1, true);
    auto *x_lu = new Matrix<double>(rows, 1, true);

    // Now we need to go and fill our matrices
    for (int i = 0; i < rows * cols; i++)
    {
        dense_mat->values[i] = init_dense_values[i];
    }

    for (int i = 0; i < rows; i++)
    {
        b->values[i] = init_b_values[i];
    }

    for (int i = 0; i < rows; i++)
    {
        x_j->values[i] = 0;
        x_gs->values[i] = 0;
        x_lu->values[i] = 0;
    }

    // testing our solver
    auto *solver_example = new Solver<double>(dense_mat, b);

    dense_mat->printMatrix();
    clock_t t = clock();
    solver_example->stationaryIterative(*x_j, tol, it_max, false);
    t = clock() - t;
    std::cout << "time: " << ((float)t) / CLOCKS_PER_SEC << std::endl;

    t = clock();
    solver_example->stationaryIterative(*x_gs, tol, it_max, true);
    t = clock() - t;
    std::cout << "time: " << ((float)t) / CLOCKS_PER_SEC << std::endl;
    x_j->printMatrix();
    x_gs->printMatrix();

    dense_mat->printMatrix();
    t = clock();
    solver_example->lu_solve(*x_lu);
    t = clock() - t;
    std::cout << "time: " << ((float)t) / CLOCKS_PER_SEC << std::endl;
    x_lu->printMatrix();

    delete dense_mat;
    delete b;
    delete x_j;
    delete x_gs;
    delete x_lu;
    delete solver_example;
}