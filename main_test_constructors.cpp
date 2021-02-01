#include <iostream>
#include <math.h>
#include <ctime>
#include <vector>
#include "Matrix.h"
#include "Matrix.cpp"
#include "Solver.h"
#include "Solver.cpp"
#include "utilities.h"

//using namespace std;

int main()
{
    int rows = 4;
    int cols = 4;
    double tol = 1e-6;
    int it_max = 1000;

    double init_dense_values[] = {10., 2., 3., 5., 1., 14., 6., 2., -1., 4., 16., -4, 5., 4., 3., 11.};

    // Testing our matrix class
    auto dense_mat = Matrix<double>(rows, cols, &init_dense_values[0]);
    std::vector<double> b = {1., 2., 3., 4.};
    std::vector<double> x_j(rows, 0);
    std::vector<double> x_gs(rows, 0);
    std::vector<double> x_lu(rows, 0);

    // testing our solver
    auto *solver_example = new Solver<double>(dense_mat, b);

    dense_mat.printMatrix();
    clock_t t = clock();
    solver_example->stationaryIterative(x_j, tol, it_max, false);
    t = clock() - t;
    std::cout << "time: " << ((float)t) / CLOCKS_PER_SEC << std::endl;

    t = clock();
    solver_example->stationaryIterative(x_gs, tol, it_max, true);
    t = clock() - t;
    std::cout << "time: " << ((float)t) / CLOCKS_PER_SEC << std::endl;
    printVector(x_j);
    printVector(x_gs);

    dense_mat.printMatrix();
    t = clock();
    solver_example->lu_solve(x_lu);
    t = clock() - t;
    std::cout << "time: " << ((float)t) / CLOCKS_PER_SEC << std::endl;
    printVector(x_lu);

    delete solver_example;
}