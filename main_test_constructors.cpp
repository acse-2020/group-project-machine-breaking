#include <iostream>
#include <math.h>
#include <ctime>
#include <vector>
#include "Matrix.h"
#include "Matrix.cpp"
#include "CSRMatrix.h"
#include "CSRMatrix.cpp"
#include "Solver.h"
#include "Solver.cpp"
#include "SparseSolver.h"
#include "SparseSolver.cpp"
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
    Matrix<double> LU(dense_mat.rows, dense_mat.cols, true);
    t = clock();
    auto piv = solver_example->lu_decomp(LU);
    solver_example->lu_solve(LU, piv, x_lu);  // would be better to input b here
    t = clock() - t;
    std::cout << "time: " << ((float)t) / CLOCKS_PER_SEC << std::endl;
    printVector(x_lu);

    delete solver_example;
    
    // sparse matrix solver. Result should be: {3.2, 7.8, 5.9, 7.3}
    std::cout << std::endl
              << "Sparse Matrix: " << std::endl;
    int nnzs = 4;
    int init_row_position[] = {0, 1, 2, 3, 4};
    int init_col_index[] = {0, 1, 2, 3};
    double init_sparse_values[] = {2, 1, 3, 7};
    std::vector<double> b_sparse = {6.4, 7.8, 56.7, 51.1};
    std::vector<double> x_sparse(rows, 0);
    std::vector<double> x_sparse_CG(rows, 0);

    auto sparse_matrix = CSRMatrix<double>(rows, cols, nnzs, &init_sparse_values[0], &init_row_position[0], &init_col_index[0]);
    auto sparse_solver = SparseSolver<double>(sparse_matrix, b_sparse);
    sparse_solver.stationaryIterative(x_sparse, tol, it_max, false);
    printVector(x_sparse);

    sparse_solver.conjugateGradient(x_sparse_CG, tol, it_max);
    printVector(x_sparse_CG);
}