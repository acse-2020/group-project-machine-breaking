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

void test_matmatmult_4x4()
{
    std::cout << std::endl
              << "Test matMatMult for 4x4 sparse matrices: " << std::endl
              << std::endl;
    int nnzs = 4;
    int init_row_position1[] = {0, 2, 3, 4, 4};
    int init_col_index1[] = {1, 3, 0, 1};
    double init_sparse_values1[] = {1, 1, 1, 2};
    auto sparse_matrix1 = CSRMatrix<double>(4, 4, nnzs, &init_sparse_values1[0], &init_row_position1[0], &init_col_index1[0]);

    int init_row_position2[] = {0, 0, 2, 3, 4};
    int init_col_index2[] = {0, 2, 3, 2};
    double init_sparse_values2[] = {1, 1, 2, 1};
    auto sparse_matrix2 = CSRMatrix<double>(4, 4, nnzs, &init_sparse_values2[0], &init_row_position2[0], &init_col_index2[0]);

    CSRMatrix<double> result = sparse_matrix1.matMatMult(sparse_matrix2);
    result.printMatrix();
}

void test_matmatmult_5x5()
{
    std::cout << std::endl
              << "Test matMatMult for 5x5 sparse matrix multiplied by itself: " << std::endl
              << std::endl;
    int nnzs = 14;
    int init_row_position1[] = {0, 2, 4, 7, 11, 14};
    int init_col_index1[] = {0, 4, 0, 1, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4};
    double init_sparse_values1[] = {10, -2, 3, 9, 7, 8, 7, 3, 8, 7, 5, 8, 9, 13};
    auto sparse_matrix = CSRMatrix<double>(5, 5, nnzs, &init_sparse_values1[0], &init_row_position1[0], &init_col_index1[0]);

    CSRMatrix<double> result = sparse_matrix.matMatMult(sparse_matrix);
    result.printMatrix();
}

int main()
{
    int rows = 4;
    int cols = 4;
    /*double tol = 1e-6;
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
    solver_example->lu_solve(LU, piv, x_lu); // would be better to input b here
    t = clock() - t;
    std::cout << "time: " << ((float)t) / CLOCKS_PER_SEC << std::endl;
    printVector(x_lu);

    delete solver_example;
    */
    // sparse matrix solver. Result should be: {3.2, 7.8, 5.9, 7.3}
    std::cout << std::endl
              << "Sparse Matrix: " << std::endl;
    int nnzs = 4;
    //int init_row_position[] = {0, 1, 2, 3, 4};
    //int init_col_index[] = {0, 1, 2, 3};
    //double init_sparse_values[] = {2, 1, 3, 7};
    //std::vector<double> b_sparse = {6.4, 7.8, 56.7, 51.1};
    int init_row_position[] = {0, 2, 4, 7, 11, 14};
    int init_col_index[] = {0, 4, 0, 1, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4};
    double init_sparse_values[] = {10, -2, 3, 9, 7, 8, 7, 3, 8, 7, 5, 8, 9, 13};
    std::vector<double> b_sparse = {6.4, 7.8, 56.7, 51.1};
    
    std::vector<double> x_sparse(rows, 0);
    std::vector<double> x_sparse_CG(rows, 0);
    std::vector<double> x_sparse_LU(rows, 0);
    
    auto sparse_matrix = CSRMatrix<double>(rows, cols, nnzs, &init_sparse_values[0], &init_row_position[0], &init_col_index[0]);
    auto sparse_solver = SparseSolver<double>(sparse_matrix, b_sparse);
    /*sparse_solver.stationaryIterative(x_sparse, tol, it_max, false);
    printVector(x_sparse);

    sparse_solver.conjugateGradient(x_sparse_CG, tol, it_max);
    printVector(x_sparse_CG);
    */

    auto LU_sparse = CSRMatrix<double>(rows, cols, nnzs, true);
    //auto piv = sparse_solver.lu_decomp(LU_sparse);
    std::vector<int> piv = {0, 1, 2, 3};
    LU_sparse.print2DMatrix();
    sparse_solver.lu_solve(LU_sparse, piv, x_sparse_LU);
    sparse_matrix.print2DMatrix();
    LU_sparse.print2DMatrix();
    printVector(x_sparse_LU);

    //test_matmatmult_4x4();
    //test_matmatmult_5x5();
}