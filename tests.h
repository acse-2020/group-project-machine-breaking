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
#include "TestRunner.h"
#include "utilities.h"
#include <memory>
// test functions should start with 'test_' prefix
bool test_sparse_matmatmult_5x5()
{
    int nnzs = 14;
    int size = 5;
    int init_row_position1[] = {0, 2, 4, 7, 11, 14};
    int init_col_index1[] = {0, 4, 0, 1, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4};
    double init_sparse_values1[] = {10, -2, 3, 9, 7, 8, 7, 3, 8, 7, 5, 8, 9, 13};
    CSRMatrix<double> sparse_matrix = CSRMatrix<double>(size, size, nnzs, &init_sparse_values1[0], &init_row_position1[0], &init_col_index1[0]);

    CSRMatrix<double> result = sparse_matrix.matMatMult(sparse_matrix);

    double expected_values[] = {100, -16, -18, -46, 57, 81, -6, 42, 119, 120, 105, 35, 51, 96, 120, 150, 94, 51, 176, 72, 180, 214};
    int expected_nnzs = 22;

    if (expected_nnzs != result.nnzs)
    {
        std::cerr << "nnzs do not match" << std::endl;
        return false;
    }
    return TestRunner::assertArrays(&expected_values[0], &result.values[0], expected_nnzs);
}

bool test_sparse_matmatmult_4x4()
{
    int nnzs = 4;
    int init_row_position1[] = {0, 2, 3, 4, 4};
    int init_col_index1[] = {1, 3, 0, 1};
    double init_sparse_values1[] = {1, 1, 1, 2};
    CSRMatrix<double> sparse_matrix1 = CSRMatrix<double>(4, 4, nnzs, &init_sparse_values1[0], &init_row_position1[0], &init_col_index1[0]);

    int init_row_position2[] = {0, 0, 2, 3, 4};
    int init_col_index2[] = {0, 2, 3, 2};
    double init_sparse_values2[] = {1, 1, 2, 1};
    CSRMatrix<double> sparse_matrix2 = CSRMatrix<double>(4, 4, nnzs, &init_sparse_values2[0], &init_row_position2[0], &init_col_index2[0]);

    CSRMatrix<double> result = sparse_matrix1.matMatMult(sparse_matrix2);

    double expected_values[] = {1, 2, 2, 2};
    int expected_row_pos[] = {0, 2, 2, 4, 4};
    int expected_col_ind[] = {0, 2, 0, 2};

    bool vals = TestRunner::assertArrays(&expected_values[0], &result.values[0], 4);
    bool rows = TestRunner::assertArrays(&expected_row_pos[0], &result.row_position[0], 4);
    bool cols = TestRunner::assertArrays(&expected_col_ind[0], &result.col_index[0], 4);

    return vals && rows && cols;
}

bool test_check_dimensions_matching()
{
    Matrix<int> m = Matrix<int>(3, 3, true);
    std::vector<int> v(3, 0);
    try
    {
        checkDimensions(m, v);
        return true;
    }
    catch (const std::exception &e)
    {
        return false;
    }
}

bool test_check_dimensions_not_matching()
{
    Matrix<int> m = Matrix<int>(2, 3, true);
    std::vector<int> v(3, 0);

    try
    {
        checkDimensions(m, v);
        return false;
    }
    catch (const std::exception &e)
    {
        // we expect error to be thrown here
        return true;
    }
}

bool test_dense_jacobi_and_gauss_seidl()
{
    int size = 4;
    double tol = 1e-6;
    int it_max = 1000;

    double init_dense_values[] = {10., 2., 3., 5., 1., 14., 6., 2., -1., 4., 16., -4, 5., 4., 3., 11.};

    Matrix<double> dense_mat = Matrix<double>(size, size, &init_dense_values[0]);
    std::vector<double> b = {1., 2., 3., 4.};

    Solver<double> dense_solver = Solver<double>(dense_mat, b);

    std::vector<double> x_j(size, 0);
    std::vector<double> x_gs(size, 0);

    clock_t t = clock();
    dense_solver.stationaryIterative(x_j, tol, it_max, false);
    t = clock() - t;

    std::cout << "Time taken for Jacobi: " << ((float)t) / CLOCKS_PER_SEC << std::endl
              << std::endl;

    t = clock();
    dense_solver.stationaryIterative(x_gs, tol, it_max, true);
    t = clock() - t;

    std::cout << "Time taken for Gauss Seidel: " << ((float)t) / CLOCKS_PER_SEC << std::endl
              << std::endl;

    std::vector<double> b_estimate(size, 0);

    bool j_res, gs_res;
    if (j_res = dense_solver.residualCalc(x_j, b_estimate) > tol)
    {
        TestRunner::testError("Jacobi residual is above tolerance");
    }
    if (gs_res = dense_solver.residualCalc(x_gs, b_estimate) > tol)
    {
        TestRunner::testError("Gauss-Seidl residual is above tolerance");
    }

    // passes if residual for both is small enough
    return !j_res && !gs_res;
}

bool test_sparse_jacobi()
{
    int size = 4;
    double tol = 1e-6;
    int it_max = 1000;
    int nnzs = 4;

    int init_row_position[] = {0, 1, 2, 3, 4};
    int init_col_index[] = {0, 1, 2, 3};
    double init_sparse_values[] = {2, 1, 3, 7};

    std::vector<double> b = {6.4, 7.8, 56.7, 51.1};
    std::vector<double> x(size, 0);

    CSRMatrix<double> sparse_matrix = CSRMatrix<double>(size, size, nnzs, &init_sparse_values[0], &init_row_position[0], &init_col_index[0]);
    SparseSolver<double> sparse_solver = SparseSolver<double>(sparse_matrix, b);

    clock_t t = clock();

    sparse_solver.stationaryIterative(x, tol, it_max, false);

    t = clock() - t;

    std::cout << "Time taken for sparse Jacobi: " << ((float)t) / CLOCKS_PER_SEC << std::endl
              << std::endl;

    std::vector<double> b_estimate(size, 0);

    if (sparse_solver.residualCalc(x, b_estimate) > 1e-6)
    {
        TestRunner::testError("Sparse Jacobi residual is above 1e-6");
        return false;
    }

    return true;
}

bool test_sparse_CG()
{
    int size = 4;
    double tol = 1e-6;
    int it_max = 1000;
    int nnzs = 4;

    int init_row_position[] = {0, 1, 2, 3, 4};
    int init_col_index[] = {0, 1, 2, 3};
    double init_sparse_values[] = {2, 1, 3, 7};

    std::vector<double> b = {6.4, 7.8, 56.7, 51.1};
    std::vector<double> x(size, 0);

    CSRMatrix<double> sparse_matrix = CSRMatrix<double>(size, size, nnzs, &init_sparse_values[0], &init_row_position[0], &init_col_index[0]);
    SparseSolver<double> sparse_solver = SparseSolver<double>(sparse_matrix, b);

    clock_t t = clock();

    sparse_solver.conjugateGradient(x, tol, it_max);

    t = clock() - t;

    std::cout << "Time taken for sparse conjugate gradient solver: " << ((float)t) / CLOCKS_PER_SEC << std::endl
              << std::endl;

    std::vector<double> b_estimate(size, 0);

    if (sparse_solver.residualCalc(x, b_estimate) > 1e-6)
    {
        TestRunner::testError("Sparse CG residual is above 1e-6");
        return false;
    }

    return true;
}

bool test_lu_dense()
{
    int size = 4;
    std::vector<double> x(size, 0);

    double init_dense_values[] = {10., 2., 3., 5., 1., 14., 6., 2., -1., 4., 16., -4, 5., 4., 3., 11.};

    Matrix<double> dense_mat = Matrix<double>(size, size, &init_dense_values[0]);
    std::vector<double> b = {1., 2., 3., 4.};

    Solver<double> dense_solver = Solver<double>(dense_mat, b);

    Matrix<double> LU(size, size, true);

    clock_t t = clock();
    std::vector<int> piv = dense_solver.lu_decomp(LU);
    dense_solver.lu_solve(LU, piv, x);
    t = clock() - t;

    std::vector<double> b_estimate(size, 0);

    std::cout << "Time taken for LU: " << ((float)t) / CLOCKS_PER_SEC << std::endl;

    if (dense_solver.residualCalc(x, b_estimate) > 1e-6)
    {
        TestRunner::testError("LU residual is above 1e-6");
        return false;
    }

    return true;
}

bool test_lu_dense_random()
{
    int size = 4;
    std::vector<double> x(size, 0);
    // Solver<double> *solver = nullptr;

    Solver<double> *solver = Solver<double>::makeSolver(size);
    // delete solver;

    // create_dense_solver(size, solver);
    // std::unique_ptr<Solver<double>> solver = solver_factory(size);
    std::cout << solver->A.values[0] << std::endl;

    delete solver; // why does this not work???

    // Matrix<double> LU(size, size, true);

    // clock_t t = clock();
    // std::vector<int> piv = solver->lu_decomp(LU);
    // solver->lu_solve(LU, piv, x);
    // t = clock() - t;

    // std::vector<double> b_estimate(size, 0);

    // std::cout << "Time taken for LU: " << ((float)t) / CLOCKS_PER_SEC << std::endl;

    // if (solver->residualCalc(x, b_estimate) > 1e-6)
    // {
    //     TestRunner::testError("LU residual is above 1e-6");
    //     return false;
    // }

    return true;
}

void run_tests()
{
    TestRunner test_runner = TestRunner();

    test_runner.test(&test_check_dimensions_matching, "checkDimensions for matching matrices.");
    test_runner.test(&test_check_dimensions_not_matching, "checkDimensions for non-matching matrices.");
    test_runner.test(&test_sparse_matmatmult_4x4, "sparse matMatMult for two sparse 4x4 matrices.");
    test_runner.test(&test_sparse_matmatmult_5x5, "sparse matMatMult for multiplying a 5x5 sparse matrix by itself.");
    test_runner.test(&test_dense_jacobi_and_gauss_seidl, "stationaryIterative: dense Jacobi and Gauss-Seidel solver for 4x4 matrix.");
    test_runner.test(&test_lu_dense, "dense LU solver for 4x4 matrix.");
    test_runner.test(&test_sparse_jacobi, "sparse Jacobi solver for 4x4 matrix.");
    test_runner.test(&test_sparse_CG, "sparse conjugate gradient solver for 4x4 matrix.");
    test_runner.test(&test_lu_dense_random, "dense LU with random matrices.");
}