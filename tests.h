#include <iostream>
#include <math.h>
#include <chrono>
#include <vector>
#include <memory>
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

// test functions should start with 'test_' prefix
bool test_sparse_matmatmult_5x5()
{
    int nnzs = 14;
    int size = 5;

    std::shared_ptr<int[]> init_row_position1(new int[size + 1] { 0, 2, 4, 7, 11, 14 });
    std::shared_ptr<int[]> init_col_index1(new int[nnzs] { 0, 4, 0, 1, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4 });
    std::shared_ptr<double[]> init_sparse_values1(new double[nnzs] { 10, -2, 3, 9, 7, 8, 7, 3, 8, 7, 5, 8, 9, 13 });
    CSRMatrix<double> sparse_matrix = CSRMatrix<double>(size, size, nnzs, init_sparse_values1, init_row_position1, init_col_index1);

    auto result = sparse_matrix.matMatMult(sparse_matrix);

    double expected_values[] = {100, -16, -18, -46, 57, 81, -6, 42, 119, 120, 105, 35, 51, 96, 120, 150, 94, 51, 176, 72, 180, 214};
    int expected_nnzs = 22;

    if (expected_nnzs != result->nnzs)
    {
        std::cerr << "nnzs do not match" << std::endl;
        return false;
    }
    bool outcome = TestRunner::assertArrays(&expected_values[0], &result->values[0], expected_nnzs);
    return outcome;
}

bool test_sparse_matmatmult_4x4()
{
    int nnzs = 4;
    int size = 4;

    std::shared_ptr<int[]> init_row_position1(new int[size + 1]{ 0, 2, 3, 4, 4 });
    std::shared_ptr<int[]> init_col_index1(new int[nnzs]{ 1, 3, 0, 1 });
    std::shared_ptr<double[]> init_sparse_values1(new double[nnzs]{ 1, 1, 1, 2 });
    CSRMatrix<double> sparse_matrix1 = CSRMatrix<double>(size, size, nnzs, init_sparse_values1, init_row_position1, init_col_index1);

    std::shared_ptr<int[]> init_row_position2(new int[4 + 1] { 0, 0, 2, 3, 4 });
    std::shared_ptr<int[]> init_col_index2(new int[nnzs] { 0, 2, 3, 2 });
    std::shared_ptr<double[]> init_sparse_values2(new double[nnzs] { 1, 1, 2, 1 });
    CSRMatrix<double> sparse_matrix2 = CSRMatrix<double>(4, 4, nnzs, init_sparse_values2, init_row_position2, init_col_index2);

    auto result = sparse_matrix1.matMatMult(sparse_matrix2);

    int expected_nnzs = 4;
    double expected_values[] = { 1, 2, 2, 2 };
    int expected_row_pos[] = { 0, 2, 2, 4, 4 };
    int expected_col_ind[] = { 0, 2, 0, 2 };

    bool vals = TestRunner::assertArrays(&expected_values[0], &result->values[0], expected_nnzs);
    bool rows = TestRunner::assertArrays(&expected_row_pos[0], &result->row_position[0], expected_nnzs);
    bool cols = TestRunner::assertArrays(&expected_col_ind[0], &result->col_index[0], expected_nnzs);
    bool outcome = vals && rows && cols;

    return outcome;
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

    std::shared_ptr<double[]> init_dense_values(new double[size * size]
    { 10., 2., 3., 5., 1., 14., 6., 2., -1., 4., 16., -4, 5., 4., 3., 11. });

    Matrix<double> dense_mat = Matrix<double>(size, size, init_dense_values);
    std::vector<double> b = {1., 2., 3., 4.};

    Solver<double> dense_solver = Solver<double>(dense_mat, b);

    std::vector<double> x_j(size, 0);
    std::vector<double> x_gs(size, 0);

    auto t1 = std::chrono::high_resolution_clock::now();
    dense_solver.stationaryIterative(x_j, tol, it_max, false);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "Time taken for Jacobi: " << duration << " s " << std::endl << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    dense_solver.stationaryIterative(x_gs, tol, it_max, true);
    t2 = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "Time taken for Gauss Seidel: " << duration << " s " << std::endl << std::endl;

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

    std::shared_ptr<int[]> init_row_position(new int[size + 1]{ 0, 1, 2, 3, 4 });
    std::shared_ptr<int[]> init_col_index(new int[nnzs] {0, 1, 2, 3});
    std::shared_ptr<double[]> init_sparse_values(new double[nnzs] { 2, 1, 3, 7 });

    std::vector<double> b = {6.4, 7.8, 56.7, 51.1};
    std::vector<double> x(size, 0);

    CSRMatrix<double> sparse_matrix = CSRMatrix<double>(size, size, nnzs, init_sparse_values, init_row_position, init_col_index);
    SparseSolver<double> sparse_solver = SparseSolver<double>(sparse_matrix, b);

    auto t1 = std::chrono::high_resolution_clock::now();
    sparse_solver.stationaryIterative(x, tol, it_max, false);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "Time taken for sparse Jacobi: " << duration << " s " << std::endl << std::endl;

    std::vector<double> b_estimate(size, 0);
    
    printVector(x);

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

    std::shared_ptr<int[]> init_row_position(new int[size + 1] {0, 1, 2, 3, 4});
    std::shared_ptr<int[]> init_col_index(new int[nnzs]{0, 1, 2, 3});
    std::shared_ptr<double[]> init_sparse_values(new double[nnzs]{ 2, 1, 3, 7 });

    std::vector<double> b = {6.4, 7.8, 56.7, 51.1};
    std::vector<double> x(size, 0);

    CSRMatrix<double> sparse_matrix = CSRMatrix<double>(size, size, nnzs, init_sparse_values, init_row_position, init_col_index);
    SparseSolver<double> sparse_solver = SparseSolver<double>(sparse_matrix, b);

    auto t1 = std::chrono::high_resolution_clock::now();
    sparse_solver.conjugateGradient(x, tol, it_max);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "Time taken for sparse conjugate gradient solver: " << duration << " s " << std::endl << std::endl;

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
    std::vector<double> b_estimate(size, 0);

    std::shared_ptr<double[]> init_dense_values(new double[size * size]
    {10., 2., 3., 5., 1., 14., 6., 2., -1., 4., 16., -4, 5., 4., 3., 11.});

    Matrix<double> dense_mat = Matrix<double>(size, size, init_dense_values);
    std::vector<double> b = {1., 2., 3., 4.};

    Solver<double> dense_solver = Solver<double>(dense_mat, b);

    Matrix<double> LU(size, size, true);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<int> piv = dense_solver.lu_decomp(LU);
    dense_solver.lu_solve(LU, piv, x);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "Time taken for LU: " << duration << " s " << std::endl << std::endl;

    if (dense_solver.residualCalc(x, b_estimate) > 1e-6)
    {
        TestRunner::testError("LU residual is above 1e-6");
        return false;
    }

    return true;
}

bool test_lu_dense_random()
{
    int size = 100;
    std::vector<double> x(size, 0);
    std::vector<double> b_estimate(size, 0);

    auto solver = Solver<double>(size);

    Matrix<double> LU(size, size, true);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<int> piv = solver.lu_decomp(LU);
    solver.lu_solve(LU, piv, x);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "Time taken for LU: " << duration << " s " << std::endl << std::endl;

    if (solver.residualCalc(x, b_estimate) > 1e-6)
    {
         TestRunner::testError("LU residual is above 1e-6");
         return false;
    }
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
    test_runner.test(&test_lu_dense_random, "dense LU with random matrix of size 100x100.");
}