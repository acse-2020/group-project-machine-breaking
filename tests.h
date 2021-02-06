#include <iostream>
#include <math.h>
#include <ctime>
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
#include <memory>

// test functions should start with 'test_' prefix
bool test_sparse_matmatmult_5x5()
{
    int nnzs = 14;
    int size = 5;

    std::shared_ptr<int[]> init_row_position1(new int[size + 1]{0, 2, 4, 7, 11, 14});
    std::shared_ptr<int[]> init_col_index1(new int[nnzs]{0, 4, 0, 1, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4});
    std::shared_ptr<double[]> init_sparse_values1(new double[nnzs]{10, -2, 3, 9, 7, 8, 7, 3, 8, 7, 5, 8, 9, 13});
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

    std::shared_ptr<int[]> init_row_position1(new int[size + 1]{0, 2, 3, 4, 4});
    std::shared_ptr<int[]> init_col_index1(new int[nnzs]{1, 3, 0, 1});
    std::shared_ptr<double[]> init_sparse_values1(new double[nnzs]{1, 1, 1, 2});
    CSRMatrix<double> sparse_matrix1 = CSRMatrix<double>(size, size, nnzs, init_sparse_values1, init_row_position1, init_col_index1);

    std::shared_ptr<int[]> init_row_position2(new int[4 + 1]{0, 0, 2, 3, 4});
    std::shared_ptr<int[]> init_col_index2(new int[nnzs]{0, 2, 3, 2});
    std::shared_ptr<double[]> init_sparse_values2(new double[nnzs]{1, 1, 2, 1});
    CSRMatrix<double> sparse_matrix2 = CSRMatrix<double>(4, 4, nnzs, init_sparse_values2, init_row_position2, init_col_index2);

    auto result = sparse_matrix1.matMatMult(sparse_matrix2);

    int expected_nnzs = 4;
    double expected_values[] = {1, 2, 2, 2};
    int expected_row_pos[] = {0, 2, 2, 4, 4};
    int expected_col_ind[] = {0, 2, 0, 2};

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

    std::shared_ptr<double[]> init_dense_values(new double[size * size]{10., 2., 3., 5., 1., 14., 6., 2., -1., 4., 16., -4, 5., 4., 3., 11.});

    Matrix<double> dense_mat = Matrix<double>(size, size, init_dense_values);
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

    std::shared_ptr<int[]> init_row_position(new int[size + 1]{0, 1, 2, 3, 4});
    std::shared_ptr<int[]> init_col_index(new int[nnzs]{0, 1, 2, 3});
    std::shared_ptr<double[]> init_sparse_values(new double[nnzs]{2, 1, 3, 7});

    std::vector<double> b = {6.4, 7.8, 56.7, 51.1};
    std::vector<double> x(size, 0);

    CSRMatrix<double> sparse_matrix = CSRMatrix<double>(size, size, nnzs, init_sparse_values, init_row_position, init_col_index);
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

    std::shared_ptr<int[]> init_row_position(new int[size + 1]{0, 1, 2, 3, 4});
    std::shared_ptr<int[]> init_col_index(new int[nnzs]{0, 1, 2, 3});
    std::shared_ptr<double[]> init_sparse_values(new double[nnzs]{2, 1, 3, 7});

    std::vector<double> b = {6.4, 7.8, 56.7, 51.1};
    std::vector<double> x(size, 0);

    CSRMatrix<double> sparse_matrix = CSRMatrix<double>(size, size, nnzs, init_sparse_values, init_row_position, init_col_index);
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

    std::shared_ptr<double[]> init_dense_values(new double[size * size]{10., 2., 3., 5., 1., 14., 6., 2., -1., 4., 16., -4, 5., 4., 3., 11.});

    Matrix<double> dense_mat = Matrix<double>(size, size, init_dense_values);
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

    //auto *solver = new Solver<double>::makeSolver(size);

    //double init_dense_values[] = { 10., 2., 3., 5., 1., 14., 6., 2., -1., 4., 16., -4, 5., 4., 3., 11. };
    std::shared_ptr<double[]> init_dense_values(new double[size * size]{10., 2., 3., 5., 1., 14., 6., 2., -1., 4., 16., -4, 5., 4., 3., 11.});
    //std::unique_ptr<double[]> init_dense_values(new double[size * size]);
    //(10., 2., 3., 5., 1., 14., 6., 2., -1., 4., 16., -4, 5., 4., 3., 11.);
    //init_dense_values = { 10., 2., 3., 5., 1., 14., 6., 2., -1., 4., 16., -4, 5., 4., 3., 11. };
    //unique_ptr<A> ptr = unique_ptr<A>(new A(1234));

    Matrix<double> dense_mat = Matrix<double>(size, size, init_dense_values);
    //Matrix<double> copy_mat = dense_mat;
    std::cout << dense_mat.values[0] << " " << &dense_mat.values[0] << std::endl;
    //std::cout << copy_mat.values[0] << " " << &copy_mat.values[0] << std::endl;
    std::vector<double> b = {1., 2., 3., 4.};

    Solver<double> dense_solver = Solver<double>(dense_mat, b);
    std::cout << dense_solver.A.values[0] << " " << &dense_solver.A.values[0] << std::endl;
    Solver<double> solver = dense_solver;
    std::cout << solver.b[0] << " " << &solver.b[0] << std::endl;
    std::cout << solver.A.values[0] << " " << &solver.A.values[0] << std::endl;
    // delete solver;

    // create_dense_solver(size, solver);
    // std::unique_ptr<Solver<double>> solver = solver_factory(size);
    //std::cout << solver->A.values[0] << std::endl;

    //delete solver; // why does this not work???

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
}

bool test_sparse_lu()
{
    int size = 6;
    int nnzs = 12;

    int init_row_position[] = {0, 4, 5, 7, 9, 11, 12};
    int init_col_index[] = {0, 2, 3, 4, 1, 0, 2, 0, 3, 0, 4, 5};
    double init_sparse_values[] = {5., 1., 1., 1., 5., 1., 5., 1., 5., 1., 5., 5.};

    std::vector<double> b = {17., 10., 16., 21., 26., 30.};
    std::vector<int> perm = {0, 1, 2, 3, 4, 5};

    CSRMatrix<double> sparse_matrix = CSRMatrix<double>(size, size, nnzs, &init_sparse_values[0], &init_row_position[0], &init_col_index[0]);
    CSRMatrix<double> LU = CSRMatrix<double>(size, size, nnzs, &init_sparse_values[0], &init_row_position[0], &init_col_index[0]);
    SparseSolver<double> sparse_solver = SparseSolver<double>(sparse_matrix, b);
    std::vector<double> x(size, 0);

    clock_t t = clock();

    std::shared_ptr<CSRMatrix<double>> new_ptr = sparse_solver.lu_decomp(LU);
    sparse_solver.lu_solve(*new_ptr, perm, x);
    t = clock() - t;

    double expected[] = {1, 2, 3, 4, 5, 6};

    for (int i = 0; i < size; i++)
    {
        if (expected[i] != x[i])
        {
            std::cout << expected[i] << " does not match " << x[i] << std::endl;
            return false;
        }
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
    test_runner.test(&test_lu_dense_random, "dense LU with random matrices.");
    test_runner.test(&test_sparse_lu, "sparse LU decomposition.");
}