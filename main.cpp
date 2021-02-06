#include <iostream>
#include "tests.h"
#include <memory>

int main()
{
    std::string answer;
    std::cout << "Would you like to run tests? Type 'yes' or 'no' and hit enter." << std::endl;
    std::cout << ">> ";
    std::cin >> answer;

    if (answer == "yes")
    {
        run_tests();
    }
    else
    {
        std::cout << "Ok, bye." << std::endl;
    }

    // int nnzs = 26;
    // int init_row_position1[] = {0, 3, 6, 9, 13, 17, 20, 22, 26};
    // int init_col_index1[] = {0, 4, 7, 1, 3, 4, 2, 3, 7, 1, 2, 3, 6, 0, 1, 4, 5, 4, 5, 7, 3, 6, 0, 2, 5, 7};
    // double init_sparse_values1[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    // int nnzs = 7;
    // int init_row_position1[] = {0, 3, 5, 7};
    // int init_col_index1[] = {0, 1, 2, 0, 1, 0, 2};
    // double init_sparse_values1[] = {25, 15, -5, 15, 18, -5, 11};

    // int nnzs = 14;
    // int init_row_position1[] = {0, 3, 6, 10, 14};
    // int init_col_index1[] = {0, 2, 3, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    // double init_sparse_values1[] = {9, -27, 18, 9, -9, -27, -27, -9, 99, -27, 18, -27, -27, 121};

    int nnzs = 9;
    int size = 3;

    std::shared_ptr<int[]> init_row_position1(new int[size + 1]{0, 3, 6, 9});
    std::shared_ptr<int[]> init_col_index1(new int[nnzs]{0, 1, 2, 0, 1, 2, 0, 1, 2});
    std::shared_ptr<double[]> init_sparse_values1(new double[nnzs]{6, 15, 55, 15, 55, 225, 55, 225, 979});

    CSRMatrix<double> sparse_matrix1 = CSRMatrix<double>(3, 3, nnzs, init_sparse_values1, init_row_position1, init_col_index1);

    sparse_matrix1.printMatrix();
    sparse_matrix1.print2DMatrix();

    std::vector<double> b(3, 0);
    std::cout << "b: ";
    for (int i = 0; i < b.size(); i++)
    {
        b[i] = i + 1;
        std::cout << " " << b[i] << " ";
    }
    std::cout << "\n";

    // CSRMatrix<double> transposed_M1 = sparse_matrix1.transpose();
    // transposed_M1.printMatrix();
    // transposed_M1.print2DMatrix();

    // int init_row_position2[] = {0, 0, 2, 3, 4};
    // int init_col_index2[] = {0, 2, 3, 2};
    // double init_sparse_values2[] = {1, 1, 2, 1};
    // auto sparse_matrix2 = CSRMatrix<double>(4, 4, nnzs, &init_sparse_values2[0], &init_row_position2[0], &init_col_index2[0]);
    // sparse_matrix2.print2DMatrix();

    // CSRMatrix<double> transposed_M2 = sparse_matrix2.transpose();
    // transposed_M2.printMatrix();
    // transposed_M2.print2DMatrix();
    // CSRMatrix<double> R = CSRMatrix<double>(3, 3, nnzs, &init_sparse_values1[0], &init_row_position1[0], &init_col_index1[0]);

    SparseSolver<double> sparse_solver = SparseSolver<double>(sparse_matrix1, b);

    std::shared_ptr<CSRMatrix<double>> R = sparse_solver.cholesky_decomp();
    std::vector<double> x(3, 0);
    sparse_solver.cholesky_solve(*R, x);

    R->printMatrix();
    R->print2DMatrix();

    std::cout << "Result: ";
    for (int i = 0; i < x.size(); i++)
    {
        std::cout << " " << x[i] << " ";
    }
    std::cout << "\n";
}