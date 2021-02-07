#pragma once
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


bool performance_lu_dense(int minsize, int maxsize)
{
    std::vector<double> x(size, 0);
    std::vector<double> b_output(size, 0);

    while (size <= maxsize)
    {
        auto solver = Solver<double>(size);

        Matrix<double> LU(size, size, true);

        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<int> piv = solver.lu_decomp(LU);
        solver.lu_solve(LU, piv, x);
        auto t2 = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "LU for size " << size << ", time = " << duration << " s " << std::endl;

        if (solver.residualCalc(x, b_output) > 1e-6)
        {
            TestRunner::testError("LU residual is above 1e-6");
            return false;
        }
        size *= 2;
    }



}


void run_performance()
{
    int minsize = 100;
    int maxsize = 1000;

    performance_lu_dense(minsize, maxsize);

}