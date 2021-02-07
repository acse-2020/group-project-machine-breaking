#pragma once
#include <iostream>
#include <math.h>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>
#include "Matrix.h"
#include "CSRMatrix.h"
#include "Solver.h"
#include "SparseSolver.h"
#include "TestRunner.h"


bool performance_lu_dense(int minsize, int maxsize)
{
    std::string filename;
    filename = "LU_dense_range_" + std::to_string(minsize) + "-" + std::to_string(maxsize) + ".txt";
    std::ofstream myfile;
    myfile.open(filename);
    int size = minsize;
    while (size <= maxsize)
    {
        std::vector<double> x(size, 0);
        std::vector<double> b_output(size, 0);
        auto solver = Solver<double>(size);

        Matrix<double> LU(size, size, true);

        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<int> piv = solver.lu_decomp(LU);
        solver.lu_solve(LU, piv, x);
        auto t2 = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "LU for size " << size << ", time = " << duration << " s " << std::endl;

        // Write to file
        if (myfile.is_open())
        {
            myfile << size << "," << duration << std::endl;
        }
        else std::cout << "Unable to open file";

        if (solver.residualCalc(x, b_output) > 1e-6)
        {
            TestRunner::testError("LU residual is above 1e-6");
            return false;
        }
        size *= 2;
    }
    myfile.close();
}


void run_performance()
{
    int minsize = 100;
    int maxsize = 1000;

    performance_lu_dense(minsize, maxsize);

}