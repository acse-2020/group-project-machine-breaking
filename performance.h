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
#include "utilities.h"

void performance_dense_jacobi_and_gauss_seidl(int minsize, int maxsize)
{
    double tol = 1e-6;
    int it_max = 10000;
    std::string filename_j, filename_gs;
    filename_j = "data/jacobi_dense_range_" + std::to_string(minsize) + "-" + std::to_string(maxsize) + ".txt";
    filename_gs = "data/gauss_dense_range_" + std::to_string(minsize) + "-" + std::to_string(maxsize) + ".txt";
    std::ofstream myfile_j;
    std::ofstream myfile_gs;
    myfile_j.open(filename_j);
    myfile_gs.open(filename_gs);

    int size = minsize;
    while (size <= maxsize)
    {
        std::vector<double> x_j(size, 0);
        std::vector<double> x_gs(size, 0);
        std::vector<double> output_b(size, 0);
        auto *solver = new Solver<double>(size);

        // Time Jacobi
        auto t1 = std::chrono::high_resolution_clock::now();
        solver->stationaryIterative(x_j, tol, it_max, false);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "Dense Jacobi for size " << size << ", time = " << duration << " s " << std::endl;

        if (solver->residualCalc(x_j, output_b) > 1e-6)
        {
            throw "Jacobi residual is above 1e-6";
        }

        // Write to file
        if (myfile_j.is_open())
        {
            myfile_j << size << "," << duration << std::endl;
        }
        else
            std::cout << "Unable to open file";

        // Time Gauss Seidel
        t1 = std::chrono::high_resolution_clock::now();
        solver->stationaryIterative(x_gs, tol, it_max, true);
        t2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "Dense Gauss-Seidel for size " << size << ", time = " << duration << " s " << std::endl;

        if (solver->residualCalc(x_gs, output_b) > 1e-6)
        {
            throw "Gauss-Seidl residual is above 1e-6";
        }

        // Write to file
        if (myfile_gs.is_open())
        {
            myfile_gs << size << "," << duration << std::endl;
        }
        else
            std::cout << "Unable to open file";

        // Delete objects to save memory usage
        delete solver;
        x_j.clear();
        x_gs.clear();
        output_b.clear();
        size *= 2;
    }
    myfile_j.close();
    myfile_gs.close();
}

void performance_lu_dense(int minsize, int maxsize)
{
    std::string filename;
    filename = "data/LU_dense_range_" + std::to_string(minsize) + "-" + std::to_string(maxsize) + ".txt";
    std::ofstream myfile;
    myfile.open(filename);
    int size = minsize;
    while (size <= maxsize)
    {
        std::vector<double> x(size, 0);
        std::vector<double> b_output(size, 0);
        auto *solver = new Solver<double>(size);

        auto *LU = new Matrix<double>(size, size, true);

        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<int> piv = solver->lu_decomp(*LU);
        solver->lu_solve(*LU, piv, x);
        auto t2 = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "LU for size " << size << ", time = " << duration << " s " << std::endl;

        // Write to file
        if (myfile.is_open())
        {
            myfile << size << "," << duration << std::endl;
        }
        else
            std::cout << "Unable to open file";

        if (solver->residualCalc(x, b_output) > 1e-6)
        {
            throw "LU residual is above 1e-6";
        }
        // Delete objects to save memory usage
        delete solver;
        delete LU;
        size *= 2;
    }
    myfile.close();
}

void run_performance()
{
    int minsize = 100;
    int maxsize = 1000;

    performance_lu_dense(minsize, maxsize);
    performance_dense_jacobi_and_gauss_seidl(minsize, maxsize);
}