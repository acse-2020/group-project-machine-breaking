#include "TestRunner.h"
#include <iostream>
#include <string>
#include <math.h>

// Colourful text output (1)
#define RED "\033[31m"
#define GREEN "\033[32m"
#define RESET "\033[0m"
#define BLUE "\033[34m"

TestRunner::TestRunner(std::string new_title) : title(new_title)
{
}

TestRunner::~TestRunner()
{
    this->completeRun();
}

void TestRunner::test(bool (*test_ptr)(), std::string title)
{
    int total = this->testsFailed + this->testsSucceeded;

    // title displayed before any terminal outputs
    std::cout << std::endl
              << BLUE << "Test " << total + 1 << ": " << title << RESET << std::endl;

    bool outcome = test_ptr();

    if (outcome)
    {
        this->testsSucceeded += 1;
        std::cout << GREEN << "Passed" << RESET << std::endl;
    }
    else
    {
        this->testsFailed += 1;
        std::cout << RED << "Failed" << RESET << std::endl;
    }
}

void TestRunner::completeRun()
{
    int total = this->testsFailed + this->testsSucceeded;
    std::cout << std::endl;
    if (this->testsFailed == 0)
    {
        std::cout << GREEN << this->title << ": " << total << "/" << total << " tests passed." << RESET << std::endl;
    }
    else
    {
        std::cout << RED << this->title << ": "
                  << " " << this->testsFailed << "/" << total << " tests failed." << RESET << std::endl
                  << std::endl;
    }
}

bool TestRunner::assertArrays(double *arr1, double *arr2, int length)
{
    // helper function to check whether all values in arrays are the same
    for (int i = 0; i < length; i++)
    {
        if (arr1[i] != arr2[i])
        {
            std::cerr << RED << "Values do not match: " << arr1[i] << " != " << arr2[i] << RESET << std::endl;
            return false;
        }
    }
    return true;
}

void TestRunner::testError(std::string message)
{
    std::cerr << RED << message << RESET << std::endl;
}

bool TestRunner::assertArrays(int *arr1, int *arr2, int length)
{
    // ideally this would be a template function to avoid duplication
    for (int i = 0; i < length; i++)
    {
        if (arr1[i] != arr2[i])
        {
            std::cerr << RED << "Values do not match: " << arr1[i] << " != " << arr2[i] << RESET << std::endl;
            return false;
        }
    }
    return true;
}

bool TestRunner::assertBelowTolerance(double val, double tol)
{

    if (isnan(val))
    {
        TestRunner::testError("Something went wrong");
        return false;
    }

    if (val > tol)
    {
        TestRunner::testError("Value is above tolerance");
        return false;
    }
    return true;
}
