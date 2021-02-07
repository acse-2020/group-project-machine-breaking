#pragma once
#include <iostream>
#include <string>

class TestRunner
{
private:
    // keeps track of every time test method is run & the outcome
    int testsFailed = 0;
    int testsSucceeded = 0;

    // called when the test run finishes - gives summary of outcomes
    void completeRun();

public:
    TestRunner();
    ~TestRunner();

    // runs the function pointed at by test_ptr
    void test(bool (*test_ptr)(), std::string title);

    // static helper methods that can be used in a test function 
    static bool assertArrays(int *arr1, int *arr2, int length);
    static bool assertArrays(double *arr1, double *arr2, int length);
    static void testError(std::string message);
};
