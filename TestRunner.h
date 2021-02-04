#pragma once
#include <iostream>

class TestRunner
{
private:
    int testsFailed = 0;
    int testsSucceeded = 0;

    void completeRun();

public:
    void test(bool (*fun_ptr)(), std::string title);
    TestRunner();
    ~TestRunner();

    static bool assertArrays(int *arr1, int *arr2, int length);
    static bool assertArrays(double *arr1, double *arr2, int length);
};
