#include <iostream>
#include <vector>
// This header file includes functions that do not fall
// under the scope of the Matrix or Solver classes

template <typename T>
void printVector(std::vector<T> vec)
{
    std::cout << "Vector is: \n";
    for (int i = 0; i < vec.size(); i++)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << "\n";
}