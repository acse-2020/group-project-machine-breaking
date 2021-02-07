#include <iostream>
#include "tests.h"
#include "performance.h"
#include <memory>

int main()
{
    std::string answer;
    std::cout << std::endl
              << "Options: " << std::endl
              << "--------" << std::endl
              << "- test [t]" << std::endl
              << "- performance [p]"
              << std::endl
              << std::endl
              << "Input the option and hit enter." << std::endl;
    std::cout << ">> ";
    std::cin >> answer;

    if (answer == "test" || answer == "t")
    {
        run_tests();
    }
    else if (answer == "performance" || answer == "p")
    {
        run_performance();
    }
    else
    {
        std::cout << "Input not recognised, exiting" << std::endl;
    }
}