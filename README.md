# ACSE-5 Group Project

## Prerequisites

This library requires a compiler with C++17 feature support.

## Matrix

Matrix is a template class, so the values can be of any type T. The matrix values are stored in form of a dynamically allocated array. The memory is managed by a shared pointer. The constructor requires the number of rows, number of columns, and optionally a shared pointer to the values array.

### Properties

- `rows`(`int`): number of rows
- `cols`(`int`) : number of columns
- `values`(`std::shared_ptr<T[]>`): Pointer to an array of the non-zero values

### Methods
- `Matrix<T> &operator=(const Matrix<T> &M2)`

## CSRMatrix

This class is a derived class of Matrix. The `values` property only contains the non-zero elements in the matrix.

### Additional Properties

- `nnzs`(`int`): The number of non-zero values in the matrix
- `col_index`(`std::shared_ptr<int[]>`): This array has the same length as `values`. Each element is the column index of the corresponding value.
- `row_position`(`std::shared_ptr<int[]>`): Pointer to array of size (rows + 1), the value in this array is the index of col_index at which the respective row starts. The last number in this array doesn't directly relate to a value in col_index, but it denotes the end of values in the last row

### Methods
- `virtual void print2DMatrix()`
- `std::shared_ptr<CSRMatrix<T>> matMatMult(CSRMatrix<T> &mat_right)`
- `std::shared_ptr<CSRMatrix<T>> transpose()`

## Solver

This class implements multiple algorithms to solve the equation `A`**`x`**`=`**`b`**. The different solver methods will return a shared pointer to the unknown **`x`**.

### Properties

- `A`(`Matrix<T>`): A matrix object
- **`b`**(`std::vector<T>`): a vector representing the right hand side of the equation.

### Usage

e.g. to solve a linear system using the Jacobi algorithm:

```cpp
double tol = 1e-6; // tolerance for a correct result
int it_max = 1000; // maximum iterations
// define A and b beforehand
auto solver = Solver<double>(A, b);
solver.stationaryIterative(x, tol, it_max, false);
```
### Methods
- `void stationaryIterative(std::vector<T> &x, double &tol, int &it_max, bool isGaussSeidel)`
- `T residualCalc(std::vector<T> &x, std::vector<T> &b_estimate)`
- `std::vector<int> lu_decomp(Matrix<T> &LU)`
- `void lu_solve(Matrix<T> &LU, std::vector<int> &piv, std::vector<T> &x)`

It is also possible to create a solver with random values in `A` and **`b`**, by calling the constructor with a `size` argument only:

```cpp
// This will create a solver class with a random 100x100 matrix A
auto random_solver = Solver<double>(100);
```

## SparseSolver

This class is similar in structure to `Solver`, but it implements algorithms to solve the equation `A`**`x`**`=`**`b`** for a _sparse matrix_ `A` of type `CSRMatrix<T>`,

### Methods
- `void stationaryIterative(std::vector<T> &x, double &tol, int &it_max, bool isGaussSeidel)`
- `T residualCalc(std::vector<T> &x, std::vector<T> &b_estimate)`
- `void conjugateGradient(std::vector<T> &x, double &tol, int &it_max)`
- `std::shared_ptr<CSRMatrix<T> > cholesky_decomp()`
- `void cholesky_solve(CSRMatrix<T> &R, std::vector<T> &x)`


## Test framework

### TestRunner

The `TestRunner` class keeps track of how many tests have passed and how many have failed. The class has a `test` method which prints the status of each test to the terminal. When the destructor is called, a summary is presented. It also includes static helper methods for testing.

```cpp
TestRunner test_runner = TestRunner();
```

### Adding tests

To add a unit test, create a function with a bool return and no arguments in `tests.h`.

```cpp
bool test_function1() {
    /* run some code */
    return result == expected;
}
```

In `run_tests()`, call the `TestRunner::test()` method with the memory address of the test function and a title which will be displayed to the user when the tests are run.

```cpp
test_runner.test(&test_function1, "Description of first test");
test_runner.test(&test_function2, "Description of second test");
```

## References

[1] Colourful text output:
<https://stackoverflow.com/questions/9158150/colored-output-in-c>

## License

[MIT](https://choosealicense.com/licenses/mit/)
