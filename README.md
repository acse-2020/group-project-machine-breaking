# ACSE-5 Group Project

## Testing

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
