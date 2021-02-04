# ACSE-5 Group Project

## Testing

### TestRunner

The `TestRunner` class keeps track of how many tests have passed and how many have failed. The class has a `test` method which prints the status of each test to the terminal. When the destructor is called, a summary is presented. It also includes static helper methods for testing.

### Adding tests

To add a unit test, create a function with a bool return and no arguments in `tests.h`. In `run_tests`, call the `TestRunner::test()` method with the memory address of the test function and a title which will be displayed to the user when the tests are run.

## References

[1] Colourful text output:
<https://stackoverflow.com/questions/9158150/colored-output-in-c>
