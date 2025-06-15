# Gaussian Process Library

C++ implementation of Gaussian Process regression with RBF kernel.

## Dependencies

- CMake (≥3.17)
- Eigen3 (≥3.3)
- C++11 compiler

## Build

```bash
mkdir build && cd build
cmake .. && make
```

## Usage

### Overview

### Example with C++11

Below is an example of how to use the library in a C++11 environment:

```cpp
#include "gaussian_process.h"
#include "rbf_kernel.h"
#include <iostream>

int main() {
    // Define training data (X_train: Eigen::MatrixXd, y_train: Eigen::VectorXd)
    Eigen::MatrixXd X_train(3, 1);
    X_train << 1.0, 2.0, 3.0;
    Eigen::VectorXd y_train(3);
    y_train << 2.0, 4.0, 6.0;

    // Define test data (X_test: Eigen::MatrixXd)
    Eigen::MatrixXd X_test(2, 1);
    X_test << 1.5, 2.5;

    // Create RBF kernel with length scale = 1.0 and variance = 1.0
    std::unique_ptr<gp::RBFKernel> kernel(new gp::RBFKernel(1.0, 1.0));

    // Initialize Gaussian Process with the kernel and noise level = 1e-6
    gp::GaussianProcess gp(std::move(kernel), 1e-6);

    // Fit the model to the training data
    gp.fit(X_train, y_train);

    // Predict on the test data
    auto predictions = gp.predict(X_test);
    Eigen::VectorXd y_pred = predictions.first;
    Eigen::VectorXd y_std = predictions.second;

    // Output predictions
    for (size_t i = 0; i < y_pred.size(); ++i) {
        std::cout << "Prediction: " << y_pred[i] << ", Std Dev: " << y_std[i] << std::endl;
    }

    return 0;
}
```

## Tests

```bash
./gp_test
./test_sine_rbf
```