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

```cpp
#include "gaussian_process.h"
#include "rbf_kernel.h"

auto kernel = std::make_unique<RBFKernel>(1.0, 1.0);
GaussianProcess gp(std::move(kernel), 1e-6);

gp.fit(X_train, y_train);
auto [y_pred, y_std] = gp.predict(X_test);
```

## Tests

```bash
./gp_test
./test_sine_rbf
```