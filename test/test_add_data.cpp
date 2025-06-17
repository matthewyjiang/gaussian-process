#include "gaussian_process.h"
#include "rbf_kernel.h"
#include <iostream>
#include <cassert>

using namespace gp;

int main() {
    std::cout << "Testing add data point functions..." << std::endl;
    
    // Create GP with RBF kernel
    std::unique_ptr<KernelBase> kernel(new RBFKernel(1.0, 1.0));
    GaussianProcess gp(std::move(kernel), 0.1);
    
    // Test add single data point (not fitted)
    Eigen::VectorXd x1(1);
    x1 << 0.0;
    gp.add_data_point(x1, 1.0);
    assert(gp.is_fitted());
    std::cout << "[PASS] Added first data point" << std::endl;
    
    // Test add single data point (already fitted)
    Eigen::VectorXd x2(1);
    x2 << 1.0;
    gp.add_data_point(x2, 2.0);
    std::cout << "[PASS] Added second data point" << std::endl;
    
    // Test add multiple data points
    Eigen::MatrixXd X_new(2, 1);
    Eigen::VectorXd y_new(2);
    X_new << 2.0, 3.0;
    y_new << 3.0, 4.0;
    gp.add_data_points(X_new, y_new);
    std::cout << "[PASS] Added multiple data points" << std::endl;
    
    // Test prediction works after adding data
    Eigen::MatrixXd X_test(1, 1);
    X_test << 0.5;
    auto pred = gp.predict(X_test);
    assert(pred.first.size() == 1);
    assert(pred.second.size() == 1);
    std::cout << "[PASS] Prediction works after adding data" << std::endl;
    
    // Test error handling - wrong dimensions
    try {
        Eigen::VectorXd x_wrong(2);
        x_wrong << 1.0, 2.0;
        gp.add_data_point(x_wrong, 5.0);
        assert(false);
    } catch (const std::exception&) {
        std::cout << "[PASS] Correctly throws error for wrong dimensions" << std::endl;
    }
    
    std::cout << "All add data point tests passed!" << std::endl;
    return 0;
}