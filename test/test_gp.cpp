#include "gaussian_process.h"
#include "rbf_kernel.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

using namespace gp;

// Simple test framework
class TestFramework {
private:
    int passed_ = 0;
    int failed_ = 0;
    
public:
    void assert_true(bool condition, const std::string& test_name) {
        if (condition) {
            std::cout << "[PASS] " << test_name << std::endl;
            passed_++;
        } else {
            std::cout << "[FAIL] " << test_name << std::endl;
            failed_++;
        }
    }
    
    void assert_near(double a, double b, double tolerance, const std::string& test_name) {
        bool condition = std::abs(a - b) < tolerance;
        if (condition) {
            std::cout << "[PASS] " << test_name << " (|" << a << " - " << b << "| < " << tolerance << ")" << std::endl;
            passed_++;
        } else {
            std::cout << "[FAIL] " << test_name << " (|" << a << " - " << b << "| = " << std::abs(a - b) << " >= " << tolerance << ")" << std::endl;
            failed_++;
        }
    }
    
    void summary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Passed: " << passed_ << std::endl;
        std::cout << "Failed: " << failed_ << std::endl;
        std::cout << "Total:  " << (passed_ + failed_) << std::endl;
        if (failed_ == 0) {
            std::cout << "All tests passed!" << std::endl;
        }
    }
    
    int get_failed_count() const { return failed_; }
};

// Test data generation
std::pair<Eigen::MatrixXd, Eigen::VectorXd> generate_test_data(int n_samples = 20) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<> dis(-3.0, 3.0);
    
    Eigen::MatrixXd X(n_samples, 1);
    Eigen::VectorXd y(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        X(i, 0) = dis(gen);
        // True function: sin(x) + noise
        y(i) = std::sin(X(i, 0)) + 0.1 * std::normal_distribution<>(0, 1)(gen);
    }
    
    return std::make_pair(X, y);
}

void test_rbf_kernel(TestFramework& test) {
    std::cout << "\n=== Testing RBF Kernel ===" << std::endl;
    
    // Test kernel creation
    RBFKernel kernel(1.0, 1.0);
    test.assert_near(kernel.variance(), 1.0, 1e-10, "RBF kernel variance initialization");
    test.assert_near(kernel.lengthscale(), 1.0, 1e-10, "RBF kernel lengthscale initialization");
    
    // Test parameter setting
    kernel.set_params({2.0, 0.5});
    test.assert_near(kernel.variance(), 2.0, 1e-10, "RBF kernel variance setting");
    test.assert_near(kernel.lengthscale(), 0.5, 1e-10, "RBF kernel lengthscale setting");
    
    // Test kernel computation
    Eigen::MatrixXd X(2, 1);
    X << 0.0, 1.0;
    
    Eigen::MatrixXd K = kernel.compute(X, X);
    test.assert_near(K(0, 0), 2.0, 1e-10, "RBF kernel diagonal element");
    test.assert_near(K(1, 1), 2.0, 1e-10, "RBF kernel diagonal element");
    test.assert_true(K(0, 1) == K(1, 0), "RBF kernel symmetry");
    
    // Test that closer points have higher covariance
    double expected_01 = 2.0 * std::exp(-0.5 * 1.0 / 0.25); // dist=1, lengthscale=0.5
    test.assert_near(K(0, 1), expected_01, 1e-10, "RBF kernel off-diagonal element");
}

void test_gaussian_process_basic(TestFramework& test) {
    std::cout << "\n=== Testing Gaussian Process Basic Functionality ===" << std::endl;
    
    // Create GP with RBF kernel
    std::unique_ptr<KernelBase> kernel(new RBFKernel(1.0, 1.0));
    GaussianProcess gp(std::move(kernel), 0.1);
    
    test.assert_true(!gp.is_fitted(), "GP not fitted initially");
    test.assert_near(gp.noise_variance(), 0.1, 1e-10, "GP noise variance initialization");
    
    // Generate simple test data
    Eigen::MatrixXd X_train(3, 1);
    Eigen::VectorXd y_train(3);
    X_train << 0.0, 1.0, 2.0;
    y_train << 0.0, 1.0, 0.0;
    
    // Fit GP
    gp.fit(X_train, y_train);
    test.assert_true(gp.is_fitted(), "GP fitted after fit() call");
    
    // Test log marginal likelihood computation
    double log_ml = gp.log_marginal_likelihood();
    test.assert_true(std::isfinite(log_ml), "Log marginal likelihood is finite");
    
    // Test predictions
    Eigen::MatrixXd X_test(2, 1);
    X_test << 0.5, 1.5;
    
    std::pair<Eigen::VectorXd, Eigen::VectorXd> pred_result = gp.predict(X_test, true);
    Eigen::VectorXd y_mean = pred_result.first;
    Eigen::VectorXd y_std = pred_result.second;
    test.assert_true(y_mean.size() == 2, "Prediction mean size correct");
    test.assert_true(y_std.size() == 2, "Prediction std size correct");
    test.assert_true((y_std.array() > 0).all(), "Prediction std positive");
    
    // Test prediction without std
    std::pair<Eigen::VectorXd, Eigen::VectorXd> pred_result_no_std = gp.predict(X_test, false);
    Eigen::VectorXd y_mean_only = pred_result_no_std.first;
    Eigen::VectorXd y_std_empty = pred_result_no_std.second;
    test.assert_true(y_mean_only.size() == 2, "Prediction mean only size correct");
    test.assert_true(y_std_empty.size() == 0, "Prediction std empty when not requested");
}

void test_gaussian_process_regression(TestFramework& test) {
    std::cout << "\n=== Testing Gaussian Process Regression ===" << std::endl;
    
    // Generate test data
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> train_data = generate_test_data(10);
    Eigen::MatrixXd X_train = train_data.first;
    Eigen::VectorXd y_train = train_data.second;
    
    // Create GP with RBF kernel
    std::unique_ptr<KernelBase> kernel(new RBFKernel(1.0, 1.0));
    GaussianProcess gp(std::move(kernel), 0.01);
    
    // Fit GP
    gp.fit(X_train, y_train);
    
    // Test interpolation property: predictions at training points should be close to training targets
    std::pair<Eigen::VectorXd, Eigen::VectorXd> interp_result = gp.predict(X_train, true);
    Eigen::VectorXd y_pred = interp_result.first;
    Eigen::VectorXd y_std = interp_result.second;
    
    double max_error = (y_pred - y_train).cwiseAbs().maxCoeff();
    test.assert_true(max_error < 0.5, "GP interpolation reasonably accurate");
    
    // Test that uncertainty is lower at training points
    Eigen::MatrixXd X_test(1, 1);
    X_test << 10.0; // Far from training data
    std::pair<Eigen::VectorXd, Eigen::VectorXd> far_result = gp.predict(X_test, true);
    Eigen::VectorXd y_far = far_result.first;
    Eigen::VectorXd y_std_far = far_result.second;
    
    double avg_train_std = y_std.mean();
    test.assert_true(y_std_far(0) > avg_train_std, "Higher uncertainty far from training data");
}

void test_hyperparameter_optimization(TestFramework& test) {
    std::cout << "\n=== Testing Hyperparameter Optimization ===" << std::endl;
    
    // Generate test data
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> train_data = generate_test_data(15);
    Eigen::MatrixXd X_train = train_data.first;
    Eigen::VectorXd y_train = train_data.second;
    
    // Create GP with RBF kernel
    std::unique_ptr<KernelBase> kernel(new RBFKernel(0.5, 0.5));
    GaussianProcess gp(std::move(kernel), 0.01);
    
    // Fit GP
    gp.fit(X_train, y_train);
    
    // Get initial log marginal likelihood
    double initial_log_ml = gp.log_marginal_likelihood();
    
    // Define parameter grid for optimization
    std::vector<std::vector<double>> param_grid;
    for (double var : {0.1, 0.5, 1.0, 2.0}) {
        for (double length : {0.1, 0.5, 1.0, 2.0}) {
            param_grid.push_back({var, length});
        }
    }
    
    // Optimize hyperparameters
    gp.optimize_hyperparameters(param_grid);
    
    // Get optimized log marginal likelihood
    double optimized_log_ml = gp.log_marginal_likelihood();
    
    test.assert_true(optimized_log_ml >= initial_log_ml - 1e-10, "Hyperparameter optimization improves or maintains likelihood");
    
    // Test that optimized parameters are within the grid
    auto params = gp.kernel()->get_params();
    test.assert_true(params.size() == 2, "Optimized parameters have correct size");
    test.assert_true(params[0] >= 0.1 && params[0] <= 2.0, "Optimized variance within grid range");
    test.assert_true(params[1] >= 0.1 && params[1] <= 2.0, "Optimized lengthscale within grid range");
}

void test_error_handling(TestFramework& test) {
    std::cout << "\n=== Testing Error Handling ===" << std::endl;
    
    // Test invalid kernel parameters
    try {
        RBFKernel kernel(-1.0, 1.0);
        test.assert_true(false, "Should throw exception for negative variance");
    } catch (const std::exception&) {
        test.assert_true(true, "Throws exception for negative variance");
    }
    
    try {
        RBFKernel kernel(1.0, -1.0);
        test.assert_true(false, "Should throw exception for negative lengthscale");
    } catch (const std::exception&) {
        test.assert_true(true, "Throws exception for negative lengthscale");
    }
    
    // Test invalid GP parameters
    try {
        std::unique_ptr<KernelBase> kernel(new RBFKernel(1.0, 1.0));
        GaussianProcess gp(std::move(kernel), -0.1);
        test.assert_true(false, "Should throw exception for negative noise variance");
    } catch (const std::exception&) {
        test.assert_true(true, "Throws exception for negative noise variance");
    }
    
    // Test prediction before fitting
    std::unique_ptr<KernelBase> kernel(new RBFKernel(1.0, 1.0));
    GaussianProcess gp(std::move(kernel), 0.1);
    
    Eigen::MatrixXd X_test(1, 1);
    X_test << 0.0;
    
    try {
        std::pair<Eigen::VectorXd, Eigen::VectorXd> pred_result = gp.predict(X_test);
        Eigen::VectorXd y_mean = pred_result.first;
        Eigen::VectorXd y_std = pred_result.second;
        test.assert_true(false, "Should throw exception for prediction before fitting");
    } catch (const std::exception&) {
        test.assert_true(true, "Throws exception for prediction before fitting");
    }
    
    // Test mismatched dimensions
    Eigen::MatrixXd X_train(2, 1);
    Eigen::VectorXd y_train(3);
    X_train << 0.0, 1.0;
    y_train << 0.0, 1.0, 2.0;
    
    try {
        gp.fit(X_train, y_train);
        test.assert_true(false, "Should throw exception for mismatched dimensions");
    } catch (const std::exception&) {
        test.assert_true(true, "Throws exception for mismatched dimensions");
    }
}

int main() {
    std::cout << "=== Gaussian Process Library Test Suite ===" << std::endl;
    
    TestFramework test;
    
    try {
        test_rbf_kernel(test);
        test_gaussian_process_basic(test);
        test_gaussian_process_regression(test);
        test_hyperparameter_optimization(test);
        test_error_handling(test);
        
        test.summary();
        
        return test.get_failed_count();
        
    } catch (const std::exception& e) {
        std::cerr << "Test suite failed with exception: " << e.what() << std::endl;
        return 1;
    }
}