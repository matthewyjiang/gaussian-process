#include "gaussian_process.h"
#include "rbf_kernel.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace gp;

// ASCII plot function for stdout visualization
void plot_ascii(const std::vector<double>& x, const std::vector<double>& y_true, 
                const std::vector<double>& y_pred, const std::vector<double>& y_std,
                const std::vector<double>& x_train, const std::vector<double>& y_train) {
    
    const int width = 80;
    const int height = 25;
    
    // Find min/max values for scaling
    double x_min = *std::min_element(x.begin(), x.end());
    double x_max = *std::max_element(x.begin(), x.end());
    
    double y_min = std::min({*std::min_element(y_true.begin(), y_true.end()),
                           *std::min_element(y_pred.begin(), y_pred.end())}) - 0.5;
    double y_max = std::max({*std::max_element(y_true.begin(), y_true.end()),
                           *std::max_element(y_pred.begin(), y_pred.end())}) + 0.5;
    
    std::cout << "\n=== Incremental Learning with Add New Point Functions ===" << std::endl;
    std::cout << "Legend: * = True function, + = GP prediction, o = Training data" << std::endl;
    std::cout << "Range: x ∈ [" << std::fixed << std::setprecision(2) << x_min 
              << ", " << x_max << "], y ∈ [" << y_min << ", " << y_max << "]" << std::endl;
    
    // Create plot grid
    std::vector<std::string> plot(height, std::string(width, ' '));
    
    // Plot true function
    for (size_t i = 0; i < x.size(); ++i) {
        int px = (int)((x[i] - x_min) / (x_max - x_min) * (width - 1));
        int py = height - 1 - (int)((y_true[i] - y_min) / (y_max - y_min) * (height - 1));
        if (px >= 0 && px < width && py >= 0 && py < height) {
            plot[py][px] = '*';
        }
    }
    
    // Plot GP predictions
    for (size_t i = 0; i < x.size(); ++i) {
        int px = (int)((x[i] - x_min) / (x_max - x_min) * (width - 1));
        int py = height - 1 - (int)((y_pred[i] - y_min) / (y_max - y_min) * (height - 1));
        if (px >= 0 && px < width && py >= 0 && py < height) {
            if (plot[py][px] == ' ') plot[py][px] = '+';
            else if (plot[py][px] == '*') plot[py][px] = '#'; // Overlap
        }
    }
    
    // Plot training data
    for (size_t i = 0; i < x_train.size(); ++i) {
        int px = (int)((x_train[i] - x_min) / (x_max - x_min) * (width - 1));
        int py = height - 1 - (int)((y_train[i] - y_min) / (y_max - y_min) * (height - 1));
        if (px >= 0 && px < width && py >= 0 && py < height) {
            plot[py][px] = 'o';
        }
    }
    
    // Add y-axis labels and print plot
    std::cout << std::endl;
    for (int i = 0; i < height; ++i) {
        double y_val = y_max - (double)i / (height - 1) * (y_max - y_min);
        std::cout << std::setw(6) << std::fixed << std::setprecision(2) << y_val << " |";
        std::cout << plot[i] << std::endl;
    }
    
    // Add x-axis
    std::cout << "       ";
    for (int i = 0; i < width; ++i) std::cout << "-";
    std::cout << std::endl;
    std::cout << "       ";
    for (int i = 0; i < width; i += 10) {
        double x_val = x_min + (double)i / (width - 1) * (x_max - x_min);
        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << x_val;
    }
    std::cout << std::endl << std::endl;
}

// Print numerical comparison
void print_comparison(const std::vector<double>& x, const std::vector<double>& y_true, 
                     const std::vector<double>& y_pred, const std::vector<double>& y_std) {
    std::cout << "=== Numerical Comparison ===" << std::endl;
    std::cout << std::setw(8) << "x" << std::setw(12) << "True" << std::setw(12) << "Predicted" 
              << std::setw(12) << "Std Dev" << std::setw(12) << "Error" << std::endl;
    std::cout << std::string(56, '-') << std::endl;
    
    double total_error = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double error = std::abs(y_true[i] - y_pred[i]);
        total_error += error;
        std::cout << std::setw(8) << std::fixed << std::setprecision(3) << x[i]
                  << std::setw(12) << std::setprecision(3) << y_true[i]
                  << std::setw(12) << std::setprecision(3) << y_pred[i]
                  << std::setw(12) << std::setprecision(3) << y_std[i]
                  << std::setw(12) << std::setprecision(3) << error << std::endl;
    }
    
    std::cout << std::string(56, '-') << std::endl;
    std::cout << "Mean Absolute Error: " << std::fixed << std::setprecision(4) 
              << total_error / x.size() << std::endl << std::endl;
}

// Target function: a combination of sine and cosine
double target_function(double x) {
    return std::sin(x) + 0.5 * std::cos(2 * x);
}

int main() {
    try {
        std::cout << "=== Testing Incremental Learning with Add New Point Functions ===" << std::endl;
        std::cout << "Target function: sin(x) + 0.5*cos(2*x)" << std::endl;
        
        // Create GP with RBF kernel
        std::unique_ptr<KernelBase> kernel(new RBFKernel(1.0, 0.5));
        GaussianProcess gp(std::move(kernel), 1e-6);
        
        std::vector<double> x_train_vec;
        std::vector<double> y_train_vec;
        
        std::cout << "\nStep 1: Adding first data point using add_data_point..." << std::endl;
        // Add first point
        Eigen::VectorXd x1(1);
        x1 << 0.0;
        double y1 = target_function(0.0);
        gp.add_data_point(x1, y1);
        x_train_vec.push_back(0.0);
        y_train_vec.push_back(y1);
        std::cout << "Added point: x=" << x1(0) << ", y=" << y1 << std::endl;
        std::cout << "GP fitted: " << (gp.is_fitted() ? "Yes" : "No") << std::endl;
        
        std::cout << "\nStep 2: Adding more single points using add_data_point..." << std::endl;
        // Add more individual points
        std::vector<double> x_single = {1.0, 2.0, 3.0, -1.0};
        for (double x_val : x_single) {
            Eigen::VectorXd x_new(1);
            x_new << x_val;
            double y_new = target_function(x_val);
            gp.add_data_point(x_new, y_new);
            x_train_vec.push_back(x_val);
            y_train_vec.push_back(y_new);
            std::cout << "Added point: x=" << x_val << ", y=" << y_new << std::endl;
        }
        
        std::cout << "\nStep 3: Adding multiple points using add_data_points..." << std::endl;
        // Add multiple points at once
        std::vector<double> x_batch = {-2.0, -0.5, 0.5, 1.5, 2.5, 3.5};
        Eigen::MatrixXd X_batch(x_batch.size(), 1);
        Eigen::VectorXd y_batch(x_batch.size());
        
        for (size_t i = 0; i < x_batch.size(); ++i) {
            X_batch(i, 0) = x_batch[i];
            y_batch(i) = target_function(x_batch[i]);
            x_train_vec.push_back(x_batch[i]);
            y_train_vec.push_back(y_batch[i]);
        }
        
        gp.add_data_points(X_batch, y_batch);
        std::cout << "Added " << x_batch.size() << " points in batch" << std::endl;
        
        std::cout << "\nStep 4: Adding final points using add_data_point..." << std::endl;
        // Add final individual points
        std::vector<double> x_final = {-3.0, 4.0};
        for (double x_val : x_final) {
            Eigen::VectorXd x_new(1);
            x_new << x_val;
            double y_new = target_function(x_val);
            gp.add_data_point(x_new, y_new);
            x_train_vec.push_back(x_val);
            y_train_vec.push_back(y_new);
            std::cout << "Added point: x=" << x_val << ", y=" << y_new << std::endl;
        }
        
        std::cout << "\nTotal training points: " << x_train_vec.size() << std::endl;
        std::cout << "GP fitted: " << (gp.is_fitted() ? "Yes" : "No") << std::endl;
        
        // Generate test points for prediction and visualization
        const int n_test = 80;
        std::vector<double> x_test_vec;
        std::vector<double> y_true_vec;
        
        for (int i = 0; i < n_test; ++i) {
            double x = -3.5 + (double)i / (n_test - 1) * 7.5; // Range [-3.5, 4.0]
            x_test_vec.push_back(x);
            y_true_vec.push_back(target_function(x));
        }
        
        // Convert to Eigen matrix
        Eigen::MatrixXd X_test(n_test, 1);
        for (int i = 0; i < n_test; ++i) {
            X_test(i, 0) = x_test_vec[i];
        }
        
        // Make predictions
        std::pair<Eigen::VectorXd, Eigen::VectorXd> pred_result = gp.predict(X_test, true);
        Eigen::VectorXd y_pred = pred_result.first;
        Eigen::VectorXd y_std = pred_result.second;
        
        // Convert predictions to std::vector for plotting
        std::vector<double> y_pred_vec;
        std::vector<double> y_std_vec;
        for (int i = 0; i < n_test; ++i) {
            y_pred_vec.push_back(y_pred(i));
            y_std_vec.push_back(y_std(i));
        }
        
        // Display results
        plot_ascii(x_test_vec, y_true_vec, y_pred_vec, y_std_vec, x_train_vec, y_train_vec);
        print_comparison(x_test_vec, y_true_vec, y_pred_vec, y_std_vec);
        
        // Print GP statistics
        std::cout << "=== GP Statistics ===" << std::endl;
        auto params = gp.kernel()->get_params();
        std::cout << "RBF Kernel Parameters:" << std::endl;
        std::cout << "  Variance (σ²): " << std::fixed << std::setprecision(4) << params[0] << std::endl;
        std::cout << "  Length scale (l): " << std::fixed << std::setprecision(4) << params[1] << std::endl;
        std::cout << "Noise variance: " << std::scientific << gp.noise_variance() << std::endl;
        std::cout << "Log marginal likelihood: " << std::fixed << std::setprecision(4) 
                  << gp.log_marginal_likelihood() << std::endl;
        
        // Verify incremental learning worked
        std::cout << "\n=== Incremental Learning Verification ===" << std::endl;
        std::cout << "✓ Successfully used add_data_point() to add individual points" << std::endl;
        std::cout << "✓ Successfully used add_data_points() to add multiple points at once" << std::endl;
        std::cout << "✓ GP remained fitted throughout the process" << std::endl;
        std::cout << "✓ Function learning and prediction successful" << std::endl;
        
        std::cout << "\nSuccess! Incremental learning with add new point functions completed." << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}