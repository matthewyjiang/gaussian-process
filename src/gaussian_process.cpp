#include "gaussian_process.h"
#include <exception>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace gp {

GaussianProcess::GaussianProcess(std::unique_ptr<KernelBase> kernel,
                                double noise_variance)
    : kernel_(std::move(kernel)), noise_variance_(noise_variance), is_fitted_(false) {
  if (noise_variance <= 0.0) {
    throw std::invalid_argument("Noise variance must be positive");
  }
}

void GaussianProcess::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
  if (X.rows() != y.size()) {
    throw std::invalid_argument("X and y must have the same number of rows");
  }
  if (X.rows() == 0) {
    throw std::invalid_argument("Training data cannot be empty");
  }

  X_train_ = X;
  y_train_ = y;
  
  compute_alpha();
  is_fitted_ = true;
}

void GaussianProcess::add_data_point(const Eigen::VectorXd& X_new, const double y_new) {
  if (!is_fitted_) {
    X_train_ = X_new.transpose();
    y_train_ = Eigen::VectorXd(1);
    y_train_(0) = y_new;
    fit(X_train_, y_train_);
    return;
  }

  if (X_new.size() != X_train_.cols()) {
      throw std::invalid_argument("New point must have same dimensionality as training data");
  }

  const int n = X_train_.rows();
  X_train_.conservativeResize(n + 1, Eigen::NoChange);
  X_train_.row(n) = X_new.transpose();

  y_train_.conservativeResize(n + 1);
  y_train_(n) = y_new;

  compute_alpha();
}

void GaussianProcess::add_data_points(const Eigen::MatrixXd& X_new, const Eigen::VectorXd& y_new) {
  if (!is_fitted_) {
    fit(X_new, y_new);
    return;
  }

  if (X_new.rows() != y_new.size()) {
    throw std::invalid_argument("X_new and y_new must have the same number of rows");
  }

  const int n = X_train_.rows();
  const int m = X_new.rows();
  X_train_.conservativeResize(n + m, Eigen::NoChange);
  X_train_.bottomRows(m) = X_new;
  

  y_train_.conservativeResize(n + m);
  y_train_.tail(m) = y_new;

  compute_alpha();

}

void GaussianProcess::compute_alpha() {
  const int n = X_train_.rows();
  
  // Compute covariance matrix K
  Eigen::MatrixXd K = kernel_->compute(X_train_);
  
  // Add noise to diagonal
  K.diagonal().array() += noise_variance_;
  
  // Compute Cholesky decomposition for numerical stability
  Eigen::LLT<Eigen::MatrixXd> llt(K);
  if (llt.info() != Eigen::Success) {
    throw std::runtime_error("Cholesky decomposition failed - matrix not positive definite");
  }
  
  // Solve K * alpha = y
  alpha_ = llt.solve(y_train_);
  
  // Store inverse for predictions
  K_inv_ = llt.solve(Eigen::MatrixXd::Identity(n, n));
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
GaussianProcess::predict(const Eigen::MatrixXd &X_test, bool return_std) const {
  if (!is_fitted_) {
    throw std::runtime_error("GP must be fitted before making predictions");
  }
  if (X_test.cols() != X_train_.cols()) {
    throw std::invalid_argument("X_test must have same number of features as training data");
  }

  const int n_test = X_test.rows();
  
  // Compute cross-covariance K_star
  Eigen::MatrixXd K_star = kernel_->compute(X_train_, X_test);
  
  // Compute mean predictions
  Eigen::VectorXd y_mean = K_star.transpose() * alpha_;
  
  Eigen::VectorXd y_std;
  if (return_std) {
    // Compute test covariance matrix
    Eigen::MatrixXd K_star_star = kernel_->compute(X_test);
    
    // Compute predictive variance
    Eigen::MatrixXd v = K_inv_ * K_star;
    Eigen::VectorXd y_var(n_test);
    
    for (int i = 0; i < n_test; ++i) {
      y_var(i) = K_star_star(i, i) - K_star.col(i).transpose() * v.col(i);
      // Add noise variance for predictive uncertainty
      y_var(i) += noise_variance_;
      // Ensure non-negative variance
      y_var(i) = std::max(y_var(i), 1e-12);
    }
    
    y_std = y_var.cwiseSqrt();
  }
  
  return std::make_pair(y_mean, y_std);
}

double GaussianProcess::log_marginal_likelihood() const {
  if (!is_fitted_) {
    throw std::runtime_error("GP must be fitted before computing log marginal likelihood");
  }

  const int n = X_train_.rows();
  
  // Compute K + noise*I
  Eigen::MatrixXd K = kernel_->compute(X_train_);
  K.diagonal().array() += noise_variance_;
  
  // Compute Cholesky decomposition
  Eigen::LLT<Eigen::MatrixXd> llt(K);
  if (llt.info() != Eigen::Success) {
    throw std::runtime_error("Cholesky decomposition failed");
  }
  
  // Log marginal likelihood = -0.5 * y^T * K^{-1} * y - 0.5 * log|K| - 0.5 * n * log(2π)
  double log_likelihood = -0.5 * y_train_.transpose() * alpha_;
  
  // Compute log determinant using Cholesky decomposition
  Eigen::MatrixXd L = llt.matrixL();
  double log_det = 2.0 * L.diagonal().array().log().sum();
  log_likelihood -= 0.5 * log_det;
  
  // Add normalization constant
  log_likelihood -= 0.5 * n * std::log(2.0 * M_PI);
  
  return log_likelihood;
}

double GaussianProcess::compute_log_determinant(const Eigen::MatrixXd &K) const {
  Eigen::LLT<Eigen::MatrixXd> llt(K);
  if (llt.info() != Eigen::Success) {
    throw std::runtime_error("Matrix is not positive definite");
  }
  Eigen::MatrixXd L = llt.matrixL();
  return 2.0 * L.diagonal().array().log().sum();
}

void GaussianProcess::optimize_hyperparameters(const std::vector<std::vector<double>>& param_grid) {
  if (!is_fitted_) {
    throw std::runtime_error("GP must be fitted before optimizing hyperparameters");
  }
  
  double best_likelihood = -std::numeric_limits<double>::infinity();
  std::vector<double> best_params;
  
  for (const auto& params : param_grid) {
    if (params.size() != kernel_->num_params()) {
      throw std::invalid_argument("Parameter vector size must match kernel parameter count");
    }
    
    try {
      // Set kernel parameters
      kernel_->set_params(params);
      
      // Recompute alpha with new parameters
      compute_alpha();
      
      // Compute log marginal likelihood
      double likelihood = log_marginal_likelihood();
      
      if (likelihood > best_likelihood) {
        best_likelihood = likelihood;
        best_params = params;
      }
    } catch (const std::exception&) {
      // Skip invalid parameter combinations
      continue;
    }
  }
  
  if (best_params.empty()) {
    throw std::runtime_error("No valid parameter combinations found");
  }
  
  // Set best parameters
  kernel_->set_params(best_params);
  compute_alpha();
}

} // namespace gp
