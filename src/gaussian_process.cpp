#include "gaussian_process.h"
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
  Eigen::MatrixXd X_matrix = X_new.transpose();
  Eigen::VectorXd y_vector(1);
  y_vector(0) = y_new;
  
  if (!is_fitted_) {
    fit(X_matrix, y_vector);
    return;
  }

  validate_new_data(X_matrix, y_vector);

  const int n = X_train_.rows();
  X_train_.conservativeResize(n + 1, Eigen::NoChange);
  X_train_.row(n) = X_new.transpose();

  y_train_.conservativeResize(n + 1);
  y_train_(n) = y_new;

  update_alpha_incremental(X_new, y_new);
}

void GaussianProcess::add_data_points(const Eigen::MatrixXd& X_new, const Eigen::VectorXd& y_new) {
  if (!is_fitted_) {
    fit(X_new, y_new);
    return;
  }

  if (X_new.rows() != y_new.size()) {
    throw std::invalid_argument("X_new and y_new must have the same number of rows");
  }

  if (X_new.cols() != X_train_.cols()) {
    throw std::invalid_argument("X_new and X_train must have same number of columns");
  }

  const int n = X_train_.rows();
  const int m = X_new.rows();
  X_train_.conservativeResize(n + m, Eigen::NoChange);
  X_train_.bottomRows(m) = X_new;

  y_train_.conservativeResize(n + m);
  y_train_.tail(m) = y_new;

  update_alpha_batch(X_new, y_new);
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

void GaussianProcess::update_alpha_incremental(const Eigen::VectorXd& x_new, double y_new) {
  const int n = X_train_.rows() - 1; // Size before adding new point
  
  // Compute cross-covariance between new point and existing points
  Eigen::MatrixXd X_old = X_train_.topRows(n);
  Eigen::VectorXd k_star = kernel_->compute(X_old, x_new.transpose());
  
  // Compute variance of new point
  Eigen::MatrixXd x_new_matrix = x_new.transpose();
  double k_star_star = kernel_->compute(x_new_matrix, x_new_matrix)(0, 0) + noise_variance_;
  
  // Sherman-Morrison formula for inverse update
  // K_inv_new = [K_inv + (K_inv * k * k^T * K_inv) / (k_star_star - k^T * K_inv * k), -K_inv * k / c]
  //             [-k^T * K_inv / c,                                                    1 / c]
  // where c = k_star_star - k^T * K_inv * k
  
  Eigen::VectorXd K_inv_k = K_inv_ * k_star;
  double c = k_star_star - k_star.transpose() * K_inv_k;
  
  if (std::abs(c) < 1e-12) {
    // Fallback to full recomputation if numerically unstable
    compute_alpha();
    return;
  }
  
  // Update K_inv using block matrix formula
  Eigen::MatrixXd K_inv_new(n + 1, n + 1);
  K_inv_new.topLeftCorner(n, n) = K_inv_ + (K_inv_k * K_inv_k.transpose()) / c;
  K_inv_new.topRightCorner(n, 1) = -K_inv_k / c;
  K_inv_new.bottomLeftCorner(1, n) = -K_inv_k.transpose() / c;
  K_inv_new(n, n) = 1.0 / c;
  
  K_inv_ = K_inv_new;
  
  // Update alpha incrementally
  // alpha_new = K_inv_new * y_train_
  alpha_ = K_inv_ * y_train_;
}

void GaussianProcess::update_alpha_batch(const Eigen::MatrixXd& X_new, const Eigen::VectorXd& y_new) {
  // For batch updates, use block matrix inversion formula
  // This is more efficient than repeated Sherman-Morrison updates
  const int n = X_train_.rows() - X_new.rows(); // Original size
  const int m = X_new.rows(); // Number of new points
  
  if (m == 1) {
    // Use single point method for efficiency
    update_alpha_incremental(X_new.row(0).transpose(), y_new(0));
    return;
  }
  
  // For larger batches, fall back to full recomputation
  // Block matrix updates become complex and may not provide significant speedup
  // for small to medium sized batches due to overhead
  if (m > n / 4) {
    compute_alpha();
    return;
  }
  
  // Compute cross-covariances
  Eigen::MatrixXd X_old = X_train_.topRows(n);
  Eigen::MatrixXd K_star = kernel_->compute(X_old, X_new);
  Eigen::MatrixXd K_star_star = kernel_->compute(X_new);
  K_star_star.diagonal().array() += noise_variance_;
  
  // Block matrix inversion using Schur complement
  // [A  B]^-1 = [A^-1 + A^-1*B*S^-1*B^T*A^-1,  -A^-1*B*S^-1]
  // [B^T C]     [-S^-1*B^T*A^-1,                 S^-1      ]
  // where S = C - B^T*A^-1*B (Schur complement)
  
  Eigen::MatrixXd K_inv_k_star = K_inv_ * K_star;
  Eigen::MatrixXd S = K_star_star - K_star.transpose() * K_inv_k_star;
  
  // Check if Schur complement is well-conditioned
  Eigen::LLT<Eigen::MatrixXd> llt_S(S);
  if (llt_S.info() != Eigen::Success) {
    // Fallback to full recomputation
    compute_alpha();
    return;
  }
  
  Eigen::MatrixXd S_inv = llt_S.solve(Eigen::MatrixXd::Identity(m, m));
  
  // Update K_inv using block matrix formula
  Eigen::MatrixXd K_inv_new(n + m, n + m);
  K_inv_new.topLeftCorner(n, n) = K_inv_ + K_inv_k_star * S_inv * K_inv_k_star.transpose();
  K_inv_new.topRightCorner(n, m) = -K_inv_k_star * S_inv;
  K_inv_new.bottomLeftCorner(m, n) = -S_inv * K_inv_k_star.transpose();
  K_inv_new.bottomRightCorner(m, m) = S_inv;
  
  K_inv_ = K_inv_new;
  
  // Update alpha
  alpha_ = K_inv_ * y_train_;
}

void GaussianProcess::validate_new_data(const Eigen::MatrixXd& X_new, const Eigen::VectorXd& y_new) const {
  if (X_new.rows() != y_new.size()) {
    throw std::invalid_argument("X_new and y_new must have the same number of rows");
  }
  if (X_new.rows() == 0) {
    throw std::invalid_argument("Cannot add empty data");
  }
  if (is_fitted_ && X_new.cols() != X_train_.cols()) {
    throw std::invalid_argument("X_new must have same number of columns as training data");
  }
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
