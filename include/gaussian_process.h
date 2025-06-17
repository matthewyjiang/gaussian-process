#ifndef _GAUSSIAN_PROCESS_H_
#define _GAUSSIAN_PROCESS_H_

#include "kernel_base.h"
#include <Eigen/Dense>
#include <memory>

namespace gp {

class GaussianProcess {
private:
  std::unique_ptr<KernelBase> kernel_;
  Eigen::MatrixXd X_train_;
  Eigen::VectorXd y_train_;
  Eigen::MatrixXd K_inv_;
  Eigen::VectorXd alpha_;
  double noise_variance_;
  bool is_fitted_;

  // Helper methods
  void compute_alpha();
  void update_alpha_incremental(const Eigen::VectorXd& x_new, double y_new);
  void update_alpha_batch(const Eigen::MatrixXd& X_new, const Eigen::VectorXd& y_new);
  double compute_log_determinant(const Eigen::MatrixXd &K) const;
  void validate_new_data(const Eigen::MatrixXd& X_new, const Eigen::VectorXd& y_new) const;

public:
  explicit GaussianProcess(std::unique_ptr<KernelBase> kernel,
                          double noise_variance = 1e-6);

  // Fit the GP to training data
  void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
  void add_data_point(const Eigen::VectorXd &X_new, const double y_new);
  void add_data_points(const Eigen::MatrixXd &X_new, const Eigen::VectorXd &y_new);

  // Make predictions
  std::pair<Eigen::VectorXd, Eigen::VectorXd>
  predict(const Eigen::MatrixXd &X_test, bool return_std = true) const;

  // Compute log marginal likelihood
  double log_marginal_likelihood() const;

  // Getters and setters
  double noise_variance() const { return noise_variance_; }
  void set_noise_variance(double noise_variance) { 
    noise_variance_ = noise_variance; 
    if (is_fitted_) {
      compute_alpha();
    }
  }

  KernelBase* kernel() const { return kernel_.get(); }
  bool is_fitted() const { return is_fitted_; }

  // Optimize hyperparameters (simple grid search)
  void optimize_hyperparameters(const std::vector<std::vector<double>>& param_grid);
};

} // namespace gp

#endif // _GAUSSIAN_PROCESS_H_
