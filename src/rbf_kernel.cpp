#include "rbf_kernel.h"
#include <cmath>
#include <stdexcept>

namespace gp {

RBFKernel::RBFKernel(double variance, double lengthscale)
    : variance_(variance), lengthscale_(lengthscale) {
  if (variance <= 0.0) {
    throw std::invalid_argument("Variance must be positive");
  }
  if (lengthscale <= 0.0) {
    throw std::invalid_argument("Lengthscale must be positive");
  }
}

Eigen::MatrixXd RBFKernel::compute(const Eigen::MatrixXd &X1,
                                   const Eigen::MatrixXd &X2) const {
  const int n1 = X1.rows();
  const int n2 = X2.rows();
  const int dim = X1.cols();

  if (X2.cols() != dim) {
    throw std::invalid_argument("X1 and X2 must have same number of columns");
  }

  Eigen::MatrixXd K(n1, n2);
  const double lengthscale_sq = lengthscale_ * lengthscale_;

  // Compute squared distances and apply RBF kernel
  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < n2; ++j) {
      double sq_dist = (X1.row(i) - X2.row(j)).squaredNorm();
      K(i, j) = variance_ * std::exp(-0.5 * sq_dist / lengthscale_sq);
    }
  }

  return K;
}

std::vector<double> RBFKernel::get_params() const {
  return {variance_, lengthscale_};
}

void RBFKernel::set_params(const std::vector<double> &params) {
  if (params.size() != 2) {
    throw std::invalid_argument("RBF kernel expects exactly 2 parameters");
  }
  if (params[0] <= 0.0 || params[1] <= 0.0) {
    throw std::invalid_argument("RBF kernel parameters must be positive");
  }
  variance_ = params[0];
  lengthscale_ = params[1];
}

size_t RBFKernel::num_params() const { return 2; }

std::vector<Eigen::MatrixXd>
RBFKernel::compute_gradients(const Eigen::MatrixXd &X1,
                             const Eigen::MatrixXd &X2) const {
  const int n1 = X1.rows();
  const int n2 = X2.rows();
  const int dim = X1.cols();

  if (X2.cols() != dim) {
    throw std::invalid_argument("X1 and X2 must have same number of columns");
  }

  std::vector<Eigen::MatrixXd> gradients(2);
  gradients[0] = Eigen::MatrixXd(n1, n2); // d/d(variance)
  gradients[1] = Eigen::MatrixXd(n1, n2); // d/d(lengthscale)

  const double lengthscale_sq = lengthscale_ * lengthscale_;
  const double lengthscale_cubed = lengthscale_sq * lengthscale_;

  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < n2; ++j) {
      double sq_dist = (X1.row(i) - X2.row(j)).squaredNorm();
      double exp_term = std::exp(-0.5 * sq_dist / lengthscale_sq);

      // Gradient w.r.t. variance: K / variance
      gradients[0](i, j) = exp_term;

      // Gradient w.r.t. lengthscale: K * sq_dist / lengthscale^3
      gradients[1](i, j) = variance_ * exp_term * sq_dist / lengthscale_cubed;
    }
  }

  return gradients;
}

} // namespace gp
